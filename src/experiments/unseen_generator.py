"""Leave-one-generator-out generalisation experiment.

###############################################################################
RESEARCH QUESTION
###############################################################################

Train on generators A/B/C and evaluate on generator D, which is absent from training
and model-selection data. A performance drop relative to the baseline estimates how
well learned cues transfer to a new generation process. Repeating D across generators
distinguishes a general pattern from one unusually easy or difficult generator.

The unseen test needs real negatives under a declared comparison policy. Keep that
real pool fixed where appropriate, and disclose that generator-wise test sets may
therefore share negatives.

###############################################################################
PARTITION POLICY
###############################################################################

The persisted GenImage split file is used unchanged so that the baseline and every
unseen/recovery run draw from identical samples. The protocol's logical partitions
are derived from it, which is what keeps them disjoint by construction:

* development train      = split ``train``, fakes restricted to the known generators
                           plus the shared real pool.
* development validation = split ``validation``, same generator restriction. Used for
                           checkpoint selection and the decision threshold.
* adaptation pool        = split ``train``, fakes from the held-out generator only.
                           Never touched by this runner; it exists for the recovery
                           experiment, which is why it must stay out of the test set.
* final unseen test      = split ``test``, fakes from the held-out generator plus the
                           fixed real test pool.

Because the importer guarantees that no ``source_group`` crosses the official
train/val boundary, the adaptation pool and the final unseen test are provenance
disjoint. Both facts are asserted at runtime rather than assumed.

The result this runner saves is the 0%-adaptation point of the recovery curve.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.datasets.detector_dataset import AIDetectionDataset
from src.datasets.schema import DatasetRecord
from src.evaluation.evaluator import collect_predictions, save_predictions
from src.evaluation.metrics import compute_binary_metrics, threshold_scores
from src.experiments.common import (
    build_data_loader,
    build_detector,
    build_training_stack,
    build_transforms,
    evaluate_records,
    finalise_run,
    load_manifest_with_splits,
    prepare_experiment,
    resolve_runtime_paths,
    select_records,
)
from src.models.checkpointing import load_checkpoint
from src.training.engine import fit
from src.utils.config import load_config

LOGGER = logging.getLogger(__name__)

# Candidate operating points considered when selecting a threshold on validation data.
THRESHOLD_GRID = tuple(index / 100 for index in range(1, 100))


def validate_unseen_protocol(config: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Confirm exactly one held-out generator that is absent from development data."""

    generators = config["generators"]
    unseen = generators.get("unseen")
    if not isinstance(unseen, str) or not unseen.strip():
        raise ValueError("generators.unseen must name exactly one held-out generator")
    if unseen == str(config["data"]["real_generator_name"]):
        raise ValueError("the held-out generator cannot be the reserved real-image name")
    known = list(generators["train"])
    if not known:
        raise ValueError("generators.train must list at least one known generator")
    if unseen in known or unseen in list(generators["validation"]):
        raise ValueError(f"held-out generator {unseen!r} appears in development generators")
    if list(generators["test"]) != [unseen]:
        raise ValueError(
            "generators.test must contain exactly the held-out generator for this protocol; "
            f"found {list(generators['test'])}"
        )
    if len(set(known)) != len(known):
        raise ValueError("generators.train contains duplicates")
    return unseen, known


def assert_unseen_absent_from_development(
    selections: Mapping[str, Sequence[DatasetRecord]], *, unseen_generator: str
) -> None:
    """Fail loudly if any held-out fake sample reached a model-development selection.

    The invariant is about what the model actually sees, not about the split file. The
    held-out generator's samples do legitimately sit inside the ``train`` split — that
    is where the adaptation pool comes from — so membership of a split proves nothing.
    What must never happen is such a sample entering the data used for gradient steps,
    checkpoint selection, early stopping, or threshold selection.
    """

    violations: dict[str, list[str]] = {}
    for name, records in selections.items():
        offenders = [
            record.sample_id
            for record in records
            if record.label == 1 and record.generator == unseen_generator
        ]
        if offenders:
            violations[name] = offenders[:10]
    if violations:
        raise ValueError(
            f"held-out generator {unseen_generator!r} leaked into development selections: "
            f"{violations}"
        )


def assert_pools_group_disjoint(
    adaptation: Sequence[DatasetRecord], final_test: Sequence[DatasetRecord]
) -> None:
    """Assert the adaptation pool and final test share no provenance group or sample."""

    adaptation_groups = {record.source_group for record in adaptation if record.source_group}
    test_groups = {record.source_group for record in final_test if record.source_group}
    shared_groups = sorted(adaptation_groups.intersection(test_groups))
    if shared_groups:
        raise ValueError(f"adaptation and final test share source groups: {shared_groups[:10]}")
    shared_ids = sorted(
        {record.sample_id for record in adaptation}.intersection(
            record.sample_id for record in final_test
        )
    )
    if shared_ids:
        raise ValueError(f"adaptation and final test share sample IDs: {shared_ids[:10]}")


def select_threshold_on_validation(
    *,
    model: Any,
    records: Sequence[DatasetRecord],
    config: Mapping[str, Any],
    device: Any,
    metric: str = "f1",
) -> tuple[float, float]:
    """Choose a decision threshold using development validation scores only.

    Returns the selected threshold and its validation metric value. This runs before
    any unseen test label is read, so the operating point cannot be tuned on the
    held-out generator.
    """

    if metric not in {"f1", "accuracy"}:
        raise ValueError("threshold selection metric must be f1 or accuracy")
    transform = build_transforms(config, training=False)
    dataset = AIDetectionDataset(list(records), transform=transform)
    loader = build_data_loader(dataset, config, training=False)
    predictions = collect_predictions(
        model=model,
        data_loader=loader,
        device=device,
        split_name="validation",
        checkpoint_id="threshold_selection",
        threshold=0.5,
    )
    labels = [item.label for item in predictions]
    scores = [item.score for item in predictions]
    best_threshold = 0.5
    best_value = float("-inf")
    for candidate in THRESHOLD_GRID:
        metrics = compute_binary_metrics(labels, scores, threshold=candidate)
        value = metrics.f1 if metric == "f1" else metrics.accuracy
        if value > best_value:
            best_threshold, best_value = float(candidate), float(value)
    # Confirm the reported threshold reproduces the scored decisions it claims to.
    threshold_scores(scores, threshold=best_threshold)
    return best_threshold, best_value


def run_unseen_generator(config_path: Path) -> Path:
    """Train without one generator and evaluate its untouched final partition."""

    loaded = load_config(config_path)
    if loaded.values["experiment"]["type"] != "unseen_generator":
        raise ValueError("run_unseen_generator requires experiment.type=unseen_generator")
    config = resolve_runtime_paths(loaded.values, loaded.source_path)
    unseen, known = validate_unseen_protocol(config)
    context = prepare_experiment(config)
    logger = logging.getLogger(f"ai_detector.{context.run_id}")
    try:
        bundle = load_manifest_with_splits(config)
        include_real = bool(config["generators"].get("include_real_images", True))
        protocol = config.get("unseen_protocol") or {}

        train_records = select_records(
            bundle.records,
            bundle.split_by_id,
            split="train",
            fake_generators=known,
            include_real=include_real,
        )
        validation_records = select_records(
            bundle.records,
            bundle.split_by_id,
            split="validation",
            fake_generators=list(config["generators"]["validation"]),
            include_real=include_real,
        )
        # The held-out generator's official-train fakes form the adaptation pool. It is
        # loaded here only to prove it stays disjoint from the final test partition.
        adaptation_pool = [
            record
            for record in bundle.records
            if record.label == 1
            and record.generator == unseen
            and bundle.split_by_id[record.sample_id] == "train"
        ]
        final_test_records = select_records(
            bundle.records,
            bundle.split_by_id,
            split="test",
            fake_generators=[unseen],
            include_real=bool(protocol.get("fixed_real_test_pool", True)) and include_real,
        )
        assert_pools_group_disjoint(adaptation_pool, final_test_records)
        # Training, checkpoint selection, early stopping, and threshold selection all
        # draw from these two selections and nothing else.
        assert_unseen_absent_from_development(
            {"train": train_records, "validation": validation_records},
            unseen_generator=unseen,
        )
        # The held-out generator's official-val fakes are deliberately unused: the final
        # test partition is drawn from that same official slice, so leaving them out
        # keeps the adaptation pool anchored to official training data.
        unused_held_out = [
            record.sample_id
            for record in bundle.records
            if record.label == 1
            and record.generator == unseen
            and bundle.split_by_id[record.sample_id] == "validation"
        ]

        logger.info(
            "unseen protocol generator=%s known=%s train=%d validation=%d "
            "adaptation_pool=%d final_unseen_test=%d manifest_sha256=%s",
            unseen,
            ",".join(known),
            len(train_records),
            len(validation_records),
            len(adaptation_pool),
            len(final_test_records),
            bundle.manifest_sha256,
        )

        train_transform = build_transforms(config, training=True)
        evaluation_transform = build_transforms(config, training=False)
        train_loader = build_data_loader(
            AIDetectionDataset(train_records, transform=train_transform), config, training=True
        )
        validation_loader = build_data_loader(
            AIDetectionDataset(validation_records, transform=evaluation_transform),
            config,
            training=False,
        )

        model = build_detector(config, device=context.device)
        epochs = int(config["training"]["epochs"])
        stack = build_training_stack(
            config,
            model,
            device=context.device,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )
        result = fit(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            optimizer=stack.optimizer,
            scheduler=stack.scheduler,
            loss_fn=stack.loss_fn,
            device=context.device,
            epochs=epochs,
            output_dir=context.run_dir,
            early_stopping=stack.early_stopping,
            scaler=stack.scaler,
            gradient_clip_norm=config["training"].get("gradient_clip_norm"),
            checkpoint_metric=str(config["training"]["checkpoint_metric"]),
            resolved_config=config,
            seed=context.seed,
        )
        best_checkpoint_path = Path(result["best_checkpoint"])
        best_checkpoint = load_checkpoint(best_checkpoint_path, map_location=str(context.device))
        model.load_state_dict(best_checkpoint["model_state"], strict=True)

        # Operating point fixed on development validation, before any unseen test label.
        selected_threshold, threshold_metric = select_threshold_on_validation(
            model=model,
            records=validation_records,
            config=config,
            device=context.device,
            metric=str(config["training"]["early_stopping"].get("metric", "f1")),
        )
        logger.info(
            "validation-selected threshold=%.2f (validation score %.4f)",
            selected_threshold,
            threshold_metric,
        )

        # Report at the configured default threshold and at the validation-selected one.
        # Neither is tuned on the held-out generator.
        outcome = evaluate_records(
            model=model,
            records=final_test_records,
            config=config,
            device=context.device,
            split_name=str(protocol.get("final_test_split_name", "unseen_test")),
            checkpoint_id=best_checkpoint_path.name,
        )
        save_predictions(outcome.predictions, context.run_dir / "unseen_test_predictions.csv")
        labels = [item.label for item in outcome.predictions]
        scores = [item.score for item in outcome.predictions]
        at_selected_threshold = compute_binary_metrics(
            labels, scores, threshold=selected_threshold
        )

        # Same trained model measured on the in-distribution test split, so the unseen
        # drop is read against a genuinely comparable number rather than a prior run.
        in_distribution_records = select_records(
            bundle.records,
            bundle.split_by_id,
            split="test",
            fake_generators=known,
            include_real=include_real,
        )
        in_distribution = evaluate_records(
            model=model,
            records=in_distribution_records,
            config=config,
            device=context.device,
            split_name="in_distribution_test",
            checkpoint_id=best_checkpoint_path.name,
        )
        save_predictions(
            in_distribution.predictions,
            context.run_dir / "in_distribution_test_predictions.csv",
        )

        payload: dict[str, Any] = {
            "protocol": "leave_one_generator_out",
            "adaptation_percentage": 0.0,
            "held_out_generator": unseen,
            "known_generators": known,
            "seed": context.seed,
            "manifest_sha256": bundle.manifest_sha256,
            "best_checkpoint": str(best_checkpoint_path),
            "best_epoch": result["best_epoch"],
            "best_validation_score": result["best_score"],
            "decision_threshold_default": float(config["model"]["decision_threshold"]),
            "decision_threshold_validation_selected": selected_threshold,
            "validation_threshold_score": threshold_metric,
            "sample_counts": {
                "train": len(train_records),
                "validation": len(validation_records),
                "adaptation_pool_unused_here": len(adaptation_pool),
                "held_out_validation_slice_deliberately_unused": len(unused_held_out),
                "final_unseen_test": len(final_test_records),
                "in_distribution_test": len(in_distribution_records),
            },
            "unseen_test": {
                "at_default_threshold": asdict(outcome.overall),
                "at_validation_selected_threshold": asdict(at_selected_threshold),
                "per_generator": {
                    name: asdict(metrics) for name, metrics in outcome.per_generator.items()
                },
            },
            "in_distribution_test": {
                "at_default_threshold": asdict(in_distribution.overall),
                "per_generator": {
                    name: asdict(metrics) for name, metrics in in_distribution.per_generator.items()
                },
            },
            "generalisation_gap": {
                "metric": "f1",
                "in_distribution": in_distribution.overall.f1,
                "unseen": outcome.overall.f1,
                "absolute_drop": in_distribution.overall.f1 - outcome.overall.f1,
            },
        }
        (context.run_dir / "unseen_generator_metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info(
            "unseen generator %s: in-distribution f1=%.4f, unseen f1=%.4f (drop %.4f)",
            unseen,
            in_distribution.overall.f1,
            outcome.overall.f1,
            in_distribution.overall.f1 - outcome.overall.f1,
        )
        finalise_run(context, status="completed")
        return context.run_dir
    except Exception:
        logger.exception("unseen generator experiment failed")
        finalise_run(context, status="failed")
        raise


# IMPLEMENTATION CHECKLIST
# [x] Assert held-out generator exclusion from every model-development partition.
# [x] Keep adaptation and final unseen test pools disjoint by provenance group.
# [x] Document and fix the real-image comparison pool.
# [x] Reuse identical model/training settings for comparable baseline and unseen runs.
# [x] Save the zero-shot/0%-adaptation prediction table for each seed/generator.
# [ ] Repeat each declared generator as held out before generalising conclusions.
#     (Driven by running this config once per held-out generator; see configs/.)
