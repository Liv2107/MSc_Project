"""Limited-data fine-tuning recovery experiment.

###############################################################################
RESEARCH QUESTION
###############################################################################

After performance on an unseen generator is measured, how much can the detector
recover using limited labelled data from that generator? The planned budgets are
5%, 10%, 20%, and 50% of a *predefined adaptation pool*, never the final test set.

Why these values:
    5% tests very-low-data adaptation; 10% tests whether modest evidence is enough;
    20% reveals whether improvement continues beyond the smallest budgets; and 50%
    provides a substantial-but-still-limited reference. Actual sample counts must be
    reported because identical percentages can represent very different evidence.

Nested subsets (5% contained in 10%, etc.) reduce one source of comparison noise.
Repeated subset seeds remain important because which examples are labelled can matter
as much as the nominal percentage.

###############################################################################
SMALL-DATA MODEL SELECTION (PREDECLARED)
###############################################################################

Limited labelled data has to cover model selection too; borrowing a large clean
validation set would measure a budget the experiment does not actually have. The
adaptation pool is therefore split ONCE, group-safely, into an adaptation-train pool
and an adaptation-validation pool. Each budget then takes nested prefixes from BOTH
pools, so:

* the reported budget covers every labelled image the run consumed, selection included;
* 5% train data is contained in 10% train data, and likewise for validation data;
* the final unseen test partition is never read during fitting or selection.

Every cell reloads the same untouched starting checkpoint. Budgets are independent
conditions, not a continual-learning sequence.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.datasets.detector_dataset import AIDetectionDataset
from src.datasets.schema import DatasetRecord
from src.evaluation.evaluator import save_predictions
from src.evaluation.metrics import compute_binary_metrics
from src.experiments.common import (
    ExperimentContext,
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
from src.experiments.unseen_generator import (
    THRESHOLD_PROVENANCE_DEFAULT,
    THRESHOLD_PROVENANCE_SEEN_VALIDATION,
    assert_pools_group_disjoint,
    assert_unseen_absent_from_development,
    build_balanced_final_test,
    select_threshold_on_validation,
    validate_unseen_protocol,
)
from src.models.checkpointing import load_checkpoint
from src.models.clip_detector import configure_trainable_layers
from src.training.engine import fit
from src.utils.config import load_config
from src.utils.reproducibility import seed_everything

LOGGER = logging.getLogger(__name__)

REQUIRED_FRACTIONS = (0.05, 0.10, 0.20, 0.50)

THRESHOLD_PROVENANCE_ADAPTATION = (
    "selected_on_adaptation_validation_only__grid_search__counts_against_budget"
)


@dataclass(frozen=True, slots=True)
class AdaptationBudget:
    """One limited-label condition and its reproducible sample identity."""

    fraction: float
    subset_seed: int
    train_sample_ids: tuple[str, ...] = ()
    validation_sample_ids: tuple[str, ...] = ()

    @property
    def train_count(self) -> int:
        return len(self.train_sample_ids)

    @property
    def validation_count(self) -> int:
        return len(self.validation_sample_ids)

    @property
    def labelled_count(self) -> int:
        """Total labelled images this condition consumed, model selection included."""
        return self.train_count + self.validation_count

    def describe(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "subset_seed": self.subset_seed,
            "adaptation_train_count": self.train_count,
            "adaptation_validation_count": self.validation_count,
            "labelled_images_consumed": self.labelled_count,
        }


GroupList = tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class AdaptationPools:
    """The one-time, group-safe division of the adaptation pool.

    Both pools stay keyed by class label. Budgets are drawn per class so that even the
    smallest percentage contains real and fake images; a single shuffled prefix over
    both classes could otherwise yield a one-class training set.
    """

    train_groups: Mapping[int, GroupList]
    validation_groups: Mapping[int, GroupList]
    by_id: Mapping[str, DatasetRecord] = field(default_factory=dict)


def _group_key(record: DatasetRecord) -> str:
    return record.source_group or f"sample:{record.sample_id}"


def _grouped_by_class(
    records: Sequence[DatasetRecord],
) -> dict[int, list[tuple[str, tuple[str, ...]]]]:
    """Collect sample IDs into provenance groups, keyed by class label."""

    per_class: dict[int, dict[str, list[str]]] = {0: {}, 1: {}}
    for record in records:
        per_class[record.label].setdefault(_group_key(record), []).append(record.sample_id)
    return {
        label: sorted(
            ((group, tuple(sorted(ids))) for group, ids in groups.items()),
            key=lambda item: item[0],
        )
        for label, groups in per_class.items()
    }


def split_adaptation_pool(
    adaptation_pool: Sequence[DatasetRecord],
    *,
    validation_fraction: float,
    seed: int,
) -> AdaptationPools:
    """Divide the adaptation pool once into train and validation pools.

    The division is group-safe and class-stratified, and happens before any budget is
    applied so that every budget draws from the same two fixed pools.
    """

    if not 0 < validation_fraction < 1:
        raise ValueError("adaptation_validation_fraction must be in (0, 1)")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    per_class = _grouped_by_class(adaptation_pool)
    if not per_class[0] or not per_class[1]:
        raise ValueError("the adaptation pool must contain both real and fake images")
    train_groups: dict[int, GroupList] = {}
    validation_groups: dict[int, GroupList] = {}
    for label in (0, 1):
        groups = list(per_class[label])
        if len(groups) < 2:
            raise ValueError(
                f"class {label} needs at least two provenance groups to split the pool"
            )
        random.Random(f"{seed}:pool:{label}").shuffle(groups)
        wanted = max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))
        validation_groups[label] = tuple(groups[:wanted])
        train_groups[label] = tuple(groups[wanted:])
    shared = {group for groups in train_groups.values() for group, _ in groups}.intersection(
        group for groups in validation_groups.values() for group, _ in groups
    )
    if shared:
        raise RuntimeError(f"adaptation pool split shares groups: {sorted(shared)[:5]}")
    return AdaptationPools(
        train_groups=train_groups,
        validation_groups=validation_groups,
        by_id={record.sample_id: record for record in adaptation_pool},
    )


def _nested_prefixes(
    groups_by_class: Mapping[int, GroupList],
    fractions: Sequence[float],
    *,
    seed: int,
    stratum: str,
) -> dict[float, list[str]]:
    """Shuffle groups once per class, then take increasing prefixes so subsets nest.

    Rounding rule: ``ceil(group_count * fraction)`` per class, floored at one group, so
    the smallest budget is never empty and never one-class. Actual counts are always
    reported alongside the nominal percentage because the two are not interchangeable.
    Selecting within each class independently preserves both nesting and group integrity.
    """

    shuffled: dict[int, list[tuple[str, tuple[str, ...]]]] = {}
    for label, groups in sorted(groups_by_class.items()):
        order = list(groups)
        random.Random(f"{seed}:{stratum}:{label}").shuffle(order)
        shuffled[label] = order
    selected: dict[float, list[str]] = {}
    for fraction in fractions:
        ids: list[str] = []
        for order in shuffled.values():
            wanted = max(1, math.ceil(len(order) * fraction))
            for _, group_ids in order[:wanted]:
                ids.extend(group_ids)
        selected[fraction] = sorted(ids)
    return selected


def build_nested_adaptation_subsets(
    *,
    adaptation_pool: Sequence[DatasetRecord],
    pools: AdaptationPools,
    fractions: Sequence[float],
    seed: int,
    final_test_records: Sequence[DatasetRecord] = (),
) -> dict[float, AdaptationBudget]:
    """Return nested sample-ID sets for fair data-budget comparison."""

    ordered = list(fractions)
    if not ordered:
        raise ValueError("at least one adaptation fraction is required")
    if any(not 0 < fraction <= 1 for fraction in ordered):
        raise ValueError("adaptation fractions must lie in (0, 1]")
    if len(set(ordered)) != len(ordered):
        raise ValueError("adaptation fractions must be unique")
    if ordered != sorted(ordered):
        raise ValueError("adaptation fractions must be given in increasing order")

    pool_ids = {record.sample_id for record in adaptation_pool}
    test_ids = {record.sample_id for record in final_test_records}
    forbidden = sorted(pool_ids.intersection(test_ids))
    if forbidden:
        raise ValueError(f"adaptation pool overlaps the final test partition: {forbidden[:10]}")
    test_groups = {_group_key(record) for record in final_test_records}
    shared_groups = sorted({_group_key(record) for record in adaptation_pool} & test_groups)
    if shared_groups:
        raise ValueError(f"adaptation pool shares final-test groups: {shared_groups[:10]}")

    train_selection = _nested_prefixes(pools.train_groups, ordered, seed=seed, stratum="train")
    validation_selection = _nested_prefixes(
        pools.validation_groups, ordered, seed=seed, stratum="validation"
    )
    budgets: dict[float, AdaptationBudget] = {}
    for fraction in ordered:
        budgets[fraction] = AdaptationBudget(
            fraction=fraction,
            subset_seed=seed,
            train_sample_ids=tuple(train_selection[fraction]),
            validation_sample_ids=tuple(validation_selection[fraction]),
        )
    # Nesting is a contract the recovery curve depends on; verify it rather than trust it.
    for smaller, larger in zip(ordered, ordered[1:], strict=False):
        for attribute in ("train_sample_ids", "validation_sample_ids"):
            small = set(getattr(budgets[smaller], attribute))
            large = set(getattr(budgets[larger], attribute))
            if not small.issubset(large):
                raise RuntimeError(
                    f"{attribute} for {smaller} is not nested inside {larger}"
                )
    for budget in budgets.values():
        overlap = set(budget.train_sample_ids).intersection(budget.validation_sample_ids)
        if overlap:
            raise RuntimeError(f"adaptation train/validation overlap: {sorted(overlap)[:5]}")
    return budgets


def save_adaptation_subsets(
    budgets: Mapping[int, Mapping[float, AdaptationBudget]], destination: Path
) -> None:
    """Persist subset IDs so ablations reuse exactly the same labelled images."""

    payload = {
        str(subset_seed): {
            f"{fraction:.2f}": {
                "fraction": budget.fraction,
                "subset_seed": budget.subset_seed,
                "adaptation_train_sample_ids": list(budget.train_sample_ids),
                "adaptation_validation_sample_ids": list(budget.validation_sample_ids),
                "adaptation_train_count": budget.train_count,
                "adaptation_validation_count": budget.validation_count,
                "labelled_images_consumed": budget.labelled_count,
            }
            for fraction, budget in sorted(per_seed.items())
        }
        for subset_seed, per_seed in sorted(budgets.items())
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_adaptation_subsets(source: Path) -> dict[int, dict[float, AdaptationBudget]]:
    """Reload persisted subset IDs, used by the ablation to control the data exactly."""

    raw = json.loads(source.read_text(encoding="utf-8"))
    restored: dict[int, dict[float, AdaptationBudget]] = {}
    for subset_seed, per_seed in raw.items():
        restored[int(subset_seed)] = {
            float(entry["fraction"]): AdaptationBudget(
                fraction=float(entry["fraction"]),
                subset_seed=int(entry["subset_seed"]),
                train_sample_ids=tuple(entry["adaptation_train_sample_ids"]),
                validation_sample_ids=tuple(entry["adaptation_validation_sample_ids"]),
            )
            for entry in per_seed.values()
        }
    return restored


def resolve_starting_checkpoint(config: Mapping[str, Any], source_path: Path) -> Path:
    """Locate the 0%-adaptation checkpoint every cell must start from."""

    settings = config.get("fine_tuning") or {}
    declared = settings.get("starting_checkpoint")
    if not declared:
        raise ValueError(
            "fine_tuning.starting_checkpoint must name the checkpoint produced by the "
            "unseen-generator run; recovery is measured relative to that 0% result"
        )
    candidate = Path(str(declared))
    if not candidate.is_absolute():
        candidate = (source_path.parent.parent / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"starting checkpoint not found: {candidate}")
    return candidate


def assert_starting_checkpoint_compatible(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    unseen_generator: str,
    known_generators: Sequence[str],
    manifest_sha256: str | None,
) -> dict[str, Any]:
    """Refuse a starting checkpoint that came from an incomparable setup."""

    stored = checkpoint.get("resolved_config") or {}
    findings: dict[str, Any] = {"warnings": []}
    stored_model = (stored.get("model") or {}).get("clip_model_name")
    if stored_model and stored_model != config["model"]["clip_model_name"]:
        raise ValueError(
            f"starting checkpoint uses CLIP model {stored_model!r}, config expects "
            f"{config['model']['clip_model_name']!r}"
        )
    stored_revision = (stored.get("model") or {}).get("clip_revision")
    if stored_revision and stored_revision != config["model"].get("clip_revision"):
        raise ValueError("starting checkpoint was trained against a different CLIP revision")
    stored_generators = stored.get("generators") or {}
    stored_unseen = stored_generators.get("unseen")
    if stored_unseen and stored_unseen != unseen_generator:
        raise ValueError(
            f"starting checkpoint held out {stored_unseen!r}, this run adapts to "
            f"{unseen_generator!r}"
        )
    stored_known = list(stored_generators.get("train") or [])
    if stored_known and sorted(stored_known) != sorted(known_generators):
        raise ValueError(
            "starting checkpoint trained on different known generators: "
            f"{sorted(stored_known)} vs {sorted(known_generators)}"
        )
    stored_manifest = (stored.get("data") or {}).get("manifest_path")
    if stored_manifest and str(stored_manifest) != str(config["data"]["manifest_path"]):
        findings["warnings"].append(
            f"starting checkpoint manifest path {stored_manifest!r} differs from this run's"
        )
    findings["starting_checkpoint_metadata"] = dict(checkpoint.get("metadata") or {})
    findings["manifest_sha256"] = manifest_sha256
    return findings


def _records_for_ids(
    pools: AdaptationPools, ids: Sequence[str], *, description: str
) -> list[DatasetRecord]:
    missing = [identifier for identifier in ids if identifier not in pools.by_id]
    if missing:
        raise ValueError(f"{description} references unknown adaptation IDs: {missing[:10]}")
    return [pools.by_id[identifier] for identifier in ids]


@dataclass(frozen=True, slots=True)
class CellResult:
    """One (freeze mode, budget, subset seed, training seed) measurement."""

    record: dict[str, Any]
    predictions_path: Path


def run_adaptation_cell(
    *,
    config: Mapping[str, Any],
    context: ExperimentContext,
    pools: AdaptationPools,
    budget: AdaptationBudget,
    final_test_records: Sequence[DatasetRecord],
    starting_checkpoint_path: Path,
    fine_tune_mode: str,
    training_seed: int,
    cell_id: str,
    baseline_threshold: float,
    baseline_threshold_provenance: str,
    retain_cell_checkpoints: bool = False,
    learning_rate: float | None = None,
    epochs: int | None = None,
) -> CellResult:
    """Fine-tune one independent condition and score the fixed final test partition.

    Always reloads ``starting_checkpoint_path`` so budgets stay independent conditions
    rather than a continual-learning chain.
    """

    logger = logging.getLogger(f"ai_detector.{context.run_id}")
    cell_dir = context.run_dir / "cells" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=False)

    seed_everything(
        training_seed, deterministic=bool(config["reproducibility"]["deterministic_algorithms"])
    )
    train_records = _records_for_ids(pools, budget.train_sample_ids, description="adaptation train")
    validation_records = _records_for_ids(
        pools, budget.validation_sample_ids, description="adaptation validation"
    )
    for name, records in (("train", train_records), ("validation", validation_records)):
        present_classes = {record.label for record in records}
        if present_classes != {0, 1}:
            raise ValueError(
                f"adaptation {name} at fraction {budget.fraction} holds classes "
                f"{present_classes}; the budget is too small to contain both classes"
            )
    assert_pools_group_disjoint(train_records + validation_records, final_test_records)
    # The adaptation pool mixes held-out-generator fakes with shared authentic images, so
    # a bare percentage is ambiguous. Record both counts: the held-out count is the one a
    # reader means by "labelled images from the new generator".
    budget_records = train_records + validation_records
    held_out_fake_count = sum(1 for record in budget_records if record.label == 1)
    authentic_count = sum(1 for record in budget_records if record.label == 0)

    model = build_detector(config, device=context.device, fine_tune_mode=fine_tune_mode)
    starting = load_checkpoint(starting_checkpoint_path, map_location=str(context.device))
    model.load_state_dict(starting["model_state"], strict=True)
    # Freeze policy is applied AFTER loading so the optimiser below only ever receives
    # parameters this mode is allowed to change.
    configure_trainable_layers(model, fine_tune_mode)

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
    resolved_epochs = int(config["training"]["epochs"] if epochs is None else epochs)
    stack = build_training_stack(
        config,
        model,
        device=context.device,
        steps_per_epoch=len(train_loader),
        epochs=resolved_epochs,
        learning_rate=learning_rate,
    )
    started = time.perf_counter()
    result = fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=stack.optimizer,
        scheduler=stack.scheduler,
        loss_fn=stack.loss_fn,
        device=context.device,
        epochs=resolved_epochs,
        output_dir=cell_dir,
        early_stopping=stack.early_stopping,
        scaler=stack.scaler,
        gradient_clip_norm=config["training"].get("gradient_clip_norm"),
        checkpoint_metric=str(config["training"]["checkpoint_metric"]),
        resolved_config=config,
        seed=training_seed,
        logger=logger,
        progress_label=f"cell[{cell_id}]",
    )
    training_seconds = time.perf_counter() - started
    best_path = Path(result["best_checkpoint"])
    best = load_checkpoint(best_path, map_location=str(context.device))
    model.load_state_dict(best["model_state"], strict=True)

    # Operating point selected on adaptation validation ONLY. Those images are part of
    # this budget's labelled allocation, so the threshold costs nothing that is not
    # already counted in labelled_images_consumed, and the final test stays untouched.
    adaptation_threshold, adaptation_threshold_score = select_threshold_on_validation(
        model=model,
        records=validation_records,
        config=config,
        device=context.device,
        metric=str(config["training"]["early_stopping"].get("metric", "f1")),
    )

    outcome = evaluate_records(
        model=model,
        records=final_test_records,
        config=config,
        device=context.device,
        split_name=str(
            (config.get("fine_tuning") or {}).get("final_test_split_name", "unseen_test")
        ),
        checkpoint_id=f"{cell_id}:{best_path.name}",
    )
    predictions_path = cell_dir / "unseen_test_predictions.csv"
    save_predictions(outcome.predictions, predictions_path)

    # Three operating points on the same saved scores, so a change in F1 between budgets
    # can be attributed to weight adaptation or to calibration rather than confounding
    # the two. `overall` stays the config default for continuity with earlier runs.
    test_labels = [item.label for item in outcome.predictions]
    test_scores = [item.score for item in outcome.predictions]
    at_adaptation_threshold = compute_binary_metrics(
        test_labels, test_scores, threshold=adaptation_threshold
    )
    at_baseline_threshold = compute_binary_metrics(
        test_labels, test_scores, threshold=baseline_threshold
    )

    checkpoint_digests = {
        path.name: path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").split()[0]
        for path in (best_path, Path(result["last_checkpoint"]))
        if path.with_suffix(path.suffix + ".sha256").is_file()
    }
    record = {
        "cell_id": cell_id,
        "fine_tune_mode": fine_tune_mode,
        "adaptation_percentage": budget.fraction,
        "subset_seed": budget.subset_seed,
        "training_seed": training_seed,
        "starting_checkpoint": str(starting_checkpoint_path),
        **budget.describe(),
        "held_out_fake_count": held_out_fake_count,
        "authentic_count": authentic_count,
        "best_epoch": result["best_epoch"],
        "best_adaptation_validation_score": result["best_score"],
        "trainable_parameters": model.trainability_summary.get("trainable_parameters"),
        "total_parameters": model.trainability_summary.get("total_parameters"),
        "trainable_parameter_names": model.trainability_summary.get("trainable_parameter_names"),
        "epochs": resolved_epochs,
        "learning_rate": float(
            config["training"]["learning_rate"] if learning_rate is None else learning_rate
        ),
        "training_seconds": training_seconds,
        "cell_checkpoint_sha256": checkpoint_digests,
        "thresholds": {
            "default": {
                "value": float(config["model"]["decision_threshold"]),
                "provenance": THRESHOLD_PROVENANCE_DEFAULT,
            },
            "adaptation_validation_selected": {
                "value": adaptation_threshold,
                "provenance": THRESHOLD_PROVENANCE_ADAPTATION,
                "selection_score": adaptation_threshold_score,
                "selection_sample_count": len(validation_records),
                "counted_in_adaptation_budget": True,
            },
            "baseline_unchanged": {
                "value": baseline_threshold,
                "provenance": baseline_threshold_provenance,
                "held_out_samples_used": 0,
            },
        },
        "overall": asdict(outcome.overall),
        "at_adaptation_selected_threshold": asdict(at_adaptation_threshold),
        "at_baseline_threshold": asdict(at_baseline_threshold),
        "per_generator": {
            name: asdict(metrics) for name, metrics in outcome.per_generator.items()
        },
    }
    if not retain_cell_checkpoints:
        # Each cell checkpoint is a full CLIP copy (~350 MB); a full grid would run to
        # tens of gigabytes. Predictions, metrics, and the checkpoint digest above are
        # what the results depend on, so the weights themselves are released here.
        for path in (best_path, Path(result["last_checkpoint"])):
            path.unlink(missing_ok=True)
        logger.info("released cell checkpoints for %s (digests retained)", cell_id)
    return CellResult(record=record, predictions_path=predictions_path)


def evaluate_starting_checkpoint(
    *,
    config: Mapping[str, Any],
    context: ExperimentContext,
    starting_checkpoint_path: Path,
    final_test_records: Sequence[DatasetRecord],
    fine_tune_mode: str,
    seen_validation_records: Sequence[DatasetRecord],
) -> tuple[dict[str, Any], float]:
    """Measure the 0% condition inside this run, with identical evaluation code.

    Recomputing it here rather than reading another run's JSON guarantees the recovery
    curve's origin was produced by the same code path as every adapted point.

    Also derives the baseline operating threshold from SEEN-generator validation data
    only. Every adaptation cell is additionally reported at this unchanged threshold, so
    calibration shift can be separated from weight adaptation. Returns the 0% row and
    that threshold.
    """

    model = build_detector(config, device=context.device, fine_tune_mode=fine_tune_mode)
    starting = load_checkpoint(starting_checkpoint_path, map_location=str(context.device))
    model.load_state_dict(starting["model_state"], strict=True)

    # Held-out samples contribute nothing here: seen_validation_records is the
    # known-generator validation split, asserted free of the held-out generator.
    baseline_threshold, baseline_score = select_threshold_on_validation(
        model=model,
        records=seen_validation_records,
        config=config,
        device=context.device,
        metric=str(config["training"]["early_stopping"].get("metric", "f1")),
    )

    outcome = evaluate_records(
        model=model,
        records=final_test_records,
        config=config,
        device=context.device,
        split_name="unseen_test",
        checkpoint_id="starting_checkpoint_0pct",
    )
    save_predictions(
        outcome.predictions, context.run_dir / "zero_percent_unseen_test_predictions.csv"
    )
    labels = [item.label for item in outcome.predictions]
    scores = [item.score for item in outcome.predictions]
    at_baseline = compute_binary_metrics(labels, scores, threshold=baseline_threshold)
    row = {
        "cell_id": "zero_percent_reference",
        "fine_tune_mode": "none",
        "adaptation_percentage": 0.0,
        "subset_seed": None,
        "training_seed": None,
        "starting_checkpoint": str(starting_checkpoint_path),
        "adaptation_train_count": 0,
        "adaptation_validation_count": 0,
        "labelled_images_consumed": 0,
        "held_out_fake_count": 0,
        "authentic_count": 0,
        "held_out_samples_used_for_fitting_or_selection": 0,
        "thresholds": {
            "default": {
                "value": float(config["model"]["decision_threshold"]),
                "provenance": THRESHOLD_PROVENANCE_DEFAULT,
            },
            "baseline_unchanged": {
                "value": baseline_threshold,
                "provenance": THRESHOLD_PROVENANCE_SEEN_VALIDATION,
                "selection_score": baseline_score,
                "selection_sample_count": len(seen_validation_records),
                "held_out_samples_used": 0,
            },
        },
        "overall": asdict(outcome.overall),
        "at_baseline_threshold": asdict(at_baseline),
        # The 0% model has no adaptation validation, so there is no adaptation-selected
        # operating point; the baseline threshold is the only legitimate one here.
        "at_adaptation_selected_threshold": asdict(at_baseline),
        "per_generator": {
            name: asdict(metrics) for name, metrics in outcome.per_generator.items()
        },
    }
    return row, baseline_threshold


def build_recovery_pools(
    config: Mapping[str, Any], bundle: Any, unseen: str, known: Sequence[str]
) -> tuple[list[DatasetRecord], list[DatasetRecord], dict[str, Any]]:
    """Return the adaptation pool, the fixed final unseen test partition, and its metadata.

    The adaptation pool is the held-out generator's official-train fakes plus the
    shared real training images, so adaptation sees new-generator evidence against the
    same negatives the detector already knows.

    The final test set is built by the same balanced-50/50 fixed-real-pool routine the
    unseen-generator runner uses, so the 0% baseline and every adaptation budget are
    scored on byte-identical membership at identical prevalence.
    """

    include_real = bool(config["generators"].get("include_real_images", True))
    adaptation_pool = [
        record
        for record in bundle.records
        if bundle.split_by_id[record.sample_id] == "train"
        and (
            (record.label == 1 and record.generator == unseen)
            or (record.label == 0 and include_real)
        )
    ]
    final_test_records, final_test_metadata = build_balanced_final_test(
        bundle.records, bundle.split_by_id, unseen_generator=unseen
    )
    assert_pools_group_disjoint(adaptation_pool, final_test_records)
    if not any(record.label == 1 and record.generator == unseen for record in adaptation_pool):
        raise ValueError(
            f"the adaptation pool contains no {unseen!r} fakes; there is nothing to adapt to"
        )
    if not any(record.label == 1 and record.generator == unseen for record in final_test_records):
        raise ValueError(f"the final test partition contains no {unseen!r} fakes")
    return adaptation_pool, final_test_records, final_test_metadata


def summarise_recovery(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate cells into a tidy recovery table with across-seed variability."""

    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["fine_tune_mode"]), float(row["adaptation_percentage"]))
        grouped.setdefault(key, []).append(row)
    zero = next((row for row in rows if float(row["adaptation_percentage"]) == 0.0), None)
    baseline_f1 = float(zero["overall"]["f1"]) if zero else None
    baseline_by_metric: dict[str, float | None] = {}
    if zero:
        for name in ("f1", "accuracy", "roc_auc", "average_precision"):
            value = zero["overall"].get(name)
            baseline_by_metric[name] = None if value is None else float(value)
    summary = []
    for (mode, fraction), cells in sorted(grouped.items()):
        for metric in ("f1", "accuracy", "roc_auc", "average_precision"):
            values = [
                float(cell["overall"][metric])
                for cell in cells
                if cell["overall"].get(metric) is not None
            ]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = (
                sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                if len(values) > 1
                else 0.0
            )
            entry = {
                "fine_tune_mode": mode,
                "adaptation_percentage": fraction,
                "metric": metric,
                "runs": len(values),
                "mean": mean,
                "standard_deviation": math.sqrt(variance),
                "minimum": min(values),
                "maximum": max(values),
                "labelled_images_consumed": sorted(
                    {int(cell.get("labelled_images_consumed") or 0) for cell in cells}
                ),
            }
            baseline_value = baseline_by_metric.get(metric)
            if baseline_value is not None:
                entry["recovery_vs_zero_percent"] = mean - baseline_value
            summary.append(entry)
    return {"zero_percent_f1": baseline_f1, "rows": summary}


def _write_cell_table(rows: Sequence[Mapping[str, Any]], destination: Path) -> None:
    columns = [
        "cell_id",
        "held_out_generator",
        "fine_tune_mode",
        "adaptation_percentage",
        "subset_seed",
        "training_seed",
        "adaptation_train_count",
        "adaptation_validation_count",
        "labelled_images_consumed",
        "held_out_fake_count",
        "authentic_count",
        "trainable_parameters",
        "total_parameters",
        "best_epoch",
        "epochs",
        "learning_rate",
        "training_seconds",
        "best_adaptation_validation_score",
        "starting_checkpoint",
        # Threshold-free metrics first: they do not depend on the operating point.
        "roc_auc",
        "average_precision",
        "support",
        # Threshold-dependent metrics, reported at each declared operating point.
        "threshold_default",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "threshold_adaptation_selected",
        "accuracy_at_adaptation_threshold",
        "precision_at_adaptation_threshold",
        "recall_at_adaptation_threshold",
        "f1_at_adaptation_threshold",
        "threshold_baseline_unchanged",
        "accuracy_at_baseline_threshold",
        "precision_at_baseline_threshold",
        "recall_at_baseline_threshold",
        "f1_at_baseline_threshold",
        "threshold_provenance_adaptation",
        "threshold_provenance_baseline",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in columns}
            overall = row["overall"]
            for metric_key in ("roc_auc", "average_precision", "support"):
                flat[metric_key] = overall.get(metric_key)
            for metric_key in ("accuracy", "precision", "recall", "f1"):
                flat[metric_key] = overall.get(metric_key)
            flat["threshold_default"] = overall.get("threshold")
            thresholds = row.get("thresholds") or {}
            adaptation = thresholds.get("adaptation_validation_selected") or {}
            baseline = thresholds.get("baseline_unchanged") or {}
            flat["threshold_adaptation_selected"] = adaptation.get("value")
            flat["threshold_baseline_unchanged"] = baseline.get("value")
            flat["threshold_provenance_adaptation"] = adaptation.get("provenance")
            flat["threshold_provenance_baseline"] = baseline.get("provenance")
            for source_key, suffix in (
                ("at_adaptation_selected_threshold", "at_adaptation_threshold"),
                ("at_baseline_threshold", "at_baseline_threshold"),
            ):
                block = row.get(source_key) or {}
                for metric_key in ("accuracy", "precision", "recall", "f1"):
                    flat[f"{metric_key}_{suffix}"] = block.get(metric_key)
            writer.writerow(flat)


def run_fine_tuning(config_path: Path) -> Path:
    """Fine-tune the unseen-generator checkpoint at each limited data budget."""

    loaded = load_config(config_path)
    if loaded.values["experiment"]["type"] != "fine_tuning":
        raise ValueError("run_fine_tuning requires experiment.type=fine_tuning")
    config = resolve_runtime_paths(loaded.values, loaded.source_path)
    unseen, known = validate_unseen_protocol(config)
    settings = config.get("fine_tuning") or {}
    fractions = [float(value) for value in settings["percentages"]]
    if tuple(fractions) != REQUIRED_FRACTIONS:
        raise ValueError(f"fine_tuning.percentages must be {list(REQUIRED_FRACTIONS)}")
    subset_seeds = [int(value) for value in settings["subset_seeds"]]
    if not subset_seeds or len(set(subset_seeds)) != len(subset_seeds):
        raise ValueError("fine_tuning.subset_seeds must be a non-empty list of unique integers")
    training_seeds = [int(value) for value in config["reproducibility"]["experiment_seeds"]]
    if not training_seeds:
        raise ValueError("reproducibility.experiment_seeds must not be empty")
    if not settings.get("nested_subsets", True):
        raise ValueError(
            "fine_tuning.nested_subsets must stay true; the recovery comparison assumes it"
        )
    if not settings.get("reload_starting_checkpoint_each_run", True):
        # Every cell always reloads the original checkpoint. Accepting `false` would let a
        # config claim cumulative adaptation while the code does the opposite.
        raise ValueError(
            "fine_tuning.reload_starting_checkpoint_each_run must be true: each budget is "
            "an independent condition restarted from the 0% checkpoint, never a "
            "continuation of a smaller budget"
        )
    fine_tune_mode = str(settings.get("fine_tune_mode") or config["training"]["fine_tune_mode"])
    validation_fraction = float(settings.get("adaptation_validation_fraction", 0.25))
    starting_checkpoint_path = resolve_starting_checkpoint(config, loaded.source_path)

    context = prepare_experiment(config)
    logger = logging.getLogger(f"ai_detector.{context.run_id}")
    try:
        bundle = load_manifest_with_splits(config)
        adaptation_pool, final_test_records, final_test_metadata = build_recovery_pools(
            config, bundle, unseen, known
        )
        # Seen-generator validation, used only to derive the unchanged baseline threshold.
        seen_validation_records = select_records(
            bundle.records,
            bundle.split_by_id,
            split="validation",
            fake_generators=list(config["generators"]["validation"]),
            include_real=bool(config["generators"].get("include_real_images", True)),
        )
        assert_unseen_absent_from_development(
            {"seen_validation_for_baseline_threshold": seen_validation_records},
            unseen_generator=unseen,
        )
        starting = load_checkpoint(starting_checkpoint_path, map_location="cpu")
        compatibility = assert_starting_checkpoint_compatible(
            starting,
            config,
            unseen_generator=unseen,
            known_generators=known,
            manifest_sha256=bundle.manifest_sha256,
        )
        del starting
        pools = split_adaptation_pool(
            adaptation_pool, validation_fraction=validation_fraction, seed=subset_seeds[0]
        )
        budgets_by_seed: dict[int, dict[float, AdaptationBudget]] = {}
        for subset_seed in subset_seeds:
            budgets_by_seed[subset_seed] = build_nested_adaptation_subsets(
                adaptation_pool=adaptation_pool,
                pools=pools,
                fractions=fractions,
                seed=subset_seed,
                final_test_records=final_test_records,
            )
        subset_path = context.run_dir / "adaptation_subsets.json"
        save_adaptation_subsets(budgets_by_seed, subset_path)
        logger.info(
            "recovery protocol generator=%s adaptation_pool=%d final_unseen_test=%d "
            "mode=%s cells=%d",
            unseen,
            len(adaptation_pool),
            len(final_test_records),
            fine_tune_mode,
            len(fractions) * len(subset_seeds) * len(training_seeds),
        )

        zero_row, baseline_threshold = evaluate_starting_checkpoint(
            config=config,
            context=context,
            starting_checkpoint_path=starting_checkpoint_path,
            final_test_records=final_test_records,
            fine_tune_mode=fine_tune_mode,
            seen_validation_records=seen_validation_records,
        )
        rows: list[dict[str, Any]] = [zero_row]
        logger.info(
            "0%% reference roc_auc=%s f1@default=%.4f f1@baseline_threshold(%.2f)=%.4f",
            zero_row["overall"]["roc_auc"],
            zero_row["overall"]["f1"],
            baseline_threshold,
            zero_row["at_baseline_threshold"]["f1"],
        )
        for fraction in fractions:
            for subset_seed in subset_seeds:
                for training_seed in training_seeds:
                    cell_id = (
                        f"{fine_tune_mode}_p{int(round(fraction * 100)):02d}"
                        f"_s{subset_seed}_t{training_seed}"
                    )
                    cell = run_adaptation_cell(
                        config=config,
                        context=context,
                        pools=pools,
                        budget=budgets_by_seed[subset_seed][fraction],
                        final_test_records=final_test_records,
                        starting_checkpoint_path=starting_checkpoint_path,
                        fine_tune_mode=fine_tune_mode,
                        training_seed=training_seed,
                        cell_id=cell_id,
                        baseline_threshold=baseline_threshold,
                        baseline_threshold_provenance=THRESHOLD_PROVENANCE_SEEN_VALIDATION,
                    )
                    rows.append(cell.record)
                    logger.info(
                        "cell %s roc_auc=%s f1@adapt_t=%.4f f1@baseline_t=%.4f (labelled %d)",
                        cell_id,
                        cell.record["overall"]["roc_auc"],
                        cell.record["at_adaptation_selected_threshold"]["f1"],
                        cell.record["at_baseline_threshold"]["f1"],
                        cell.record["labelled_images_consumed"],
                    )

        for row in rows:
            row["held_out_generator"] = unseen
        _write_cell_table(rows, context.run_dir / "recovery_cells.csv")
        summary = summarise_recovery(rows)
        payload = {
            "protocol": "limited_data_recovery",
            "held_out_generator": unseen,
            "known_generators": known,
            "fine_tune_mode": fine_tune_mode,
            "percentages": fractions,
            "subset_seeds": subset_seeds,
            "training_seeds": training_seeds,
            "adaptation_validation_fraction": validation_fraction,
            "adaptation_pool_size": len(adaptation_pool),
            "final_unseen_test_size": len(final_test_records),
            "final_test_composition": final_test_metadata,
            "final_test_sample_ids": sorted(record.sample_id for record in final_test_records),
            "baseline_threshold": baseline_threshold,
            "baseline_threshold_provenance": THRESHOLD_PROVENANCE_SEEN_VALIDATION,
            "manifest_sha256": bundle.manifest_sha256,
            "starting_checkpoint_compatibility": compatibility,
            "adaptation_subsets_path": str(subset_path),
            "cells": rows,
            "summary": summary,
        }
        (context.run_dir / "recovery_metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        finalise_run(context, status="completed")
        return context.run_dir
    except Exception:
        logger.exception("fine-tuning recovery experiment failed")
        finalise_run(context, status="failed")
        raise


# IMPLEMENTATION CHECKLIST
# [x] Partition adaptation and final test pools before sampling percentages.
# [x] Build, save, and test nested 5/10/20/50% ID sets across subset seeds.
# [x] Reload identical starting weights for every independent condition.
# [x] Define small-data validation/early-stopping without final-test leakage.
# [x] Hold final test, real pool, metric, and threshold policy constant.
# [x] Report actual counts plus subset/training seed variability.
# [x] Include the measured 0% condition in recovery plots.
