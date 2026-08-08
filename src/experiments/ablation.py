"""Fine-tuning-depth ablation experiment.

###############################################################################
RESEARCH QUESTION
###############################################################################

Does recovery require changing CLIP broadly, or can a lightweight classifier update
adapt existing features? Compare:

- ``head_only``: tests whether existing CLIP features already separate the new source.
- ``last_block``: tests whether limited high-level representation adaptation suffices.
- ``full``: tests maximum adaptability, with higher compute and overfitting risk.

Only the trainable layers should differ. Reuse the same starting checkpoint, subset
IDs, test samples, seeds, selection metric, and—unless explicitly studying it—training
budget. Learning rates may need mode-specific values, but if so they must be tuned by
a predeclared validation procedure and reported as part of the comparison.

###############################################################################
HOW FAIRNESS IS ENFORCED HERE
###############################################################################

The ablation deliberately reuses the recovery experiment's own cell runner, so the
data path cannot drift between the two experiments. On top of that it asserts, rather
than assumes, the controls declared in ``configs/ablation.yaml``:

* ``control_starting_checkpoint`` -- every cell reloads one identical checkpoint file.
* ``control_subset_ids``          -- the adaptation subsets are built once and each
                                     mode is handed the byte-identical ID lists.
* ``control_final_test_ids``      -- one fixed final-test record set scores every cell.
* ``training_budget_policy``      -- recorded, and any mode-specific override is
                                     flagged in the output as a fairness caveat.

The full run matrix is written to disk before execution so the intended grid is
auditable independently of what completed.

Interpretation warning carried into the output: head-only success implies the frozen
features already separate the new generator; last-block gains imply high-level
adaptation is needed; full-model gains imply wider change helps, while full-model
UNDER-performance is at least as likely to reflect small-data overfitting or
optimisation difficulty as a lack of capacity.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.experiments.common import (
    finalise_run,
    load_manifest_with_splits,
    prepare_experiment,
    resolve_runtime_paths,
)
from src.experiments.fine_tuning import (
    REQUIRED_FRACTIONS,
    AdaptationBudget,
    _write_cell_table,
    assert_starting_checkpoint_compatible,
    build_nested_adaptation_subsets,
    build_recovery_pools,
    evaluate_starting_checkpoint,
    resolve_starting_checkpoint,
    run_adaptation_cell,
    save_adaptation_subsets,
    split_adaptation_pool,
    summarise_recovery,
)
from src.experiments.unseen_generator import validate_unseen_protocol
from src.models.checkpointing import load_checkpoint
from src.utils.config import load_config

LOGGER = logging.getLogger(__name__)

SUPPORTED_MODES = ("head_only", "last_block", "full")
BUDGET_POLICIES = ("equal_epochs",)

INTERPRETATION_NOTE = (
    "head_only gains indicate the frozen CLIP features already separate the held-out "
    "generator; last_block gains indicate high-level representation adaptation is "
    "required; full gains indicate broader change helps. Full-model underperformance "
    "must not be read as insufficient capacity: with these label budgets it is at "
    "least as likely to reflect overfitting or optimisation difficulty."
)


def validate_ablation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Check the freeze modes, controls, and budget policy before any training."""

    settings = config.get("ablation") or {}
    modes = list(settings.get("modes") or [])
    if not modes:
        raise ValueError("ablation.modes must list at least one freeze mode")
    unknown = sorted(set(modes).difference(SUPPORTED_MODES))
    if unknown:
        raise ValueError(f"unknown ablation modes: {', '.join(unknown)}")
    if len(set(modes)) != len(modes):
        raise ValueError("ablation.modes contains duplicates")
    if set(modes) != set(SUPPORTED_MODES):
        LOGGER.warning(
            "ablation covers %s rather than all of %s; this is a deliberately reduced "
            "study and must be described as such",
            modes,
            list(SUPPORTED_MODES),
        )
    policy = str(settings.get("training_budget_policy", "equal_epochs"))
    if policy not in BUDGET_POLICIES:
        raise ValueError(
            f"unsupported ablation.training_budget_policy {policy!r}; supported: "
            f"{list(BUDGET_POLICIES)}"
        )
    for control in ("control_starting_checkpoint", "control_subset_ids", "control_final_test_ids"):
        if not bool(settings.get(control, True)):
            raise ValueError(
                f"ablation.{control} must stay true; the depth comparison is only "
                "interpretable when it is held constant"
            )
    return {
        "modes": modes,
        "training_budget_policy": policy,
        "record_trainable_parameter_names": bool(
            settings.get("record_trainable_parameter_names", True)
        ),
        "record_training_time": bool(settings.get("record_training_time", True)),
    }


def resolve_mode_overrides(
    config: Mapping[str, Any], modes: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Read optional per-mode optimisation overrides and flag any unequal budget.

    ``mode_overrides: {head_only: null, ...}`` means "use the shared training settings",
    which is the fair default. A mode that overrides ``epochs`` breaks the equal-epoch
    budget, so it is surfaced as a caveat rather than silently applied.
    """

    raw = config.get("mode_overrides") or {}
    resolved: dict[str, dict[str, Any]] = {}
    for mode in modes:
        override = raw.get(mode)
        if override is None:
            resolved[mode] = {}
            continue
        if not isinstance(override, Mapping):
            raise ValueError(f"mode_overrides.{mode} must be a mapping or null")
        unknown = sorted(set(override).difference({"learning_rate", "epochs"}))
        if unknown:
            raise ValueError(
                f"mode_overrides.{mode} supports only learning_rate and epochs; "
                f"found {', '.join(unknown)}"
            )
        entry: dict[str, Any] = {}
        if "learning_rate" in override:
            value = float(override["learning_rate"])
            if value <= 0:
                raise ValueError(f"mode_overrides.{mode}.learning_rate must be positive")
            entry["learning_rate"] = value
        if "epochs" in override:
            value_int = int(override["epochs"])
            if value_int <= 0:
                raise ValueError(f"mode_overrides.{mode}.epochs must be positive")
            entry["epochs"] = value_int
        resolved[mode] = entry
    return resolved


def build_run_matrix(
    *,
    modes: Sequence[str],
    fractions: Sequence[float],
    subset_seeds: Sequence[int],
    training_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Enumerate the full grid up front so the intended study is auditable."""

    matrix: list[dict[str, Any]] = []
    for mode in modes:
        for fraction in fractions:
            for subset_seed in subset_seeds:
                for training_seed in training_seeds:
                    matrix.append(
                        {
                            "cell_id": (
                                f"{mode}_p{int(round(fraction * 100)):02d}"
                                f"_s{subset_seed}_t{training_seed}"
                            ),
                            "fine_tune_mode": mode,
                            "adaptation_percentage": fraction,
                            "subset_seed": subset_seed,
                            "training_seed": training_seed,
                        }
                    )
    return matrix


def assert_subset_ids_controlled(
    budgets_by_seed: Mapping[int, Mapping[float, AdaptationBudget]],
    *,
    fractions: Sequence[float],
    subset_seeds: Sequence[int],
) -> dict[str, Any]:
    """Record the exact labelled-image identity each cell will reuse across modes.

    Every mode is handed the same ``AdaptationBudget`` objects, so control is
    structural. This returns a digest per (fraction, subset seed) so the output proves
    which IDs were shared rather than merely claiming it.
    """

    digests: dict[str, Any] = {}
    for subset_seed in subset_seeds:
        for fraction in fractions:
            budget = budgets_by_seed[subset_seed][fraction]
            joined = "|".join(
                (*sorted(budget.train_sample_ids), "::", *sorted(budget.validation_sample_ids))
            )
            digests[f"s{subset_seed}_p{int(round(fraction * 100)):02d}"] = {
                "sample_id_sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
                "adaptation_train_count": budget.train_count,
                "adaptation_validation_count": budget.validation_count,
                "labelled_images_consumed": budget.labelled_count,
            }
    return digests


def summarise_by_mode(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare freeze modes at equal budget, including cost and trainable size."""

    grouped: dict[tuple[float, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if float(row["adaptation_percentage"]) == 0.0:
            continue
        key = (float(row["adaptation_percentage"]), str(row["fine_tune_mode"]))
        grouped.setdefault(key, []).append(row)
    table: list[dict[str, Any]] = []
    for (fraction, mode), cells in sorted(grouped.items()):
        f1_values = [float(cell["overall"]["f1"]) for cell in cells]
        seconds = [
            float(cell["training_seconds"]) for cell in cells if cell.get("training_seconds")
        ]
        table.append(
            {
                "adaptation_percentage": fraction,
                "fine_tune_mode": mode,
                "runs": len(f1_values),
                "mean_f1": sum(f1_values) / len(f1_values),
                "minimum_f1": min(f1_values),
                "maximum_f1": max(f1_values),
                "trainable_parameters": sorted(
                    {int(cell.get("trainable_parameters") or 0) for cell in cells}
                ),
                "mean_training_seconds": (sum(seconds) / len(seconds)) if seconds else None,
            }
        )
    return table


def run_ablation(config_path: Path) -> Path:
    """Compare head-only, last-block, and full adaptation fairly."""

    loaded = load_config(config_path)
    if loaded.values["experiment"]["type"] != "ablation":
        raise ValueError("run_ablation requires experiment.type=ablation")
    config = resolve_runtime_paths(loaded.values, loaded.source_path)
    unseen, known = validate_unseen_protocol(config)
    ablation_settings = validate_ablation_settings(config)
    modes = ablation_settings["modes"]
    overrides = resolve_mode_overrides(config, modes)

    settings = config.get("fine_tuning") or {}
    fractions = [float(value) for value in settings["percentages"]]
    if tuple(fractions) != REQUIRED_FRACTIONS:
        raise ValueError(f"fine_tuning.percentages must be {list(REQUIRED_FRACTIONS)}")
    subset_seeds = [int(value) for value in settings["subset_seeds"]]
    training_seeds = [int(value) for value in config["reproducibility"]["experiment_seeds"]]
    if not subset_seeds or not training_seeds:
        raise ValueError("subset seeds and training seeds must both be non-empty")
    validation_fraction = float(settings.get("adaptation_validation_fraction", 0.25))
    starting_checkpoint_path = resolve_starting_checkpoint(config, loaded.source_path)

    context = prepare_experiment(config)
    logger = logging.getLogger(f"ai_detector.{context.run_id}")
    try:
        bundle = load_manifest_with_splits(config)
        adaptation_pool, final_test_records = build_recovery_pools(config, bundle, unseen, known)
        starting = load_checkpoint(starting_checkpoint_path, map_location="cpu")
        compatibility = assert_starting_checkpoint_compatible(
            starting,
            config,
            unseen_generator=unseen,
            known_generators=known,
            manifest_sha256=bundle.manifest_sha256,
        )
        del starting

        # Subsets are built ONCE and shared by every mode: control by construction.
        pools = split_adaptation_pool(
            adaptation_pool, validation_fraction=validation_fraction, seed=subset_seeds[0]
        )
        budgets_by_seed: dict[int, dict[float, AdaptationBudget]] = {
            subset_seed: build_nested_adaptation_subsets(
                adaptation_pool=adaptation_pool,
                pools=pools,
                fractions=fractions,
                seed=subset_seed,
                final_test_records=final_test_records,
            )
            for subset_seed in subset_seeds
        }
        save_adaptation_subsets(budgets_by_seed, context.run_dir / "adaptation_subsets.json")
        subset_digests = assert_subset_ids_controlled(
            budgets_by_seed, fractions=fractions, subset_seeds=subset_seeds
        )
        final_test_ids = sorted(record.sample_id for record in final_test_records)

        matrix = build_run_matrix(
            modes=modes,
            fractions=fractions,
            subset_seeds=subset_seeds,
            training_seeds=training_seeds,
        )
        (context.run_dir / "run_matrix.json").write_text(
            json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info(
            "ablation matrix modes=%s cells=%d budget_policy=%s",
            ",".join(modes),
            len(matrix),
            ablation_settings["training_budget_policy"],
        )

        rows: list[dict[str, Any]] = [
            evaluate_starting_checkpoint(
                config=config,
                context=context,
                starting_checkpoint_path=starting_checkpoint_path,
                final_test_records=final_test_records,
                fine_tune_mode=modes[0],
            )
        ]
        logger.info("0%% reference f1=%.4f", rows[0]["overall"]["f1"])

        trainable_by_mode: dict[str, Any] = {}
        for entry in matrix:
            mode = str(entry["fine_tune_mode"])
            override = overrides[mode]
            cell = run_adaptation_cell(
                config=config,
                context=context,
                pools=pools,
                budget=budgets_by_seed[int(entry["subset_seed"])][
                    float(entry["adaptation_percentage"])
                ],
                final_test_records=final_test_records,
                starting_checkpoint_path=starting_checkpoint_path,
                fine_tune_mode=mode,
                training_seed=int(entry["training_seed"]),
                cell_id=str(entry["cell_id"]),
                learning_rate=override.get("learning_rate"),
                epochs=override.get("epochs"),
            )
            record = dict(cell.record)
            # Freeze policy is an invariant: the same mode must always expose the same
            # trainable parameters, whatever the budget or seed.
            names = tuple(record.get("trainable_parameter_names") or ())
            if mode in trainable_by_mode and trainable_by_mode[mode] != names:
                raise RuntimeError(
                    f"freeze mode {mode!r} exposed different trainable parameters between cells"
                )
            trainable_by_mode[mode] = names
            if not ablation_settings["record_trainable_parameter_names"]:
                record.pop("trainable_parameter_names", None)
            if not ablation_settings["record_training_time"]:
                record.pop("training_seconds", None)
            rows.append(record)
            logger.info(
                "cell %s f1=%.4f trainable=%s labelled=%d",
                record["cell_id"],
                record["overall"]["f1"],
                record.get("trainable_parameters"),
                record["labelled_images_consumed"],
            )

        _write_cell_table(rows, context.run_dir / "ablation_cells.csv")
        unequal_budget = sorted(mode for mode, entry in overrides.items() if "epochs" in entry)
        payload = {
            "protocol": "fine_tuning_depth_ablation",
            "held_out_generator": unseen,
            "known_generators": known,
            "modes": modes,
            "percentages": fractions,
            "subset_seeds": subset_seeds,
            "training_seeds": training_seeds,
            "adaptation_validation_fraction": validation_fraction,
            "manifest_sha256": bundle.manifest_sha256,
            "controls": {
                "starting_checkpoint": str(starting_checkpoint_path),
                "starting_checkpoint_compatibility": compatibility,
                "subset_id_digests": subset_digests,
                "final_test_sample_id_count": len(final_test_ids),
                "final_test_sample_ids": final_test_ids,
                "training_budget_policy": ablation_settings["training_budget_policy"],
                "mode_overrides_applied": {
                    mode: entry for mode, entry in overrides.items() if entry
                },
                "budget_policy_violated_by": unequal_budget,
            },
            "trainable_parameter_names_by_mode": {
                mode: list(names) for mode, names in sorted(trainable_by_mode.items())
            },
            "cells": rows,
            "recovery_summary": summarise_recovery(rows),
            "mode_comparison": summarise_by_mode(rows),
            "interpretation_note": INTERPRETATION_NOTE,
        }
        if unequal_budget:
            logger.warning(
                "modes %s override the epoch budget; the comparison is no longer equal-epoch",
                unequal_budget,
            )
        (context.run_dir / "ablation_metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        finalise_run(context, status="completed")
        return context.run_dir
    except Exception:
        logger.exception("fine-tuning depth ablation failed")
        finalise_run(context, status="failed")
        raise


# IMPLEMENTATION CHECKLIST
# [x] Freeze the ablation run matrix and shared subset/checkpoint identities.
# [x] Verify trainable names/counts for head-only, last-block, and full modes.
# [x] Recreate optimiser and reload starting weights for every condition.
# [x] Define fair training/selection budget and any mode-specific hyperparameter policy.
# [x] Evaluate identical final-test IDs with identical metric semantics.
# [x] Report compute and variation across both subset and training seeds.
# [x] Separate representational conclusions from optimisation/overfitting explanations.
