"""Leakage, budget, and control contracts for the unseen/recovery/ablation protocols.

These tests guard the properties the dissertation's claims rest on: the held-out
generator never reaches development data, adaptation budgets nest and stay disjoint
from the final test partition, and the ablation cannot silently drop a control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.datasets.schema import DatasetRecord
from src.experiments.ablation import (
    build_run_matrix,
    resolve_mode_overrides,
    summarise_by_mode,
    validate_ablation_settings,
)
from src.experiments.fine_tuning import (
    build_nested_adaptation_subsets,
    load_adaptation_subsets,
    save_adaptation_subsets,
    split_adaptation_pool,
    summarise_recovery,
)
from src.experiments.unseen_generator import (
    assert_pools_group_disjoint,
    assert_unseen_absent_from_development,
    validate_unseen_protocol,
)

FRACTIONS = (0.05, 0.10, 0.20, 0.50)


def record(sample_id: str, label: int, generator: str, group: str) -> DatasetRecord:
    return DatasetRecord(
        sample_id, Path(f"/data/{sample_id}.png"), label, generator, group, "test"
    )


def adaptation_pool(fake_count: int = 40, real_count: int = 40) -> list[DatasetRecord]:
    pool = [record(f"fake-{index}", 1, "biggan", f"gf-{index}") for index in range(fake_count)]
    pool += [record(f"real-{index}", 0, "real", f"gr-{index}") for index in range(real_count)]
    return pool


def base_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "data": {"real_generator_name": "real"},
        "generators": {
            "train": ["midjourney", "adm"],
            "validation": ["midjourney", "adm"],
            "test": ["biggan"],
            "unseen": "biggan",
        },
    }
    for key, value in overrides.items():
        config[key] = value
    return config


# --------------------------------------------------------------------------- unseen


def test_unseen_protocol_requires_one_held_out_generator_absent_from_development() -> None:
    unseen, known = validate_unseen_protocol(base_config())
    assert unseen == "biggan"
    assert known == ["midjourney", "adm"]

    leaking = base_config()
    leaking["generators"]["train"] = ["midjourney", "biggan"]
    with pytest.raises(ValueError, match="appears in development generators"):
        validate_unseen_protocol(leaking)

    validation_leak = base_config()
    validation_leak["generators"]["validation"] = ["midjourney", "biggan"]
    with pytest.raises(ValueError, match="appears in development generators"):
        validate_unseen_protocol(validation_leak)

    wrong_test = base_config()
    wrong_test["generators"]["test"] = ["biggan", "adm"]
    with pytest.raises(ValueError, match="must contain exactly the held-out generator"):
        validate_unseen_protocol(wrong_test)

    reserved = base_config()
    reserved["generators"]["unseen"] = "real"
    with pytest.raises(ValueError, match="cannot be the reserved real-image name"):
        validate_unseen_protocol(reserved)


def test_held_out_fakes_in_a_development_selection_are_rejected() -> None:
    clean = {
        "train": [record("a", 1, "adm", "g1"), record("r", 0, "real", "g2")],
        "validation": [record("b", 1, "midjourney", "g3")],
    }
    assert_unseen_absent_from_development(clean, unseen_generator="biggan")

    leaked = dict(clean)
    leaked["train"] = [*clean["train"], record("leak", 1, "biggan", "g9")]
    with pytest.raises(ValueError, match="leaked into development selections"):
        assert_unseen_absent_from_development(leaked, unseen_generator="biggan")


def test_held_out_real_images_are_not_treated_as_leakage() -> None:
    """Real negatives are shared by design; only held-out FAKES constitute leakage."""

    selections = {"train": [record("r", 0, "real", "shared-real-group")]}
    assert_unseen_absent_from_development(selections, unseen_generator="biggan")


def test_adaptation_and_final_test_must_not_share_groups_or_samples() -> None:
    adaptation = [record("a", 1, "biggan", "shared")]
    test = [record("b", 1, "biggan", "shared")]
    with pytest.raises(ValueError, match="share source groups"):
        assert_pools_group_disjoint(adaptation, test)

    same_id = [record("dup", 1, "biggan", "g1")]
    other = [record("dup", 1, "biggan", "g2")]
    with pytest.raises(ValueError, match="share sample IDs"):
        assert_pools_group_disjoint(same_id, other)

    assert_pools_group_disjoint([record("a", 1, "biggan", "g1")], [record("b", 1, "biggan", "g2")])


# ------------------------------------------------------------------------- budgets


def test_adaptation_pool_split_is_group_disjoint_and_keeps_both_classes() -> None:
    pools = split_adaptation_pool(adaptation_pool(), validation_fraction=0.25, seed=42)
    assert set(pools.train_groups) == {0, 1}
    assert set(pools.validation_groups) == {0, 1}
    train_groups = {group for groups in pools.train_groups.values() for group, _ in groups}
    validation_groups = {
        group for groups in pools.validation_groups.values() for group, _ in groups
    }
    assert not train_groups & validation_groups
    for label in (0, 1):
        assert pools.train_groups[label] and pools.validation_groups[label]


def test_adaptation_pool_split_rejects_a_single_class_pool() -> None:
    only_fake = [record(f"f-{index}", 1, "biggan", f"g-{index}") for index in range(4)]
    with pytest.raises(ValueError, match="both real and fake"):
        split_adaptation_pool(only_fake, validation_fraction=0.25, seed=42)


def test_budgets_nest_and_every_budget_contains_both_classes() -> None:
    pool = adaptation_pool()
    pools = split_adaptation_pool(pool, validation_fraction=0.25, seed=42)
    budgets = build_nested_adaptation_subsets(
        adaptation_pool=pool, pools=pools, fractions=FRACTIONS, seed=42
    )
    by_id = {item.sample_id: item for item in pool}
    previous_train: set[str] | None = None
    previous_validation: set[str] | None = None
    for fraction in FRACTIONS:
        budget = budgets[fraction]
        train = set(budget.train_sample_ids)
        validation = set(budget.validation_sample_ids)
        assert not train & validation
        # The smallest budget must still be a two-class training problem.
        assert {by_id[identifier].label for identifier in train} == {0, 1}
        assert {by_id[identifier].label for identifier in validation} == {0, 1}
        if previous_train is not None:
            assert previous_train < train
            assert previous_validation <= validation
        previous_train, previous_validation = train, validation
        assert budget.labelled_count == len(train) + len(validation)


def test_budgets_are_reproducible_and_seed_sensitive() -> None:
    pool = adaptation_pool()
    pools = split_adaptation_pool(pool, validation_fraction=0.25, seed=42)
    first = build_nested_adaptation_subsets(
        adaptation_pool=pool, pools=pools, fractions=FRACTIONS, seed=42
    )
    same = build_nested_adaptation_subsets(
        adaptation_pool=pool, pools=pools, fractions=FRACTIONS, seed=42
    )
    other = build_nested_adaptation_subsets(
        adaptation_pool=pool, pools=pools, fractions=FRACTIONS, seed=123
    )
    assert first[0.05].train_sample_ids == same[0.05].train_sample_ids
    assert first[0.50].train_sample_ids != other[0.50].train_sample_ids


def test_budget_construction_refuses_final_test_overlap() -> None:
    pool = adaptation_pool()
    pools = split_adaptation_pool(pool, validation_fraction=0.25, seed=42)
    shared_sample = [record("fake-0", 1, "biggan", "unrelated-group")]
    with pytest.raises(ValueError, match="overlaps the final test partition"):
        build_nested_adaptation_subsets(
            adaptation_pool=pool,
            pools=pools,
            fractions=FRACTIONS,
            seed=42,
            final_test_records=shared_sample,
        )
    shared_group = [record("unrelated-id", 1, "biggan", "gf-0")]
    with pytest.raises(ValueError, match="shares final-test groups"):
        build_nested_adaptation_subsets(
            adaptation_pool=pool,
            pools=pools,
            fractions=FRACTIONS,
            seed=42,
            final_test_records=shared_group,
        )


@pytest.mark.parametrize(
    ("fractions", "message"),
    [
        ((0.5, 0.05), "increasing order"),
        ((0.05, 0.05), "unique"),
        ((0.0, 0.5), r"\(0, 1\]"),
        ((0.05, 1.5), r"\(0, 1\]"),
        ((), "at least one"),
    ],
)
def test_budget_fractions_are_validated(fractions: tuple[float, ...], message: str) -> None:
    pool = adaptation_pool()
    pools = split_adaptation_pool(pool, validation_fraction=0.25, seed=42)
    with pytest.raises(ValueError, match=message):
        build_nested_adaptation_subsets(
            adaptation_pool=pool, pools=pools, fractions=fractions, seed=42
        )


def test_saved_subsets_round_trip_for_exact_reuse_by_the_ablation(tmp_path: Path) -> None:
    pool = adaptation_pool()
    pools = split_adaptation_pool(pool, validation_fraction=0.25, seed=42)
    budgets = {
        seed: build_nested_adaptation_subsets(
            adaptation_pool=pool, pools=pools, fractions=FRACTIONS, seed=seed
        )
        for seed in (42, 123)
    }
    destination = tmp_path / "adaptation_subsets.json"
    save_adaptation_subsets(budgets, destination)
    restored = load_adaptation_subsets(destination)
    assert set(restored) == {42, 123}
    for seed in (42, 123):
        for fraction in FRACTIONS:
            assert (
                restored[seed][fraction].train_sample_ids
                == budgets[seed][fraction].train_sample_ids
            )
            assert (
                restored[seed][fraction].validation_sample_ids
                == budgets[seed][fraction].validation_sample_ids
            )
    assert json.loads(destination.read_text(encoding="utf-8"))["42"]["0.05"]["fraction"] == 0.05


# ------------------------------------------------------------------------ ablation


def test_ablation_controls_cannot_be_disabled() -> None:
    settings = {
        "ablation": {
            "modes": ["head_only", "last_block", "full"],
            "control_starting_checkpoint": True,
            "control_subset_ids": True,
            "control_final_test_ids": True,
            "training_budget_policy": "equal_epochs",
        }
    }
    resolved = validate_ablation_settings(settings)
    assert resolved["modes"] == ["head_only", "last_block", "full"]

    for control in (
        "control_starting_checkpoint",
        "control_subset_ids",
        "control_final_test_ids",
    ):
        broken = {"ablation": dict(settings["ablation"], **{control: False})}
        with pytest.raises(ValueError, match=f"ablation.{control} must stay true"):
            validate_ablation_settings(broken)


def test_ablation_rejects_unknown_modes_and_budget_policies() -> None:
    with pytest.raises(ValueError, match="unknown ablation modes"):
        validate_ablation_settings({"ablation": {"modes": ["head_only", "everything"]}})
    with pytest.raises(ValueError, match="contains duplicates"):
        validate_ablation_settings({"ablation": {"modes": ["full", "full"]}})
    with pytest.raises(ValueError, match="must list at least one"):
        validate_ablation_settings({"ablation": {"modes": []}})
    with pytest.raises(ValueError, match="unsupported ablation.training_budget_policy"):
        validate_ablation_settings(
            {"ablation": {"modes": ["full"], "training_budget_policy": "equal_wall_clock"}}
        )


def test_mode_overrides_are_restricted_and_null_means_shared_settings() -> None:
    modes = ["head_only", "last_block"]
    assert resolve_mode_overrides({"mode_overrides": {"head_only": None}}, modes) == {
        "head_only": {},
        "last_block": {},
    }
    resolved = resolve_mode_overrides(
        {"mode_overrides": {"last_block": {"learning_rate": 2e-5}}}, modes
    )
    assert resolved["last_block"] == {"learning_rate": 2e-5}
    with pytest.raises(ValueError, match="supports only learning_rate and epochs"):
        resolve_mode_overrides({"mode_overrides": {"full": {"optimizer": "sgd"}}}, ["full"])
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        resolve_mode_overrides({"mode_overrides": {"full": {"learning_rate": 0}}}, ["full"])


def test_run_matrix_covers_the_declared_grid_exactly() -> None:
    matrix = build_run_matrix(
        modes=["head_only", "full"],
        fractions=[0.05, 0.5],
        subset_seeds=[42, 123],
        training_seeds=[7],
    )
    assert len(matrix) == 2 * 2 * 2 * 1
    assert len({entry["cell_id"] for entry in matrix}) == len(matrix)
    assert "head_only_p05_s42_t7" in {entry["cell_id"] for entry in matrix}


# ------------------------------------------------------------------------ summaries


def cell(mode: str, fraction: float, f1: float, **extra: Any) -> dict[str, Any]:
    return {
        "fine_tune_mode": mode,
        "adaptation_percentage": fraction,
        "labelled_images_consumed": 10,
        "overall": {"f1": f1, "accuracy": f1, "roc_auc": None, "average_precision": None},
        **extra,
    }


def test_recovery_summary_reports_spread_and_uses_the_measured_zero_percent() -> None:
    rows = [
        cell("none", 0.0, 0.2),
        cell("head_only", 0.05, 0.4),
        cell("head_only", 0.05, 0.8),
    ]
    summary = summarise_recovery(rows)
    assert summary["zero_percent_f1"] == pytest.approx(0.2)
    f1_rows = [row for row in summary["rows"] if row["metric"] == "f1" and row["runs"] == 2]
    assert len(f1_rows) == 1
    entry = f1_rows[0]
    assert entry["mean"] == pytest.approx(0.6)
    assert entry["minimum"] == pytest.approx(0.4)
    assert entry["maximum"] == pytest.approx(0.8)
    assert entry["standard_deviation"] > 0
    assert entry["recovery_vs_zero_percent"] == pytest.approx(0.4)


def test_mode_comparison_excludes_the_zero_percent_row() -> None:
    rows = [
        cell("none", 0.0, 0.1),
        cell("head_only", 0.5, 0.9, trainable_parameters=769, training_seconds=1.0),
        cell("full", 0.5, 0.7, trainable_parameters=87_456_769, training_seconds=9.0),
    ]
    table = summarise_by_mode(rows)
    assert {entry["fine_tune_mode"] for entry in table} == {"head_only", "full"}
    full = next(entry for entry in table if entry["fine_tune_mode"] == "full")
    assert full["trainable_parameters"] == [87_456_769]
    assert full["mean_training_seconds"] == pytest.approx(9.0)
