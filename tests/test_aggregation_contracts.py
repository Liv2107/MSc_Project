"""Aggregation contracts: provenance, honest gaps, and comparable operating points.

The consolidated tables are what a dissertation chapter is written from, so these tests
check the properties that would otherwise put a wrong number in a thesis: a smoke run
must not be aggregated as a result, an unmeasured quantity must stay undefined rather
than becoming zero, every row must name the file it came from, and a threshold-dependent
metric must never be compared against a reference measured at a different threshold.

All fixtures are synthetic run directories built in ``tmp_path``. Nothing here reads the
real ``outputs/`` tree, so the suite stays deterministic and CPU-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.evaluation.aggregation import (
    CONSOLIDATED_COLUMNS,
    MINIMUM_RELIABLE_GAP,
    RunRecord,
    consolidate_runs,
    degradation_rows,
    describe_run,
    discover_runs,
    in_distribution_reference,
    load_metrics,
    reportable_runs,
    summarise_recovery,
    summarise_runs,
    write_table,
)


def metric_block(
    *,
    f1: float = 0.8,
    roc_auc: float | None = 0.9,
    threshold: float = 0.5,
    true_positive: int = 40,
    false_negative: int = 10,
    true_negative: int = 45,
    false_positive: int = 5,
) -> dict[str, Any]:
    return {
        "accuracy": 0.85,
        "precision": 0.889,
        "recall": 0.8,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": None if roc_auc is None else 0.91,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "support": true_positive + false_negative + true_negative + false_positive,
        "threshold": threshold,
    }


def make_run(
    root: Path,
    run_id: str,
    *,
    metrics_filename: str | None,
    metrics: dict[str, Any] | None = None,
    status: str | None = "completed",
    experiment_name: str = "tiny_experiment",
    unseen: str | None = None,
) -> Path:
    """Write a minimal but structurally faithful run directory."""

    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    config: dict[str, Any] = {
        "experiment": {"name": experiment_name, "type": run_id.split("-", 1)[0]},
        "training": {"epochs": 10, "batch_size": 32, "learning_rate": 0.001,
                     "fine_tune_mode": "head_only"},
        "model": {"clip_model_name": "openai/clip-vit-base-patch32"},
        "data": {"manifest_path": "data/manifests/tiny.csv"},
        "reproducibility": {"seed": 42},
    }
    if unseen is not None:
        config["generators"] = {"unseen": unseen, "train": ["adm"], "test": [unseen]}
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps({"python": "3.11.15", "cuda_available": False, "packages": {"torch": "2.13.0"}}),
        encoding="utf-8",
    )
    if status is not None:
        (run_dir / "status.json").write_text(
            json.dumps({"run_id": run_id, "status": status}), encoding="utf-8"
        )
    if metrics_filename is not None and metrics is not None:
        (run_dir / metrics_filename).write_text(json.dumps(metrics), encoding="utf-8")
        (run_dir / "artefacts.json").write_text(
            json.dumps([{"path": metrics_filename, "bytes": 1, "sha256": "deadbeef"}]),
            encoding="utf-8",
        )
    return run_dir


def unseen_metrics(
    *, held_out: str = "biggan", in_distribution_f1: float = 0.9, unseen_f1: float = 0.8
) -> dict[str, Any]:
    return {
        "protocol": "leave_one_generator_out",
        "held_out_generator": held_out,
        "manifest_sha256": "abc123",
        "best_epoch": 9,
        "best_validation_score": 0.83,
        "decision_threshold_default": 0.5,
        "decision_threshold_validation_selected": 0.37,
        "thresholds": {
            "default": {"provenance": "fixed_prior", "value": 0.5},
            "baseline_validation_selected": {"provenance": "seen_validation_only", "value": 0.37},
        },
        "in_distribution_test": {
            "at_default_threshold": metric_block(f1=in_distribution_f1, roc_auc=0.95),
            "per_generator": {"adm": metric_block(f1=0.7), "real": metric_block(roc_auc=None)},
        },
        "unseen_test": {
            "at_default_threshold": metric_block(f1=unseen_f1, roc_auc=0.90),
            "at_validation_selected_threshold": metric_block(f1=0.78, threshold=0.37, roc_auc=0.90),
        },
        "generalisation_gap": {
            "roc_auc": {"in_distribution": 0.95, "unseen": 0.90, "absolute_drop": 0.05},
            "f1_at_default_threshold": {
                "in_distribution": in_distribution_f1,
                "unseen": unseen_f1,
                "absolute_drop": in_distribution_f1 - unseen_f1,
            },
        },
    }


def recovery_cell(
    *, percentage: float, f1: float, roc_auc: float, mode: str = "head_only", seed: int = 42
) -> dict[str, Any]:
    return {
        "cell_id": f"{mode}_p{int(percentage * 100):02d}_s{seed}",
        "fine_tune_mode": mode if percentage > 0 else "none",
        "adaptation_percentage": percentage,
        "subset_seed": seed if percentage > 0 else None,
        "training_seed": seed if percentage > 0 else None,
        "held_out_generator": "biggan",
        "labelled_images_consumed": int(percentage * 16000),
        "trainable_parameters": 769 if percentage > 0 else None,
        "total_parameters": 87456769,
        "learning_rate": 0.001,
        "epochs": 10,
        "best_epoch": 5,
        "training_seconds": 100.0,
        "starting_checkpoint": "/somewhere/best_checkpoint.pt",
        "cell_checkpoint_sha256": {"best_checkpoint.pt": "cafe1234"},
        "overall": metric_block(f1=f1, roc_auc=roc_auc),
        "at_adaptation_selected_threshold": metric_block(f1=f1 + 0.01, roc_auc=roc_auc,
                                                         threshold=0.45),
        "at_baseline_threshold": metric_block(f1=f1 + 0.02, roc_auc=roc_auc, threshold=0.37),
        "thresholds": {
            "adaptation_validation_selected": {
                "provenance": "adaptation_validation",
                "value": 0.45,
            },
            "baseline_unchanged": {"provenance": "seen_validation_only", "value": 0.37},
        },
    }


def recovery_metrics(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "protocol": "limited_data_recovery",
        "held_out_generator": "biggan",
        "manifest_sha256": "abc123",
        "cells": cells,
    }


@pytest.fixture
def study(tmp_path: Path) -> Path:
    """A small but complete study: baseline, unseen, recovery, a smoke run, a failure."""

    root = tmp_path / "outputs"
    root.mkdir()
    make_run(
        root,
        "baseline-20260101T000000000000Z-aaaa-0001",
        metrics_filename="test_metrics.json",
        metrics={
            "best_epoch": 7,
            "best_validation_score": 0.85,
            "overall": metric_block(f1=0.86),
            "per_generator": {"biggan": metric_block(f1=0.63), "real": metric_block(roc_auc=None)},
        },
    )
    make_run(
        root,
        "unseen_generator-20260102T000000000000Z-bbbb-0002",
        metrics_filename="unseen_generator_metrics.json",
        metrics=unseen_metrics(),
        unseen="biggan",
    )
    make_run(
        root,
        "fine_tuning-20260103T000000000000Z-cccc-0003",
        metrics_filename="recovery_metrics.json",
        metrics=recovery_metrics(
            [
                recovery_cell(percentage=0.0, f1=0.80, roc_auc=0.90),
                recovery_cell(percentage=0.05, f1=0.84, roc_auc=0.94),
                recovery_cell(percentage=0.50, f1=0.90, roc_auc=0.97),
            ]
        ),
        unseen="biggan",
    )
    make_run(
        root,
        "unseen_generator-20260104T000000000000Z-dddd-0004",
        metrics_filename="unseen_generator_metrics.json",
        metrics=unseen_metrics(),
        experiment_name="SMOKE_synthetic_unseen_biggan",
        unseen="biggan",
    )
    make_run(
        root,
        "unseen_generator-20260105T000000000000Z-eeee-0005",
        metrics_filename=None,
        status="failed",
        unseen="biggan",
    )
    # An interrupted grid: no status.json at all, which is what a killed run leaves.
    make_run(
        root,
        "ablation-20260106T000000000000Z-ffff-0006",
        metrics_filename=None,
        status=None,
        unseen="biggan",
    )
    return root


# ------------------------------------------------------------------------------ discovery


def test_discovery_reports_every_run_and_its_true_status(study: Path) -> None:
    records = discover_runs(study)
    by_id = {record.run_id: record for record in records}
    assert len(records) == 6

    # A missing status.json is an interruption, not a recorded failure. Conflating the
    # two would hide a grid that never finished.
    assert by_id["ablation-20260106T000000000000Z-ffff-0006"].status == "incomplete"
    assert by_id["unseen_generator-20260105T000000000000Z-eeee-0005"].status == "failed"
    assert by_id["baseline-20260101T000000000000Z-aaaa-0001"].status == "completed"

    # Oldest first by run timestamp, so a reader can follow the study in the order it
    # happened rather than in whatever order the filesystem returns.
    starts = [record.started_at for record in records]
    assert starts == sorted(start for start in starts if start is not None)
    assert records[0].experiment_type == "baseline"
    assert records[-1].experiment_type == "ablation"


def test_discovery_reads_metadata_from_the_resolved_config(study: Path) -> None:
    record = describe_run(study / "unseen_generator-20260102T000000000000Z-bbbb-0002")
    assert record is not None
    assert record.held_out_generator == "biggan"
    assert record.fine_tune_mode == "head_only"
    assert record.model_name == "openai/clip-vit-base-patch32"
    assert record.learning_rate == 0.001
    assert record.epochs == 10
    assert record.seed == 42
    assert record.started_at == "20260102T000000000000Z"


def test_directories_that_are_not_runs_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "report").mkdir()
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    assert discover_runs(tmp_path) == []
    assert describe_run(tmp_path / "report") is None


def test_synthetic_smoke_runs_are_excluded_unless_opted_into(study: Path) -> None:
    records = discover_runs(study)
    smoke = [record for record in records if record.is_synthetic_smoke]
    assert [record.run_id for record in smoke] == [
        "unseen_generator-20260104T000000000000Z-dddd-0004"
    ]

    default = reportable_runs(records)
    assert all(not record.is_synthetic_smoke for record in default)
    assert len(default) == 3

    opted_in = reportable_runs(records, include_synthetic_smoke=True)
    assert len(opted_in) == 4
    # Even when included, the rows must still say what they are.
    assert any(record.is_synthetic_smoke for record in opted_in)


def test_loading_metrics_from_a_run_without_them_raises(study: Path) -> None:
    record = describe_run(study / "unseen_generator-20260105T000000000000Z-eeee-0005")
    assert record is not None
    assert record.has_metrics is False
    with pytest.raises(FileNotFoundError, match="unseen_generator_metrics.json"):
        load_metrics(record)


# -------------------------------------------------------------------------- consolidation


def test_every_consolidated_row_names_its_source_file_and_digest(study: Path) -> None:
    rows = consolidate_runs(reportable_runs(discover_runs(study)))
    assert rows
    for row in rows:
        assert row["run_id"]
        assert row["source_file"] in {
            "test_metrics.json",
            "unseen_generator_metrics.json",
            "recovery_metrics.json",
        }
        # Traceability is the whole point: a plotted number a reader cannot verify
        # against a hashed artefact is not evidence.
        assert row["source_sha256"] == "deadbeef"
        assert set(row) == set(CONSOLIDATED_COLUMNS)


def test_prevalence_is_derived_from_saved_counts(study: Path) -> None:
    rows = consolidate_runs(reportable_runs(discover_runs(study)))
    row = next(row for row in rows if row["experiment_type"] == "baseline")
    # 40 true positives + 10 false negatives out of 100 samples.
    assert row["support"] == 100
    assert row["positive_prevalence"] == pytest.approx(0.5)


def test_undefined_metrics_stay_undefined_rather_than_becoming_zero(study: Path) -> None:
    rows = consolidate_runs(reportable_runs(discover_runs(study)))
    real = next(
        row
        for row in rows
        if row["experiment_type"] == "baseline" and row["generator"] == "real"
    )
    # A single-class slice has no ROC-AUC. Zero would read as "very bad" rather than
    # "not measurable", and would drag any mean computed over the column.
    assert real["roc_auc"] is None
    assert real["average_precision"] is None


def test_each_cell_is_reported_at_every_saved_operating_point(study: Path) -> None:
    rows = consolidate_runs(reportable_runs(discover_runs(study)))
    adapted = [
        row
        for row in rows
        if row["experiment_type"] == "fine_tuning"
        and row["adaptation_percentage"] == 0.05
        and row["generator"] is None
    ]
    assert {row["operating_point"] for row in adapted} == {
        "default",
        "adaptation_selected",
        "baseline_unchanged",
    }
    # Threshold-free metrics must be identical across operating points; only the
    # threshold-dependent ones may move.
    assert len({row["roc_auc"] for row in adapted}) == 1
    assert len({row["f1"] for row in adapted}) == 3


def test_trainable_parameter_fraction_is_derived_only_when_both_counts_exist(
    study: Path,
) -> None:
    rows = consolidate_runs(reportable_runs(discover_runs(study)))
    adapted = next(
        row for row in rows if row["experiment_type"] == "fine_tuning"
        and row["adaptation_percentage"] == 0.5 and row["operating_point"] == "default"
    )
    assert adapted["trainable_parameters"] == 769
    assert adapted["trainable_parameter_fraction"] == pytest.approx(769 / 87456769)

    zero = next(
        row for row in rows if row["experiment_type"] == "fine_tuning"
        and row["adaptation_percentage"] == 0.0 and row["operating_point"] == "default"
    )
    # Nothing was fitted at 0%, so there is no trainable-parameter count to report.
    assert zero["trainable_parameters"] is None
    assert zero["trainable_parameter_fraction"] is None


def test_run_summary_counts_conditions_and_keeps_failed_runs_visible(study: Path) -> None:
    records = discover_runs(study)
    rows = consolidate_runs(reportable_runs(records))
    summary = summarise_runs(records, rows)
    by_id = {row["run_id"]: row for row in summary}

    assert len(summary) == 6
    assert by_id["unseen_generator-20260105T000000000000Z-eeee-0005"]["status"] == "failed"
    assert by_id["unseen_generator-20260105T000000000000Z-eeee-0005"]["evaluated_conditions"] == 0
    assert by_id["fine_tuning-20260103T000000000000Z-cccc-0003"]["cells"] == 3
    assert by_id["baseline-20260101T000000000000Z-aaaa-0001"]["torch"] == "2.13.0"


# ------------------------------------------------------------------------------ recovery


def test_recovery_is_measured_against_the_zero_percent_row(study: Path) -> None:
    records = reportable_runs(discover_runs(study))
    rows = consolidate_runs(records)
    summary = summarise_recovery(rows, records=records)

    entry = next(
        row
        for row in summary
        if row["metric"] == "roc_auc"
        and row["adaptation_percentage"] == 0.05
        and row["operating_point"] == "default"
    )
    assert entry["zero_percent_reference"] == pytest.approx(0.90)
    assert entry["absolute_recovery"] == pytest.approx(0.04)
    assert entry["relative_improvement"] == pytest.approx(0.04 / 0.90)
    # The 0% condition is the reference, not a recovery result of its own.
    assert all(row["adaptation_percentage"] != 0.0 for row in summary)


def test_a_single_run_reports_unmeasured_spread_rather_than_zero_uncertainty(
    study: Path,
) -> None:
    records = reportable_runs(discover_runs(study))
    summary = summarise_recovery(consolidate_runs(records), records=records)
    entry = next(row for row in summary if row["metric"] == "f1")
    assert entry["runs"] == 1
    assert entry["standard_deviation"] == 0.0
    # A standard error of 0.0 would claim a precision one run cannot support.
    assert entry["standard_error"] is None


def test_repeated_seeds_produce_a_measured_spread(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    make_run(
        root,
        "fine_tuning-20260101T000000000000Z-aaaa-0001",
        metrics_filename="recovery_metrics.json",
        metrics=recovery_metrics(
            [
                recovery_cell(percentage=0.0, f1=0.80, roc_auc=0.90),
                recovery_cell(percentage=0.05, f1=0.84, roc_auc=0.94, seed=42),
                recovery_cell(percentage=0.05, f1=0.88, roc_auc=0.96, seed=123),
            ]
        ),
        unseen="biggan",
    )
    records = reportable_runs(discover_runs(root))
    summary = summarise_recovery(consolidate_runs(records), records=records)
    entry = next(
        row for row in summary if row["metric"] == "f1" and row["operating_point"] == "default"
    )
    assert entry["runs"] == 2
    assert entry["mean"] == pytest.approx(0.86)
    assert entry["standard_deviation"] == pytest.approx(0.0282842712, abs=1e-6)
    assert entry["standard_error"] == pytest.approx(0.02, abs=1e-6)
    assert entry["minimum"] == pytest.approx(0.84)
    assert entry["maximum"] == pytest.approx(0.88)


def test_gap_closed_is_flagged_unreliable_when_the_measured_gap_is_tiny(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    # An in-distribution/unseen gap far below the reliability threshold.
    tiny = unseen_metrics(in_distribution_f1=0.805, unseen_f1=0.80)
    assert MINIMUM_RELIABLE_GAP > 0.805 - 0.80
    make_run(
        root,
        "unseen_generator-20260101T000000000000Z-aaaa-0001",
        metrics_filename="unseen_generator_metrics.json",
        metrics=tiny,
        unseen="biggan",
    )
    make_run(
        root,
        "fine_tuning-20260102T000000000000Z-bbbb-0002",
        metrics_filename="recovery_metrics.json",
        metrics=recovery_metrics(
            [
                recovery_cell(percentage=0.0, f1=0.80, roc_auc=0.90),
                recovery_cell(percentage=0.5, f1=0.95, roc_auc=0.99),
            ]
        ),
        unseen="biggan",
    )
    records = reportable_runs(discover_runs(root))
    summary = summarise_recovery(consolidate_runs(records), records=records)
    entry = next(
        row for row in summary if row["metric"] == "f1" and row["operating_point"] == "default"
    )
    assert entry["generalisation_gap"] == pytest.approx(0.005, abs=1e-9)
    # The fraction is still reported -- it is not fabricated or hidden -- but a reader is
    # told the denominator cannot support the claim.
    assert entry["gap_closed_fraction"] is not None
    assert entry["gap_closed_is_reliable"] is False


def test_threshold_dependent_metrics_get_no_reference_at_a_shifted_operating_point(
    study: Path,
) -> None:
    records = reportable_runs(discover_runs(study))
    summary = summarise_recovery(consolidate_runs(records), records=records)

    shifted_f1 = next(
        row
        for row in summary
        if row["metric"] == "f1" and row["operating_point"] == "adaptation_selected"
    )
    # The saved in-distribution reference was measured at the default threshold. Quoting
    # it against an F1 at an adaptation-selected threshold would compare two operating
    # points and call the difference a generalisation gap.
    assert shifted_f1["in_distribution_reference"] is None
    assert shifted_f1["generalisation_gap"] is None
    assert shifted_f1["gap_closed_fraction"] is None
    # The threshold-free ranking metric is unaffected by the operating point, so it keeps
    # its reference.
    shifted_auc = next(
        row
        for row in summary
        if row["metric"] == "roc_auc" and row["operating_point"] == "adaptation_selected"
    )
    assert shifted_auc["in_distribution_reference"] == pytest.approx(0.95)


def test_missing_unseen_run_leaves_the_reference_undefined(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    make_run(
        root,
        "fine_tuning-20260101T000000000000Z-aaaa-0001",
        metrics_filename="recovery_metrics.json",
        metrics=recovery_metrics(
            [
                recovery_cell(percentage=0.0, f1=0.80, roc_auc=0.90),
                recovery_cell(percentage=0.5, f1=0.90, roc_auc=0.97),
            ]
        ),
        unseen="biggan",
    )
    records = reportable_runs(discover_runs(root))
    value, run_id = in_distribution_reference(records, "roc_auc", held_out_generator="biggan")
    assert value is None and run_id is None

    summary = summarise_recovery(consolidate_runs(records), records=records)
    entry = next(row for row in summary if row["metric"] == "roc_auc")
    # Recovery against the 0% point is still measurable; only the ceiling is unknown.
    assert entry["absolute_recovery"] == pytest.approx(0.07)
    assert entry["in_distribution_reference"] is None
    assert entry["gap_closed_fraction"] is None


def test_the_reference_is_taken_from_the_matching_held_out_generator(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    make_run(
        root,
        "unseen_generator-20260101T000000000000Z-aaaa-0001",
        metrics_filename="unseen_generator_metrics.json",
        metrics=unseen_metrics(held_out="biggan", in_distribution_f1=0.90),
        unseen="biggan",
    )
    make_run(
        root,
        "unseen_generator-20260102T000000000000Z-bbbb-0002",
        metrics_filename="unseen_generator_metrics.json",
        metrics=unseen_metrics(held_out="glide", in_distribution_f1=0.70),
        unseen="glide",
    )
    records = reportable_runs(discover_runs(root))
    value, run_id = in_distribution_reference(records, "f1", held_out_generator="biggan")
    assert value == pytest.approx(0.90)
    assert run_id == "unseen_generator-20260101T000000000000Z-aaaa-0001"
    # A reference must never be borrowed from a different held-out generator's run.
    other, _ = in_distribution_reference(records, "f1", held_out_generator="glide")
    assert other == pytest.approx(0.70)


# --------------------------------------------------------------------------- degradation


def test_degradation_pairs_the_two_sides_at_one_operating_point(study: Path) -> None:
    rows = consolidate_runs(reportable_runs(discover_runs(study)))
    degradation = degradation_rows(rows)
    entry = next(
        row
        for row in degradation
        if row["metric"] == "roc_auc" and row["operating_point"] == "default"
    )
    assert entry["in_distribution"] == pytest.approx(0.95)
    assert entry["unseen"] == pytest.approx(0.90)
    assert entry["absolute_drop"] == pytest.approx(0.05)
    assert entry["relative_drop"] == pytest.approx(0.05 / 0.95)
    assert entry["unseen_prevalence"] == pytest.approx(0.5)


def test_degradation_leaves_the_drop_undefined_when_one_side_is_missing(study: Path) -> None:
    rows = consolidate_runs(reportable_runs(discover_runs(study)))
    entry = next(
        row
        for row in degradation_rows(rows)
        if row["operating_point"] == "validation_selected" and row["metric"] == "f1"
    )
    # The fixture never evaluated in-distribution at the selected threshold, so there is
    # nothing to subtract from. Substituting the default-threshold value would report a
    # threshold change as a generalisation gap.
    assert entry["unseen"] is not None
    assert entry["in_distribution"] is None
    assert entry["absolute_drop"] is None


def test_degradation_ignores_runs_that_measured_only_one_side(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    make_run(
        root,
        "baseline-20260101T000000000000Z-aaaa-0001",
        metrics_filename="test_metrics.json",
        metrics={"overall": metric_block(), "per_generator": {}},
    )
    rows = consolidate_runs(reportable_runs(discover_runs(root)))
    # A baseline measured no held-out generator, so it contributes no degradation row.
    assert degradation_rows(rows) == []


# -------------------------------------------------------------------------------- output


def test_written_table_uses_the_declared_schema(tmp_path: Path) -> None:
    destination = write_table(
        [{"a": 1, "b": None, "ignored": "x"}], ["a", "b", "c"], tmp_path / "out.csv"
    )
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "a,b,c"
    # A missing value is an empty cell, never a zero and never a placeholder number.
    assert lines[1] == "1,,"


def test_consolidating_a_run_without_metrics_is_skipped_not_fatal(study: Path) -> None:
    records = discover_runs(study)
    failed = [
        record
        for record in records
        if record.run_id == "unseen_generator-20260105T000000000000Z-eeee-0005"
    ]
    assert consolidate_runs(failed) == []


def test_run_record_reportability_requires_completion_metrics_and_non_smoke() -> None:
    base = {
        "run_id": "x",
        "run_dir": Path("x"),
        "experiment_type": "baseline",
        "experiment_name": "x",
        "metrics_path": None,
    }
    assert not RunRecord(**base, status="completed", is_synthetic_smoke=False).is_reportable
    assert not RunRecord(**base, status="failed", is_synthetic_smoke=False).is_reportable
