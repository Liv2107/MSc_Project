"""Generate Chapter 4 tables and figures from saved run outputs.

This script is strictly a reader. It never loads a model, never scores an image, and
never recomputes a metric from raw data: every number it emits is copied from a
``*_metrics.json`` or ``*_predictions.csv`` file that an experiment run already wrote
and hashed into its ``artefacts.json``. That is what makes the generated chapter
material auditable -- each figure and table cell traces back to a specific run
directory, and re-running this script cannot change a result.

Curves are the one exception worth naming: ROC and precision-recall coordinates are
derived here from the *saved per-sample scores*, using the same
``src.evaluation.metrics`` functions the runners use. No new decisions are taken.

The report has two halves:

* **Per-experiment sections** -- one run of each protocol, reported in depth (curves,
  confusion matrices, composition and threshold tables).
* **A consolidated section** -- *every* completed run aggregated into one tidy table,
  plus the cross-run comparison figures and the Chapter 4 metric table. This half is
  built by :mod:`src.evaluation.aggregation`, and it is what future notebooks should
  read instead of walking ``outputs/`` themselves.

Synthetic smoke runs (experiment names prefixed ``SMOKE_``) are excluded from both
halves by default, because they are pipeline evidence on procedurally generated images
rather than detector performance. ``--include-synthetic-smoke`` opts them back in and
every generated table then carries an ``is_synthetic_smoke`` column.

Usage
-----
    python -m scripts.build_report --output outputs/report
    python -m scripts.build_report --run outputs/unseen_generator-... --run outputs/fine_tuning-...

With no ``--run``, the newest completed run of each experiment type under the output
root is used, and the choice is recorded in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.aggregation import (
    CONSOLIDATED_COLUMNS,
    EXPERIMENT_METRIC_FILES,
    RECOVERY_SUMMARY_COLUMNS,
    RUN_SUMMARY_COLUMNS,
    SUMMARY_METRICS,
    RunRecord,
    consolidate_runs,
    degradation_rows,
    describe_run,
    reportable_runs,
    summarise_recovery,
    summarise_runs,
    write_table,
)
from src.evaluation.aggregation import (
    discover_runs as discover_all_runs,
)
from src.evaluation.metrics import (
    compute_binary_metrics,
    confusion_matrix,
    precision_recall_curve_data,
    roc_curve_data,
    threshold_scores,
)
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_fine_tuning_recovery,
    plot_generalisation_degradation,
    plot_generator_performance,
    plot_parameter_efficiency,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_training_curves,
)

#: Protocols a complete study is expected to contain, in the order they must be run.
EXPECTED_PROTOCOLS: tuple[str, ...] = (
    "baseline",
    "unseen_generator",
    "fine_tuning",
    "ablation",
)


@dataclass(frozen=True, slots=True)
class DiscoveredRun:
    experiment_type: str
    run_dir: Path
    metrics_path: Path


def _is_completed(run_dir: Path) -> bool:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        return bool(isinstance(status, dict) and status.get("status") == "completed")
    except json.JSONDecodeError:
        return False


def _is_synthetic_smoke(run_dir: Path) -> bool:
    record = describe_run(run_dir)
    return record is not None and record.is_synthetic_smoke


def discover_runs(
    output_root: Path,
    explicit: Sequence[Path] = (),
    *,
    include_synthetic_smoke: bool = False,
) -> list[DiscoveredRun]:
    """Find the runs to report on, preferring explicit paths over newest-completed.

    Synthetic smoke runs are skipped during automatic discovery unless opted into: a
    smoke run that happened to finish most recently would otherwise be picked up as the
    newest run of its protocol and reported as a result. An explicitly named ``--run``
    is always honoured, because naming it is the opt-in.
    """

    discovered: list[DiscoveredRun] = []
    if explicit:
        for run_dir in explicit:
            resolved = run_dir.resolve()
            if not resolved.is_dir():
                raise FileNotFoundError(f"run directory not found: {resolved}")
            matched = [
                (experiment_type, resolved / filename)
                for experiment_type, filename in EXPERIMENT_METRIC_FILES.items()
                if (resolved / filename).is_file()
            ]
            if not matched:
                raise ValueError(f"no recognised metrics file inside {resolved}")
            for experiment_type, metrics_path in matched:
                discovered.append(DiscoveredRun(experiment_type, resolved, metrics_path))
        return discovered

    for experiment_type, filename in EXPERIMENT_METRIC_FILES.items():
        candidates = [
            path
            for path in sorted(output_root.glob(f"{experiment_type}-*"))
            if path.is_dir()
            and (path / filename).is_file()
            and _is_completed(path)
            and (include_synthetic_smoke or not _is_synthetic_smoke(path))
        ]
        if candidates:
            newest = candidates[-1]
            discovered.append(DiscoveredRun(experiment_type, newest, newest / filename))
    return discovered


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"prediction table is empty: {path}")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                "sample_id": row["sample_id"],
                "label": int(row["label"]),
                "score": float(row["score"]),
                "generator": row["generator"],
            }
        )
    return parsed


def _read_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    history: list[dict[str, Any]] = []
    for row in rows:
        entry: dict[str, Any] = {"epoch": int(row["epoch"]), "split": row["split"]}
        for key, value in row.items():
            if key in {"epoch", "split"} or value in {"", None}:
                continue
            entry[key] = float(value)
        history.append(entry)
    return history


def _save_figure(figure: Any, destination_stem: Path) -> list[str]:
    """Write vector and raster copies; both are wanted for a dissertation."""

    destination_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for suffix in (".pdf", ".png"):
        path = destination_stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight")
        written.append(path.name)
    return written


def _write_markdown_table(
    rows: Sequence[Sequence[Any]], headers: Sequence[str], destination: Path
) -> None:
    lines = ["| " + " | ".join(str(header) for header in headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        rendered = []
        for value in row:
            if value is None:
                rendered.append("undefined")
            elif isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv_table(
    rows: Sequence[Sequence[Any]], headers: Sequence[str], destination: Path
) -> None:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _emit_table(
    rows: Sequence[Sequence[Any]], headers: Sequence[str], stem: Path, artefacts: list[str]
) -> None:
    _write_markdown_table(rows, headers, stem.with_suffix(".md"))
    _write_csv_table(rows, headers, stem.with_suffix(".csv"))
    artefacts.extend([stem.with_suffix(".md").name, stem.with_suffix(".csv").name])


def report_baseline(run: DiscoveredRun, output_dir: Path) -> dict[str, Any]:
    """In-distribution reference: overall metrics, per-generator table, curves."""

    metrics = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    artefacts: list[str] = []
    overall = metrics["overall"]
    _emit_table(
        [[key, overall[key]] for key in sorted(overall)],
        ["quantity", "value"],
        output_dir / "table_baseline_overall",
        artefacts,
    )
    per_generator = metrics.get("per_generator") or {}
    _emit_table(
        [
            [
                name,
                values["support"],
                values["accuracy"],
                values["precision"],
                values["recall"],
                values["f1"],
                values["roc_auc"],
                values["average_precision"],
            ]
            for name, values in sorted(per_generator.items())
        ],
        [
            "generator",
            "support",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "avg_precision",
        ],
        output_dir / "table_baseline_per_generator",
        artefacts,
    )

    predictions_path = run.run_dir / "test_predictions.csv"
    if predictions_path.is_file():
        predictions = _read_predictions(predictions_path)
        labels = [row["label"] for row in predictions]
        scores = [row["score"] for row in predictions]
        threshold = float(overall["threshold"])
        figure, _ = plot_confusion_matrix(
            confusion_matrix(labels, threshold_scores(scores, threshold=threshold)),
            title=f"Baseline in-distribution test (threshold {threshold:g})",
        )
        artefacts.extend(_save_figure(figure, output_dir / "figure_baseline_confusion_matrix"))
        if len(set(labels)) == 2:
            recomputed = compute_binary_metrics(labels, scores, threshold=threshold)
            figure, _ = plot_roc_curves(
                {"in-distribution test": roc_curve_data(labels, scores)},
                areas={"in-distribution test": recomputed.roc_auc},
                supports={"in-distribution test": recomputed.support},
                title="Baseline ROC",
            )
            artefacts.extend(_save_figure(figure, output_dir / "figure_baseline_roc"))
            figure, _ = plot_precision_recall_curves(
                {"in-distribution test": precision_recall_curve_data(labels, scores)},
                prevalence=sum(labels) / len(labels),
                average_precisions={"in-distribution test": recomputed.average_precision},
                supports={"in-distribution test": recomputed.support},
                title="Baseline precision-recall",
            )
            artefacts.extend(_save_figure(figure, output_dir / "figure_baseline_pr"))
            # Guard against a figure drifting from the metrics file it claims to show.
            if abs(recomputed.f1 - float(overall["f1"])) > 1e-9:
                raise ValueError(
                    "recomputed baseline f1 disagrees with the saved metrics file; "
                    "predictions and metrics are inconsistent"
                )

    history_path = run.run_dir / "train_history.csv"
    if history_path.is_file():
        figure, _ = plot_training_curves(
            _read_history(history_path), best_epoch=metrics.get("best_epoch")
        )
        artefacts.extend(_save_figure(figure, output_dir / "figure_baseline_training_history"))
    return {"run_dir": str(run.run_dir), "artefacts": artefacts}


def report_unseen_generator(run: DiscoveredRun, output_dir: Path) -> dict[str, Any]:
    """The headline generalisation result: in-distribution versus held-out generator."""

    metrics = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    artefacts: list[str] = []
    held_out = metrics["held_out_generator"]
    unseen_default = metrics["unseen_test"]["at_default_threshold"]
    unseen_selected = metrics["unseen_test"]["at_validation_selected_threshold"]
    in_distribution = metrics["in_distribution_test"]["at_default_threshold"]

    _emit_table(
        [
            [
                "in-distribution test",
                float(metrics["decision_threshold_default"]),
                in_distribution["support"],
                in_distribution["accuracy"],
                in_distribution["precision"],
                in_distribution["recall"],
                in_distribution["f1"],
                in_distribution["roc_auc"],
            ],
            [
                f"unseen ({held_out})",
                float(metrics["decision_threshold_default"]),
                unseen_default["support"],
                unseen_default["accuracy"],
                unseen_default["precision"],
                unseen_default["recall"],
                unseen_default["f1"],
                unseen_default["roc_auc"],
            ],
            [
                f"unseen ({held_out}), validation-selected threshold",
                float(metrics["decision_threshold_validation_selected"]),
                unseen_selected["support"],
                unseen_selected["accuracy"],
                unseen_selected["precision"],
                unseen_selected["recall"],
                unseen_selected["f1"],
                unseen_selected["roc_auc"],
            ],
        ],
        ["condition", "threshold", "support", "accuracy", "precision", "recall", "f1", "roc_auc"],
        output_dir / "table_unseen_generalisation",
        artefacts,
    )

    curves: dict[str, Any] = {}
    areas: dict[str, float | None] = {}
    supports: dict[str, int] = {}
    prevalences: list[float] = []
    pr_curves: dict[str, Any] = {}
    average_precisions: dict[str, float | None] = {}
    for label, filename in (
        ("in-distribution test", "in_distribution_test_predictions.csv"),
        (f"unseen ({held_out})", "unseen_test_predictions.csv"),
    ):
        path = run.run_dir / filename
        if not path.is_file():
            continue
        predictions = _read_predictions(path)
        labels = [row["label"] for row in predictions]
        scores = [row["score"] for row in predictions]
        if len(set(labels)) != 2:
            continue
        computed = compute_binary_metrics(labels, scores)
        curves[label] = roc_curve_data(labels, scores)
        areas[label] = computed.roc_auc
        supports[label] = computed.support
        pr_curves[label] = precision_recall_curve_data(labels, scores)
        average_precisions[label] = computed.average_precision
        prevalences.append(sum(labels) / len(labels))
    if curves:
        figure, _ = plot_roc_curves(
            curves,
            areas=areas,
            supports=supports,
            title=f"Ranking transfer to the held-out generator ({held_out})",
        )
        artefacts.extend(_save_figure(figure, output_dir / "figure_unseen_roc"))
        figure, _ = plot_precision_recall_curves(
            pr_curves,
            prevalence=sum(prevalences) / len(prevalences),
            average_precisions=average_precisions,
            supports=supports,
            title=f"Precision-recall on the held-out generator ({held_out})",
        )
        artefacts.extend(_save_figure(figure, output_dir / "figure_unseen_pr"))

    unseen_predictions = run.run_dir / "unseen_test_predictions.csv"
    if unseen_predictions.is_file():
        predictions = _read_predictions(unseen_predictions)
        labels = [row["label"] for row in predictions]
        scores = [row["score"] for row in predictions]
        for name, threshold in (
            ("default", float(metrics["decision_threshold_default"])),
            ("validation_selected", float(metrics["decision_threshold_validation_selected"])),
        ):
            figure, _ = plot_confusion_matrix(
                confusion_matrix(labels, threshold_scores(scores, threshold=threshold)),
                title=f"Unseen {held_out} at {name} threshold {threshold:g}",
            )
            artefacts.extend(
                _save_figure(figure, output_dir / f"figure_unseen_confusion_{name}")
            )

    per_generator = metrics["unseen_test"].get("per_generator") or {}
    rows = [
        {
            "generator": name,
            "support": values["support"],
            "f1": values["f1"],
            "roc_auc": values["roc_auc"],
        }
        for name, values in per_generator.items()
    ]
    if rows:
        for metric_name in ("f1", "roc_auc"):
            if any(row[metric_name] is not None for row in rows):
                figure, _ = plot_generator_performance(rows, metric_name=metric_name)
                artefacts.extend(
                    _save_figure(figure, output_dir / f"figure_unseen_per_generator_{metric_name}")
                )

    # Threshold-free metrics are listed first: on an unseen generator a fixed-threshold
    # gap largely reflects calibration drift, so ROC-AUC and average precision are the
    # defensible headline. Both test sets are prevalence matched (see the composition
    # tables), which is what makes precision/F1/PR-AUC comparable at all.
    gap = metrics["generalisation_gap"]
    gap_rows: list[list[Any]] = []
    for metric_name, key in (
        ("roc_auc (threshold-free)", "roc_auc"),
        ("average_precision (threshold-free)", "average_precision"),
        ("f1 @ default threshold", "f1_at_default_threshold"),
    ):
        entry = gap.get(key) or {}
        gap_rows.append(
            [
                metric_name,
                entry.get("in_distribution"),
                entry.get("unseen"),
                entry.get("absolute_drop"),
            ]
        )
    selected = gap.get("f1_at_baseline_validation_selected_threshold") or {}
    if "unseen" in selected:
        gap_rows.append(
            ["f1 @ validation-selected threshold", None, selected.get("unseen"), None]
        )
    _emit_table(
        gap_rows,
        ["metric", "in_distribution", "unseen", "absolute_drop"],
        output_dir / "table_unseen_gap",
        artefacts,
    )
    _emit_table(
        [
            ["prevalence matched", gap.get("prevalence_matched")],
            ["unseen positive prevalence", gap.get("unseen_prevalence")],
            ["in-distribution positive prevalence", gap.get("in_distribution_prevalence")],
            *[
                [f"unseen test: {key}", value]
                for key, value in sorted((metrics.get("final_test_composition") or {}).items())
            ],
            *[
                [f"threshold {name}: {field}", value]
                for name, block in sorted((metrics.get("thresholds") or {}).items())
                for field, value in sorted(block.items())
            ],
        ],
        ["quantity", "value"],
        output_dir / "table_unseen_test_composition_and_thresholds",
        artefacts,
    )
    return {"run_dir": str(run.run_dir), "held_out_generator": held_out, "artefacts": artefacts}


def _recovery_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten saved cells into plot-ready rows without recomputing anything."""

    rows: list[dict[str, Any]] = []
    for cell in metrics["cells"]:
        overall = cell["overall"]
        rows.append(
            {
                "cell_id": cell.get("cell_id"),
                "fine_tune_mode": cell.get("fine_tune_mode"),
                "adaptation_percentage": float(cell["adaptation_percentage"]),
                "subset_seed": cell.get("subset_seed"),
                "training_seed": cell.get("training_seed"),
                "labelled_images_consumed": cell.get("labelled_images_consumed"),
                "held_out_fake_count": cell.get("held_out_fake_count"),
                "authentic_count": cell.get("authentic_count"),
                "trainable_parameters": cell.get("trainable_parameters"),
                "training_seconds": cell.get("training_seconds"),
                "accuracy": overall.get("accuracy"),
                "f1": overall.get("f1"),
                "roc_auc": overall.get("roc_auc"),
                "average_precision": overall.get("average_precision"),
                "threshold_default": overall.get("threshold"),
                # Threshold-free metrics are identical at every operating point; only the
                # threshold-dependent ones change, which is exactly what separates
                # calibration shift from weight adaptation.
                "f1_at_adaptation_threshold": (
                    cell.get("at_adaptation_selected_threshold") or {}
                ).get("f1"),
                "f1_at_baseline_threshold": (cell.get("at_baseline_threshold") or {}).get("f1"),
                "threshold_adaptation_selected": (
                    (cell.get("thresholds") or {}).get("adaptation_validation_selected") or {}
                ).get("value"),
                "threshold_baseline_unchanged": (
                    (cell.get("thresholds") or {}).get("baseline_unchanged") or {}
                ).get("value"),
                "held_out_generator": cell.get("held_out_generator"),
            }
        )
    return rows


def report_fine_tuning(run: DiscoveredRun, output_dir: Path) -> dict[str, Any]:
    """The recovery curve plus the tidy per-cell and summary tables behind it."""

    metrics = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    artefacts: list[str] = []
    rows = _recovery_rows(metrics)

    _emit_table(
        [
            [
                row["cell_id"],
                row["fine_tune_mode"],
                row["adaptation_percentage"],
                row["subset_seed"],
                row["training_seed"],
                row["held_out_fake_count"],
                row["labelled_images_consumed"],
                row["roc_auc"],
                row["average_precision"],
                row["threshold_default"],
                row["f1"],
                row["threshold_adaptation_selected"],
                row["f1_at_adaptation_threshold"],
                row["threshold_baseline_unchanged"],
                row["f1_at_baseline_threshold"],
            ]
            for row in rows
        ],
        [
            "cell",
            "mode",
            "percentage",
            "subset_seed",
            "training_seed",
            "held_out_images",
            "labelled_images_total",
            "roc_auc",
            "pr_auc",
            "threshold_default",
            "f1_at_default",
            "threshold_adaptation_selected",
            "f1_at_adaptation_threshold",
            "threshold_baseline",
            "f1_at_baseline_threshold",
        ],
        output_dir / "table_recovery_cells",
        artefacts,
    )
    summary = metrics.get("summary") or {}
    _emit_table(
        [
            [
                entry["fine_tune_mode"],
                entry["adaptation_percentage"],
                entry["metric"],
                entry["runs"],
                entry["mean"],
                entry["standard_deviation"],
                entry["minimum"],
                entry["maximum"],
                entry.get("recovery_vs_zero_percent"),
            ]
            for entry in summary.get("rows", [])
        ],
        [
            "mode",
            "percentage",
            "metric",
            "runs",
            "mean",
            "std_dev",
            "min",
            "max",
            "recovery_vs_0pct",
        ],
        output_dir / "table_recovery_summary",
        artefacts,
    )
    held_out = str(metrics.get("held_out_generator") or "unknown")
    # One figure per reported quantity. The threshold-dependent ones are plotted at EVERY
    # declared operating point, because that is what separates a calibration change from a
    # genuine ranking improvement; a single F1 curve cannot distinguish them.
    for metric_name, description in (
        ("roc_auc", "threshold-free"),
        ("average_precision", "threshold-free"),
        ("f1", "at the fixed 0.5 prior"),
        ("f1_at_adaptation_threshold", "at the adaptation-validation threshold"),
        ("f1_at_baseline_threshold", "at the unchanged baseline threshold"),
    ):
        if not any(row.get(metric_name) is not None for row in rows):
            continue
        figure, axes = plot_fine_tuning_recovery(rows, metric_name=metric_name)
        axes.set_title(f"Recovery on held-out {held_out}: {metric_name} ({description})")
        artefacts.extend(
            _save_figure(figure, output_dir / f"figure_recovery_{held_out}_{metric_name}")
        )
    return {
        "run_dir": str(run.run_dir),
        "held_out_generator": metrics.get("held_out_generator"),
        "artefacts": artefacts,
    }


def report_ablation(run: DiscoveredRun, output_dir: Path) -> dict[str, Any]:
    """Depth comparison at equal budget, with the fairness controls printed."""

    metrics = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    artefacts: list[str] = []
    rows = _recovery_rows(metrics)

    _emit_table(
        [
            [
                entry["adaptation_percentage"],
                entry["fine_tune_mode"],
                entry["runs"],
                entry["mean_f1"],
                entry["minimum_f1"],
                entry["maximum_f1"],
                ", ".join(str(value) for value in entry["trainable_parameters"]),
                entry.get("mean_training_seconds"),
            ]
            for entry in metrics.get("mode_comparison", [])
        ],
        [
            "percentage",
            "mode",
            "runs",
            "mean_f1",
            "min_f1",
            "max_f1",
            "trainable_parameters",
            "mean_training_seconds",
        ],
        output_dir / "table_ablation_mode_comparison",
        artefacts,
    )
    for metric_name in ("f1", "roc_auc"):
        if any(row.get(metric_name) is not None for row in rows):
            figure, _ = plot_fine_tuning_recovery(rows, metric_name=metric_name)
            artefacts.extend(_save_figure(figure, output_dir / f"figure_ablation_{metric_name}"))

    controls = metrics.get("controls") or {}
    _emit_table(
        [
            ["starting checkpoint", controls.get("starting_checkpoint")],
            ["training budget policy", controls.get("training_budget_policy")],
            [
                "modes overriding the epoch budget",
                ", ".join(controls.get("budget_policy_violated_by") or []) or "none",
            ],
            [
                "mode overrides applied",
                json.dumps(controls.get("mode_overrides_applied") or {}, sort_keys=True),
            ],
            ["final test samples", controls.get("final_test_sample_id_count")],
            [
                "shared adaptation subset digests",
                json.dumps(
                    {
                        key: value["sample_id_sha256"][:12]
                        for key, value in sorted((controls.get("subset_id_digests") or {}).items())
                    },
                    sort_keys=True,
                ),
            ],
        ],
        ["control", "value"],
        output_dir / "table_ablation_controls",
        artefacts,
    )
    (output_dir / "ablation_interpretation_note.md").write_text(
        f"# Interpreting the depth ablation\n\n{metrics.get('interpretation_note', '')}\n",
        encoding="utf-8",
    )
    artefacts.append("ablation_interpretation_note.md")
    return {"run_dir": str(run.run_dir), "artefacts": artefacts}


REPORTERS = {
    "baseline": report_baseline,
    "unseen_generator": report_unseen_generator,
    "fine_tuning": report_fine_tuning,
    "ablation": report_ablation,
}


# --------------------------------------------------------------- consolidated section


#: Column order of the Chapter 4 headline table.
CHAPTER4_COLUMNS: Sequence[str] = (
    "experiment",
    "condition",
    "evaluation_set",
    "operating_point",
    "threshold",
    "support",
    "prevalence",
    *SUMMARY_METRICS,
    "trainable_parameters",
    "labelled_images_consumed",
    "run_id",
)


def _condition_label(row: Mapping[str, Any]) -> str:
    """A human-readable name for one evaluated condition, built from saved fields."""

    percentage = row.get("adaptation_percentage")
    mode = row.get("fine_tune_mode")
    if row["experiment_type"] == "baseline":
        return "in-distribution baseline"
    if row["experiment_type"] == "unseen_generator":
        if row.get("evaluation_set") == "in_distribution_test":
            return "in-distribution reference (prevalence matched)"
        return f"unseen {row.get('held_out_generator')} @ 0% adaptation"
    return (
        f"unseen {row.get('held_out_generator')} @ "
        f"{float(percentage or 0.0) * 100:g}% adaptation, {mode}"
    )


#: Reading order of the Chapter 4 table: the protocols as the study runs them, then
#: increasing adaptation budget, then operating point. Run start time -- the order the
#: consolidated table happens to be in -- would put the baseline last.
_PROTOCOL_ORDER = {name: index for index, name in enumerate(EXPECTED_PROTOCOLS)}
_OPERATING_POINT_ORDER = {
    "default": 0,
    "validation_selected": 1,
    "adaptation_selected": 2,
    "baseline_unchanged": 3,
}


def _chapter4_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _PROTOCOL_ORDER.get(str(row.get("experiment_type")), len(_PROTOCOL_ORDER)),
        str(row.get("held_out_generator") or ""),
        float(row.get("adaptation_percentage") or 0.0),
        str(row.get("fine_tune_mode") or ""),
        str(row.get("evaluation_set") or ""),
        _OPERATING_POINT_ORDER.get(str(row.get("operating_point")), 9),
    )


def _chapter4_rows(consolidated: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """Headline rows only: aggregate conditions, never the per-generator breakdowns."""

    rows: list[list[Any]] = []
    for row in sorted(consolidated, key=_chapter4_sort_key):
        if row.get("generator") is not None:
            continue
        rows.append(
            [
                row.get("experiment_type"),
                _condition_label(row),
                row.get("evaluation_set"),
                row.get("operating_point"),
                row.get("threshold"),
                row.get("support"),
                row.get("positive_prevalence"),
                *[row.get(metric) for metric in SUMMARY_METRICS],
                row.get("trainable_parameters"),
                row.get("labelled_images_consumed"),
                row.get("run_id"),
            ]
        )
    return rows


def _missing_results(
    records: Sequence[RunRecord], consolidated: Sequence[Mapping[str, Any]]
) -> list[str]:
    """State what a complete study would contain that these artefacts do not.

    Every statement is derived from what was found, never from a hard-coded expectation
    about what the numbers *should* be. This is the section that stops a gap being read
    as a result.
    """

    notes: list[str] = []
    reportable = reportable_runs(records)
    by_type = {record.experiment_type for record in reportable}
    for protocol in EXPECTED_PROTOCOLS:
        if protocol in by_type:
            continue
        partial = [
            record
            for record in records
            if record.experiment_type == protocol and record.status != "completed"
        ]
        detail = (
            f" (found {len(partial)} non-completed run(s): "
            + ", ".join(f"{record.run_id} [{record.status}]" for record in partial)
            + ")"
            if partial
            else ""
        )
        notes.append(f"- **No completed `{protocol}` run.**{detail}")

    # Which generators exist at all, taken from the baseline's own per-generator block.
    available = sorted(
        {
            str(row["generator"])
            for row in consolidated
            if row.get("experiment_type") == "baseline"
            and row.get("generator")
            and str(row["generator"]) != "real"
        }
    )
    held_out = sorted(
        {
            str(record.held_out_generator)
            for record in reportable
            if record.experiment_type == "unseen_generator" and record.held_out_generator
        }
    )
    if available:
        # The baseline trains on every generator, so `available` already contains the
        # ones that have since been held out; union rather than sum.
        every_generator = sorted(set(available) | set(held_out))
        remaining = [name for name in every_generator if name not in held_out]
        notes.append(
            f"- **Leave-one-generator-out covers {len(held_out)} of {len(every_generator)} "
            f"generators** ({', '.join(held_out) or 'none'}). Not yet held out: "
            f"{', '.join(remaining) or 'none'}."
        )

    modes = sorted(
        {
            str(row["fine_tune_mode"])
            for row in consolidated
            if row.get("experiment_type") in {"fine_tuning", "ablation"}
            and row.get("fine_tune_mode")
            and str(row["fine_tune_mode"]) != "none"
        }
    )
    if modes:
        absent = [name for name in ("head_only", "last_block", "full") if name not in modes]
        notes.append(
            f"- **Fine-tuning depths measured: {', '.join(modes)}.** "
            f"Missing from the depth comparison: {', '.join(absent) or 'none'}."
        )

    replication = {
        (
            row.get("run_id"),
            row.get("fine_tune_mode"),
            row.get("adaptation_percentage"),
        )
        for row in consolidated
        if row.get("experiment_type") in {"fine_tuning", "ablation"}
        and row.get("generator") is None
    }
    seeds = {
        (row.get("subset_seed"), row.get("training_seed"))
        for row in consolidated
        if row.get("experiment_type") in {"fine_tuning", "ablation"}
        and row.get("subset_seed") is not None
    }
    if replication and len(seeds) <= 1:
        notes.append(
            "- **No repeated seeds.** Every adaptation cell was fitted once "
            f"({len(seeds)} distinct (subset_seed, training_seed) pair). Standard "
            "deviations are therefore reported as 0.0 with `runs = 1`, which means "
            "*unmeasured*, not *zero variance*; no error bars are drawn."
        )
    return notes


def report_consolidated(
    records: Sequence[RunRecord], output_dir: Path, *, include_synthetic_smoke: bool = False
) -> dict[str, Any]:
    """Aggregate every completed run into the cross-run tables and figures."""

    artefacts: list[str] = []
    reportable = reportable_runs(records, include_synthetic_smoke=include_synthetic_smoke)
    consolidated = consolidate_runs(reportable)
    if not consolidated:
        raise ValueError("no completed run produced a metrics file to consolidate")

    # 1. The machine-readable artefact future notebooks consume.
    write_table(consolidated, CONSOLIDATED_COLUMNS, output_dir / "consolidated_results.csv")
    artefacts.append("consolidated_results.csv")

    run_summary = summarise_runs(records, consolidated)
    degradation = degradation_rows(consolidated)
    recovery = summarise_recovery(consolidated, records=reportable)

    (output_dir / "consolidated_results.json").write_text(
        json.dumps(
            {
                "schema": {
                    "consolidated_results": list(CONSOLIDATED_COLUMNS),
                    "experiment_inventory": list(RUN_SUMMARY_COLUMNS),
                    "recovery_summary": list(RECOVERY_SUMMARY_COLUMNS),
                },
                "includes_synthetic_smoke_runs": include_synthetic_smoke,
                "runs_discovered": len(records),
                "runs_consolidated": len(reportable),
                "experiment_inventory": run_summary,
                "consolidated_results": consolidated,
                "unseen_generator_degradation": degradation,
                "recovery_summary": recovery,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    artefacts.append("consolidated_results.json")

    # 2. Inventory of every run found, completed or not.
    _emit_table(
        [[row.get(column) for column in RUN_SUMMARY_COLUMNS] for row in run_summary],
        list(RUN_SUMMARY_COLUMNS),
        output_dir / "table_experiment_inventory",
        artefacts,
    )

    # 3. The Chapter 4 metric comparison table.
    _emit_table(
        _chapter4_rows(consolidated),
        list(CHAPTER4_COLUMNS),
        output_dir / "table_chapter4_metrics",
        artefacts,
    )

    # 4. Degradation and recovery.
    if degradation:
        _emit_table(
            [
                [
                    row["held_out_generator"],
                    row["operating_point"],
                    row["metric"],
                    row["in_distribution"],
                    row["unseen"],
                    row["absolute_drop"],
                    row["relative_drop"],
                    row["unseen_support"],
                    row["unseen_prevalence"],
                    row["run_id"],
                ]
                for row in degradation
            ],
            [
                "held_out_generator",
                "operating_point",
                "metric",
                "in_distribution",
                "unseen",
                "absolute_drop",
                "relative_drop",
                "unseen_support",
                "unseen_prevalence",
                "run_id",
            ],
            output_dir / "table_unseen_degradation",
            artefacts,
        )
        for operating_point in sorted({str(row["operating_point"]) for row in degradation}):
            try:
                figure, _ = plot_generalisation_degradation(
                    degradation, operating_point=operating_point
                )
            except ValueError:
                continue
            artefacts.extend(
                _save_figure(figure, output_dir / f"figure_degradation_{operating_point}")
            )

    if recovery:
        _emit_table(
            [[row.get(column) for column in RECOVERY_SUMMARY_COLUMNS] for row in recovery],
            list(RECOVERY_SUMMARY_COLUMNS),
            output_dir / "table_recovery_budget",
            artefacts,
        )

    # 5. Cross-run recovery and parameter-efficiency figures. These pool every completed
    #    recovery and ablation run, so a depth comparison appears as soon as the ablation
    #    exists without this script changing.
    cell_rows = [
        row
        for row in consolidated
        if row.get("experiment_type") in {"fine_tuning", "ablation"}
        and row.get("generator") is None
        and row.get("operating_point") == "default"
    ]
    for metric in ("roc_auc", "average_precision", "f1", "accuracy"):
        if not any(row.get(metric) is not None for row in cell_rows):
            continue
        try:
            figure, axes = plot_fine_tuning_recovery(cell_rows, metric_name=metric)
        except ValueError:
            continue
        axes.set_title(f"Limited-data recovery, all completed runs ({metric})")
        artefacts.extend(_save_figure(figure, output_dir / f"figure_recovery_all_runs_{metric}"))

    adapted = [row for row in cell_rows if float(row.get("adaptation_percentage") or 0.0) > 0.0]
    zero_by_metric = {
        metric: next(
            (
                float(row[metric])
                for row in cell_rows
                if float(row.get("adaptation_percentage") or 0.0) == 0.0
                and row.get(metric) is not None
            ),
            None,
        )
        for metric in ("roc_auc", "f1")
    }
    for metric in ("roc_auc", "f1"):
        try:
            figure, _ = plot_parameter_efficiency(
                adapted, metric_name=metric, zero_percent_reference=zero_by_metric[metric]
            )
        except ValueError:
            continue
        artefacts.extend(_save_figure(figure, output_dir / f"figure_parameter_efficiency_{metric}"))

    # 6. What is absent. Written every time, so a reader never has to infer it.
    notes = _missing_results(records, consolidated)
    (output_dir / "missing_results.md").write_text(
        "# Expected results that these artefacts do not contain\n\n"
        "Derived from the runs actually found under the output root. An entry here means "
        "the experiment has not been run or has not completed; no value anywhere in this "
        "report has been substituted for it.\n\n"
        + ("\n".join(notes) if notes else "- Nothing missing was detected.")
        + "\n",
        encoding="utf-8",
    )
    artefacts.append("missing_results.md")

    return {
        "runs_discovered": len(records),
        "runs_consolidated": [record.run_id for record in reportable],
        "rows": len(consolidated),
        "artefacts": artefacts,
    }


def build_report(
    *,
    output_root: Path,
    destination: Path,
    explicit: Sequence[Path] = (),
    include_synthetic_smoke: bool = False,
) -> Path:
    runs = discover_runs(output_root, explicit, include_synthetic_smoke=include_synthetic_smoke)
    if not runs:
        raise ValueError(
            f"no completed runs with recognised metrics files were found under {output_root}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"sections": {}, "source_runs": {}}
    synthetic_warning = False
    for run in runs:
        section_dir = destination / run.experiment_type
        section_dir.mkdir(parents=True, exist_ok=True)
        result = REPORTERS[run.experiment_type](run, section_dir)
        manifest["sections"][run.experiment_type] = result
        manifest["source_runs"][run.experiment_type] = str(run.run_dir)
        resolved_config = run.run_dir / "resolved_config.yaml"
        if resolved_config.is_file() and "SMOKE" in resolved_config.read_text(encoding="utf-8"):
            synthetic_warning = True

    # The consolidated half sees every run under the output root, not just the newest of
    # each protocol, so it is built from its own discovery rather than from `runs`.
    records = discover_all_runs(output_root)
    consolidated_dir = destination / "consolidated"
    consolidated_dir.mkdir(parents=True, exist_ok=True)
    manifest["sections"]["consolidated"] = report_consolidated(
        records, consolidated_dir, include_synthetic_smoke=include_synthetic_smoke
    )
    manifest["runs_discovered"] = [
        {"run_id": record.run_id, "status": record.status, "smoke": record.is_synthetic_smoke}
        for record in records
    ]
    manifest["contains_synthetic_smoke_runs"] = synthetic_warning
    (destination / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = ["# Generated results", ""]
    if synthetic_warning:
        lines += [
            "> **WARNING: this report includes SMOKE runs on synthetic fixture data.**",
            "> Those numbers verify the pipeline and are not detector performance.",
            "",
        ]
    lines += ["Every table and figure below is derived from these run directories:", ""]
    for experiment_type, run_dir in sorted(manifest["source_runs"].items()):
        lines.append(f"- `{experiment_type}`: `{run_dir}`")
    lines += [
        "",
        "`consolidated/` aggregates **every** completed run, not just the newest of each "
        "protocol. Start from `consolidated/consolidated_results.csv`; every row names the "
        "run and the metrics file it came from, with that file's SHA-256.",
        "",
        "See `consolidated/missing_results.md` for what a complete study would contain that "
        "these artefacts do not.",
        "",
    ]
    for experiment_type, result in sorted(manifest["sections"].items()):
        lines.append(f"## {experiment_type}")
        lines.append("")
        for artefact in sorted(result["artefacts"]):
            lines.append(f"- `{experiment_type}/{artefact}`")
        lines.append("")
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/report"))
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        default=[],
        help="Explicit run directory to report on; repeatable. Overrides discovery.",
    )
    parser.add_argument(
        "--include-synthetic-smoke",
        action="store_true",
        help=(
            "Include SMOKE_ runs on synthetic fixture data. They verify the pipeline and "
            "are not detector performance; excluded by default."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    destination = build_report(
        output_root=args.output_root,
        destination=args.output,
        explicit=args.run,
        include_synthetic_smoke=args.include_synthetic_smoke,
    )
    print(f"Report written to {destination}")


if __name__ == "__main__":
    main()
