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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    plot_generator_performance,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_training_curves,
)

EXPERIMENT_METRIC_FILES = {
    "baseline": "test_metrics.json",
    "unseen_generator": "unseen_generator_metrics.json",
    "fine_tuning": "recovery_metrics.json",
    "ablation": "ablation_metrics.json",
}


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


def discover_runs(output_root: Path, explicit: Sequence[Path] = ()) -> list[DiscoveredRun]:
    """Find the runs to report on, preferring explicit paths over newest-completed."""

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
            if path.is_dir() and (path / filename).is_file() and _is_completed(path)
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
            "labelled_images",
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
    for metric_name in ("f1", "roc_auc"):
        if any(row.get(metric_name) is not None for row in rows):
            figure, _ = plot_fine_tuning_recovery(rows, metric_name=metric_name)
            artefacts.extend(_save_figure(figure, output_dir / f"figure_recovery_{metric_name}"))
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


def build_report(*, output_root: Path, destination: Path, explicit: Sequence[Path] = ()) -> Path:
    runs = discover_runs(output_root, explicit)
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
    lines.append("")
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    destination = build_report(
        output_root=args.output_root, destination=args.output, explicit=args.run
    )
    print(f"Report written to {destination}")


if __name__ == "__main__":
    main()
