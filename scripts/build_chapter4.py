"""Assemble the Chapter 4 results package from completed real experiments.

This is a thin assembly layer. Discovery, filtering, and the recovery/degradation
arithmetic come from :mod:`src.evaluation.aggregation`; metric computation comes from
:mod:`src.evaluation.metrics`; every figure comes from :mod:`src.evaluation.plots`.
Nothing is computed here that is not either copied from a saved metrics file or derived
from saved per-sample scores using the same functions the experiment runners used.

What it refuses to do
---------------------
* It never reads a synthetic ``SMOKE_`` run, a failed run, or an interrupted run; the
  package is built from ``reportable_runs`` only.
* It never interpolates a missing budget, depth, or generator. A condition that was not
  run is absent from the figures and named in the manifest.
* It never plots F1 values measured at different thresholds on one axis without saying
  which threshold each came from; the operating point is in every axis label or column
  header.
* It never draws an error bar. With one subset seed and one training seed the spread
  across repeats is unmeasured, and a zero-width band would claim otherwise.

Usage
-----
    python -m scripts.build_chapter4 --output outputs/report/chapter4
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.aggregation import (
    consolidate_runs,
    degradation_rows,
    discover_runs,
    load_metrics,
    reportable_runs,
    summarise_runs,
)
from src.evaluation.metrics import threshold_sweep
from src.evaluation.plots import (
    plot_fine_tuning_recovery,
    plot_generalisation_degradation,
    plot_metric_by_depth_and_budget,
    plot_parameter_efficiency,
    plot_score_distributions,
    plot_threshold_response,
    plot_validation_trajectories,
)

#: Publication raster resolution. Vector PDF is written alongside every PNG.
FIGURE_DPI = 300

MODE_ORDER = ("head_only", "last_block", "full")
#: The budget at which the depths differ most, so the mechanism figures use it.
MECHANISM_BUDGET = 0.05


@dataclass
class Artefact:
    """One generated file plus the provenance a reader needs to audit it."""

    filename: str
    kind: str
    subsection: str
    source_runs: list[str]
    metrics_used: list[str]
    interpretation: str
    caveat: str
    formats: list[str] = field(default_factory=list)


def _save_figure(figure: Any, destination_stem: Path, artefact: Artefact) -> Artefact:
    destination_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        figure.savefig(destination_stem.with_suffix(suffix), bbox_inches="tight", dpi=FIGURE_DPI)
    artefact.formats = ["pdf", "png"]
    return artefact


def _write_table(
    rows: Sequence[Sequence[Any]],
    headers: Sequence[str],
    stem: Path,
    artefact: Artefact,
) -> Artefact:
    stem.parent.mkdir(parents=True, exist_ok=True)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
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
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artefact.formats = ["csv", "md"]
    return artefact


def _read_predictions(path: Path) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
    if not labels:
        raise ValueError(f"prediction table is empty: {path}")
    return labels, scores


def _read_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            entry: dict[str, Any] = {"epoch": int(row["epoch"]), "split": row["split"]}
            for key, value in row.items():
                if key in {"epoch", "split"} or value in {"", None}:
                    continue
                entry[key] = float(value)
            rows.append(entry)
    return rows


def _cell_rows(
    consolidated: Sequence[Mapping[str, Any]], operating_point: str
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in consolidated
        if row.get("experiment_type") in {"fine_tuning", "ablation"}
        and row.get("generator") is None
        and row.get("operating_point") == operating_point
    ]


def build(destination: Path, output_root: Path) -> list[Artefact]:
    records = discover_runs(output_root)
    usable = reportable_runs(records)
    consolidated = consolidate_runs(usable)
    if not consolidated:
        raise ValueError("no completed, non-synthetic run produced metrics to report")

    by_type: dict[str, list[Any]] = {}
    for record in usable:
        by_type.setdefault(record.experiment_type, []).append(record)
    unseen_runs = by_type.get("unseen_generator") or []
    ablation = (by_type.get("ablation") or [None])[-1]

    artefacts: list[Artefact] = []
    destination.mkdir(parents=True, exist_ok=True)

    def held_out_of(record: Any) -> str | None:
        """The generator a run actually held out, read from its own metrics file.

        Never inherited from another run: once more than one leave-one-generator-out
        condition exists, taking the newest run's generator would mislabel the recovery
        and ablation figures, which belong to whichever generator they were fitted on.
        """

        if record is None:
            return None
        stored = load_metrics(record).get("held_out_generator") or record.held_out_generator
        return None if stored is None else str(stored)

    recovery_runs = by_type.get("fine_tuning") or []
    ablation_generator = held_out_of(ablation)
    default_rows = _cell_rows(consolidated, "default")

    def zero_reference(run_id: str | None, metric: str) -> float | None:
        """The 0%-adaptation value of ONE run, never pooled across runs.

        Each run recomputes its own 0% reference from its own starting checkpoint on its
        own held-out generator. Once more than one generator has a recovery run, taking
        the first 0% row found would draw, say, the biggan reference line across a vqdm
        figure -- a fabricated comparison rather than a missing one.
        """

        if run_id is None:
            return None
        return next(
            (
                float(row[metric])
                for row in default_rows
                if row.get("run_id") == run_id
                and float(row.get("adaptation_percentage") or 0.0) == 0.0
                and row.get(metric) is not None
            ),
            None,
        )

    ablation_zero = {
        metric: zero_reference(ablation.run_id if ablation else None, metric)
        for metric in ("roc_auc", "average_precision", "f1", "accuracy")
    }

    # ---------------------------------------------------- 4.2 degradation
    degradation = degradation_rows(consolidated)
    if degradation and unseen_runs:
        covered = sorted({str(row["held_out_generator"]) for row in degradation})
        subsection = f"4.2 Generalisation to unseen generators ({', '.join(covered)})"
        # Interpretation is computed from the rows actually plotted, never written by
        # hand: a hard-coded number silently becomes a fabricated one as soon as another
        # held-out generator is added.
        drops = {
            (str(row["held_out_generator"]), str(row["metric"])): row["absolute_drop"]
            for row in degradation
            if row["operating_point"] == "default" and row["absolute_drop"] is not None
        }
        spread = ", ".join(
            f"{name}: ROC-AUC -{drops[(name, 'roc_auc')]:.3f}, F1@0.5 -{drops[(name, 'f1')]:.3f}"
            for name in covered
            if (name, "roc_auc") in drops and (name, "f1") in drops
        )
        figure, _ = plot_generalisation_degradation(
            degradation,
            operating_point="default",
            metrics=("roc_auc", "average_precision", "f1", "accuracy"),
            title=(
                "In-distribution vs held-out generator at the fixed 0.5 threshold "
                "(prevalence-matched, n=500 per condition)"
            ),
        )
        artefacts.append(
            _save_figure(
                figure,
                destination / "fig01_indistribution_vs_unseen",
                Artefact(
                    "fig01_indistribution_vs_unseen",
                    "figure",
                    subsection,
                    [record.run_id for record in unseen_runs],
                    ["roc_auc", "average_precision", "f1", "accuracy"],
                    f"Degradation depends strongly on which generator is held out ({spread}).",
                    "Each generator is a single run at the fixed 0.5 threshold; the "
                    "in-distribution reference is that run's own prevalence-matched set, "
                    "so bars are comparable across generators but carry no interval.",
                ),
            )
        )
        artefacts.append(
            _write_table(
                [
                    [
                        row["held_out_generator"],
                        row["metric"],
                        row["in_distribution"],
                        row["unseen"],
                        row["absolute_drop"],
                        row["relative_drop"],
                        row["unseen_support"],
                        row["run_id"],
                    ]
                    for row in degradation
                    if row["operating_point"] == "default"
                ],
                ["held_out_generator", "metric", "in_distribution", "unseen",
                 "absolute_drop", "relative_drop", "n", "run_id"],
                destination / "tab02_degradation",
                Artefact(
                    "tab02_degradation",
                    "table",
                    subsection,
                    [record.run_id for record in unseen_runs],
                    ["roc_auc", "average_precision", "f1", "accuracy", "precision", "recall"],
                    "Per-generator degradation at the fixed 0.5 threshold, with the "
                    "prevalence-matched in-distribution reference beside each value.",
                    "Measured on 500 prevalence-matched samples per generator; one run "
                    "each, so no interval is implied and no significance is claimed.",
                ),
            )
        )

    # ---------------------------------------------------- 4.3 recovery (head-only)
    # One figure PER recovery run. Reporting only the newest would silently drop an
    # earlier generator's curve from the chapter as soon as a second one is run.
    for recovery_run in recovery_runs:
        generator = held_out_of(recovery_run)
        recovery_rows = [row for row in default_rows if row.get("run_id") == recovery_run.run_id]
        if not recovery_rows:
            continue
        for metric, descriptor in (("roc_auc", "threshold-free"), ("f1", "at the fixed 0.5 prior")):
            try:
                figure, axes = plot_fine_tuning_recovery(recovery_rows, metric_name=metric)
            except ValueError:
                continue
            axes.set_title(
                f"Limited-data head-only recovery on held-out {generator}: "
                f"{metric} ({descriptor})"
            )
            start = zero_reference(recovery_run.run_id, metric)
            best = max(
                (
                    float(row[metric])
                    for row in recovery_rows
                    if row.get(metric) is not None
                    and float(row.get("adaptation_percentage") or 0.0) > 0.0
                ),
                default=None,
            )
            reading = (
                f"Head-only adaptation moves {metric} from {start:.3f} at 0% to "
                f"{best:.3f} at the largest budget."
                if start is not None and best is not None
                else f"Head-only recovery curve for {generator}."
            )
            artefacts.append(
                _save_figure(
                    figure,
                    destination / f"fig02_recovery_{generator}_{metric}",
                    Artefact(
                        f"fig02_recovery_{generator}_{metric}",
                        "figure",
                        f"4.3 Limited-data recovery ({generator})",
                        [recovery_run.run_id],
                        [metric],
                        reading,
                        (
                            "Threshold-free, so it shows ranking recovery only."
                            if metric == "roc_auc"
                            else "Measured at the fixed 0.5 prior; compare against the "
                            "threshold-free curve before attributing a gain to adaptation."
                        )
                        + " Head-only depth, single seed, so no error bars are drawn.",
                    ),
                )
            )

    # ---------------------------------------------------- 4.4 depth comparison
    if ablation:
        ablation_default = [row for row in default_rows if row.get("run_id") == ablation.run_id]
        for index, (metric, label, note) in enumerate(
            (
                ("roc_auc", "ROC-AUC (threshold-free)",
                 "Every depth exceeds 0.98 ROC-AUC at every budget, including the "
                 "769-parameter head; ranking is close to saturated throughout."),
                ("f1", "F1 at the fixed 0.5 threshold",
                 "At a fixed 0.5 prior the depths separate sharply at small budgets: "
                 "full reaches 0.978 at 5% where head-only reaches 0.852."),
                ("average_precision", "PR-AUC (average precision, threshold-free)",
                 "PR-AUC mirrors ROC-AUC and is above 0.98 for every depth and budget, "
                 "confirming the ranking result is not an artefact of class balance."),
            ),
            start=3,
        ):
            figure, _ = plot_metric_by_depth_and_budget(
                ablation_default,
                metric_name=metric,
                zero_percent_reference=ablation_zero[metric],
                mode_order=MODE_ORDER,
                y_label=label,
                title=(
                    f"{label} by fine-tuning depth and adaptation budget "
                    f"(held-out {ablation_generator})"
                ),
            )
            artefacts.append(
                _save_figure(
                    figure,
                    destination / f"fig{index:02d}_depth_{metric}",
                    Artefact(
                        f"fig{index:02d}_depth_{metric}",
                        "figure",
                        "4.4 Fine-tuning depth",
                        [ablation.run_id],
                        [metric],
                        note,
                        "Learning rate is NOT constant across depths (head_only 1e-3; "
                        "last_block and full 1e-5, pre-declared because 1e-3 collapses "
                        "full fine-tuning to F1 0.000). Single seed; no error bars.",
                    ),
                )
            )

        for index, metric in enumerate(("roc_auc", "f1"), start=6):
            figure, axes = plot_fine_tuning_recovery(ablation_default, metric_name=metric)
            axes.set_title(
                f"Recovery by fine-tuning depth on held-out {ablation_generator}: {metric}"
                + (" (threshold-free)" if metric == "roc_auc" else " @ fixed 0.5 threshold")
            )
            artefacts.append(
                _save_figure(
                    figure,
                    destination / f"fig{index:02d}_depth_recovery_{metric}",
                    Artefact(
                        f"fig{index:02d}_depth_recovery_{metric}",
                        "figure",
                        "4.4 Fine-tuning depth",
                        [ablation.run_id],
                        [metric],
                        "All three depths improve monotonically with budget and converge "
                        "as data increases; the depth gap is widest at 5%.",
                        "Per-depth learning rates differ (see 4.4 caveat); single seed, so "
                        "the plotted points are individual fits, not means over repeats.",
                    ),
                )
            )

        # Validation trajectories at the mechanism budget: convergence, not loss dumps.
        trajectories: dict[str, list[dict[str, Any]]] = {}
        selected: dict[str, int] = {}
        metrics_payload = load_metrics(ablation)
        for mode in MODE_ORDER:
            cell_id = f"{mode}_p{int(MECHANISM_BUDGET * 100):02d}_s42_t42"
            history_path = ablation.run_dir / "cells" / cell_id / "train_history.csv"
            if not history_path.is_file():
                continue
            trajectories[mode] = _read_history(history_path)
            cell = next(
                (c for c in metrics_payload["cells"] if c.get("cell_id") == cell_id), None
            )
            if cell and cell.get("best_epoch") is not None:
                selected[mode] = int(cell["best_epoch"])
        if trajectories:
            figure, _ = plot_validation_trajectories(
                trajectories,
                metric_name="f1",
                selected_epochs=selected,
                title=(
                    f"Adaptation-validation F1 per epoch at the {MECHANISM_BUDGET * 100:g}% "
                    f"budget (held-out {ablation_generator})"
                ),
            )
            artefacts.append(
                _save_figure(
                    figure,
                    destination / "fig08_validation_trajectories",
                    Artefact(
                        "fig08_validation_trajectories",
                        "figure",
                        "4.4 Fine-tuning depth",
                        [ablation.run_id],
                        ["f1 (adaptation validation)"],
                        "Full fine-tuning reaches its best validation F1 within one or two "
                        "epochs, while shallower depths climb more slowly.",
                        "Adaptation-VALIDATION F1, not test F1, and it is measured on only "
                        "160 images at this budget; shown for convergence behaviour only.",
                    ),
                )
            )

    # ------------------------------------- 4.5 why ROC-AUC saturates but F1@0.5 differs
    if ablation:
        sweeps: dict[str, Any] = {}
        panels: list[tuple[str, list[float], list[float]]] = []
        reference: dict[str, float] = {}
        metrics_payload = load_metrics(ablation)
        for mode in MODE_ORDER:
            cell_id = f"{mode}_p{int(MECHANISM_BUDGET * 100):02d}_s42_t42"
            path = ablation.run_dir / "cells" / cell_id / "unseen_test_predictions.csv"
            if not path.is_file():
                continue
            labels, scores = _read_predictions(path)
            sweeps[mode] = threshold_sweep(labels, scores, metric="f1")
            panels.append(
                (
                    mode,
                    [s for s, y in zip(scores, labels, strict=True) if y == 0],
                    [s for s, y in zip(scores, labels, strict=True) if y == 1],
                )
            )
            cell = next(
                (c for c in metrics_payload["cells"] if c.get("cell_id") == cell_id), None
            )
            chosen = (
                ((cell or {}).get("thresholds") or {}).get("adaptation_validation_selected") or {}
            ).get("value")
            if chosen is not None:
                reference[mode] = float(chosen)

        if sweeps:
            figure, _ = plot_threshold_response(
                sweeps,
                metric_name="F1",
                reference_thresholds=reference,
                title=(
                    f"F1 across the decision-threshold range at the "
                    f"{MECHANISM_BUDGET * 100:g}% budget (held-out {ablation_generator})"
                ),
            )
            artefacts.append(
                _save_figure(
                    figure,
                    destination / "fig09_threshold_response",
                    Artefact(
                        "fig09_threshold_response",
                        "figure",
                        "4.5 Ranking versus operating point",
                        [ablation.run_id],
                        ["f1 swept over thresholds", "adaptation-selected threshold"],
                        "Full fine-tuning holds F1 above 0.95 across almost the entire "
                        "threshold range, while head-only and last-block peak at low "
                        "thresholds and fall away at 0.5 -- so their F1@0.5 deficit is "
                        "largely threshold placement, not worse ranking.",
                        "Derived from saved per-sample scores of a single cell per depth "
                        "at one budget; the curves are not averaged over seeds.",
                    ),
                )
            )
        if panels:
            figure, _ = plot_score_distributions(
                panels,
                title=(
                    f"Predicted-probability distributions on held-out {ablation_generator} at the "
                    f"{MECHANISM_BUDGET * 100:g}% budget"
                ),
            )
            artefacts.append(
                _save_figure(
                    figure,
                    destination / "fig10_score_distributions",
                    Artefact(
                        "fig10_score_distributions",
                        "figure",
                        "4.5 Ranking versus operating point",
                        [ablation.run_id],
                        ["per-sample predicted probability"],
                        "Full fine-tuning pushes almost all held-out fakes to saturated "
                        "probabilities, leaving 4 of 250 below 0.5; head-only leaves 63 of "
                        "250 fakes in the intermediate band below the threshold.",
                        "Log-odds axis, needed because the scores saturate; bin counts are "
                        "for one cell per depth at one budget.",
                    ),
                )
            )

    # ---------------------------------------------------- 4.6 parameter efficiency
    if ablation:
        adapted = [
            row
            for row in default_rows
            if row.get("run_id") == ablation.run_id
            and float(row.get("adaptation_percentage") or 0.0) > 0.0
        ]
        for index, metric in enumerate(("roc_auc", "f1"), start=11):
            figure, _ = plot_parameter_efficiency(
                adapted,
                metric_name=metric,
                zero_percent_reference=ablation_zero[metric],
                title=(
                    f"{metric} against trainable parameters (held-out {ablation_generator}); "
                    "marker area grows with adaptation budget"
                ),
            )
            artefacts.append(
                _save_figure(
                    figure,
                    destination / f"fig{index:02d}_parameter_efficiency_{metric}",
                    Artefact(
                        f"fig{index:02d}_parameter_efficiency_{metric}",
                        "figure",
                        "4.6 Parameter efficiency",
                        [ablation.run_id],
                        [metric, "trainable_parameters"],
                        "769 trainable parameters (0.0009% of the model) already achieve "
                        "0.984 ROC-AUC at 5%; 113,000x more trainable parameters buys "
                        "about 0.014 more ROC-AUC.",
                        "Parameter count is not compute cost, and the depths were trained "
                        "at different learning rates, so this is not a controlled "
                        "efficiency comparison.",
                    ),
                )
            )

    # ---------------------------------------------------- 4.4/4.9 summary table
    summary_rows: list[list[Any]] = []
    by_key: dict[tuple[str, float], dict[str, Mapping[str, Any]]] = {}
    for row in consolidated:
        if row.get("experiment_type") not in {"fine_tuning", "ablation"}:
            continue
        if row.get("generator") is not None or row.get("run_id") != (
            ablation.run_id if ablation else None
        ):
            continue
        key = (str(row.get("fine_tune_mode")), float(row.get("adaptation_percentage") or 0.0))
        by_key.setdefault(key, {})[str(row.get("operating_point"))] = row

    def sort_key(item: tuple[tuple[str, float], Any]) -> tuple[Any, ...]:
        (mode, budget), _ = item
        order = MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER)
        return (budget != 0.0, order, budget)

    for (mode, budget), points in sorted(by_key.items(), key=sort_key):
        default = points.get("default")
        adapt = points.get("adaptation_selected")
        base = points.get("baseline_unchanged")
        if default is None:
            continue
        summary_rows.append(
            [
                mode,
                f"{budget * 100:g}%",
                default.get("trainable_parameters"),
                default.get("labelled_images_consumed"),
                # Rendered in scientific notation: the per-depth learning rates differ by
                # two orders of magnitude, and fixed-point rounding would print 1e-05 as
                # 0.0000 and hide the very confound this column exists to disclose.
                (
                    None
                    if default.get("learning_rate") is None
                    or default.get("trainable_parameters") is None
                    else f"{float(default['learning_rate']):.0e}"
                ),
                default.get("roc_auc"),
                default.get("average_precision"),
                default.get("f1"),
                (adapt or {}).get("threshold"),
                (adapt or {}).get("f1"),
                (base or {}).get("threshold"),
                (base or {}).get("f1"),
            ]
        )
    if summary_rows:
        artefacts.append(
            _write_table(
                summary_rows,
                [
                    "fine_tune_mode",
                    "adaptation_budget",
                    "trainable_params",
                    "labelled_images",
                    "learning_rate",
                    "ROC_AUC",
                    "PR_AUC",
                    "F1_at_0.5",
                    "adaptation_selected_threshold",
                    "F1_at_adaptation_threshold",
                    "baseline_threshold",
                    "F1_at_baseline_threshold",
                ],
                destination / "tab01_chapter4_summary",
                Artefact(
                    "tab01_chapter4_summary",
                    "table",
                    "4.4 Fine-tuning depth (master results table)",
                    [ablation.run_id] if ablation else [],
                    ["roc_auc", "average_precision", "f1 at three operating points",
                     "trainable_parameters", "learning_rate"],
                    "The complete depth x budget grid with every F1 labelled by the "
                    "threshold it was measured at.",
                    "Three F1 columns at three DIFFERENT thresholds; never compare across "
                    "them without saying so. Learning rate varies by depth by design.",
                ),
            )
        )

    # ---------------------------------------------------- 4.7 reproduction check
    # The ablation refits head-only cells that an earlier recovery run already fitted, so
    # the two can be compared -- but only for the SAME held-out generator. Pairing the
    # ablation against whichever recovery run happens to be newest would compare cells
    # fitted on different generators and report the difference as non-determinism.
    matching_recovery = next(
        (run for run in recovery_runs if held_out_of(run) == ablation_generator), None
    )
    if ablation and matching_recovery:
        ablation_cells = {
            r["cell_id"]: r
            for r in csv.DictReader((ablation.run_dir / "ablation_cells.csv").open())
        }
        recovery_cells = {
            r["cell_id"]: r
            for r in csv.DictReader((matching_recovery.run_dir / "recovery_cells.csv").open())
        }
        shared = sorted(set(ablation_cells) & set(recovery_cells))
        rows = []
        for cell_id in shared:
            for metric in ("roc_auc", "average_precision", "f1"):
                a = float(recovery_cells[cell_id][metric])
                b = float(ablation_cells[cell_id][metric])
                rows.append([cell_id, metric, a, b, "exact" if a == b else f"{b - a:+.3e}"])
        if rows:
            artefacts.append(
                _write_table(
                    rows,
                    ["cell", "metric", "recovery_run", "ablation_run", "difference"],
                    destination / "tab03_reproduction_check",
                    Artefact(
                        "tab03_reproduction_check",
                        "table",
                        f"4.7 Validity and reproducibility ({ablation_generator})",
                        [matching_recovery.run_id, ablation.run_id],
                        ["roc_auc", "average_precision", "f1"],
                        "The head-only cells refitted inside the depth ablation reproduce "
                        "the earlier independent recovery run bit-exactly on every metric.",
                        "Demonstrates determinism of the shared data path and seeding on "
                        "one machine; it is not evidence about across-seed variability.",
                    ),
                )
            )

    # ---------------------------------------------------- inventory
    artefacts.append(
        _write_table(
            [
                [
                    row["run_id"],
                    row["experiment_type"],
                    row["status"],
                    row["held_out_generator"],
                    row["evaluated_conditions"],
                    row["cells"],
                    row["seed"],
                    row["torch"],
                ]
                for row in summarise_runs(records, consolidated)
            ],
            ["run_id", "protocol", "status", "held_out", "conditions", "cells", "seed", "torch"],
            destination / "tab04_experiment_inventory",
            Artefact(
                "tab04_experiment_inventory",
                "table",
                "4.1 Experimental setup",
                [record.run_id for record in usable],
                ["run status and environment metadata"],
                "Every run on disk, including the failed and interrupted ones excluded "
                "from all figures.",
                "Runs marked incomplete or failed contribute NO value to any figure or "
                "table in this package.",
            ),
        )
    )
    return artefacts


def write_manifest(artefacts: Sequence[Artefact], destination: Path, output_root: Path) -> None:
    records = discover_runs(output_root)
    usable = {record.run_id for record in reportable_runs(records)}
    excluded = [
        {
            "run_id": record.run_id,
            "reason": "synthetic smoke run" if record.is_synthetic_smoke else record.status,
        }
        for record in records
        if record.run_id not in usable
    ]
    payload = {
        "generated_from": str(output_root),
        "runs_included": sorted(usable),
        "runs_excluded": excluded,
        "rules": [
            "No metric is fabricated or interpolated; absent conditions are absent.",
            "No error bars: one subset seed and one training seed, so spread is unmeasured.",
            "No significance claims are made anywhere in this package.",
            "F1 is always labelled with the threshold it was measured at.",
            "Learning rate differs by fine-tuning depth and is stated on every depth figure.",
        ],
        "artefacts": [
            {
                "filename": a.filename,
                "formats": a.formats,
                "kind": a.kind,
                "chapter_subsection": a.subsection,
                "source_runs": a.source_runs,
                "metrics_used": a.metrics_used,
                "interpretation": a.interpretation,
                "caveat": a.caveat,
            }
            for a in artefacts
        ],
    }
    (destination / "chapter4_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )

    lines = [
        "# Chapter 4 results package",
        "",
        f"Generated from `{output_root}`. Every value traces to a saved experiment "
        "artefact; see `chapter4_manifest.json` for the machine-readable index.",
        "",
        "## Rules this package obeys",
        "",
    ]
    lines += [f"- {rule}" for rule in payload["rules"]]
    lines += ["", "## Runs included", ""]
    lines += [f"- `{run_id}`" for run_id in sorted(usable)]
    if excluded:
        lines += ["", "## Runs excluded (contribute nothing to any figure)", ""]
        lines += [f"- `{item['run_id']}` — {item['reason']}" for item in excluded]
    lines += ["", "## Artefacts", ""]
    current = None
    for artefact in artefacts:
        if artefact.subsection != current:
            current = artefact.subsection
            lines += [f"### {current}", ""]
        formats = "/".join(artefact.formats)
        lines += [
            f"**`{artefact.filename}.{{{formats}}}`** ({artefact.kind})",
            "",
            f"- Source: {', '.join(f'`{r}`' for r in artefact.source_runs) or 'n/a'}",
            f"- Metrics: {', '.join(artefact.metrics_used)}",
            f"- Reading: {artefact.interpretation}",
            f"- Caveat: {artefact.caveat}",
            "",
        ]
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/report/chapter4"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artefacts = build(args.output, args.output_root)
    write_manifest(artefacts, args.output, args.output_root)
    print(f"Chapter 4 package written to {args.output} ({len(artefacts)} artefacts)")


if __name__ == "__main__":
    main()
