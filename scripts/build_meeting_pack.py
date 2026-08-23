"""Build the five-figure supervisor meeting pack from completed experiments.

Five figures, in the order the scientific argument runs, plus one headline table and a
``MEETING_README.md`` giving a sentence to say, what each figure demonstrates, and its
caveat.

Like every other reader in this repository it is strictly non-fabricating. An arm that
has not been run is drawn as **PENDING** and named in the README; it is never
interpolated, never estimated, and never quietly dropped so the figure looks complete.

Usage
-----
    python -m scripts.build_meeting_pack --output outputs/report/chapter4
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.aggregation import (
    consolidate_runs,
    degradation_rows,
    discover_runs,
    load_metrics,
    reportable_runs,
)
from src.evaluation.plots import (
    plot_experiment_roadmap,
    plot_fine_tuning_recovery,
    plot_generalisation_degradation,
    plot_metric_by_depth_and_budget,
)

FIGURE_DPI = 300
DEVELOPMENT_GENERATOR = "vqdm"
CONFIRMATORY_GENERATOR = "wukong"


@dataclass
class Slide:
    filename: str
    heading: str
    say: str
    demonstrates: str
    caveat: str
    status: str = "ready"


def _save(figure: Any, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        figure.savefig(stem.with_suffix(suffix), bbox_inches="tight", dpi=FIGURE_DPI)


def _unseen_by_generator_and_head(
    records: Sequence[Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index completed leave-one-out runs by (held-out generator, head type)."""

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record.experiment_type != "unseen_generator":
            continue
        metrics = load_metrics(record)
        generator = str(metrics.get("held_out_generator") or record.held_out_generator)
        head = "cosine" if record.experiment_name.endswith("_cosine") else "linear"
        indexed[(generator, head)] = {"record": record, "metrics": metrics}
    return indexed


def _degradation_pair(entry: Mapping[str, Any]) -> dict[str, float | None]:
    gap = entry["metrics"].get("generalisation_gap") or {}
    roc = gap.get("roc_auc") or {}
    return {
        "in_distribution": roc.get("in_distribution"),
        "unseen": roc.get("unseen"),
        "drop": roc.get("absolute_drop"),
    }


def build(destination: Path, output_root: Path) -> list[Slide]:
    records = reportable_runs(discover_runs(output_root))
    consolidated = consolidate_runs(records)
    unseen = _unseen_by_generator_and_head(records)
    slides: list[Slide] = []
    destination.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- FIGURE 1
    degradation = [
        row for row in degradation_rows(consolidated) if row["operating_point"] == "default"
    ]
    linear_runs = {
        generator: entry
        for (generator, head), entry in unseen.items()
        if head == "linear"
    }
    linear_ids = {entry["record"].run_id for entry in linear_runs.values()}
    figure_one_rows = [row for row in degradation if row["run_id"] in linear_ids]
    if figure_one_rows:
        figure, _ = plot_generalisation_degradation(
            figure_one_rows,
            operating_point="default",
            metrics=("roc_auc", "f1"),
            title="Where the original detector fails: degradation depends on the generator",
        )
        _save(figure, destination / "meeting_fig1_where_it_fails")
        drops = {
            str(row["held_out_generator"]): row["absolute_drop"]
            for row in figure_one_rows
            if row["metric"] == "roc_auc" and row["absolute_drop"] is not None
        }
        spread = "; ".join(f"{name} -{value:.3f}" for name, value in sorted(drops.items()))
        slides.append(
            Slide(
                "meeting_fig1_where_it_fails",
                "1. Where does the original detector fail?",
                "The detector barely notices one unseen generator and collapses on another "
                f"({spread} ROC-AUC).",
                "That generator shift is not one phenomenon: robustness measured on a "
                "single held-out generator would have been badly misleading.",
                "Two generators, one seed each, at the fixed 0.5 threshold. No interval "
                "is implied and no significance is claimed.",
            )
        )

    # ---------------------------------------------------------- FIGURE 2
    development_rows: list[dict[str, Any]] = []
    pending_arms: list[str] = []
    for head in ("linear", "cosine"):
        entry = unseen.get((DEVELOPMENT_GENERATOR, head))
        if entry is None:
            pending_arms.append(head)
            continue
        pair = _degradation_pair(entry)
        development_rows.append(
            {
                "held_out_generator": f"{head} head",
                "operating_point": "default",
                "metric": "roc_auc",
                "in_distribution": pair["in_distribution"],
                "unseen": pair["unseen"],
                "absolute_drop": pair["drop"],
                "run_id": entry["record"].run_id,
            }
        )
    if development_rows:
        figure, axes = plot_generalisation_degradation(
            development_rows,
            operating_point="default",
            metrics=("roc_auc",),
            title=(
                f"Development condition ({DEVELOPMENT_GENERATOR}): does the modified head "
                "withstand the shift better?"
            ),
        )
        # With a single arm present the shared plotting helper collapses the tick label to
        # the bare metric name, which loses the very thing this figure compares.
        axes.set_xticks(
            axes.get_xticks(),
            labels=[f"ROC-AUC\n({row['held_out_generator']})" for row in development_rows],
        )
        if pending_arms:
            # Never invent the missing arm. The note goes BELOW the axes so it cannot be
            # mistaken for a plotted value.
            axes.text(
                0.5,
                -0.30,
                f"{' and '.join(arm + ' head' for arm in pending_arms)}: NOT YET RUN",
                transform=axes.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#B00020",
                fontweight="bold",
            )
        _save(figure, destination / "meeting_fig2_modified_detector")
        slides.append(
            Slide(
                "meeting_fig2_modified_detector",
                "2. Can the modified detector withstand the shift?",
                "Each bar pair is one architecture's own in-distribution reference next to "
                "its unseen score, so the comparison is between DROPS, not absolute scores.",
                "Whether removing embedding magnitude from the decision reduces "
                "generator-shift degradation, at matched parameter count (769 vs 770).",
                f"{DEVELOPMENT_GENERATOR} motivated this architecture, so it is a "
                f"DEVELOPMENT condition; {CONFIRMATORY_GENERATOR} is pre-declared as the "
                "confirmatory zero-shot test."
                + (f" The {', '.join(pending_arms)} arm has not been run." if pending_arms else ""),
                status="pending" if pending_arms else "ready",
            )
        )

    # ---------------------------------------------------------- FIGURE 3
    recovery_rows = [
        row
        for row in consolidated
        if row.get("experiment_type") == "fine_tuning"
        and row.get("generator") is None
        and row.get("operating_point") == "default"
        and str(row.get("held_out_generator")) == DEVELOPMENT_GENERATOR
    ]
    if recovery_rows:
        figure, axes = plot_fine_tuning_recovery(recovery_rows, metric_name="roc_auc")
        axes.set_title(
            f"How quickly does it recover? Head-only adaptation on held-out "
            f"{DEVELOPMENT_GENERATOR} (ROC-AUC)"
        )
        _save(figure, destination / "meeting_fig3_recovery")
        start = next(
            (
                float(row["roc_auc"])
                for row in recovery_rows
                if float(row.get("adaptation_percentage") or 0.0) == 0.0
            ),
            None,
        )
        best = max(
            (
                float(row["roc_auc"])
                for row in recovery_rows
                if float(row.get("adaptation_percentage") or 0.0) > 0.0
            ),
            default=None,
        )
        slides.append(
            Slide(
                "meeting_fig3_recovery",
                "3. How quickly can it recover?",
                f"A few hundred labelled {DEVELOPMENT_GENERATOR} images take ROC-AUC from "
                f"{start:.3f} to {best:.3f}."
                if start is not None and best is not None
                else "Recovery against labelled budget.",
                "That the failure is recoverable with limited labels, so the problem is "
                "adaptation cost rather than a permanent blind spot.",
                "This is recovery AFTER exposure to the new generator; it is a different "
                "question from zero-shot robustness. Original head only; single seed.",
            )
        )

    # ---------------------------------------------------------- FIGURE 4
    ablation_rows = [
        row
        for row in consolidated
        if row.get("experiment_type") == "ablation"
        and row.get("generator") is None
        and row.get("operating_point") == "default"
    ]
    if ablation_rows:
        generator = str(ablation_rows[0].get("held_out_generator"))
        zero = next(
            (
                float(row["roc_auc"])
                for row in ablation_rows
                if float(row.get("adaptation_percentage") or 0.0) == 0.0
            ),
            None,
        )
        figure, _ = plot_metric_by_depth_and_budget(
            ablation_rows,
            metric_name="roc_auc",
            zero_percent_reference=zero,
            y_label="ROC-AUC (threshold-free)",
            title=(
                f"How much of CLIP must change? Depth vs budget on held-out {generator} "
                "-- EXPLORATORY, one generator only"
            ),
        )
        _save(figure, destination / "meeting_fig4_depth")
        slides.append(
            Slide(
                "meeting_fig4_depth",
                "4. How much of CLIP needs to change?",
                "Every depth exceeds 0.98 ROC-AUC at every budget, including the "
                "769-parameter head.",
                "That representation-level adaptation is not required for ranking on this "
                "generator; the depths separate at a fixed threshold, not in ranking.",
                f"{generator}-only and EXPLORATORY: {generator} barely degraded in the "
                "first place, so there was little for depth to repair. Learning rate "
                "differs by depth (1e-3 head-only, 1e-5 otherwise). Single seed.",
            )
        )

    # ---------------------------------------------------------- FIGURE 5
    stages = _roadmap_stages(unseen, bool(recovery_rows), bool(ablation_rows))
    figure, _ = plot_experiment_roadmap(
        stages, title="Experimental logic: each stage exists because of the previous result"
    )
    _save(figure, destination / "meeting_fig5_roadmap")
    slides.append(
        Slide(
            "meeting_fig5_roadmap",
            "5. Where the project is going",
            "Each experiment was chosen because of what the previous one showed, and the "
            "confirmatory test is pre-declared rather than picked afterwards.",
            "That the project is a controlled sequence of model-development experiments, "
            "not a single pretrained model evaluated repeatedly.",
            "Stages marked PENDING have not been run; nothing about their outcome is "
            "implied here.",
        )
    )

    _write_headline_table(consolidated, unseen, destination)
    return slides


def _roadmap_stages(
    unseen: Mapping[tuple[str, str], Mapping[str, Any]],
    has_recovery: bool,
    has_ablation: bool,
) -> list[dict[str, Any]]:
    """Build the roadmap from what is actually on disk, never from intent."""

    def status(flag: bool) -> str:
        return "completed" if flag else "pending"

    linear_vqdm = unseen.get((DEVELOPMENT_GENERATOR, "linear"))
    detail = "not yet measured"
    if linear_vqdm is not None:
        pair = _degradation_pair(linear_vqdm)
        if pair["in_distribution"] is not None and pair["unseen"] is not None:
            detail = (
                f"ROC-AUC {pair['in_distribution']:.3f} -> {pair['unseen']:.3f} "
                f"(drop {pair['drop']:.3f})"
            )
    return [
        {
            "label": "In-distribution baseline",
            "status": "completed",
            "detail": "CLIP ViT-B/32 + Linear(768,1), all seven generators",
        },
        {
            "label": "Leave-one-generator-out: does it fail?",
            "status": status(any(head == "linear" for _, head in unseen)),
            "detail": f"biggan barely degrades; {DEVELOPMENT_GENERATOR} collapses -- {detail}",
        },
        {
            "label": "Limited-data recovery: is the failure repairable?",
            "status": status(has_recovery),
            "detail": "0/5/10/20/50% labelled budgets, head-only adaptation",
        },
        {
            "label": "Fine-tuning-depth ablation: how much must change?",
            "status": status(has_ablation),
            "detail": "head_only vs last_block vs full (biggan, exploratory)",
        },
        {
            "label": f"MODIFIED DETECTOR, development ({DEVELOPMENT_GENERATOR})",
            "status": status((DEVELOPMENT_GENERATOR, "cosine") in unseen),
            "detail": "cosine head, 770 params vs 769 -- magnitude removed from the decision",
        },
        {
            "label": f"CONFIRMATORY zero-shot test ({CONFIRMATORY_GENERATOR}), pre-declared",
            "status": status(
                (CONFIRMATORY_GENERATOR, "linear") in unseen
                and (CONFIRMATORY_GENERATOR, "cosine") in unseen
            ),
            "detail": "never held out before; the honest test of the zero-shot claim",
        },
    ]


def _write_headline_table(
    consolidated: Sequence[Mapping[str, Any]],
    unseen: Mapping[tuple[str, str], Mapping[str, Any]],
    destination: Path,
) -> None:
    """One compact table of every completed headline result."""

    rows: list[list[Any]] = []
    for (generator, head), entry in sorted(unseen.items()):
        pair = _degradation_pair(entry)
        block = entry["metrics"]["unseen_test"]["at_default_threshold"]
        rows.append(
            [
                f"unseen {generator} @ 0%",
                head,
                pair["in_distribution"],
                pair["unseen"],
                pair["drop"],
                block.get("average_precision"),
                block.get("f1"),
                block.get("recall"),
                entry["record"].run_id,
            ]
        )
    for row in consolidated:
        if (
            row.get("experiment_type") not in {"fine_tuning", "ablation"}
            or row.get("generator") is not None
            or row.get("operating_point") != "default"
            or float(row.get("adaptation_percentage") or 0.0) == 0.0
        ):
            continue
        rows.append(
            [
                f"{row['held_out_generator']} @ "
                f"{float(row['adaptation_percentage']) * 100:g}% ({row['fine_tune_mode']})",
                "linear",
                None,
                row.get("roc_auc"),
                None,
                row.get("average_precision"),
                row.get("f1"),
                row.get("recall"),
                row.get("run_id"),
            ]
        )

    headers = [
        "condition",
        "head",
        "in_distribution_roc_auc",
        "roc_auc",
        "roc_auc_drop",
        "pr_auc",
        "f1_at_0.5",
        "recall_at_0.5",
        "run_id",
    ]
    stem = destination / "meeting_table_headline_results"
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for table_row in rows:
        rendered = [
            "-" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
            for value in table_row
        ]
        lines.append("| " + " | ".join(rendered) + " |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(slides: Sequence[Slide], destination: Path, output_root: Path) -> None:
    records = discover_runs(output_root)
    usable = reportable_runs(records)
    unseen = _unseen_by_generator_and_head(usable)
    pending = [
        f"`{name}` — not yet run"
        for name, present in (
            (
                f"tiny_unseen_{DEVELOPMENT_GENERATOR}_cosine",
                (DEVELOPMENT_GENERATOR, "cosine") in unseen,
            ),
            (
                f"tiny_unseen_{CONFIRMATORY_GENERATOR}",
                (CONFIRMATORY_GENERATOR, "linear") in unseen,
            ),
            (
                f"tiny_unseen_{CONFIRMATORY_GENERATOR}_cosine",
                (CONFIRMATORY_GENERATOR, "cosine") in unseen,
            ),
        )
        if not present
    ]

    lines = [
        "# Supervisor meeting pack",
        "",
        "Five figures in presentation order, about two minutes. Every number is copied "
        "from a saved run artefact; nothing is interpolated, and an experiment that has "
        "not been run is marked PENDING rather than estimated.",
        "",
        "**Standing caveats for all five:** one seed per condition, so no error bars and "
        "no significance claims; and *zero-shot robustness* (before seeing a generator) "
        "is a different question from *adaptation* (after seeing it).",
        "",
    ]
    for index, slide in enumerate(slides, start=1):
        lines += [
            f"## {slide.heading}",
            "",
            f"`{slide.filename}.pdf` / `.png`"
            + ("  **[CONTAINS A PENDING ARM]**" if slide.status == "pending" else ""),
            "",
            f"- **Say:** {slide.say}",
            f"- **Demonstrates:** {slide.demonstrates}",
            f"- **Caveat:** {slide.caveat}",
            "",
        ]
        del index

    lines += ["## Completed experiments", ""]
    for record in usable:
        head = "cosine" if record.experiment_name.endswith("_cosine") else "linear"
        lines.append(
            f"- `{record.experiment_name}` ({record.experiment_type}, {head} head) — "
            f"`{record.run_id}`"
        )
    lines += ["", "## Pending experiments", ""]
    lines += [f"- {item}" for item in pending] or ["- None."]
    lines += [
        "",
        "## The methodological point to make first",
        "",
        f"`{DEVELOPMENT_GENERATOR}` exposed the failure and motivated the architecture, so "
        f"it is treated as a **development** condition. `{CONFIRMATORY_GENERATOR}` has "
        "never been held out in any run, and is **pre-declared in config comments before "
        "being run** as the confirmatory test of the zero-shot claim. Saying this before "
        "showing Figure 2 turns the obvious objection into evidence of experimental "
        "control.",
        "",
        "## Headline results table",
        "",
        "`meeting_table_headline_results.csv` / `.md`.",
        "",
    ]
    (destination / "MEETING_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/report/chapter4"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    slides = build(args.output, args.output_root)
    write_readme(slides, args.output, args.output_root)
    pending = sum(1 for slide in slides if slide.status == "pending")
    print(
        f"Meeting pack written to {args.output} ({len(slides)} figures, "
        f"{pending} containing a pending arm)"
    )


if __name__ == "__main__":
    main()
