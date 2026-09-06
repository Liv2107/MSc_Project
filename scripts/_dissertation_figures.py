"""The combined figures the dissertation needs that the Chapter 4 package does not have.

Two figures only, both of which replace an act of visual arithmetic the reader would
otherwise have to perform across two separate plots:

``fig13_depth_vs_budget_tradeoff_<generator>``
    Puts adaptation depth and labelled budget on the same axes, with the two conditions
    the trade-off argument rests on annotated in place. Reading this off ``fig02``
    (budget, head-only) beside ``fig03`` (depth grid) requires the reader to hold a
    number in their head and compare it across two coordinate systems.

``fig14_confusion_panel``
    One panel of the five confusion matrices worth showing, on a shared colour scale so
    the counts are comparable. Five separate figures are not comparable by eye and cost
    five figure slots.

Both are generated from the saved cell metrics. Neither loads a model or a prediction
file it does not need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.evaluation import dissertation as D  # noqa: E402

#: One colour per depth, kept identical across both new figures and readable in greyscale.
DEPTH_STYLE: dict[str, dict[str, Any]] = {
    "head_only": {"colour": "#4C72B0", "marker": "o", "label": "head only (769 params)"},
    "last_block": {"colour": "#DD8452", "marker": "s", "label": "last block (7.09M params)"},
    "full": {"colour": "#55A868", "marker": "^", "label": "full (87.46M params)"},
}


def _save(figure: Figure, stem: Path) -> list[Path]:
    written = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, bbox_inches="tight", dpi=200)
        written.append(path)
    plt.close(figure)
    return written


def depth_vs_budget_tradeoff(
    ctx: D.Context, generator: str, destination: Path, *, metric: str = "roc_auc"
) -> list[Path]:
    """Depth curves over budget, with the cheap-deep against dear-shallow pair annotated."""

    rows = [
        r
        for r in D.depth_table(ctx)
        if r["held_out_generator"] == generator and r[metric] is not None
    ]
    if not rows:
        return []
    zero = next(
        (
            r
            for r in D.recovery_table(ctx)
            if r["held_out_generator"] == generator
            and r["protocol"] == "ablation"
            and r["fine_tune_mode"] == "none"
        ),
        None,
    )

    figure, axes = plt.subplots(figsize=(7.2, 5.0))
    for mode, style in DEPTH_STYLE.items():
        curve = sorted(
            (r for r in rows if r["fine_tune_mode"] == mode),
            key=lambda r: r["adaptation_percentage"],
        )
        if not curve:
            continue
        xs = [r["labelled_fake_images"] for r in curve]
        ys = [r[metric] for r in curve]
        if zero is not None and zero[metric] is not None:
            xs = [0, *xs]
            ys = [zero[metric], *ys]
        axes.plot(
            xs,
            ys,
            marker=style["marker"],
            color=style["colour"],
            label=style["label"],
            linewidth=1.8,
            markersize=6,
        )

    cheap_deep = next(
        (r for r in rows if r["fine_tune_mode"] == "full" and r["adaptation_percentage"] == 0.05),
        None,
    )
    dear_shallow = next(
        (
            r
            for r in rows
            if r["fine_tune_mode"] == "head_only" and r["adaptation_percentage"] == 0.50
        ),
        None,
    )
    if cheap_deep and dear_shallow:
        # Label positions are given in data coordinates and aimed at the two regions the
        # curves leave empty: above the rising full curve on the left, and below the
        # head-only curve on the right. Offset-point placement put both boxes in the
        # same crowded band around y=0.94, where they overlapped each other.
        span = max(r[metric] for r in rows) - min(
            [zero[metric]] if zero and zero[metric] is not None else [] + [r[metric] for r in rows]
        )
        top = max(r[metric] for r in rows)
        bottom = zero[metric] if zero and zero[metric] is not None else min(
            r[metric] for r in rows
        )
        for row, note, target in (
            (cheap_deep, "full, 5% budget", (2.0, top - span * 0.04)),
            (dear_shallow, "head only, 50% budget", (150.0, bottom + span * 0.22)),
        ):
            axes.annotate(
                f"{note}\n{D.METRIC_LABELS.get(metric, metric)} {row[metric]:.4f}\n"
                f"{row['labelled_fake_images']} labelled images",
                xy=(row["labelled_fake_images"], row[metric]),
                xytext=target,
                textcoords="data",
                fontsize=8,
                ha="left",
                va="center",
                bbox={"boxstyle": "round,pad=0.35", "fc": "#FFFFFF", "ec": "#999999", "lw": 0.7},
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#666666",
                    "lw": 0.8,
                    "connectionstyle": "arc3,rad=0.12",
                },
            )
        axes.axhline(
            dear_shallow[metric], color="#999999", linestyle=":", linewidth=1.0, zorder=0
        )

    axes.set_xscale("symlog", linthresh=100)
    axes.set_xticks([0, 100, 200, 400, 1000])
    axes.set_xticklabels(["0", "100", "200", "400", "1000"])
    axes.set_xlabel(f"labelled held-out {generator} images used for adaptation")
    axes.set_ylabel(D.METRIC_LABELS.get(metric, metric))
    axes.set_title(
        f"Adaptation depth against labelled budget, held-out {generator}", fontsize=11
    )
    axes.grid(True, alpha=0.25, linewidth=0.6)
    axes.legend(fontsize=8, loc="lower right", frameon=True)

    rates = sorted({(r["fine_tune_mode"], r["learning_rate"]) for r in rows if r["learning_rate"]})
    caveat = "; ".join(f"{mode} {rate:.0e}" for mode, rate in rates)
    figure.text(
        0.01,
        -0.04,
        f"Learning rate is not constant across depths ({caveat}). Single subset seed and "
        "single training seed: points are individual fits, not means, and no error bars "
        "are implied.",
        fontsize=7,
        color="#444444",
        wrap=True,
    )
    return _save(figure, destination / f"fig13_depth_vs_budget_tradeoff_{generator}")


def confusion_panel(ctx: D.Context, destination: Path) -> list[Path]:
    """The five selected confusion matrices in one panel on a shared scale."""

    selected = [row for row in D.confusion_selection(ctx) if row.get("available")]
    if not selected:
        return []

    figure, axes_list = plt.subplots(1, len(selected), figsize=(3.0 * len(selected), 3.4))
    if len(selected) == 1:
        axes_list = [axes_list]
    largest = max(
        max(r["true_negative"], r["false_positive"], r["false_negative"], r["true_positive"])
        for r in selected
    )

    for axes, row in zip(axes_list, selected, strict=True):
        matrix = [
            [row["true_negative"], row["false_positive"]],
            [row["false_negative"], row["true_positive"]],
        ]
        axes.imshow(matrix, cmap="Blues", vmin=0, vmax=largest)
        for i in range(2):
            for j in range(2):
                value = matrix[i][j]
                axes.text(
                    j,
                    i,
                    f"{value}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if value > largest * 0.55 else "#222222",
                )
        axes.set_xticks([0, 1])
        axes.set_xticklabels(["pred real", "pred fake"], fontsize=8)
        axes.set_yticks([0, 1])
        axes.set_yticklabels(["real", "fake"], fontsize=8)
        axes.set_title(
            f"{row['label']}\nrecall {row['recall']:.3f}  F1 {row['f1']:.3f}", fontsize=9
        )

    figure.suptitle(
        "Held-out test confusion at the fixed 0.5 threshold (n=500, balanced)", fontsize=11
    )
    figure.text(
        0.01,
        -0.02,
        "Shared colour scale across panels. Counts at a fixed 0.5 threshold; no threshold "
        "was re-selected. Single seed per cell.",
        fontsize=7,
        color="#444444",
    )
    figure.tight_layout()
    return _save(figure, destination / "fig14_confusion_panel")


def build_all(ctx: D.Context, destination: Path) -> list[Path]:
    written: list[Path] = []
    for record in ctx.ablation_runs():
        generator = ctx.held_out_of(record.run_id)
        if generator:
            written += depth_vs_budget_tradeoff(ctx, generator, destination)
    written += confusion_panel(ctx, destination)
    return written
