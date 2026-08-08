"""Publication-quality plotting interfaces built from evaluated data.

###############################################################################
PURPOSE
###############################################################################

Figures should be deterministic views of saved predictions/history, never locations
where metrics are secretly recalculated with different conventions. Each function
should accept explicit data and return a Matplotlib Figure/Axes so notebooks decide
when and where to save or display it.

Use accessible colours, readable fonts, labelled axes, sample counts, uncertainty
where available, and vector output (PDF/SVG) for dissertation inclusion. Do not use
3D charts or truncate axes in ways that exaggerate recovery.

###############################################################################
CONVENTIONS THESE FUNCTIONS HOLD TO
###############################################################################

* No function calls ``show``, no function writes a file, and no function mutates
  global Matplotlib state. Each returns ``(Figure, Axes)`` for the caller to place.
* Metrics are never recomputed here. Values arrive already computed from saved
  predictions, so a figure can always be traced back to a prediction table.
* Undefined values (one-class ROC-AUC, for instance) are omitted and annotated as
  undefined. They are never silently drawn as zero, which would read as "very bad"
  instead of "not measurable".
* Axes covering a metric in [0, 1] are drawn on the full range, so a recovery curve
  cannot be exaggerated by cropping.
* Sample support is shown wherever unequal support could mislead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Colour-blind-safe qualitative palette (Okabe-Ito), used consistently across figures.
PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)
REAL_COLOUR = "#0072B2"
FAKE_COLOUR = "#D55E00"
FINE_TUNE_MODE_COLOURS = {
    "head_only": "#0072B2",
    "last_block": "#D55E00",
    "full": "#009E73",
    "none": "#666666",
}


def _new_figure(*, width: float = 6.4, height: float = 4.4) -> tuple[Figure, Axes]:
    """Create an isolated figure so no global style is touched."""

    figure = Figure(figsize=(width, height), dpi=150, layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    return figure, axes


def _colour_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def plot_confusion_matrix(
    matrix: object, *, normalize: bool = False, title: str | None = None
) -> tuple[Figure, Axes]:
    """Visualise fixed-order [[TN, FP], [FN, TP]] counts or rates.

    ``normalize`` divides by ROW totals, i.e. each row shows the rate within one true
    class. Rows with zero support are drawn as undefined rather than as zero, and the
    normalisation choice is written into the colour-bar label so no reader has to guess.
    """

    array = np.asarray(matrix)
    if array.shape != (2, 2):
        raise ValueError(f"confusion matrix must be 2x2; received {array.shape}")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("confusion matrix values must be finite and non-negative")
    counts = array.astype(np.float64)
    row_totals = counts.sum(axis=1)

    if normalize:
        display = np.full_like(counts, np.nan)
        for row in range(2):
            if row_totals[row] > 0:
                display[row] = counts[row] / row_totals[row]
        value_label = "Rate within true class (row-normalised)"
        limits = (0.0, 1.0)
    else:
        display = counts
        value_label = "Sample count"
        limits = (0.0, float(counts.max()) if counts.max() > 0 else 1.0)

    figure, axes = _new_figure(width=5.2, height=4.4)
    image = axes.imshow(
        np.ma.masked_invalid(display), cmap="Blues", vmin=limits[0], vmax=limits[1]
    )
    labels = ("Real (0)", "Fake (1)")
    axes.set_xticks([0, 1], labels=labels)
    axes.set_yticks([0, 1], labels=labels)
    axes.set_xlabel("Predicted class")
    axes.set_ylabel("True class")
    axes.set_title(title or "Confusion matrix")

    midpoint = (limits[0] + limits[1]) / 2
    for row in range(2):
        for column in range(2):
            if normalize and row_totals[row] == 0:
                text = "undefined\n(no support)"
                colour = "black"
            elif normalize:
                text = f"{display[row, column]:.3f}\n(n={int(counts[row, column])})"
                colour = "white" if display[row, column] > midpoint else "black"
            else:
                text = f"{int(counts[row, column])}"
                colour = "white" if display[row, column] > midpoint else "black"
            axes.text(column, row, text, ha="center", va="center", color=colour, fontsize=9)
    figure.colorbar(image, ax=axes, label=value_label)
    return figure, axes


def _curve_arrays(curve: object) -> tuple[np.ndarray, np.ndarray]:
    """Accept a CurveData-like object or an (x, y) pair."""

    x = getattr(curve, "x", None)
    y = getattr(curve, "y", None)
    if x is None or y is None:
        if not isinstance(curve, Sequence) or len(curve) < 2:
            raise TypeError("curve must expose .x/.y or be an (x, y) sequence")
        x, y = curve[0], curve[1]
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.ndim != 1 or y_array.ndim != 1 or len(x_array) != len(y_array):
        raise ValueError("curve coordinates must be one-dimensional and equal length")
    if len(x_array) == 0:
        raise ValueError("curve contains no points")
    return x_array, y_array


def plot_roc_curves(
    curves: Mapping[str, object],
    *,
    title: str | None = None,
    areas: Mapping[str, float | None] | None = None,
    supports: Mapping[str, int] | None = None,
) -> tuple[Figure, Axes]:
    """Compare ROC curves with AUC and a chance diagonal.

    ``areas`` and ``supports`` are supplied by the caller from already-computed metrics;
    nothing is integrated here. An entry whose area is ``None`` is labelled as undefined
    (a single-class slice) rather than being drawn as if it scored zero.
    """

    if not curves:
        raise ValueError("at least one ROC curve is required")
    figure, axes = _new_figure()
    axes.plot([0, 1], [0, 1], linestyle=":", color="#666666", linewidth=1, label="Chance")
    for index, (name, curve) in enumerate(sorted(curves.items())):
        x, y = _curve_arrays(curve)
        area = (areas or {}).get(name)
        support = (supports or {}).get(name)
        pieces = [name]
        pieces.append("AUC undefined" if area is None else f"AUC={area:.3f}")
        if support is not None:
            pieces.append(f"n={support}")
        axes.plot(x, y, color=_colour_for(index), linewidth=1.8, label=", ".join(pieces))
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title(title or "ROC curves")
    axes.legend(loc="lower right", fontsize=8, frameon=False)
    return figure, axes


def plot_precision_recall_curves(
    curves: Mapping[str, object],
    *,
    prevalence: float,
    title: str | None = None,
    average_precisions: Mapping[str, float | None] | None = None,
    supports: Mapping[str, int] | None = None,
) -> tuple[Figure, Axes]:
    """Compare PR curves against the positive-class prevalence baseline.

    Curves are expected as (recall, precision), matching
    ``metrics.precision_recall_curve_data``. Labels report average precision, which is
    the step-wise summary actually computed, and never call it an area under a
    trapezoidally interpolated curve.
    """

    if not curves:
        raise ValueError("at least one precision-recall curve is required")
    if not math.isfinite(prevalence) or not 0 <= prevalence <= 1:
        raise ValueError("prevalence must be a finite value in [0, 1]")
    figure, axes = _new_figure()
    axes.axhline(
        prevalence,
        linestyle=":",
        color="#666666",
        linewidth=1,
        label=f"Prevalence baseline ({prevalence:.3f})",
    )
    for index, (name, curve) in enumerate(sorted(curves.items())):
        recall, precision = _curve_arrays(curve)
        if ((recall < 0) | (recall > 1)).any() or ((precision < 0) | (precision > 1)).any():
            raise ValueError(f"curve {name!r} has coordinates outside [0, 1]")
        score = (average_precisions or {}).get(name)
        support = (supports or {}).get(name)
        pieces = [name]
        pieces.append("AP undefined" if score is None else f"AP={score:.3f}")
        if support is not None:
            pieces.append(f"n={support}")
        axes.plot(
            recall, precision, color=_colour_for(index), linewidth=1.8, label=", ".join(pieces)
        )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_title(title or "Precision-recall curves")
    axes.legend(loc="lower left", fontsize=8, frameon=False)
    return figure, axes


def plot_generator_performance(
    rows: Sequence[Mapping[str, Any]], *, metric_name: str
) -> tuple[Figure, Axes]:
    """Show per-generator performance without hiding unequal support.

    Each row needs ``generator`` and ``support``, plus the named metric. Repeated seeds
    are drawn as individual points beside the mean, so a generator measured once is
    visibly different from one measured three times. Rows whose metric is ``None`` are
    listed as undefined instead of plotted at zero. Sorted by mean metric, ascending,
    so the weakest generator reads first.
    """

    if not rows:
        raise ValueError("at least one generator row is required")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        for required in ("generator", "support"):
            if required not in row:
                raise ValueError(f"generator rows must contain {required!r}")
        if metric_name not in row:
            raise ValueError(f"generator rows must contain the metric {metric_name!r}")
        grouped.setdefault(str(row["generator"]), []).append(row)

    measured: list[tuple[str, list[float], int]] = []
    undefined: list[str] = []
    for name, entries in grouped.items():
        values = [float(entry[metric_name]) for entry in entries if entry[metric_name] is not None]
        support = sum(int(entry["support"]) for entry in entries) // max(1, len(entries))
        if values:
            measured.append((name, values, support))
        else:
            undefined.append(name)
    if not measured:
        raise ValueError(f"no generator has a defined {metric_name}")
    measured.sort(key=lambda item: sum(item[1]) / len(item[1]))

    figure, axes = _new_figure(width=6.8, height=0.5 * len(measured) + 2.2)
    positions = np.arange(len(measured), dtype=np.float64)
    means = [sum(values) / len(values) for _, values, _ in measured]
    for offset, (name, values, _) in enumerate(measured):
        colour = REAL_COLOUR if name == "real" else FAKE_COLOUR
        axes.scatter(
            values,
            np.full(len(values), positions[offset]),
            color=colour,
            alpha=0.55,
            s=26,
            zorder=3,
            label="individual runs" if offset == 0 else None,
        )
        axes.scatter(
            [means[offset]],
            [positions[offset]],
            marker="|",
            color="black",
            s=260,
            linewidths=1.6,
            zorder=4,
            label="mean" if offset == 0 else None,
        )
    axes.set_yticks(
        positions,
        labels=[f"{name}  (n={support}, runs={len(values)})" for name, values, support in measured],
    )
    axes.set_xlim(0.0, 1.0)
    axes.set_xlabel(metric_name)
    axes.set_title(f"Per-generator {metric_name}")
    axes.legend(loc="lower right", fontsize=8, frameon=False)
    if undefined:
        axes.text(
            0.01,
            -0.16,
            f"Undefined {metric_name} (single-class slice), not plotted: "
            + ", ".join(sorted(undefined)),
            transform=axes.transAxes,
            fontsize=7.5,
            color="#444444",
        )
    return figure, axes


def plot_training_curves(
    history_rows: Sequence[Mapping[str, Any]],
    *,
    metric_name: str = "f1",
    best_epoch: int | None = None,
) -> tuple[Figure, Axes]:
    """Plot train/validation loss and declared metrics across epochs.

    Expects tidy rows as written to ``train_history.csv``: one row per epoch and split,
    with ``epoch``, ``split``, ``loss``, and metric columns. Loss and the selection
    metric are drawn on twinned axes so neither scale hides the other, and the
    validation-selected epoch is marked because that, not the final epoch, is the model
    the results come from.
    """

    if not history_rows:
        raise ValueError("training history is empty")
    ordered = sorted(history_rows, key=lambda row: (int(row["epoch"]), str(row["split"])))
    figure, axes = _new_figure()
    metric_axes = axes.twinx()
    metric_axes.spines["top"].set_visible(False)

    for index, split in enumerate(sorted({str(row["split"]) for row in ordered})):
        rows = [row for row in ordered if str(row["split"]) == split]
        epochs = [int(row["epoch"]) for row in rows]
        axes.plot(
            epochs,
            [float(row["loss"]) for row in rows],
            color=_colour_for(index),
            linewidth=1.8,
            label=f"{split} loss",
        )
        if all(metric_name in row and row[metric_name] is not None for row in rows):
            metric_axes.plot(
                epochs,
                [float(row[metric_name]) for row in rows],
                color=_colour_for(index),
                linestyle="--",
                linewidth=1.4,
                label=f"{split} {metric_name}",
            )
    if best_epoch is not None:
        axes.axvline(
            int(best_epoch),
            color="#666666",
            linestyle="-.",
            linewidth=1,
            label=f"selected epoch ({int(best_epoch)})",
        )
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss (solid)")
    metric_axes.set_ylabel(f"{metric_name} (dashed)")
    metric_axes.set_ylim(0.0, 1.0)
    axes.set_title("Training and validation history")
    handles, labels = axes.get_legend_handles_labels()
    metric_handles, metric_labels = metric_axes.get_legend_handles_labels()
    axes.legend(
        handles + metric_handles, labels + metric_labels, loc="best", fontsize=8, frameon=False
    )
    return figure, axes


def plot_fine_tuning_recovery(
    result_rows: Sequence[Mapping[str, Any]], *, metric_name: str = "f1"
) -> tuple[Figure, Axes]:
    """Plot performance against labelled adaptation-data percentage.

    Expects the per-cell rows saved by the recovery and ablation runners: each with
    ``adaptation_percentage``, ``fine_tune_mode``, ``labelled_images_consumed``, and the
    metric. The 0% row is required, because a recovery curve without its starting point
    invites reading the y-axis as absolute skill rather than as recovery.

    Repeated subset/training seeds are shown as raw scatter points alongside the mean,
    with a mean +/- one standard deviation band drawn only where at least three runs
    exist. Below that, a band would imply a precision the data cannot support.
    """

    if not result_rows:
        raise ValueError("at least one recovery row is required")
    for row in result_rows:
        for required in ("adaptation_percentage", "fine_tune_mode"):
            if required not in row:
                raise ValueError(f"recovery rows must contain {required!r}")
    values_present = [
        row for row in result_rows if row.get(metric_name) is not None
    ]
    if not values_present:
        raise ValueError(f"no recovery row has a defined {metric_name}")
    zero_rows = [row for row in values_present if float(row["adaptation_percentage"]) == 0.0]
    if not zero_rows:
        raise ValueError(
            "the 0%-adaptation reference is missing; recovery must be plotted relative "
            "to the measured unseen-generator result"
        )
    zero_value = sum(float(row[metric_name]) for row in zero_rows) / len(zero_rows)

    adapted = [row for row in values_present if float(row["adaptation_percentage"]) > 0.0]
    modes = sorted({str(row["fine_tune_mode"]) for row in adapted})
    figure, axes = _new_figure(width=7.0, height=4.6)
    axes.axhline(
        zero_value,
        color=FINE_TUNE_MODE_COLOURS["none"],
        linestyle=":",
        linewidth=1.4,
        label=f"0% adaptation reference ({zero_value:.3f})",
    )

    counts_by_percentage: dict[float, set[int]] = {}
    for mode in modes:
        mode_rows = [row for row in adapted if str(row["fine_tune_mode"]) == mode]
        percentages = sorted({float(row["adaptation_percentage"]) for row in mode_rows})
        means: list[float] = []
        deviations: list[float] = []
        run_counts: list[int] = []
        for percentage in percentages:
            cells = [
                float(row[metric_name])
                for row in mode_rows
                if float(row["adaptation_percentage"]) == percentage
            ]
            mean = sum(cells) / len(cells)
            means.append(mean)
            run_counts.append(len(cells))
            deviations.append(
                math.sqrt(sum((value - mean) ** 2 for value in cells) / (len(cells) - 1))
                if len(cells) > 1
                else 0.0
            )
            for row in mode_rows:
                if float(row["adaptation_percentage"]) == percentage:
                    consumed = row.get("labelled_images_consumed")
                    if consumed is not None:
                        counts_by_percentage.setdefault(percentage, set()).add(int(consumed))
            axes.scatter(
                [percentage * 100] * len(cells),
                cells,
                color=FINE_TUNE_MODE_COLOURS.get(mode, _colour_for(modes.index(mode))),
                alpha=0.45,
                s=24,
                zorder=3,
            )
        colour = FINE_TUNE_MODE_COLOURS.get(mode, _colour_for(modes.index(mode)))
        x_positions = [percentage * 100 for percentage in percentages]
        axes.plot(x_positions, means, color=colour, marker="o", linewidth=1.9, label=mode, zorder=4)
        if all(count >= 3 for count in run_counts):
            axes.fill_between(
                x_positions,
                [mean - deviation for mean, deviation in zip(means, deviations, strict=True)],
                [mean + deviation for mean, deviation in zip(means, deviations, strict=True)],
                color=colour,
                alpha=0.14,
                linewidth=0,
            )

    percentages_present = sorted({float(row["adaptation_percentage"]) for row in adapted})
    tick_positions = [percentage * 100 for percentage in percentages_present]
    tick_labels = []
    for percentage in percentages_present:
        counts = sorted(counts_by_percentage.get(percentage, set()))
        if len(counts) == 1:
            tick_labels.append(f"{percentage * 100:g}%\n(n={counts[0]})")
        elif counts:
            tick_labels.append(f"{percentage * 100:g}%\n(n={counts[0]}-{counts[-1]})")
        else:
            tick_labels.append(f"{percentage * 100:g}%")
    axes.set_xticks(tick_positions, labels=tick_labels)
    # Keep the axis inside the observed budgets so no trend is implied beyond them.
    axes.set_xlim(min(tick_positions) - 2, max(tick_positions) + 2)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("Labelled adaptation data (% of adaptation pool, actual image counts shown)")
    axes.set_ylabel(metric_name)
    axes.set_title(f"Recovery on the held-out generator ({metric_name})")
    axes.legend(loc="lower right", fontsize=8, frameon=False, title="Fine-tune depth")
    return figure, axes


# IMPLEMENTATION CHECKLIST
# [x] Implement plots only after metric and output schemas are stable.
# [x] Unit-test validation and smoke-test returned Figure/Axes objects.
# [x] Use consistent real/fake and generator colour conventions.
# [x] Show supports, repeated seeds, and uncertainty where available.
# [x] Export vector and high-resolution raster figures with run IDs.
#     (Exporting is the caller's job; scripts/build_report.py writes PDF plus PNG.)
# [x] Verify every plotted number traces to a saved prediction/history file.
