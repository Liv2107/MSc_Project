"""Plotting interfaces built from evaluated data."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
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
    """Visualise fixed-order [[TN, FP], [FN, TP]] counts or rates."""
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
    """Compare ROC curves with AUC and a chance diagonal."""
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
    """Compare PR curves against the positive-class prevalence baseline."""
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
    """Show per-generator performance without hiding unequal support."""
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
    """Plot train/validation loss and declared metrics across epochs."""
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
    """Plot performance against labelled adaptation-data percentage."""
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
    held_out_by_percentage: dict[float, set[int]] = {}
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
                    held_out_count = row.get("held_out_fake_count")
                    if held_out_count is not None:
                        held_out_by_percentage.setdefault(percentage, set()).add(
                            int(held_out_count)
                        )
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

    def _range_text(values: set[int]) -> str | None:
        ordered = sorted(values)
        if not ordered:
            return None
        return str(ordered[0]) if len(ordered) == 1 else f"{ordered[0]}-{ordered[-1]}"

    for percentage in percentages_present:
        total = _range_text(counts_by_percentage.get(percentage, set()))
        held_out = _range_text(held_out_by_percentage.get(percentage, set()))
        if held_out is not None and total is not None:
            tick_labels.append(f"{percentage * 100:g}%\n{held_out} / {total}")
        elif total is not None:
            tick_labels.append(f"{percentage * 100:g}%\n(n={total})")
        else:
            tick_labels.append(f"{percentage * 100:g}%")
    axes.set_xticks(tick_positions, labels=tick_labels, fontsize=8)
    axes.set_xlim(min(tick_positions) - 2, max(tick_positions) + 2)
    axes.set_ylim(0.0, 1.0)
    if held_out_by_percentage:
        axis_label = (
            "Labelled adaptation budget (% of adaptation pool)\n"
            "Ticks show: held-out-generator images / all labelled images "
            "(including shared authentic ones)"
        )
    else:
        axis_label = (
            "Labelled adaptation budget (% of adaptation pool; n = ALL labelled images, "
            "held-out-generator and shared authentic combined)"
        )
    axes.set_xlabel(axis_label)
    axes.set_ylabel(metric_name)
    held_out_names = sorted(
        {str(row["held_out_generator"]) for row in adapted if row.get("held_out_generator")}
    )
    subject = ", ".join(held_out_names) if held_out_names else "the held-out generator"
    axes.set_title(f"Recovery on held-out {subject} ({metric_name})")
    axes.legend(loc="lower right", fontsize=8, frameon=False, title="Fine-tune depth")
    return figure, axes


IN_DISTRIBUTION_COLOUR = "#0072B2"
UNSEEN_COLOUR = "#D55E00"


def _mean_of(rows: Sequence[Mapping[str, Any]], metric_name: str) -> float:
    values = [float(row[metric_name]) for row in rows]
    return sum(values) / len(values)


def plot_generalisation_degradation(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = ("roc_auc", "average_precision", "f1", "accuracy"),
    operating_point: str = "default",
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Contrast in-distribution and held-out-generator performance, metric by metric."""
    if not rows:
        raise ValueError("at least one degradation row is required")
    selected = [row for row in rows if str(row.get("operating_point")) == operating_point]
    if not selected:
        raise ValueError(f"no degradation row at operating point {operating_point!r}")

    ordered: list[tuple[str, str, float, float, float | None]] = []
    undefined: list[str] = []
    for metric in metrics:
        for row in selected:
            if str(row.get("metric")) != metric:
                continue
            generator = str(row.get("held_out_generator") or "unknown")
            reference, unseen = row.get("in_distribution"), row.get("unseen")
            if reference is None or unseen is None:
                undefined.append(f"{generator}/{metric}")
                continue
            drop = row.get("absolute_drop")
            ordered.append(
                (
                    generator,
                    metric,
                    float(reference),
                    float(unseen),
                    None if drop is None else float(drop),
                )
            )
    if not ordered:
        raise ValueError("no degradation row has both an in-distribution and an unseen value")

    positions = np.arange(len(ordered), dtype=np.float64)
    width = 0.38
    figure, axes = _new_figure(width=max(6.4, 1.5 * len(ordered) + 1.6), height=4.6)
    axes.bar(
        positions - width / 2,
        [reference for _, _, reference, _, _ in ordered],
        width,
        color=IN_DISTRIBUTION_COLOUR,
        label="In-distribution test",
    )
    axes.bar(
        positions + width / 2,
        [unseen for _, _, _, unseen, _ in ordered],
        width,
        color=UNSEEN_COLOUR,
        label="Held-out (unseen) generator",
    )
    for index, (_, _, reference, unseen, drop) in enumerate(ordered):
        if drop is None:
            continue
        axes.annotate(
            f"{drop:+.3f}",
            xy=(positions[index], max(reference, unseen)),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#333333",
        )

    generators = {generator for generator, _, _, _, _ in ordered}
    axes.set_xticks(
        positions,
        labels=[
            metric if len(generators) == 1 else f"{metric}\n({generator})"
            for generator, metric, _, _, _ in ordered
        ],
    )
    axes.set_ylim(0.0, 1.0)
    axes.set_ylabel("Score")
    axes.set_xlabel(f"Metric (operating point: {operating_point})")
    subject = ", ".join(sorted(generators))
    axes.set_title(title or f"Degradation on the held-out generator ({subject})")
    handles, labels = axes.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8, frameon=False)
    if undefined:
        axes.text(
            0.01,
            -0.30,
            "Undefined on one side, not plotted: " + ", ".join(sorted(set(undefined))),
            transform=axes.transAxes,
            fontsize=7.5,
            color="#444444",
        )
    return figure, axes


def plot_parameter_efficiency(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_name: str = "roc_auc",
    zero_percent_reference: float | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot performance against the number of parameters each depth actually trained."""
    if not rows:
        raise ValueError("at least one cell row is required")
    usable = [
        row
        for row in rows
        if row.get(metric_name) is not None and row.get("trainable_parameters") is not None
    ]
    if not usable:
        raise ValueError(
            f"no row carries both {metric_name!r} and a trainable_parameters count; "
            "parameter efficiency cannot be plotted without both"
        )

    budgets = [float(row.get("adaptation_percentage") or 0.0) for row in usable]
    largest = max(budgets) or 1.0
    figure, axes = _new_figure(width=7.0, height=4.6)
    if zero_percent_reference is not None:
        axes.axhline(
            float(zero_percent_reference),
            color=FINE_TUNE_MODE_COLOURS["none"],
            linestyle=":",
            linewidth=1.4,
            label=f"0% adaptation reference ({float(zero_percent_reference):.3f})",
        )

    modes = sorted({str(row["fine_tune_mode"]) for row in usable})
    for mode in modes:
        mode_rows = [row for row in usable if str(row["fine_tune_mode"]) == mode]
        colour = FINE_TUNE_MODE_COLOURS.get(mode, _colour_for(modes.index(mode)))
        axes.scatter(
            [int(row["trainable_parameters"]) for row in mode_rows],
            [float(row[metric_name]) for row in mode_rows],
            s=[
                28 + 170 * (float(row.get("adaptation_percentage") or 0.0) / largest)
                for row in mode_rows
            ],
            color=colour,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            label=mode,
        )
        by_position: dict[int, list[Mapping[str, Any]]] = {}
        for row in mode_rows:
            by_position.setdefault(int(row["trainable_parameters"]), []).append(row)
        for parameters, group in by_position.items():
            descending = _mean_of(group, metric_name) > 0.5
            ordered_group = sorted(
                group, key=lambda item: float(item[metric_name]), reverse=descending
            )
            for rank, row in enumerate(ordered_group):
                axes.annotate(
                    f"{float(row.get('adaptation_percentage') or 0.0) * 100:g}%",
                    xy=(parameters, float(row[metric_name])),
                    xytext=(8, (-1 if descending else 1) * (2 + 10 * rank)),
                    textcoords="offset points",
                    fontsize=7.5,
                    color="#444444",
                )

    axes.set_xscale("log")
    axes.set_xlabel("Trainable parameters (log scale)")
    axes.set_ylabel(metric_name)
    axes.set_ylim(0.0, 1.0)
    axes.set_title(title or f"Parameter efficiency of adaptation ({metric_name})")
    if len(modes) == 1:
        axes.text(
            0.01,
            -0.20,
            f"Only the {modes[0]!r} depth has been run; no depth comparison is shown.",
            transform=axes.transAxes,
            fontsize=7.5,
            color="#444444",
        )
    axes.legend(loc="lower right", fontsize=8, frameon=False, title="Fine-tune depth")
    return figure, axes


BUDGET_ORDER_NOTE = "Budgets are % of the held-out generator's adaptation pool."


def plot_threshold_response(
    series: Mapping[str, Any],
    *,
    metric_name: str = "f1",
    reference_thresholds: Mapping[str, float] | None = None,
    default_threshold: float = 0.5,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a threshold-dependent metric across the whole decision-threshold range."""
    if not series:
        raise ValueError("at least one threshold-response series is required")
    figure, axes = _new_figure(width=7.2, height=4.6)
    axes.axvline(
        default_threshold,
        color="#444444",
        linestyle="--",
        linewidth=1.2,
        zorder=2,
        label=f"fixed prior ({default_threshold:g})",
    )
    for index, (label, curve) in enumerate(series.items()):
        thresholds, values = curve
        colour = FINE_TUNE_MODE_COLOURS.get(str(label).split()[0], _colour_for(index))
        axes.plot(thresholds, values, color=colour, linewidth=1.9, label=str(label), zorder=3)
        marked = (reference_thresholds or {}).get(label)
        if marked is not None:
            nearest = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - marked))
            axes.scatter(
                [thresholds[nearest]],
                [values[nearest]],
                color=colour,
                marker="D",
                s=34,
                zorder=4,
                edgecolors="white",
                linewidths=0.6,
            )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("Decision threshold applied to the predicted probability")
    axes.set_ylabel(metric_name)
    axes.set_title(title or f"{metric_name} across the decision-threshold range")
    axes.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=len(series) + 1,
        fontsize=8,
        frameon=False,
    )
    if reference_thresholds:
        axes.text(
            0.01,
            -0.30,
            "Diamonds mark each model's own validation-selected threshold.",
            transform=axes.transAxes,
            fontsize=7.5,
            color="#444444",
        )
    return figure, axes


def _logit(values: np.ndarray, *, clip: float = 1e-7) -> np.ndarray:
    bounded = np.clip(values.astype(np.float64), clip, 1.0 - clip)
    return np.log(bounded / (1.0 - bounded))


def plot_score_distributions(
    panels: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    *,
    default_threshold: float = 0.5,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Show where each model places real and fake scores relative to the threshold."""
    if not panels:
        raise ValueError("at least one score-distribution panel is required")
    figure = Figure(figsize=(7.2, 2.2 * len(panels) + 1.0), dpi=200, layout="constrained")
    axes_list = figure.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
    edges = np.linspace(-17.0, 17.0, 61)
    cut = float(_logit(np.asarray([default_threshold]))[0])

    for panel_axes, (label, real_scores, fake_scores) in zip(axes_list, panels, strict=True):
        real = _logit(np.asarray(real_scores, dtype=np.float64))
        fake = _logit(np.asarray(fake_scores, dtype=np.float64))
        panel_axes.hist(real, bins=edges, color=REAL_COLOUR, alpha=0.75, label="Real (0)")
        panel_axes.hist(fake, bins=edges, color=FAKE_COLOUR, alpha=0.75, label="Fake (1)")
        panel_axes.axvline(cut, color="#444444", linestyle="--", linewidth=1.2)
        below = int(np.sum(fake < cut))
        panel_axes.set_title(
            f"{label}   -   {below} of {len(fake)} held-out fakes fall below the "
            f"{default_threshold:g} threshold",
            fontsize=9,
            loc="left",
        )
        panel_axes.set_ylabel("count")
        panel_axes.spines["top"].set_visible(False)
        panel_axes.spines["right"].set_visible(False)

    probabilities = (1e-7, 1e-4, 0.01, 0.5, 0.99, 1 - 1e-4, 1 - 1e-7)
    axes_list[-1].set_xticks(
        _logit(np.asarray(probabilities)),
        labels=[f"{p:g}" if p not in (0.5,) else "0.5\n(threshold)" for p in probabilities],
        fontsize=7.5,
    )
    axes_list[-1].set_xlabel("Predicted probability of 'fake', on a log-odds axis")
    handles, labels = axes_list[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    figure.suptitle(title or "Score distributions on the held-out generator", fontsize=11)
    return figure, axes_list[0]


def plot_metric_by_depth_and_budget(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_name: str,
    zero_percent_reference: float | None = None,
    mode_order: Sequence[str] = ("head_only", "last_block", "full"),
    title: str | None = None,
    y_label: str | None = None,
) -> tuple[Figure, Axes]:
    """Grouped bars: one budget group per x tick, one bar per fine-tuning depth."""
    usable = [
        row
        for row in rows
        if row.get(metric_name) is not None
        and float(row.get("adaptation_percentage") or 0.0) > 0.0
    ]
    if not usable:
        raise ValueError(f"no adapted row has a defined {metric_name}")
    budgets = sorted({float(row["adaptation_percentage"]) for row in usable})
    modes = [mode for mode in mode_order if any(str(r["fine_tune_mode"]) == mode for r in usable)]
    modes += sorted(
        {str(r["fine_tune_mode"]) for r in usable}.difference(modes)
    )

    figure, axes = _new_figure(width=max(6.8, 1.6 * len(budgets) + 2.2), height=4.6)
    positions = np.arange(len(budgets), dtype=np.float64)
    width = 0.8 / max(1, len(modes))
    for index, mode in enumerate(modes):
        heights: list[float] = []
        for budget in budgets:
            match = [
                float(row[metric_name])
                for row in usable
                if str(row["fine_tune_mode"]) == mode
                and float(row["adaptation_percentage"]) == budget
            ]
            heights.append(sum(match) / len(match) if match else float("nan"))
        offset = (index - (len(modes) - 1) / 2) * width
        axes.bar(
            positions + offset,
            heights,
            width * 0.94,
            color=FINE_TUNE_MODE_COLOURS.get(mode, _colour_for(index)),
            label=mode,
        )
    if zero_percent_reference is not None:
        axes.axhline(
            float(zero_percent_reference),
            color=FINE_TUNE_MODE_COLOURS["none"],
            linestyle=":",
            linewidth=1.5,
            zorder=4,
            label=f"0% adaptation ({float(zero_percent_reference):.3f})",
        )
    axes.set_xticks(positions, labels=[f"{budget * 100:g}%" for budget in budgets])
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel(f"Labelled adaptation budget   ({BUDGET_ORDER_NOTE})")
    axes.set_ylabel(y_label or metric_name)
    axes.set_title(title or f"{metric_name} by fine-tuning depth and adaptation budget")
    axes.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=len(modes) + 1,
        fontsize=8,
        frameon=False,
        title="Fine-tune depth",
    )
    return figure, axes


def plot_validation_trajectories(
    series: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    metric_name: str = "f1",
    selected_epochs: Mapping[str, int] | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Validation metric per epoch for a few named cells, on shared axes."""
    if not series:
        raise ValueError("at least one trajectory is required")
    figure, axes = _new_figure(width=7.0, height=4.4)
    for index, (label, history) in enumerate(series.items()):
        rows = [
            row
            for row in history
            if str(row.get("split")) == "validation" and row.get(metric_name) is not None
        ]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: int(row["epoch"]))
        colour = FINE_TUNE_MODE_COLOURS.get(str(label).split()[0], _colour_for(index))
        epochs = [int(row["epoch"]) for row in rows]
        values = [float(row[metric_name]) for row in rows]
        axes.plot(epochs, values, color=colour, marker="o", markersize=3.5,
                  linewidth=1.8, label=str(label))
        chosen = (selected_epochs or {}).get(label)
        if chosen is not None and chosen in epochs:
            axes.scatter([chosen], [values[epochs.index(chosen)]], color=colour, marker="D",
                         s=44, zorder=5, edgecolors="white", linewidths=0.7)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("Adaptation epoch")
    axes.set_ylabel(f"Adaptation-validation {metric_name}")
    axes.set_title(title or f"Adaptation-validation {metric_name} per epoch")
    axes.legend(loc="lower right", fontsize=8, frameon=False, title="Fine-tune depth")
    if selected_epochs:
        axes.text(0.01, -0.20, "Diamonds mark the validation-selected epoch.",
                  transform=axes.transAxes, fontsize=7.5, color="#444444")
    return figure, axes


STATUS_COLOURS = {
    "completed": "#009E73",
    "pending": "#999999",
    "planned": "#CCCCCC",
}


def plot_experiment_roadmap(
    stages: Sequence[Mapping[str, Any]], *, title: str | None = None
) -> tuple[Figure, Axes]:
    """Draw the experimental logic of the study as a status-annotated flow."""
    if not stages:
        raise ValueError("at least one stage is required")
    unknown = sorted(
        {str(stage.get("status")) for stage in stages}.difference(STATUS_COLOURS)
    )
    if unknown:
        raise ValueError(f"unknown stage status: {', '.join(unknown)}")

    height = 1.15 * len(stages) + 1.0
    figure = Figure(figsize=(7.6, height), dpi=200, layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    axes.set_xlim(0, 10)
    axes.set_ylim(0, len(stages))
    axes.axis("off")

    for index, stage in enumerate(stages):
        row = len(stages) - index - 1
        status = str(stage["status"])
        colour = STATUS_COLOURS[status]
        axes.add_patch(
            Rectangle(
                (0.4, row + 0.18),
                9.2,
                0.64,
                facecolor=colour if status == "completed" else "white",
                edgecolor=colour,
                linewidth=1.6,
                alpha=0.16 if status == "completed" else 1.0,
                zorder=2,
            )
        )
        axes.text(
            0.75,
            row + 0.62,
            f"{index + 1}. {stage['label']}",
            fontsize=10,
            fontweight="bold",
            va="center",
            zorder=3,
        )
        detail = stage.get("detail")
        if detail:
            axes.text(0.75, row + 0.36, str(detail), fontsize=8.5, va="center",
                      color="#333333", zorder=3)
        axes.text(
            9.35,
            row + 0.5,
            status.upper(),
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="center",
            color=colour if status != "completed" else "#00694F",
            zorder=3,
        )
        if index < len(stages) - 1:
            axes.annotate(
                "",
                xy=(5.0, row),
                xytext=(5.0, row + 0.18),
                arrowprops={"arrowstyle": "-|>", "color": "#666666", "linewidth": 1.2},
            )
    axes.set_title(title or "Experimental logic of the study", fontsize=11, pad=10)
    return figure, axes


def plot_embedding_norms(
    rows: Sequence[Mapping[str, Any]],
    *,
    raw_norms: Mapping[str, Sequence[float]] | None = None,
    highlight: str | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Show the L2 norm of the frozen CLIP embedding for each generator."""
    if not rows:
        raise ValueError("at least one generator row is required")
    ordered = sorted(rows, key=lambda row: float(row["mean_norm"]))
    positions = np.arange(len(ordered), dtype=np.float64)
    figure, axes = _new_figure(width=8.4, height=0.46 * len(ordered) + 2.4)

    lows = [float(row["min_norm"]) for row in ordered if row.get("min_norm") is not None]
    highs = [float(row["max_norm"]) for row in ordered if row.get("max_norm") is not None]
    if not lows or not highs:
        lows = [float(row["p10_norm"]) for row in ordered]
        highs = [float(row["p90_norm"]) for row in ordered]
    span = max(max(highs) - min(lows), 1e-6)
    axes.set_xlim(min(lows) - 0.12 * span, max(highs) + 0.12 * span)

    for index, row in enumerate(ordered):
        name = str(row["generator"])
        is_highlight = highlight is not None and name == highlight
        colour = FAKE_COLOUR if is_highlight else (REAL_COLOUR if name == "real" else "#999999")
        values = list((raw_norms or {}).get(name) or [])
        if values:
            axes.scatter(
                values,
                np.full(len(values), positions[index]),
                color=colour,
                alpha=0.28,
                s=12,
                zorder=2,
            )
        axes.plot(
            [float(row["p10_norm"]), float(row["p90_norm"])],
            [positions[index], positions[index]],
            color=colour,
            linewidth=2.4,
            solid_capstyle="butt",
            zorder=3,
        )
        axes.scatter(
            [float(row["mean_norm"])],
            [positions[index]],
            marker="|",
            color="black",
            s=200,
            linewidths=1.6,
            zorder=4,
        )

    axes.set_yticks(
        positions,
        labels=[
            f"{row['generator']}{'  (held out)' if highlight == row['generator'] else ''}"
            f"  n={row['n']}"
            for row in ordered
        ],
    )
    axes.set_xlabel("L2 norm of the frozen CLIP pooled embedding")
    axes.set_title(title or "Embedding magnitude by generator")
    axes.text(
        0.0,
        -0.16 - 0.02 * len(ordered),
        "Bars span the 10th-90th percentile; the tick marks the mean. Overlapping ranges "
        "mean magnitude does NOT separate generators.",
        transform=axes.transAxes,
        fontsize=7.5,
        color="#444444",
    )
    return figure, axes
