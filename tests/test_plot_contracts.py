"""Plotting contracts: validation, returned objects, and honest handling of gaps.

The figures carry dissertation claims, so these tests check the properties that would
otherwise mislead a reader: undefined metrics must not be drawn as zero, a recovery
curve must not be plotted without its 0% reference, and metric axes must span [0, 1]
so no gain can be exaggerated by cropping.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.evaluation.metrics import precision_recall_curve_data, roc_curve_data
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

LABELS = [0, 0, 1, 1, 0, 1]
SCORES = [0.1, 0.35, 0.7, 0.9, 0.2, 0.6]


def recovery_row(mode: str, fraction: float, f1: float | None, **extra: Any) -> dict[str, Any]:
    return {
        "fine_tune_mode": mode,
        "adaptation_percentage": fraction,
        "labelled_images_consumed": 12,
        "f1": f1,
        **extra,
    }


# --------------------------------------------------------------- confusion matrices


def test_confusion_matrix_returns_a_figure_and_validates_shape() -> None:
    figure, axes = plot_confusion_matrix(np.asarray([[3, 1], [2, 4]]))
    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    # Axis labels must name the classes, not just 0/1.
    assert [text.get_text() for text in axes.get_xticklabels()] == ["Real (0)", "Fake (1)"]

    with pytest.raises(ValueError, match="must be 2x2"):
        plot_confusion_matrix(np.asarray([[1, 2, 3], [4, 5, 6]]))
    with pytest.raises(ValueError, match="finite and non-negative"):
        plot_confusion_matrix(np.asarray([[-1, 0], [0, 1]]))


def test_row_normalised_confusion_matrix_marks_zero_support_rows_as_undefined() -> None:
    _, axes = plot_confusion_matrix(np.asarray([[0, 0], [2, 4]]), normalize=True)
    annotations = [text.get_text() for text in axes.texts]
    assert any("undefined" in text for text in annotations)
    # The populated row still shows rates alongside raw counts.
    assert any("0.333" in text and "n=2" in text for text in annotations)


# ------------------------------------------------------------------------- curves


def test_roc_and_pr_plots_span_the_full_unit_axes() -> None:
    curves = {"test": roc_curve_data(LABELS, SCORES)}
    _, axes = plot_roc_curves(curves, areas={"test": 0.89}, supports={"test": len(LABELS)})
    assert axes.get_xlim() == (0.0, 1.0)
    assert axes.get_ylim() == (0.0, 1.0)
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert any("AUC=0.890" in label and "n=6" in label for label in labels)
    assert any("Chance" in label for label in labels)

    _, pr_axes = plot_precision_recall_curves(
        {"test": precision_recall_curve_data(LABELS, SCORES)},
        prevalence=0.5,
        average_precisions={"test": 0.75},
    )
    assert pr_axes.get_ylim() == (0.0, 1.0)
    pr_labels = [text.get_text() for text in pr_axes.get_legend().get_texts()]
    # Average precision must be named as such, never as an interpolated area.
    assert any("AP=0.750" in label for label in pr_labels)
    assert any("Prevalence baseline" in label for label in pr_labels)


def test_undefined_auc_is_labelled_not_drawn_as_zero() -> None:
    _, axes = plot_roc_curves({"single class slice": roc_curve_data(LABELS, SCORES)}, areas={})
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert any("AUC undefined" in label for label in labels)
    assert not any("AUC=0.000" in label for label in labels)


def test_curve_plots_validate_their_inputs() -> None:
    with pytest.raises(ValueError, match="at least one ROC curve"):
        plot_roc_curves({})
    with pytest.raises(ValueError, match="prevalence must be"):
        plot_precision_recall_curves(
            {"a": precision_recall_curve_data(LABELS, SCORES)}, prevalence=1.4
        )
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        plot_precision_recall_curves({"a": ([0.0, 1.2], [1.0, 0.5])}, prevalence=0.5)


# -------------------------------------------------------------- generator overview


def test_generator_plot_excludes_undefined_metrics_and_shows_support() -> None:
    rows = [
        {"generator": "adm", "support": 40, "roc_auc": 0.9},
        {"generator": "biggan", "support": 40, "roc_auc": 0.6},
        {"generator": "real", "support": 40, "roc_auc": None},
    ]
    _, axes = plot_generator_performance(rows, metric_name="roc_auc")
    tick_labels = [text.get_text() for text in axes.get_yticklabels()]
    assert len(tick_labels) == 2
    assert all("n=40" in label for label in tick_labels)
    # Ascending order means the weakest generator is listed first.
    assert "biggan" in tick_labels[0]
    # The undefined slice is named in the figure rather than silently dropped.
    assert any("real" in text.get_text() for text in axes.texts)
    assert axes.get_xlim() == (0.0, 1.0)


def test_generator_plot_requires_the_named_metric_and_support() -> None:
    with pytest.raises(ValueError, match="must contain 'support'"):
        plot_generator_performance([{"generator": "adm", "f1": 0.5}], metric_name="f1")
    with pytest.raises(ValueError, match="must contain the metric"):
        plot_generator_performance([{"generator": "adm", "support": 10}], metric_name="f1")
    with pytest.raises(ValueError, match="no generator has a defined"):
        plot_generator_performance(
            [{"generator": "adm", "support": 10, "f1": None}], metric_name="f1"
        )


# ----------------------------------------------------------------- history curves


def test_training_curves_mark_the_selected_epoch() -> None:
    history = [
        {"epoch": 1, "split": "train", "loss": 0.7, "f1": 0.5},
        {"epoch": 1, "split": "validation", "loss": 0.6, "f1": 0.6},
        {"epoch": 2, "split": "train", "loss": 0.4, "f1": 0.8},
        {"epoch": 2, "split": "validation", "loss": 0.5, "f1": 0.9},
    ]
    _, axes = plot_training_curves(history, best_epoch=2)
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert any("selected epoch (2)" in label for label in labels)
    assert any("train loss" in label for label in labels)
    with pytest.raises(ValueError, match="history is empty"):
        plot_training_curves([])


# --------------------------------------------------------------- recovery curves


def test_recovery_plot_requires_the_zero_percent_reference() -> None:
    adapted_only = [recovery_row("head_only", 0.05, 0.4), recovery_row("head_only", 0.5, 0.9)]
    with pytest.raises(ValueError, match="0%-adaptation reference is missing"):
        plot_fine_tuning_recovery(adapted_only, metric_name="f1")


def test_recovery_plot_shows_counts_and_keeps_the_full_metric_axis() -> None:
    rows = [
        recovery_row("none", 0.0, 0.1),
        recovery_row("head_only", 0.05, 0.4, labelled_images_consumed=6),
        recovery_row("head_only", 0.5, 0.9, labelled_images_consumed=48),
    ]
    _, axes = plot_fine_tuning_recovery(rows, metric_name="f1")
    assert axes.get_ylim() == (0.0, 1.0)
    tick_labels = [text.get_text() for text in axes.get_xticklabels()]
    assert any("n=6" in label for label in tick_labels)
    assert any("n=48" in label for label in tick_labels)
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert any("0% adaptation reference (0.100)" in label for label in labels)
    # The axis must not extend beyond the observed budgets.
    assert axes.get_xlim()[1] <= 52


def test_recovery_band_only_appears_with_enough_repeats() -> None:
    two_runs = [
        recovery_row("none", 0.0, 0.1),
        recovery_row("head_only", 0.05, 0.3),
        recovery_row("head_only", 0.05, 0.7),
    ]
    _, axes = plot_fine_tuning_recovery(two_runs, metric_name="f1")
    assert not axes.collections or all(
        collection.get_alpha() != 0.14 for collection in axes.collections
    )

    three_runs = two_runs + [recovery_row("head_only", 0.05, 0.5)]
    _, banded = plot_fine_tuning_recovery(three_runs, metric_name="f1")
    assert any(collection.get_alpha() == 0.14 for collection in banded.collections)


def test_recovery_plot_validates_rows() -> None:
    with pytest.raises(ValueError, match="at least one recovery row"):
        plot_fine_tuning_recovery([], metric_name="f1")
    with pytest.raises(ValueError, match="must contain 'fine_tune_mode'"):
        plot_fine_tuning_recovery([{"adaptation_percentage": 0.0, "f1": 0.5}], metric_name="f1")
    with pytest.raises(ValueError, match="no recovery row has a defined"):
        plot_fine_tuning_recovery([recovery_row("none", 0.0, None)], metric_name="f1")


# ------------------------------------------------------- generalisation degradation


def degradation_row(
    metric: str,
    in_distribution: float | None,
    unseen: float | None,
    *,
    operating_point: str = "default",
) -> dict[str, Any]:
    return {
        "held_out_generator": "biggan",
        "operating_point": operating_point,
        "metric": metric,
        "in_distribution": in_distribution,
        "unseen": unseen,
        "absolute_drop": (
            None
            if in_distribution is None or unseen is None
            else in_distribution - unseen
        ),
    }


def test_degradation_plot_annotates_the_drop_and_keeps_the_full_axis() -> None:
    rows = [degradation_row("roc_auc", 0.95, 0.90), degradation_row("f1", 0.90, 0.80)]
    _, axes = plot_generalisation_degradation(rows, metrics=("roc_auc", "f1"))
    assert axes.get_ylim() == (0.0, 1.0)
    annotations = [text.get_text() for text in axes.texts]
    assert "+0.050" in annotations
    assert "+0.100" in annotations
    assert [text.get_text() for text in axes.get_xticklabels()] == ["roc_auc", "f1"]


def test_degradation_plot_only_draws_one_operating_point() -> None:
    rows = [
        degradation_row("f1", 0.90, 0.80, operating_point="default"),
        degradation_row("f1", 0.85, 0.70, operating_point="validation_selected"),
    ]
    _, axes = plot_generalisation_degradation(rows, metrics=("f1",))
    # Two bars, not four: mixing thresholds in one chart would not be a comparison.
    assert len(axes.patches) == 2
    with pytest.raises(ValueError, match="no degradation row at operating point"):
        plot_generalisation_degradation(rows, metrics=("f1",), operating_point="adaptation")


def test_degradation_plot_lists_one_sided_metrics_instead_of_drawing_them() -> None:
    rows = [degradation_row("roc_auc", 0.95, 0.90), degradation_row("f1", None, 0.80)]
    _, axes = plot_generalisation_degradation(rows, metrics=("roc_auc", "f1"))
    # The undefined pair must not be drawn against an implied zero.
    assert len(axes.patches) == 2
    assert any("Undefined on one side" in text.get_text() for text in axes.texts)

    with pytest.raises(ValueError, match="both an in-distribution and an unseen value"):
        plot_generalisation_degradation([degradation_row("f1", None, 0.8)], metrics=("f1",))
    with pytest.raises(ValueError, match="at least one degradation row"):
        plot_generalisation_degradation([])


# ------------------------------------------------------------- parameter efficiency


def efficiency_row(
    mode: str, parameters: int | None, percentage: float, metric: float | None
) -> dict[str, Any]:
    return {
        "fine_tune_mode": mode,
        "trainable_parameters": parameters,
        "adaptation_percentage": percentage,
        "roc_auc": metric,
    }


def test_parameter_efficiency_uses_a_log_axis_and_marks_the_starting_point() -> None:
    rows = [
        efficiency_row("head_only", 769, 0.05, 0.94),
        efficiency_row("last_block", 7_088_641, 0.05, 0.97),
        efficiency_row("full", 87_456_769, 0.05, 0.98),
    ]
    _, axes = plot_parameter_efficiency(rows, metric_name="roc_auc", zero_percent_reference=0.90)
    # Depths differ by orders of magnitude; a linear axis would collapse them together.
    assert axes.get_xscale() == "log"
    assert axes.get_ylim() == (0.0, 1.0)
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert any("0% adaptation reference (0.900)" in label for label in labels)
    assert {"head_only", "last_block", "full"} <= set(labels)


def test_parameter_efficiency_says_when_no_depth_comparison_exists() -> None:
    rows = [
        efficiency_row("head_only", 769, 0.05, 0.94),
        efficiency_row("head_only", 769, 0.50, 0.98),
    ]
    _, axes = plot_parameter_efficiency(rows, metric_name="roc_auc")
    # A single depth is a legitimate state of the study, but the figure must not imply a
    # trade-off was measured.
    assert any("no depth comparison is shown" in text.get_text() for text in axes.texts)
    # Budgets sharing an x position must still be labelled distinctly.
    annotations = [text.get_text() for text in axes.texts]
    assert "5%" in annotations and "50%" in annotations


def test_parameter_efficiency_requires_both_a_metric_and_a_parameter_count() -> None:
    with pytest.raises(ValueError, match="at least one cell row"):
        plot_parameter_efficiency([], metric_name="roc_auc")
    with pytest.raises(ValueError, match="trainable_parameters"):
        plot_parameter_efficiency(
            [efficiency_row("head_only", None, 0.05, 0.9)], metric_name="roc_auc"
        )
    with pytest.raises(ValueError, match="trainable_parameters"):
        plot_parameter_efficiency(
            [efficiency_row("head_only", 769, 0.05, None)], metric_name="roc_auc"
        )
