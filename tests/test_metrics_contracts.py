"""Hand-calculated specifications for binary and generator-wise evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.metrics import (
    compute_binary_metrics,
    per_generator_metrics,
    precision_recall_curve_data,
    roc_curve_data,
    threshold_scores,
)


def test_binary_metrics_match_hand_calculated_confusion_counts() -> None:
    metrics = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8])
    assert (
        metrics.true_negative,
        metrics.false_positive,
        metrics.false_negative,
        metrics.true_positive,
    ) == (1, 1, 1, 1)
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)


def test_threshold_boundary_uses_greater_than_or_equal() -> None:
    result = threshold_scores([0.499, 0.5, 0.501])
    np.testing.assert_array_equal(result, [0, 1, 1])
    assert result.dtype == np.int64
    with pytest.raises(ValueError, match="threshold"):
        threshold_scores([0.5], threshold=1.1)


def test_curves_use_continuous_scores_not_hard_predictions() -> None:
    labels = np.asarray([0, 1, 0, 1])
    scores = np.asarray([0.1, 0.7, 0.4, 0.9])
    metrics = compute_binary_metrics(labels, scores)
    assert metrics.roc_auc == pytest.approx(roc_auc_score(labels, scores))
    assert metrics.average_precision == pytest.approx(average_precision_score(labels, scores))
    assert len(roc_curve_data(labels, scores).thresholds) > 2
    assert len(precision_recall_curve_data(labels, scores).thresholds) > 1
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_binary_metrics(labels, np.column_stack((1 - scores, scores)))


def test_single_class_generator_slice_reports_undefined_auc() -> None:
    metrics = per_generator_metrics([1, 1], [0.2, 0.8], ["g", "g"])["g"]
    assert metrics.roc_auc is None
    assert metrics.average_precision is None
    assert metrics.support == 2


def test_pair_with_real_generator_policy_produces_defined_auc() -> None:
    metrics = per_generator_metrics(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["real", "real", "g", "g"], pair_with_real=True
    )
    assert metrics["g"].roc_auc == pytest.approx(1.0)
    assert metrics["g"].support == 4
