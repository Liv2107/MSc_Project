"""Strict binary classification metrics with explicit edge-case semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    average_precision: float | None
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    support: int
    threshold: float


@dataclass(frozen=True, slots=True)
class CurveData:
    x: FloatArray
    y: FloatArray
    thresholds: FloatArray


def _one_dimensional(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return array


def _binary_vectors(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[IntArray, IntArray]:
    true = _one_dimensional(y_true, name="y_true")
    pred = _one_dimensional(y_pred, name="y_pred")
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have equal lengths")
    if not np.isin(true, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("labels and predictions must contain only 0 and 1")
    return true.astype(np.int64), pred.astype(np.int64)


def validate_binary_inputs(y_true: ArrayLike, y_score: ArrayLike) -> tuple[IntArray, FloatArray]:
    true = _one_dimensional(y_true, name="y_true")
    score = _one_dimensional(y_score, name="y_score")
    if len(true) != len(score):
        raise ValueError("y_true and y_score must have equal lengths")
    if not np.isin(true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    try:
        score = score.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_score must be numeric") from exc
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("y_score must contain finite probabilities in [0, 1]")
    return true.astype(np.int64), score


def threshold_scores(y_score: ArrayLike, *, threshold: float = 0.5) -> IntArray:
    score = _one_dimensional(y_score, name="y_score").astype(np.float64)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("y_score must contain finite probabilities in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    return (score >= threshold).astype(np.int64)


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> IntArray:
    true, pred = _binary_vectors(y_true, y_pred)
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    tp = int(np.sum((true == 1) & (pred == 1)))
    return np.asarray([[tn, fp], [fn, tp]], dtype=np.int64)


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    true, pred = _binary_vectors(y_true, y_pred)
    return float(np.mean(true == pred))


def precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    matrix = confusion_matrix(y_true, y_pred)
    fp, tp = int(matrix[0, 1]), int(matrix[1, 1])
    return 0.0 if tp + fp == 0 else tp / (tp + fp)


def recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    matrix = confusion_matrix(y_true, y_pred)
    fn, tp = int(matrix[1, 0]), int(matrix[1, 1])
    return 0.0 if tp + fn == 0 else tp / (tp + fn)


def f1_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def roc_curve_data(y_true: ArrayLike, y_score: ArrayLike) -> CurveData:
    true, score = validate_binary_inputs(y_true, y_score)
    if len(np.unique(true)) != 2:
        raise ValueError("ROC curve requires both classes")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(true, score)
    return CurveData(false_positive_rate, true_positive_rate, thresholds.astype(np.float64))


def precision_recall_curve_data(y_true: ArrayLike, y_score: ArrayLike) -> CurveData:
    true, score = validate_binary_inputs(y_true, y_score)
    if 1 not in true:
        raise ValueError("precision-recall curve requires positive samples")
    p, r, thresholds = precision_recall_curve(true, score)
    return CurveData(r.astype(np.float64), p.astype(np.float64), thresholds.astype(np.float64))


def compute_binary_metrics(
    y_true: ArrayLike, y_score: ArrayLike, *, threshold: float = 0.5
) -> BinaryMetrics:
    true, score = validate_binary_inputs(y_true, y_score)
    pred = threshold_scores(score, threshold=threshold)
    matrix = confusion_matrix(true, pred)
    tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1]))
    has_both = len(np.unique(true)) == 2
    return BinaryMetrics(
        accuracy=accuracy(true, pred),
        precision=precision(true, pred),
        recall=recall(true, pred),
        f1=f1_score(true, pred),
        roc_auc=float(roc_auc_score(true, score)) if has_both else None,
        average_precision=float(average_precision_score(true, score)) if has_both else None,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        true_positive=tp,
        support=len(true),
        threshold=float(threshold),
    )


def per_generator_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    generators: Sequence[str],
    *,
    threshold: float = 0.5,
    pair_with_real: bool = False,
) -> Mapping[str, BinaryMetrics]:
    true, score = validate_binary_inputs(y_true, y_score)
    names = np.asarray(generators, dtype=object)
    if names.ndim != 1 or len(names) != len(true):
        raise ValueError("generators must be a one-dimensional vector aligned with samples")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("generator names must be non-empty strings")
    result: dict[str, BinaryMetrics] = {}
    real_mask = names == "real"
    for name in sorted(set(names.tolist())):
        mask = names == name
        if pair_with_real and name != "real":
            mask = mask | real_mask
        result[name] = compute_binary_metrics(true[mask], score[mask], threshold=threshold)
    return result
