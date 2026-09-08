"""Figures and a table for the external contemporary-generator challenge.

Read-only. Everything is drawn from the saved external run and the saved internal runs it
compares against; no metric is recomputed from a model, and nothing is interpolated or
smoothed. Curves are drawn from the per-sample prediction files, which is the same data
the reported scalar metrics were computed from.

Three figures, each answering something a table cannot:

``fig15_external_vs_internal_<metric>``
    The frozen detector's discrimination on three evaluation sets side by side: the
    in-distribution internal test, the unseen internal generator, and the external
    challenge set. This is the comparison the external study exists to make.

``fig16_external_roc_pr``
    ROC and precision-recall curves for the primary detector on the external set beside
    its internal unseen curve, so the shape of the difference is visible and not only its
    area.

``fig17_external_score_distributions``
    Where the external authentic and generated scores actually fall relative to the fixed
    0.5 operating point, which is what determines the reported F1.

The external set is 100 images per class. That is the pre-registered minimum-reportable
tier, not the recommended 250, and every figure caption says so via the subtitle.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.evaluation.metrics import (  # noqa: E402
    precision_recall_curve_data,
    roc_curve_data,
)

#: One colour per evaluation set, kept stable across every figure here.
SET_STYLE: dict[str, dict[str, Any]] = {
    "in_distribution": {
        "colour": "#4C72B0",
        "label": "internal, in-distribution\n(known generators)",
    },
    "unseen": {"colour": "#DD8452", "label": "internal, unseen generator\n(vqdm / biggan)"},
    "external": {"colour": "#55A868", "label": "external challenge\n(Astra-mediated)"},
}

METRIC_LABELS = {
    "average_precision": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "f1": "F1 at threshold 0.5",
}

#: Compact axis labels: the full role name is carried in the table, not the tick.
ROLE_LABELS = {
    "primary_vqdm_held_out_linear": "vqdm held out\nlinear\n(primary)",
    "all_seven_generators_seen_baseline": "all seven seen\nbaseline",
    "vqdm_held_out_cosine": "vqdm held out\ncosine",
    "biggan_held_out_linear": "biggan held out\nlinear",
}

SUBTITLE = (
    "External set: 100 generated + 100 authentic images (pre-registered "
    "minimum-reportable tier). Generator identity is not reported by the mediation "
    "route, so no architectural claim attaches to these numbers."
)

#: Written alongside the exports, because ``outputs/report/`` is documented as
#: safe to delete and regenerate. Kept as a template file so its markdown tables
#: are not subject to the source line-length rule.
EXTERNAL_README_TEMPLATE = (
    Path(__file__).with_name("report_readmes") / "external_challenge.md"
)


def _save(figure: Figure, stem: Path) -> list[Path]:
    written = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, bbox_inches="tight", dpi=200)
        written.append(path)
    plt.close(figure)
    return written


def _read_predictions(path: Path) -> tuple[list[int], list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [int(row["label"]) for row in rows], [float(row["score"]) for row in rows]


def _detector_series(metrics: Mapping[str, Any], metric: str) -> list[dict[str, Any]]:
    """One entry per detector, carrying its three comparable numbers where they exist."""

    series: list[dict[str, Any]] = []
    for detector in metrics["detectors"]:
        reference = detector["internal_reference"]
        in_distribution = reference.get("in_distribution_test") or {}
        unseen = reference.get("unseen_test") or {}
        external = detector["external_test"]["at_default_threshold"]
        series.append(
            {
                "role": str(detector["role"]),
                "is_primary": bool(detector["is_primary"]),
                "held_out": reference.get("held_out_generator"),
                "in_distribution": in_distribution.get(metric),
                "unseen": unseen.get(metric),
                "external": external.get(metric),
            }
        )
    series.sort(key=lambda item: (not item["is_primary"], item["role"]))
    return series


def external_vs_internal(
    metrics: Mapping[str, Any], destination: Path, *, metric: str
) -> list[Path]:
    """Grouped bars: each frozen detector on each evaluation set it has a number for."""

    series = _detector_series(metrics, metric)
    if not series:
        return []
    keys = ("in_distribution", "unseen", "external")
    figure, axes = plt.subplots(figsize=(9.4, 5.2))
    width = 0.26
    positions = np.arange(len(series), dtype=float)
    for offset, key in enumerate(keys):
        style = SET_STYLE[key]
        values = [item[key] for item in series]
        bar_positions = positions + (offset - 1) * width
        paired = list(zip(bar_positions, values, strict=True))
        drawn = [(position, value) for position, value in paired if value is not None]
        if not drawn:
            continue
        axes.bar(
            [position for position, _ in drawn],
            [value for _, value in drawn],
            width=width,
            color=style["colour"],
            edgecolor="white",
            linewidth=0.6,
            label=style["label"],
            zorder=3,
        )
        for position, value in drawn:
            axes.annotate(
                f"{value:.3f}",
                (position, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7.5,
            )
        # A detector with no number for a set gets an explicit gap marker rather than a
        # zero bar, so an absent measurement cannot read as a poor one.
        for position, value in paired:
            if value is None:
                axes.annotate(
                    "not measured",
                    (position, 0.02),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#888888",
                )
    axes.set_xticks(positions)
    axes.set_xticklabels(
        [
            ROLE_LABELS.get(str(item["role"]), str(item["role"]).replace("_", " "))
            for item in series
        ],
        fontsize=8,
    )
    axes.set_ylabel(METRIC_LABELS.get(metric, metric))
    axes.set_ylim(0.0, 1.10)
    axes.axhline(0.5, color="#999999", linewidth=0.8, linestyle=":", zorder=1)
    axes.grid(axis="y", alpha=0.25, zorder=0)
    axes.set_title(
        f"Frozen detectors: {METRIC_LABELS.get(metric, metric)} on internal and external sets",
        fontsize=11,
    )
    axes.legend(
        fontsize=7.5,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
    )
    axes.text(
        0.0,
        -0.32,
        SUBTITLE,
        transform=axes.transAxes,
        fontsize=7,
        va="top",
        wrap=True,
    )
    return _save(figure, destination / f"fig15_external_vs_internal_{metric}")


def external_roc_pr(
    metrics: Mapping[str, Any], run_dir: Path, output_root: Path, destination: Path
) -> list[Path]:
    """ROC and PR curves for the primary detector, external beside internal unseen."""

    primary = next((item for item in metrics["detectors"] if item["is_primary"]), None)
    if primary is None:
        return []
    curves: list[dict[str, Any]] = []
    external_path = run_dir / str(primary["predictions_file"])
    if external_path.is_file():
        labels, scores = _read_predictions(external_path)
        curves.append(
            {
                "key": "external",
                "labels": labels,
                "scores": scores,
                "roc_auc": primary["external_test"]["at_default_threshold"]["roc_auc"],
                "pr_auc": primary["external_test"]["at_default_threshold"]["average_precision"],
            }
        )
    internal_dir = output_root / str(primary["run_id"])
    for key, filename in (
        ("unseen", "unseen_test_predictions.csv"),
        ("in_distribution", "in_distribution_test_predictions.csv"),
    ):
        path = internal_dir / filename
        if not path.is_file():
            continue
        labels, scores = _read_predictions(path)
        reference = primary["internal_reference"].get(
            "unseen_test" if key == "unseen" else "in_distribution_test"
        ) or {}
        curves.append(
            {
                "key": key,
                "labels": labels,
                "scores": scores,
                "roc_auc": reference.get("roc_auc"),
                "pr_auc": reference.get("average_precision"),
            }
        )
    if not curves:
        return []

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for entry in curves:
        style = SET_STYLE[str(entry["key"])]
        roc = roc_curve_data(entry["labels"], entry["scores"])
        label = style["label"].replace("\n", " ")
        auc = entry["roc_auc"]
        left.plot(
            roc.x,
            roc.y,
            color=style["colour"],
            linewidth=1.8,
            label=f"{label} (ROC-AUC {auc:.3f})" if auc is not None else label,
        )
        pr = precision_recall_curve_data(entry["labels"], entry["scores"])
        pr_auc = entry["pr_auc"]
        right.plot(
            pr.x,
            pr.y,
            color=style["colour"],
            linewidth=1.8,
            label=f"{label} (PR-AUC {pr_auc:.3f})" if pr_auc is not None else label,
        )
    left.plot([0, 1], [0, 1], color="#999999", linewidth=0.8, linestyle=":")
    left.set_xlabel("false positive rate")
    left.set_ylabel("true positive rate")
    left.set_title("ROC", fontsize=10)
    left.grid(alpha=0.25)
    left.legend(fontsize=7, loc="lower right")
    right.axhline(0.5, color="#999999", linewidth=0.8, linestyle=":")
    right.set_xlabel("recall")
    right.set_ylabel("precision")
    right.set_title("Precision-recall", fontsize=10)
    right.set_ylim(0.0, 1.02)
    right.grid(alpha=0.25)
    right.legend(fontsize=7, loc="lower left")
    figure.suptitle(
        f"Primary frozen detector ({primary['role'].replace('_', ' ')}): "
        "external challenge against its own internal partitions",
        fontsize=11,
    )
    figure.text(0.01, -0.04, SUBTITLE, fontsize=7, va="top")
    return _save(figure, destination / "fig16_external_roc_pr")


def external_score_distributions(
    metrics: Mapping[str, Any], run_dir: Path, destination: Path
) -> list[Path]:
    """Where the external scores fall relative to the fixed 0.5 operating point."""

    detectors = [
        item
        for item in metrics["detectors"]
        if (run_dir / str(item["predictions_file"])).is_file()
    ]
    if not detectors:
        return []
    detectors.sort(key=lambda item: (not item["is_primary"], str(item["role"])))
    columns = len(detectors)
    figure, axes_row = plt.subplots(
        1, columns, figsize=(3.3 * columns, 3.6), sharey=True, squeeze=False
    )
    bins = np.linspace(0.0, 1.0, 21)
    for axes, detector in zip(axes_row[0], detectors, strict=True):
        labels, scores = _read_predictions(run_dir / str(detector["predictions_file"]))
        array = np.asarray(scores, dtype=float)
        mask = np.asarray(labels, dtype=int) == 1
        axes.hist(
            array[~mask],
            bins=bins,
            color="#4C72B0",
            alpha=0.75,
            label=f"authentic (n={int((~mask).sum())})",
        )
        axes.hist(
            array[mask],
            bins=bins,
            color="#C44E52",
            alpha=0.75,
            label=f"generated (n={int(mask.sum())})",
        )
        axes.axvline(0.5, color="black", linewidth=1.0, linestyle="--")
        summary = detector["external_test"]["at_default_threshold"]
        axes.set_title(
            ROLE_LABELS.get(
                str(detector["role"]), str(detector["role"]).replace("_", " ")
            ).replace("\n", " ")
            + f"\nROC-AUC {summary['roc_auc']:.3f}  "
            f"PR-AUC {summary['average_precision']:.3f}",
            fontsize=8.5,
        )
        axes.set_xlabel("score (P(generated))")
        axes.grid(alpha=0.2)
        axes.legend(fontsize=7)
    axes_row[0][0].set_ylabel("images")
    figure.suptitle(
        "External challenge score distributions at the fixed 0.5 threshold",
        fontsize=11,
        y=1.10,
    )
    figure.text(0.01, -0.08, SUBTITLE, fontsize=7, va="top")
    return _save(figure, destination / "fig17_external_score_distributions")


def external_table(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The external result as one flat table, internal references alongside."""

    rows: list[dict[str, Any]] = []
    for detector in metrics["detectors"]:
        external = detector["external_test"]["at_default_threshold"]
        reference = detector["internal_reference"]
        in_distribution = reference.get("in_distribution_test") or {}
        unseen = reference.get("unseen_test") or {}
        rows.append(
            {
                "detector_role": detector["role"],
                "is_primary": detector["is_primary"],
                "run_id": detector["run_id"],
                "head_type": detector["head_type"],
                "held_out_generator": reference.get("held_out_generator"),
                "checkpoint_sha256_prefix": str(detector["checkpoint_sha256"])[:16],
                "external_roc_auc": external["roc_auc"],
                "external_pr_auc": external["average_precision"],
                "external_f1": external["f1"],
                "external_precision": external["precision"],
                "external_recall": external["recall"],
                "external_accuracy": external["accuracy"],
                "external_tp": external["true_positive"],
                "external_fn": external["false_negative"],
                "external_fp": external["false_positive"],
                "external_tn": external["true_negative"],
                "external_support": external["support"],
                "internal_in_distribution_roc_auc": in_distribution.get("roc_auc"),
                "internal_in_distribution_pr_auc": in_distribution.get("average_precision"),
                "internal_unseen_roc_auc": unseen.get("roc_auc"),
                "internal_unseen_pr_auc": unseen.get("average_precision"),
                "external_minus_unseen_roc_auc": (
                    None
                    if unseen.get("roc_auc") is None or external["roc_auc"] is None
                    else external["roc_auc"] - unseen["roc_auc"]
                ),
                "external_minus_in_distribution_roc_auc": (
                    None
                    if in_distribution.get("roc_auc") is None or external["roc_auc"] is None
                    else external["roc_auc"] - in_distribution["roc_auc"]
                ),
            }
        )
    rows.sort(key=lambda row: (not row["is_primary"], str(row["detector_role"])))
    return rows


def _write_table(rows: Sequence[Mapping[str, Any]], destination: Path) -> list[Path]:
    csv_path = destination / "tab04_external_challenge.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for key in columns:
            value = row[key]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append("" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    md_path = destination / "tab04_external_challenge.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def build(run_dir: Path, output_root: Path, destination: Path) -> list[Path]:
    metrics = json.loads(
        (run_dir / "external_challenge_metrics.json").read_text(encoding="utf-8")
    )
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in ("average_precision", "roc_auc", "f1"):
        written += external_vs_internal(metrics, destination, metric=metric)
    written += external_roc_pr(metrics, run_dir, output_root, destination)
    written += external_score_distributions(metrics, run_dir, destination)
    written += _write_table(external_table(metrics), destination)
    readme = destination / "README.md"
    readme.write_text(EXTERNAL_README_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    written.append(readme)
    return written


def _latest_external_run(output_root: Path) -> Path:
    candidates = sorted(
        path
        for path in output_root.glob("external_challenge-*")
        if (path / "external_challenge_metrics.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"no completed external challenge run under {output_root}; run "
            "python main.py --config configs/external_challenge_v1.yaml first"
        )
    return candidates[-1]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--destination", type=Path, default=Path("outputs/report/external_challenge")
    )
    args = parser.parse_args(argv)
    run_dir = args.run_dir or _latest_external_run(args.output_root)
    written = build(run_dir, args.output_root, args.destination)
    print(f"External challenge figures from {run_dir.name}:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
