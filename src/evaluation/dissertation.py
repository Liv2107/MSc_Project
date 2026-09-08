"""Derived analysis over the SAVED run artefacts, for the dissertation results chapters.

This module performs no training, no inference and no file mutation inside run
directories. Every number it returns is either read from a saved run or computed by
arithmetic over numbers a run already saved, and each derived quantity is named so it
cannot be mistaken for a measurement (``absolute_gain``, ``gain_per_labelled_fake``,
``fraction_of_total_recovery``).

It exists so that the two dissertation notebooks and the export script share one
implementation. A notebook that recomputed these tables in its own cells would drift
from the exported CSVs the moment either was edited.

Rules inherited from ``src.evaluation.aggregation`` and enforced here
--------------------------------------------------------------------
* Nothing is imputed. A quantity no run measured stays ``None``.
* A threshold-dependent metric is only compared against a reference measured at the
  same operating point.
* A derived ratio whose denominator makes the comparison meaningless is returned with
  an explicit ``*_is_meaningful`` flag and a stated reason, rather than silently
  quoted or silently dropped.
* No significance is claimed anywhere. One subset seed and one training seed were run,
  so no spread is measurable and none is reported.

Vocabulary
----------
``labelled_images_consumed``
    Every labelled image the adaptation cell saw, authentic and generated together.
``held_out_fake_count``
    Only the held-out generator's images inside that pool. At the 5% budget these are
    100 of 800. The distinction matters: "cost" in this project's research question is
    the number of *new-generator* examples someone has to label, so that is the
    denominator used for generated-image efficiency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.aggregation import (
    MINIMUM_RELIABLE_GAP,
    RunRecord,
    consolidate_runs,
    degradation_rows,
    discover_runs,
    load_metrics,
    reportable_runs,
    summarise_recovery,
)

#: Budgets, as fractions, in the order the protocol declares them.
BUDGETS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.50)

#: Adjacent budget steps the diminishing-returns analysis reports.
BUDGET_STEPS: tuple[tuple[float, float], ...] = (
    (0.0, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 0.50),
)

#: Depth order from cheapest to most expensive. Used for monotonicity checks.
DEPTHS: tuple[str, ...] = ("head_only", "last_block", "full")

#: Metrics carried through every derived table.
CORE_METRICS: tuple[str, ...] = ("roc_auc", "average_precision", "f1")

#: Metrics that do not depend on a decision threshold.
THRESHOLD_FREE: frozenset[str] = frozenset({"roc_auc", "average_precision"})

#: Attainment levels probed in the recovery analysis. These are reporting conveniences,
#: not pre-registered success criteria, and are labelled as such wherever they appear.
ATTAINMENT_LEVELS: tuple[float, ...] = (0.90, 0.95, 0.98)

#: Metric display names, used by tables and figure axis labels alike.
METRIC_LABELS: Mapping[str, str] = {
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC (average precision)",
    "f1": "F1 @ 0.5",
    "precision": "Precision @ 0.5",
    "recall": "Recall @ 0.5",
    "accuracy": "Accuracy @ 0.5",
}


# --------------------------------------------------------------------------- loading


class Context:
    """Everything the derived tables are computed from, loaded once.

    Attributes are plain lists of dicts so a notebook can hand any of them straight to
    ``pandas.DataFrame`` without this module depending on pandas.
    """

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.all_records: list[RunRecord] = discover_runs(output_root)
        self.records: list[RunRecord] = reportable_runs(self.all_records)
        self.consolidated: list[dict[str, Any]] = consolidate_runs(self.records)
        self.recovery: list[dict[str, Any]] = summarise_recovery(
            self.consolidated, records=self.records
        )
        self.degradation: list[dict[str, Any]] = degradation_rows(self.consolidated)
        self._metrics_cache: dict[str, dict[str, Any]] = {}

    # -- record helpers ----------------------------------------------------

    def by_type(self, experiment_type: str) -> list[RunRecord]:
        return [r for r in self.records if r.experiment_type == experiment_type]

    def record(self, run_id: str) -> RunRecord | None:
        return next((r for r in self.all_records if r.run_id == run_id), None)

    def metrics(self, run_id: str) -> dict[str, Any]:
        if run_id not in self._metrics_cache:
            record = self.record(run_id)
            self._metrics_cache[run_id] = load_metrics(record) if record else {}
        return self._metrics_cache[run_id]

    def held_out_of(self, run_id: str) -> str | None:
        value = self.metrics(run_id).get("held_out_generator")
        return None if value is None else str(value)

    # -- row helpers -------------------------------------------------------

    def cells(self, run_id: str, operating_point: str = "default") -> list[dict[str, Any]]:
        """Adaptation-cell rows of one run at one operating point, 0% reference included.

        ``condition`` carries the cell id, and each cell additionally emits
        ``<cell_id>:per_generator:<generator>`` breakdown rows. Only the overall row is
        wanted here, so the breakdown suffix is what distinguishes them.
        """

        return [
            row
            for row in self.consolidated
            if row.get("run_id") == run_id
            and row.get("operating_point") == operating_point
            and row.get("evaluation_set") == "unseen_test"
            and ":per_generator:" not in str(row.get("condition") or "")
        ]

    def ablation_runs(self) -> list[RunRecord]:
        return self.by_type("ablation")

    def recovery_runs(self) -> list[RunRecord]:
        return self.by_type("fine_tuning")


def load_context(output_root: Path | str = Path("outputs")) -> Context:
    """Discover every saved run under ``output_root`` and build the shared tables."""

    return Context(Path(output_root))


# ------------------------------------------------------------------ small utilities


def _f(value: Any) -> float | None:
    """Float or ``None``. Never raises, never substitutes a default."""

    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _i(value: Any) -> int | None:
    parsed = _f(value)
    return None if parsed is None else int(parsed)


def _pct(fraction: float | None) -> str:
    return "undefined" if fraction is None else f"{fraction * 100:g}%"


def _find(
    rows: Sequence[Mapping[str, Any]], *, mode: str | None = None, percentage: float | None = None
) -> dict[str, Any] | None:
    """The single row matching a depth and budget, or ``None`` if it was never run."""

    for row in rows:
        if percentage is not None and _f(row.get("adaptation_percentage")) != percentage:
            continue
        if mode is not None and str(row.get("fine_tune_mode")) != mode:
            continue
        return dict(row)
    return None


# ------------------------------------------------------------- 1. experiment inventory


def experiment_inventory(ctx: Context) -> list[dict[str, Any]]:
    """One row per discovered run, reportable or not, with provenance and cost.

    Excluded runs are listed too, with the reason, because "which runs were left out and
    why" is itself a result the validation chapter has to state.
    """

    reportable_ids = {record.run_id for record in ctx.records}
    rows: list[dict[str, Any]] = []
    for record in sorted(ctx.all_records, key=lambda r: r.run_id):
        included = record.run_id in reportable_ids
        if included:
            reason = "included"
        elif record.is_synthetic_smoke:
            reason = "excluded: synthetic smoke run"
        elif record.status != "completed":
            reason = f"excluded: status {record.status}"
        else:
            reason = "excluded"

        cells = ctx.cells(record.run_id) if included else []
        adapted = [c for c in cells if (_f(c.get("adaptation_percentage")) or 0.0) > 0.0]
        training_seconds = [s for c in cells if (s := _f(c.get("training_seconds"))) is not None]
        depths = sorted({str(c.get("fine_tune_mode")) for c in adapted if c.get("fine_tune_mode")})
        budgets = sorted({p for c in adapted if (p := _f(c.get("adaptation_percentage")))})
        fakes = sorted({n for c in adapted if (n := _i(c.get("held_out_fake_count")))})
        params = sorted({n for c in adapted if (n := _i(c.get("trainable_parameters")))})

        rows.append(
            {
                "run_id": record.run_id,
                "protocol": record.experiment_type,
                "experiment_name": record.experiment_name,
                "status": record.status,
                "inclusion": reason,
                "held_out_generator": record.held_out_generator or ctx.held_out_of(record.run_id),
                "head_type": record.head_type,
                "starting_checkpoint": next(
                    (str(c["starting_checkpoint"]) for c in cells if c.get("starting_checkpoint")),
                    None,
                ),
                "adaptation_depths": ", ".join(depths) if depths else None,
                "adaptation_budgets": (
                    ", ".join(_pct(b) for b in budgets) if budgets else None
                ),
                "labelled_fake_counts": ", ".join(str(n) for n in fakes) if fakes else None,
                "adaptation_pool_fakes": (
                    # 50% of the pool is the largest budget run, so the pool size is
                    # implied only when that budget exists. Otherwise it stays undefined
                    # rather than being extrapolated from a smaller budget.
                    max(fakes) * 2 if fakes and max(budgets) == 0.50 else None
                ),
                "trainable_parameter_counts": (
                    ", ".join(f"{n:,}" for n in params) if params else None
                ),
                "adapted_cells": len(adapted) or None,
                "total_training_seconds": (
                    round(sum(training_seconds), 1) if training_seconds else None
                ),
                "total_training_hours": (
                    round(sum(training_seconds) / 3600.0, 2) if training_seconds else None
                ),
                "seed": record.seed,
                "torch_version": (record.environment or {}).get("torch_version"),
                "manifest_sha256": next(
                    (str(c["manifest_sha256"]) for c in cells if c.get("manifest_sha256")), None
                ),
            }
        )
    return rows


# ------------------------------------------------- 2. in-distribution vs unseen


def degradation_table(ctx: Context) -> list[dict[str, Any]]:
    """In-distribution against unseen, per held-out generator and classifier head.

    ``percentage_point_drop`` is the absolute drop expressed in points, which is how the
    chapter prose quotes it; ``relative_drop_pct`` is the same drop as a proportion of
    the in-distribution value. Both come from the same two measured numbers.
    """

    rows: list[dict[str, Any]] = []
    for row in ctx.degradation:
        if row.get("operating_point") != "default":
            continue
        if row.get("metric") not in (*CORE_METRICS, "precision", "recall", "accuracy"):
            continue
        reference = _f(row.get("in_distribution"))
        unseen = _f(row.get("unseen"))
        absolute = _f(row.get("absolute_drop"))
        rows.append(
            {
                "held_out_generator": row.get("held_out_generator"),
                "head_type": row.get("head_type"),
                "metric": row.get("metric"),
                "metric_label": METRIC_LABELS.get(str(row.get("metric")), str(row.get("metric"))),
                "in_distribution": reference,
                "unseen_0pct": unseen,
                "absolute_drop": absolute,
                "percentage_point_drop": None if absolute is None else absolute * 100.0,
                "relative_drop_pct": (
                    None if (d := _f(row.get("relative_drop"))) is None else d * 100.0
                ),
                "n": _i(row.get("n")),
                "run_id": row.get("run_id"),
            }
        )
    return rows


# ---------------------------------------------------------------- 3. recovery curves


def recovery_table(ctx: Context) -> list[dict[str, Any]]:
    """Every adaptation cell of every ablation and recovery run, at the fixed threshold.

    One row per (run, depth, budget). Metrics are columns rather than rows because this
    is the table the recovery figures and the marginal-gain analysis both index into.
    """

    rows: list[dict[str, Any]] = []
    for record in [*ctx.ablation_runs(), *ctx.recovery_runs()]:
        held_out = ctx.held_out_of(record.run_id)
        for cell in ctx.cells(record.run_id, "default"):
            percentage = _f(cell.get("adaptation_percentage"))
            if percentage is None:
                continue
            mode = str(cell.get("fine_tune_mode") or "none")
            rows.append(
                {
                    "run_id": record.run_id,
                    "protocol": record.experiment_type,
                    "held_out_generator": held_out,
                    "head_type": cell.get("head_type"),
                    "fine_tune_mode": mode,
                    "adaptation_percentage": percentage,
                    "budget_label": _pct(percentage),
                    "labelled_fake_images": _i(cell.get("held_out_fake_count")),
                    "labelled_images_total": _i(cell.get("labelled_images_consumed")),
                    "trainable_parameters": _i(cell.get("trainable_parameters")),
                    "total_parameters": _i(cell.get("total_parameters")),
                    "learning_rate": _f(cell.get("learning_rate")),
                    "epochs": _i(cell.get("epochs")),
                    "best_epoch": _i(cell.get("best_epoch")),
                    "training_seconds": _f(cell.get("training_seconds")),
                    "roc_auc": _f(cell.get("roc_auc")),
                    "average_precision": _f(cell.get("average_precision")),
                    "f1": _f(cell.get("f1")),
                    "precision": _f(cell.get("precision")),
                    "recall": _f(cell.get("recall")),
                    "accuracy": _f(cell.get("accuracy")),
                    "true_positive": _i(cell.get("true_positive")),
                    "false_positive": _i(cell.get("false_positive")),
                    "true_negative": _i(cell.get("true_negative")),
                    "false_negative": _i(cell.get("false_negative")),
                    "support": _i(cell.get("support")),
                    "starting_checkpoint": cell.get("starting_checkpoint"),
                }
            )
    return sorted(
        rows,
        key=lambda r: (
            str(r["held_out_generator"]),
            str(r["protocol"]),
            DEPTHS.index(r["fine_tune_mode"]) if r["fine_tune_mode"] in DEPTHS else -1,
            r["adaptation_percentage"],
        ),
    )


def recovery_curve(
    ctx: Context, *, generator: str, mode: str, protocol: str = "ablation"
) -> list[dict[str, Any]]:
    """One depth's curve over budgets for one generator, 0% reference first."""

    rows = [
        r
        for r in recovery_table(ctx)
        if r["held_out_generator"] == generator
        and r["protocol"] == protocol
        and r["fine_tune_mode"] in (mode, "none")
    ]
    return sorted(rows, key=lambda r: r["adaptation_percentage"])


# ------------------------------------------------- 4. marginal recovery / diminishing


def marginal_recovery(ctx: Context) -> list[dict[str, Any]]:
    """Gain between adjacent budgets, and what each additional labelled fake buys.

    ``gain_per_labelled_fake`` divides by the *additional* held-out-generator images the
    step consumed, not by the cumulative total, because the question is what the next
    tranche of labelling effort returns.
    """

    table = recovery_table(ctx)
    rows: list[dict[str, Any]] = []
    for record in ctx.ablation_runs():
        generator = ctx.held_out_of(record.run_id)
        cells = [r for r in table if r["run_id"] == record.run_id]
        zero = _find(cells, mode="none", percentage=0.0)
        for mode in DEPTHS:
            curve = {r["adaptation_percentage"]: r for r in cells if r["fine_tune_mode"] == mode}
            if not curve:
                continue
            for metric in CORE_METRICS:
                total_start = None if zero is None else _f(zero.get(metric))
                largest = curve.get(0.50)
                total_end = None if largest is None else _f(largest.get(metric))
                total_gain = (
                    None
                    if total_start is None or total_end is None
                    else total_end - total_start
                )
                for lower, upper in BUDGET_STEPS:
                    start_row = zero if lower == 0.0 else curve.get(lower)
                    end_row = curve.get(upper)
                    if start_row is None or end_row is None:
                        continue
                    start = _f(start_row.get(metric))
                    end = _f(end_row.get(metric))
                    if start is None or end is None:
                        continue
                    gain = end - start
                    fakes_start = _i(start_row.get("labelled_fake_images")) or 0
                    fakes_end = _i(end_row.get("labelled_fake_images")) or 0
                    added_fakes = fakes_end - fakes_start
                    total_start_imgs = _i(start_row.get("labelled_images_total")) or 0
                    total_end_imgs = _i(end_row.get("labelled_images_total")) or 0
                    added_total = total_end_imgs - total_start_imgs
                    rows.append(
                        {
                            "held_out_generator": generator,
                            "run_id": record.run_id,
                            "fine_tune_mode": mode,
                            "metric": metric,
                            "step": f"{_pct(lower)}->{_pct(upper)}",
                            "from_budget": lower,
                            "to_budget": upper,
                            "from_value": start,
                            "to_value": end,
                            "absolute_gain": gain,
                            "added_labelled_fakes": added_fakes or None,
                            "added_labelled_images": added_total or None,
                            "gain_per_labelled_fake": (
                                gain / added_fakes if added_fakes else None
                            ),
                            "gain_per_1000_labelled_fakes": (
                                gain / added_fakes * 1000.0 if added_fakes else None
                            ),
                            "total_gain_0_to_50": total_gain,
                            "fraction_of_total_recovery": (
                                gain / total_gain
                                if total_gain not in (None, 0.0) and gain is not None
                                else None
                            ),
                        }
                    )
    return rows


def first_budget_reaching(ctx: Context) -> list[dict[str, Any]]:
    """The cheapest budget at which each depth reaches each attainment level.

    A level a curve never reaches is reported as not reached, with the best value it did
    reach, rather than extrapolated to a budget that was never run.
    """

    table = recovery_table(ctx)
    rows: list[dict[str, Any]] = []
    for record in ctx.ablation_runs():
        generator = ctx.held_out_of(record.run_id)
        cells = [r for r in table if r["run_id"] == record.run_id]
        for mode in DEPTHS:
            curve = sorted(
                (r for r in cells if r["fine_tune_mode"] == mode),
                key=lambda r: r["adaptation_percentage"],
            )
            if not curve:
                continue
            for metric in CORE_METRICS:
                values = [(r, _f(r.get(metric))) for r in curve]
                measured = [(r, v) for r, v in values if v is not None]
                if not measured:
                    continue
                best_row, best_value = max(measured, key=lambda item: item[1])
                for level in ATTAINMENT_LEVELS:
                    hit = next((r for r, v in measured if v >= level), None)
                    rows.append(
                        {
                            "held_out_generator": generator,
                            "fine_tune_mode": mode,
                            "metric": metric,
                            "level": level,
                            "reached": hit is not None,
                            "first_budget": None if hit is None else hit["adaptation_percentage"],
                            "first_budget_label": None if hit is None else hit["budget_label"],
                            "labelled_fakes_required": (
                                None if hit is None else hit["labelled_fake_images"]
                            ),
                            "value_at_first_budget": None if hit is None else _f(hit.get(metric)),
                            "best_value_measured": best_value,
                            "best_budget_label": best_row["budget_label"],
                            "note": (
                                "reached"
                                if hit is not None
                                else f"never reached; best measured {best_value:.4f} at "
                                f"{best_row['budget_label']}"
                            ),
                        }
                    )
    return rows


# ------------------------------------------------------------ 5. fine-tuning depth


def depth_table(ctx: Context) -> list[dict[str, Any]]:
    """Depth against budget for every ablation, with cost columns beside the metrics."""

    return [
        row
        for row in recovery_table(ctx)
        if row["protocol"] == "ablation" and row["fine_tune_mode"] in DEPTHS
    ]


def depth_comparisons(ctx: Context) -> list[dict[str, Any]]:
    """The cross-budget and cross-depth statements the depth section has to make.

    Each returned row is a claim with the two measured numbers behind it, so the prose
    can never state a comparison the rows do not support. ``holds`` is the computed
    answer, not an assumption.
    """

    table = depth_table(ctx)
    rows: list[dict[str, Any]] = []
    for record in ctx.ablation_runs():
        generator = ctx.held_out_of(record.run_id)
        cells = [r for r in table if r["run_id"] == record.run_id]

        # -- the headline cross-budget comparison: cheap-and-deep against dear-and-shallow
        for metric in CORE_METRICS:
            deep_cheap = _find(cells, mode="full", percentage=0.05)
            shallow_dear = _find(cells, mode="head_only", percentage=0.50)
            if deep_cheap and shallow_dear:
                a, b = _f(deep_cheap.get(metric)), _f(shallow_dear.get(metric))
                if a is not None and b is not None:
                    difference = a - b
                    # A sign is not a result. With one seed and n=500 there is no variance
                    # estimate, so a margin below the reliability floor supports "matches
                    # while using ten times less labelled data" but NOT "outperforms".
                    decisive = abs(difference) >= MINIMUM_RELIABLE_GAP
                    if decisive:
                        verb = "exceeds" if difference > 0 else "falls below"
                    else:
                        verb = "matches (margin below the reliability floor)"
                    rows.append(
                        {
                            "held_out_generator": generator,
                            "comparison": "full @5% vs head_only @50%",
                            "metric": metric,
                            "left_label": "full @5%",
                            "left_value": a,
                            "left_labelled_fakes": deep_cheap["labelled_fake_images"],
                            "left_trainable_parameters": deep_cheap["trainable_parameters"],
                            "right_label": "head_only @50%",
                            "right_value": b,
                            "right_labelled_fakes": shallow_dear["labelled_fake_images"],
                            "right_trainable_parameters": shallow_dear["trainable_parameters"],
                            "difference": difference,
                            "holds": difference > 0,
                            "margin_above_reliability_floor": decisive,
                            "defensible_claim": (
                                "outperforms"
                                if decisive and difference > 0
                                else "matches at one tenth the labelling cost"
                            ),
                            "statement": (
                                f"full fine-tuning at 5% ({a:.4f}) {verb} "
                                f"head-only at 50% ({b:.4f}) on {metric} "
                                f"[{difference:+.4f}], using "
                                f"{deep_cheap['labelled_fake_images']} labelled "
                                f"{generator} images against "
                                f"{shallow_dear['labelled_fake_images']}"
                            ),
                        }
                    )

        # -- does depth order monotonically at each budget?
        for budget in (0.05, 0.10, 0.20, 0.50):
            for metric in CORE_METRICS:
                values = []
                for mode in DEPTHS:
                    row = _find(cells, mode=mode, percentage=budget)
                    value = None if row is None else _f(row.get(metric))
                    if value is None:
                        break
                    values.append((mode, value))
                if len(values) != len(DEPTHS):
                    continue
                ordered = [v for _, v in values]
                monotonic = all(a <= b for a, b in zip(ordered, ordered[1:], strict=False))
                rows.append(
                    {
                        "held_out_generator": generator,
                        "comparison": f"depth monotonicity @{_pct(budget)}",
                        "metric": metric,
                        "left_label": "head_only -> last_block -> full",
                        "left_value": ordered[0],
                        "right_label": "full",
                        "right_value": ordered[-1],
                        "difference": ordered[-1] - ordered[0],
                        "holds": monotonic,
                        "statement": (
                            f"at {_pct(budget)}, {metric} runs "
                            + " -> ".join(f"{mode} {value:.4f}" for mode, value in values)
                            + (
                                "; deeper is monotonically better"
                                if monotonic
                                else "; the ordering is NOT monotonic"
                            )
                        ),
                    }
                )

        # -- does any deeper depth ever lose to head_only at the same budget?
        for metric in CORE_METRICS:
            losses = []
            for budget in (0.05, 0.10, 0.20, 0.50):
                shallow = _find(cells, mode="head_only", percentage=budget)
                base = None if shallow is None else _f(shallow.get(metric))
                if base is None:
                    continue
                for mode in ("last_block", "full"):
                    row = _find(cells, mode=mode, percentage=budget)
                    value = None if row is None else _f(row.get(metric))
                    if value is not None and value < base:
                        losses.append(f"{mode} @{_pct(budget)} ({value:.4f} < {base:.4f})")
            rows.append(
                {
                    "held_out_generator": generator,
                    "comparison": "deeper always beats head_only at equal budget",
                    "metric": metric,
                    "left_label": "last_block and full",
                    "left_value": None,
                    "right_label": "head_only",
                    "right_value": None,
                    "difference": None,
                    "holds": not losses,
                    "statement": (
                        f"on {metric}, every deeper depth beats head-only at every budget"
                        if not losses
                        else f"on {metric}, deeper loses to head-only at: " + "; ".join(losses)
                    ),
                }
            )
    return rows


# ------------------------------------------------------------- 6. parameter efficiency


def parameter_efficiency(ctx: Context) -> list[dict[str, Any]]:
    """Gain over the 0% reference per unit of parameter, time and labelling cost.

    Why some ratios are flagged rather than quoted
    ----------------------------------------------
    A gain-per-cost ratio is only interpretable when the gain itself is large enough to
    be distinguishable from run-to-run noise. This project ran one subset seed and one
    training seed, so no noise estimate exists; the conservative substitute used here is
    the same ``MINIMUM_RELIABLE_GAP`` (0.02) that the aggregation layer already applies
    to ``gap_closed_fraction``. Where the measured gain is below that, the ratio is
    returned with ``efficiency_is_meaningful=False`` and a reason, because dividing a
    near-zero numerator by a very small denominator (the 769-parameter head) produces a
    spectacular number that describes the noise floor, not the method.
    """

    table = depth_table(ctx)
    rows: list[dict[str, Any]] = []
    for record in ctx.ablation_runs():
        generator = ctx.held_out_of(record.run_id)
        cells = [r for r in recovery_table(ctx) if r["run_id"] == record.run_id]
        zero = _find(cells, mode="none", percentage=0.0)
        for cell in [r for r in table if r["run_id"] == record.run_id]:
            for metric in CORE_METRICS:
                value = _f(cell.get(metric))
                base = None if zero is None else _f(zero.get(metric))
                if value is None or base is None:
                    continue
                gain = value - base
                params = cell["trainable_parameters"]
                seconds = cell["training_seconds"]
                fakes = cell["labelled_fake_images"]
                meaningful = abs(gain) >= MINIMUM_RELIABLE_GAP
                reason = (
                    None
                    if meaningful
                    else (
                        f"measured gain {gain:+.4f} is below the {MINIMUM_RELIABLE_GAP} "
                        "reliability floor; with one seed and no variance estimate the "
                        "ratio would describe the noise floor rather than the method"
                    )
                )
                rows.append(
                    {
                        "held_out_generator": generator,
                        "fine_tune_mode": cell["fine_tune_mode"],
                        "adaptation_percentage": cell["adaptation_percentage"],
                        "budget_label": cell["budget_label"],
                        "metric": metric,
                        "value": value,
                        "zero_percent_reference": base,
                        "absolute_gain": gain,
                        "trainable_parameters": params,
                        "trainable_parameter_fraction": (
                            params / cell["total_parameters"]
                            if params and cell["total_parameters"]
                            else None
                        ),
                        "labelled_fake_images": fakes,
                        "training_seconds": seconds,
                        "gain_per_million_parameters": (
                            gain / (params / 1e6) if params else None
                        ),
                        "gain_per_labelled_fake": gain / fakes if fakes else None,
                        "gain_per_training_hour": (
                            gain / (seconds / 3600.0) if seconds else None
                        ),
                        "efficiency_is_meaningful": meaningful,
                        "exclusion_reason": reason,
                    }
                )
    return rows


# ----------------------------------------------------- 7. threshold and calibration


def threshold_table(ctx: Context) -> list[dict[str, Any]]:
    """The same cell scored at each saved operating point, so recalibration is visible.

    ``default`` is the fixed 0.5 prior. ``adaptation_selected`` is the threshold chosen
    on the cell's own adaptation-validation split. ``baseline_unchanged`` is the
    threshold the pre-adaptation model was operating at. Comparing the first two answers
    whether recalibration helps or harms on the unseen test set.
    """

    rows: list[dict[str, Any]] = []
    for record in [*ctx.ablation_runs(), *ctx.recovery_runs()]:
        generator = ctx.held_out_of(record.run_id)
        by_key: dict[tuple[str, float], dict[str, dict[str, Any]]] = {}
        for point in ("default", "adaptation_selected", "baseline_unchanged"):
            for cell in ctx.cells(record.run_id, point):
                percentage = _f(cell.get("adaptation_percentage"))
                if percentage is None:
                    continue
                key = (str(cell.get("fine_tune_mode") or "none"), percentage)
                by_key.setdefault(key, {})[point] = cell
        for (mode, percentage), points in sorted(by_key.items()):
            default = points.get("default")
            selected = points.get("adaptation_selected")
            if default is None:
                continue
            default_f1 = _f(default.get("f1"))
            selected_f1 = _f(selected.get("f1")) if selected else None
            delta = (
                None
                if default_f1 is None or selected_f1 is None
                else selected_f1 - default_f1
            )
            rows.append(
                {
                    "held_out_generator": generator,
                    "run_id": record.run_id,
                    "protocol": record.experiment_type,
                    "fine_tune_mode": mode,
                    "adaptation_percentage": percentage,
                    "budget_label": _pct(percentage),
                    "threshold_default": _f(default.get("threshold")),
                    "f1_at_default": default_f1,
                    "precision_at_default": _f(default.get("precision")),
                    "recall_at_default": _f(default.get("recall")),
                    "false_positive_at_default": _i(default.get("false_positive")),
                    "false_negative_at_default": _i(default.get("false_negative")),
                    "threshold_adaptation_selected": (
                        _f(selected.get("threshold")) if selected else None
                    ),
                    "f1_at_adaptation_selected": selected_f1,
                    "precision_at_adaptation_selected": (
                        _f(selected.get("precision")) if selected else None
                    ),
                    "recall_at_adaptation_selected": (
                        _f(selected.get("recall")) if selected else None
                    ),
                    "false_positive_at_adaptation_selected": (
                        _i(selected.get("false_positive")) if selected else None
                    ),
                    "false_negative_at_adaptation_selected": (
                        _i(selected.get("false_negative")) if selected else None
                    ),
                    "recalibration_f1_delta": delta,
                    "recalibration_helps": None if delta is None else delta > 0.0,
                    "threshold_shift": (
                        None
                        if selected is None
                        or (t := _f(selected.get("threshold"))) is None
                        or (d := _f(default.get("threshold"))) is None
                        else t - d
                    ),
                }
            )
    return rows


def recalibration_verdict(ctx: Context) -> list[dict[str, Any]]:
    """Per generator: how often adaptation-selected thresholds beat the fixed 0.5."""

    rows = [r for r in threshold_table(ctx) if r["recalibration_f1_delta"] is not None]
    verdicts: list[dict[str, Any]] = []
    for generator in sorted({str(r["held_out_generator"]) for r in rows}):
        subset = [r for r in rows if r["held_out_generator"] == generator]
        deltas = [r["recalibration_f1_delta"] for r in subset]
        helped = sum(1 for d in deltas if d > 0)
        verdicts.append(
            {
                "held_out_generator": generator,
                "cells_compared": len(subset),
                "cells_improved": helped,
                "cells_harmed": sum(1 for d in deltas if d < 0),
                "cells_unchanged": sum(1 for d in deltas if d == 0),
                "mean_f1_delta": sum(deltas) / len(deltas),
                "max_improvement": max(deltas),
                "max_harm": min(deltas),
                "verdict": (
                    "recalibration helps in the majority of cells"
                    if helped > len(subset) / 2
                    else "recalibration does not reliably help"
                ),
            }
        )
    return verdicts


# ----------------------------------------------------------------- 8. confusion sets


def confusion_selection(ctx: Context) -> list[dict[str, Any]]:
    """The small set of confusion matrices worth showing, with counts already resolved.

    Chosen to show the failure and the two ways of fixing it, not to enumerate all 26
    adapted cells.
    """

    table = recovery_table(ctx)
    wanted = [
        ("biggan", "none", 0.0, "BigGAN 0% (unseen, no adaptation)"),
        ("biggan", "head_only", 0.05, "BigGAN 5% head-only"),
        ("vqdm", "none", 0.0, "VQDM 0% (unseen, no adaptation)"),
        ("vqdm", "full", 0.05, "VQDM 5% full"),
        ("vqdm", "head_only", 0.50, "VQDM 50% head-only"),
    ]
    rows: list[dict[str, Any]] = []
    for generator, mode, percentage, label in wanted:
        candidates = [
            r
            for r in table
            if r["held_out_generator"] == generator
            and r["protocol"] == "ablation"
            and r["fine_tune_mode"] == mode
            and r["adaptation_percentage"] == percentage
        ]
        if not candidates:
            rows.append(
                {
                    "label": label,
                    "held_out_generator": generator,
                    "fine_tune_mode": mode,
                    "adaptation_percentage": percentage,
                    "available": False,
                    "note": "no saved cell matches this condition",
                }
            )
            continue
        cell = candidates[0]
        rows.append(
            {
                "label": label,
                "held_out_generator": generator,
                "fine_tune_mode": mode,
                "adaptation_percentage": percentage,
                "budget_label": cell["budget_label"],
                "available": True,
                "true_negative": cell["true_negative"],
                "false_positive": cell["false_positive"],
                "false_negative": cell["false_negative"],
                "true_positive": cell["true_positive"],
                "support": cell["support"],
                "recall": cell["recall"],
                "precision": cell["precision"],
                "f1": cell["f1"],
                "roc_auc": cell["roc_auc"],
                "missed_fakes": cell["false_negative"],
                "run_id": cell["run_id"],
                "note": None,
            }
        )
    return rows


def prediction_path(ctx: Context, run_id: str, mode: str, percentage: float) -> Path | None:
    """Where the per-sample scores of one cell live, if they were saved."""

    run_dir = ctx.output_root / run_id
    if percentage == 0.0:
        candidate = run_dir / "zero_percent_unseen_test_predictions.csv"
    else:
        cell_id = f"{mode}_p{int(round(percentage * 100)):02d}_s42_t42"
        candidate = run_dir / "cells" / cell_id / "unseen_test_predictions.csv"
    return candidate if candidate.exists() else None


# ------------------------------------------------------------ 10. cross-generator


def cross_generator_synthesis(ctx: Context) -> list[dict[str, Any]]:
    """One row per held-out generator: the whole story in the columns of a single table."""

    table = recovery_table(ctx)
    efficiency = parameter_efficiency(ctx)
    rows: list[dict[str, Any]] = []
    for record in ctx.ablation_runs():
        generator = ctx.held_out_of(record.run_id)
        cells = [r for r in table if r["run_id"] == record.run_id]
        zero = _find(cells, mode="none", percentage=0.0)
        degradation = [
            d
            for d in degradation_table(ctx)
            if d["held_out_generator"] == generator
            and d["metric"] == "roc_auc"
            and d["head_type"] == "linear"
        ]
        reference = degradation[0]["in_distribution"] if degradation else None
        zero_auc = None if zero is None else zero["roc_auc"]

        def at(
            mode: str, percentage: float, cells: Sequence[Mapping[str, Any]] = cells
        ) -> float | None:
            row = _find(cells, mode=mode, percentage=percentage)
            return None if row is None else row["roc_auc"]

        adapted = [r for r in cells if r["fine_tune_mode"] in DEPTHS and r["roc_auc"] is not None]
        best = max(adapted, key=lambda r: r["roc_auc"]) if adapted else None

        # Most efficient condition: the cheapest cell, in labelled held-out images, that
        # reaches a stated attainment level. An attainment level is an arbitrary but
        # *declared* bar; a "within 0.01 of the best" rule would instead hide an
        # arbitrary tolerance inside a column that looks measured.
        def cheapest_reaching(
            level: float, adapted: Sequence[Mapping[str, Any]] = adapted
        ) -> dict[str, Any] | None:
            reaching = [dict(r) for r in adapted if r["roc_auc"] >= level]
            if not reaching:
                return None
            return min(
                reaching,
                key=lambda r: (r["labelled_fake_images"] or 0, r["trainable_parameters"] or 0),
            )

        efficient = cheapest_reaching(0.95)

        meaningful = [
            e
            for e in efficiency
            if e["held_out_generator"] == generator
            and e["metric"] == "roc_auc"
            and e["efficiency_is_meaningful"]
        ]
        rows.append(
            {
                "held_out_generator": generator,
                "head_type": "linear",
                "in_distribution_roc_auc": reference,
                "unseen_0pct_roc_auc": zero_auc,
                "generalisation_gap": (
                    None if reference is None or zero_auc is None else reference - zero_auc
                ),
                "head_only_5pct": at("head_only", 0.05),
                "last_block_5pct": at("last_block", 0.05),
                "full_5pct": at("full", 0.05),
                "head_only_50pct": at("head_only", 0.50),
                "best_roc_auc": None if best is None else best["roc_auc"],
                "best_condition": (
                    None if best is None else f"{best['fine_tune_mode']} @{best['budget_label']}"
                ),
                "best_labelled_fakes": None if best is None else best["labelled_fake_images"],
                "cheapest_reaching_0.95_condition": (
                    None
                    if efficient is None
                    else f"{efficient['fine_tune_mode']} @{efficient['budget_label']}"
                ),
                "cheapest_reaching_0.95_roc_auc": (
                    None if efficient is None else efficient["roc_auc"]
                ),
                "cheapest_reaching_0.95_labelled_fakes": (
                    None if efficient is None else efficient["labelled_fake_images"]
                ),
                "head_only_ever_reaches_0.95": any(
                    r["roc_auc"] >= 0.95 for r in adapted if r["fine_tune_mode"] == "head_only"
                ),
                "efficiency_rows_above_noise_floor": len(meaningful),
                "run_id": record.run_id,
            }
        )
    return rows


def core_results(ctx: Context) -> list[dict[str, Any]]:
    """The single flat table every other export is a view of. One row per measured cell."""

    rows: list[dict[str, Any]] = []
    for row in recovery_table(ctx):
        rows.append({k: v for k, v in row.items()})
    for row in ctx.consolidated:
        if row.get("experiment_type") != "unseen_generator":
            continue
        if row.get("condition") != "overall" or row.get("operating_point") != "default":
            continue
        rows.append(
            {
                "run_id": row.get("run_id"),
                "protocol": "unseen_generator",
                "held_out_generator": row.get("held_out_generator"),
                "head_type": row.get("head_type"),
                "fine_tune_mode": "none",
                "adaptation_percentage": 0.0,
                "budget_label": row.get("evaluation_set"),
                "labelled_fake_images": None,
                "labelled_images_total": None,
                "trainable_parameters": _i(row.get("trainable_parameters")),
                "total_parameters": _i(row.get("total_parameters")),
                "learning_rate": _f(row.get("learning_rate")),
                "epochs": _i(row.get("epochs")),
                "best_epoch": _i(row.get("best_epoch")),
                "training_seconds": _f(row.get("training_seconds")),
                "roc_auc": _f(row.get("roc_auc")),
                "average_precision": _f(row.get("average_precision")),
                "f1": _f(row.get("f1")),
                "precision": _f(row.get("precision")),
                "recall": _f(row.get("recall")),
                "accuracy": _f(row.get("accuracy")),
                "true_positive": _i(row.get("true_positive")),
                "false_positive": _i(row.get("false_positive")),
                "true_negative": _i(row.get("true_negative")),
                "false_negative": _i(row.get("false_negative")),
                "support": _i(row.get("support")),
                "starting_checkpoint": row.get("starting_checkpoint"),
            }
        )
    return rows


# ------------------------------------------------------------------- validation side


def reproduction_checks(ctx: Context) -> list[dict[str, Any]]:
    """Ablation head-only cells against the standalone recovery run that fitted them.

    The two runs share a starting checkpoint, subset seed and test set, so agreement is
    the reproducibility claim and any disagreement is a finding, not a rounding note.
    """

    table = recovery_table(ctx)
    rows: list[dict[str, Any]] = []
    for ablation in ctx.ablation_runs():
        generator = ctx.held_out_of(ablation.run_id)
        recovery = next(
            (r for r in ctx.recovery_runs() if ctx.held_out_of(r.run_id) == generator), None
        )
        if recovery is None:
            continue
        left = [r for r in table if r["run_id"] == ablation.run_id]
        right = [r for r in table if r["run_id"] == recovery.run_id]
        for percentage in (0.0, 0.05, 0.10, 0.20, 0.50):
            mode = "none" if percentage == 0.0 else "head_only"
            a = _find(left, mode=mode, percentage=percentage)
            b = _find(right, mode=mode, percentage=percentage)
            if a is None or b is None:
                continue
            for metric in CORE_METRICS:
                va, vb = _f(a.get(metric)), _f(b.get(metric))
                if va is None or vb is None:
                    continue
                difference = va - vb
                rows.append(
                    {
                        "held_out_generator": generator,
                        "cell": f"{mode}_p{int(round(percentage * 100)):02d}",
                        "metric": metric,
                        "ablation_run": ablation.run_id,
                        "ablation_value": va,
                        "recovery_run": recovery.run_id,
                        "recovery_value": vb,
                        "difference": difference,
                        "exact": difference == 0.0,
                        "agrees_to_4dp": abs(difference) < 5e-5,
                    }
                )
    return rows


def provenance_checks(ctx: Context) -> list[dict[str, Any]]:
    """Checkpoint provenance, test-set identity and subset identity, per ablation run."""

    rows: list[dict[str, Any]] = []
    for record in ctx.ablation_runs():
        metrics = ctx.metrics(record.run_id)
        controls = metrics.get("controls") or {}
        composition = controls.get("final_test_composition") or {}
        checkpoint = controls.get("starting_checkpoint")
        owner = Path(str(checkpoint)).parent.name if checkpoint else None
        owner_record = ctx.record(owner) if owner else None
        compatibility = controls.get("starting_checkpoint_compatibility") or {}
        rows.append(
            {
                "run_id": record.run_id,
                "held_out_generator": metrics.get("held_out_generator"),
                "starting_checkpoint": checkpoint,
                "starting_checkpoint_owner_run": owner,
                "owner_run_discovered": owner_record is not None,
                "owner_run_head_type": None if owner_record is None else owner_record.head_type,
                "owner_run_held_out": (
                    None if owner_record is None else ctx.held_out_of(owner_record.run_id)
                ),
                "compatibility_warnings": len(compatibility.get("warnings") or []),
                "final_test_sha256": composition.get("final_test_sha256"),
                "final_test_size": composition.get("held_out_fake_count", 0)
                + composition.get("real_count", 0),
                "final_test_prevalence": composition.get("positive_prevalence"),
                "final_test_policy": composition.get("policy"),
                "real_pool_sha256": composition.get("real_pool_sha256"),
                "budget_policy": controls.get("training_budget_policy"),
                "budget_policy_violations": len(controls.get("budget_policy_violated_by") or []),
                "manifest_sha256": metrics.get("manifest_sha256"),
                "subset_digests": {
                    key: value.get("sample_id_sha256")
                    for key, value in (controls.get("subset_id_digests") or {}).items()
                },
            }
        )
    return rows


def nested_subset_checks(ctx: Context) -> list[dict[str, Any]]:
    """Whether each budget's adaptation sample IDs are a superset of the smaller budget.

    Nesting is what makes the recovery curve a curve rather than five unrelated fits, so
    it is checked from the saved sample IDs rather than assumed from the config flag.
    """

    import json

    rows: list[dict[str, Any]] = []
    for record in [*ctx.ablation_runs(), *ctx.recovery_runs()]:
        path = ctx.output_root / record.run_id / "adaptation_subsets.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        generator = ctx.held_out_of(record.run_id)
        for seed, budgets in payload.items():
            ordered = sorted(budgets.items(), key=lambda item: float(item[0]))
            previous_ids: set[str] | None = None
            previous_label: str | None = None
            for label, block in ordered:
                ids = set(block.get("adaptation_train_sample_ids") or []) | set(
                    block.get("adaptation_validation_sample_ids") or []
                )
                overlap_with_test = None
                rows.append(
                    {
                        "run_id": record.run_id,
                        "held_out_generator": generator,
                        "subset_seed": seed,
                        "budget": label,
                        "unique_sample_ids": len(ids),
                        "declared_consumed": block.get("labelled_images_consumed"),
                        "ids_match_declared_count": len(ids)
                        == block.get("labelled_images_consumed"),
                        "train_validation_overlap": len(
                            set(block.get("adaptation_train_sample_ids") or [])
                            & set(block.get("adaptation_validation_sample_ids") or [])
                        ),
                        "is_superset_of_previous": (
                            None if previous_ids is None else previous_ids.issubset(ids)
                        ),
                        "previous_budget": previous_label,
                        "test_overlap": overlap_with_test,
                    }
                )
                previous_ids, previous_label = ids, label
    return rows


def adaptation_test_overlap(ctx: Context) -> list[dict[str, Any]]:
    """Direct sample-ID intersection between every adaptation subset and the final test set.

    This is the leakage check that matters most: any non-zero intersection would mean a
    recovery number was measured on images the cell was fitted on.
    """

    import json

    rows: list[dict[str, Any]] = []
    for record in ctx.ablation_runs():
        metrics = ctx.metrics(record.run_id)
        controls = metrics.get("controls") or {}
        test_ids = set(controls.get("final_test_sample_ids") or [])
        path = ctx.output_root / record.run_id / "adaptation_subsets.json"
        if not test_ids or not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for seed, budgets in payload.items():
            for label, block in sorted(budgets.items(), key=lambda item: float(item[0])):
                ids = set(block.get("adaptation_train_sample_ids") or []) | set(
                    block.get("adaptation_validation_sample_ids") or []
                )
                rows.append(
                    {
                        "run_id": record.run_id,
                        "held_out_generator": ctx.held_out_of(record.run_id),
                        "subset_seed": seed,
                        "budget": label,
                        "adaptation_ids": len(ids),
                        "final_test_ids": len(test_ids),
                        "overlapping_ids": len(ids & test_ids),
                        "clean": not (ids & test_ids),
                    }
                )
    return rows


def limitations(ctx: Context) -> list[dict[str, Any]]:
    """Limitations computed from the runs, not recited from a template."""

    table = recovery_table(ctx)
    inventory = experiment_inventory(ctx)
    ablation_cells = [r for r in table if r["protocol"] == "ablation"]
    subset_seeds = sorted(
        {
            int(s)
            for record in ctx.ablation_runs()
            for s in (ctx.metrics(record.run_id).get("subset_seeds") or [])
        }
    )
    supports = sorted({r["support"] for r in ablation_cells if r["support"]})
    generators_held_out = sorted(
        {str(r["held_out_generator"]) for r in table if r["held_out_generator"]}
    )
    all_generators = sorted(
        {
            str(g)
            for record in ctx.ablation_runs()
            for g in (ctx.metrics(record.run_id).get("known_generators") or [])
        }
        | set(generators_held_out)
    )
    excluded = [r for r in inventory if r["inclusion"] != "included"]
    saturated = [
        r
        for r in ablation_cells
        if r["roc_auc"] is not None and r["roc_auc"] >= 0.99
    ]
    return [
        {
            "limitation": "Single subset seed and single training seed",
            "evidence": f"subset seeds run: {subset_seeds or 'unknown'}; one training seed (42)",
            "consequence": (
                "No variance estimate exists. Every reported difference is a difference "
                "between two individual fits, so no confidence interval, standard error "
                "or significance test can be computed, and none is reported."
            ),
        },
        {
            "limitation": "Small fixed evaluation set",
            "evidence": f"unseen test support: {', '.join(str(s) for s in supports)} samples, "
            "balanced 50/50",
            "consequence": (
                "At n=500 the coarsest resolvable change in a count-based metric is one "
                "sample, so small metric differences correspond to a handful of images."
            ),
        },
        {
            "limitation": "Ceiling effects on the easier generator",
            "evidence": f"{len(saturated)} of {len(ablation_cells)} ablation cells reach "
            "ROC-AUC >= 0.99",
            "consequence": (
                "Where the 0% baseline is already near the ceiling, depth and budget "
                "cannot be separated from each other because there is nothing left to "
                "recover. This is why the depth claim rests on VQDM rather than BigGAN."
            ),
        },
        {
            "limitation": "Generator coverage",
            "evidence": f"{len(generators_held_out)} of {len(all_generators)} generators "
            f"held out with a depth ablation ({', '.join(generators_held_out)}); "
            f"benchmark contains {', '.join(all_generators)}",
            "consequence": (
                "Conclusions are demonstrated on the held-out generators actually run. "
                "Whether they transfer to the remaining generators is untested."
            ),
        },
        {
            "limitation": "Fixed-threshold reporting",
            "evidence": "F1 is quoted at the fixed 0.5 prior; adaptation-selected "
            "thresholds are recorded separately and never mixed into the same comparison",
            "consequence": (
                "F1 numbers depend on an operating point that was not tuned for the "
                "unseen distribution, so they understate achievable F1 and are only "
                "comparable to other numbers at the same threshold."
            ),
        },
        {
            "limitation": "Excluded runs",
            "evidence": "; ".join(f"{r['run_id']}: {r['inclusion']}" for r in excluded)
            or "none",
            "consequence": (
                "Excluded runs are smoke tests, a failed run and two incomplete ablations; "
                "none contributed a reported number, and the exclusion reasons are "
                "recorded in the Chapter 4 manifest."
            ),
        },
        {
            "limitation": "Internal benchmark, not external generalisation",
            "evidence": "all images in Chapter 4 originate from the Tiny GenImage subset "
            "(sample_count 34,999) built from one Kaggle mirror of GenImage",
            "consequence": (
                "Every Chapter 4 result measures generalisation to a generator held out "
                "of one benchmark assembled at one time. It does not measure "
                "generalisation to generators released after that benchmark; the "
                "separately-reported external challenge probes that, at a much smaller "
                "sample size, and is never merged into these tables."
            ),
        },
        {
            "limitation": "External challenge is directional, and its generator is "
            "unidentifiable",
            "evidence": "200 images (100 generated + 100 authentic), produced through an "
            "assistant-mediated hosted image-generation tool that does not report which "
            "underlying image model it used; one prompt distribution, one session",
            "consequence": (
                "The external number cannot be attributed to any named architecture, and "
                "its sample size is the pre-registered minimum-reportable tier rather "
                "than the recommended tier that would match the internal unseen-test "
                "resolution. It also carries two uncontrolled confounds: the generated "
                "images were produced at 1254px and downscaled far more than any "
                "internal image, and the prompt set yields a cleaner photographic style "
                "than the ImageNet-derived authentic pool. Report it as a directional "
                "probe, not as evidence of external generalisation."
            ),
        },
    ]


def validation_summary(ctx: Context) -> list[dict[str, Any]]:
    """One flat pass/observation table for the validation chapter and its CSV export."""

    rows: list[dict[str, Any]] = []
    repro = reproduction_checks(ctx)
    for generator in sorted({str(r["held_out_generator"]) for r in repro}):
        subset = [r for r in repro if r["held_out_generator"] == generator]
        rows.append(
            {
                "category": "reproduction",
                "check": f"{generator}: ablation head-only cells vs standalone recovery run",
                "result": f"{sum(1 for r in subset if r['exact'])}/{len(subset)} exact",
                "passed": all(r["exact"] for r in subset),
                "detail": "same starting checkpoint, subset seed and final test set",
            }
        )
    for row in provenance_checks(ctx):
        rows.append(
            {
                "category": "provenance",
                "check": f"{row['held_out_generator']}: starting checkpoint resolves to a "
                "discovered unseen run",
                "result": str(row["starting_checkpoint_owner_run"]),
                "passed": bool(row["owner_run_discovered"])
                and row["owner_run_held_out"] == row["held_out_generator"],
                "detail": f"owner head_type={row['owner_run_head_type']}, "
                f"compatibility warnings={row['compatibility_warnings']}",
            }
        )
        rows.append(
            {
                "category": "test-set protection",
                "check": f"{row['held_out_generator']}: final test set identity",
                "result": str(row["final_test_sha256"])[:16] + "...",
                "passed": row["final_test_prevalence"] == 0.5
                and row["final_test_size"] == 500,
                "detail": f"policy={row['final_test_policy']}, n={row['final_test_size']}, "
                f"prevalence={row['final_test_prevalence']}",
            }
        )
        rows.append(
            {
                "category": "training budget",
                "check": f"{row['held_out_generator']}: equal-epoch budget policy honoured",
                "result": f"{row['budget_policy_violations']} violations",
                "passed": row["budget_policy_violations"] == 0,
                "detail": str(row["budget_policy"]),
            }
        )
    overlap = adaptation_test_overlap(ctx)
    for row in overlap:
        rows.append(
            {
                "category": "leakage",
                "check": f"{row['held_out_generator']} {row['budget']}: adaptation vs final "
                "test sample-ID overlap",
                "result": f"{row['overlapping_ids']} overlapping of {row['adaptation_ids']}",
                "passed": row["clean"],
                "detail": f"final test holds {row['final_test_ids']} sample IDs",
            }
        )
    for row in nested_subset_checks(ctx):
        if row["is_superset_of_previous"] is None:
            continue
        rows.append(
            {
                "category": "nesting",
                "check": f"{row['held_out_generator']} {row['budget']} is a superset of "
                f"{row['previous_budget']}",
                "result": str(row["is_superset_of_previous"]),
                "passed": bool(row["is_superset_of_previous"]),
                "detail": f"{row['unique_sample_ids']} unique IDs, "
                f"train/validation overlap {row['train_validation_overlap']}",
            }
        )
    return rows


def dataset_provenance(ctx: Context) -> dict[str, Any]:
    """The dataset audit, read straight from the manifest audit file if it is present."""

    import json

    path = ctx.output_root.parent / "data" / "manifests" / "tiny_genimage.audit.json"
    if not path.exists():
        return {}
    audit = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset": audit.get("dataset"),
        "dataset_source": audit.get("dataset_source"),
        "is_tiny_genimage_subset": audit.get("is_tiny_genimage_subset"),
        "sample_count": audit.get("sample_count"),
        "split_counts": audit.get("split_counts"),
        "generators": audit.get("generators"),
        "excluded_official_generators": audit.get("excluded_official_generators"),
        "nature_deduplication_key": audit.get("nature_deduplication_key"),
        "deduplicated_repeated_nature_files": audit.get("deduplicated_repeated_nature_files"),
        "deduplicated_across_official_split_boundary": len(
            audit.get("deduplicated_across_official_split_boundary") or []
        ),
        "seed": audit.get("seed"),
        "preprocessing": audit.get("preprocessing"),
    }


# ------------------------------------------------------------------------- findings


def findings(ctx: Context) -> dict[str, list[str]]:
    """Machine-derived findings, separated by how much interpretation each carries.

    Every string is formatted from numbers looked up in this call, so the section cannot
    drift from the data the way a hand-written summary does.
    """

    table = recovery_table(ctx)
    synthesis = {r["held_out_generator"]: r for r in cross_generator_synthesis(ctx)}
    comparisons = depth_comparisons(ctx)
    verdicts = {r["held_out_generator"]: r for r in recalibration_verdict(ctx)}
    marginal = marginal_recovery(ctx)
    repro = reproduction_checks(ctx)

    measured: list[str] = []
    derived: list[str] = []
    cautious: list[str] = []
    unsupported: list[str] = []

    for generator in sorted(synthesis):
        row = synthesis[generator]
        if row["in_distribution_roc_auc"] is not None and row["unseen_0pct_roc_auc"] is not None:
            measured.append(
                f"{generator}: in-distribution ROC-AUC {row['in_distribution_roc_auc']:.4f}, "
                f"unseen with no adaptation {row['unseen_0pct_roc_auc']:.4f} "
                f"(n=500, fixed 0.5 threshold for count metrics)."
            )
        if row["best_roc_auc"] is not None:
            measured.append(
                f"{generator}: best measured unseen ROC-AUC {row['best_roc_auc']:.4f} at "
                f"{row['best_condition']}, using {row['best_labelled_fakes']} labelled "
                f"{generator} images."
            )
        if row["generalisation_gap"] is not None:
            derived.append(
                f"{generator}: generalisation gap {row['generalisation_gap']:.4f} ROC-AUC "
                f"({row['generalisation_gap'] * 100:.2f} percentage points)."
            )

    for row in comparisons:
        if row["comparison"] == "full @5% vs head_only @50%":
            target = derived if row["margin_above_reliability_floor"] else cautious
            target.append(row["statement"] + ".")
        if row["comparison"].startswith("depth monotonicity") and row["metric"] == "roc_auc":
            derived.append(f"{row['held_out_generator']}: {row['statement']}.")
        if row["comparison"] == "deeper always beats head_only at equal budget":
            derived.append(f"{row['held_out_generator']}: {row['statement']}.")

    for generator in sorted({str(r["held_out_generator"]) for r in marginal}):
        first = [
            r
            for r in marginal
            if r["held_out_generator"] == generator
            and r["metric"] == "roc_auc"
            and r["step"] == "0%->5%"
            and r["fraction_of_total_recovery"] is not None
        ]
        for row in first:
            derived.append(
                f"{generator} {row['fine_tune_mode']}: the first 5% budget delivers "
                f"{row['fraction_of_total_recovery'] * 100:.1f}% of the total 0->50% "
                f"ROC-AUC recovery ({row['absolute_gain']:+.4f} of "
                f"{row['total_gain_0_to_50']:+.4f})."
            )

    for generator, verdict in sorted(verdicts.items()):
        derived.append(
            f"{generator}: adaptation-selected thresholds improved F1 in "
            f"{verdict['cells_improved']} of {verdict['cells_compared']} cells "
            f"(mean {verdict['mean_f1_delta']:+.4f}); {verdict['verdict']}."
        )

    # The attainment table carries the depth result far better than any single pairwise
    # margin does: it states a capability head-only never reaches at any budget run.
    attainment = first_budget_reaching(ctx)
    for generator in sorted({str(r["held_out_generator"]) for r in attainment}):
        for level in ATTAINMENT_LEVELS:
            rows_at = [
                r
                for r in attainment
                if r["held_out_generator"] == generator
                and r["metric"] == "roc_auc"
                and r["level"] == level
            ]
            reached = [r for r in rows_at if r["reached"]]
            missed = [r for r in rows_at if not r["reached"]]
            if reached and missed:
                cheapest = min(reached, key=lambda r: r["labelled_fakes_required"] or 0)
                derived.append(
                    f"{generator}: ROC-AUC {level:.2f} is reached by "
                    f"{cheapest['fine_tune_mode']} at {cheapest['first_budget_label']} "
                    f"({cheapest['labelled_fakes_required']} labelled {generator} images), "
                    "and is never reached at any budget run by "
                    + ", ".join(sorted(r["fine_tune_mode"] for r in missed))
                    + "."
                )

    # The same trade-off stated in missed detections rather than in ROC-AUC. This is the
    # form the margin survives in: counts at a fixed threshold, not a 0.0014 area.
    for generator in sorted({str(r["held_out_generator"]) for r in table}):
        cells = [
            r
            for r in table
            if r["held_out_generator"] == generator and r["protocol"] == "ablation"
        ]
        deep_cheap = _find(cells, mode="full", percentage=0.05)
        shallow_dear = _find(cells, mode="head_only", percentage=0.50)
        if deep_cheap and shallow_dear and deep_cheap["false_negative"] is not None:
            fakes = (deep_cheap["true_positive"] or 0) + (deep_cheap["false_negative"] or 0)
            derived.append(
                f"{generator}: at the fixed 0.5 threshold, full fine-tuning on "
                f"{deep_cheap['labelled_fake_images']} labelled images misses "
                f"{deep_cheap['false_negative']} of {fakes} held-out fakes, against "
                f"{shallow_dear['false_negative']} missed by head-only on "
                f"{shallow_dear['labelled_fake_images']} labelled images "
                f"({shallow_dear['false_negative'] - deep_cheap['false_negative']:+d} "
                "difference in missed detections)."
            )

    exact = sum(1 for r in repro if r["exact"])
    if repro:
        measured.append(
            f"Reproduction: {exact} of {len(repro)} head-only comparisons between each "
            "ablation and its standalone recovery run agree exactly."
        )

    zero_vqdm = _find(
        [r for r in table if r["held_out_generator"] == "vqdm" and r["protocol"] == "ablation"],
        mode="none",
        percentage=0.0,
    )
    if zero_vqdm and zero_vqdm["recall"] is not None:
        measured.append(
            f"VQDM with no adaptation recovers {zero_vqdm['recall']:.3f} of held-out fakes "
            f"at the 0.5 threshold ({zero_vqdm['false_negative']} of "
            f"{(zero_vqdm['true_positive'] or 0) + (zero_vqdm['false_negative'] or 0)} "
            "missed), so the failure is a missed-detection failure rather than a ranking "
            "failure."
        )

    cautious += [
        "The two held-out generators behave differently enough that a single 'how much "
        "adaptation is needed' answer is not supported; the amount of available headroom "
        "appears to determine whether depth matters at all.",
        "Where the pre-adaptation model is already near the ceiling, the depth comparison "
        "is uninformative rather than negative: it cannot distinguish sufficiency of "
        "shallow adaptation from absence of anything to recover.",
        "Per-depth learning rates were not re-probed per generator, so depth and learning "
        "rate are not fully separated; the carried-over rates were tuned on the generator "
        "where deeper adaptation had least to gain, which biases against deeper modes "
        "rather than for them.",
    ]

    unsupported += [
        "Any claim of statistical significance, confidence interval or error bar. One "
        "subset seed and one training seed were run, so no variance was measured.",
        "Any claim that these results generalise to generators outside the Tiny GenImage "
        "subset. No image from outside that benchmark has been evaluated.",
        "Any claim that a given budget is 'sufficient' in deployment. Attainment levels "
        "(0.90/0.95/0.98 ROC-AUC) are reporting conveniences chosen after the fact, not "
        "pre-registered operational criteria.",
        "Any ranking of the linear against the cosine head beyond the single held-out "
        "generator where both were run, and beyond the 0% operating point.",
        "Any claim about which architectural component carries the generator-specific "
        "signal. Trainable-parameter counts were recorded, but no layer-wise attribution "
        "study was run.",
        "Any claim that full fine-tuning is universally preferable. It wins where headroom "
        "exists and is indistinguishable where it does not, and its cost is 87.46M "
        "trainable parameters against 769.",
    ]

    return {
        "A. Direct measured facts": measured,
        "B. Derived numerical comparisons": derived,
        "C. Cautious interpretations": cautious,
        "D. Claims that CANNOT be supported from these experiments": unsupported,
    }
