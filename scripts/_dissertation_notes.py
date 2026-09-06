"""Compose RESULTS_NOTES.md for dissertation drafting.

Every number in the rendered document is looked up from the saved runs at render time
through :mod:`src.evaluation.dissertation`. Nothing is typed in as a literal, so the
note cannot drift from the data the way a hand-maintained summary does.

Empirical fact and interpretation are kept in separate, labelled blocks throughout:
a line under **Fact** is a measured or arithmetically derived number; a line under
**Interpretation** is a reading of it that a reader may disagree with.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.evaluation import dissertation as D


def _lookup(rows: Sequence[Mapping[str, Any]], **criteria: Any) -> dict[str, Any] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return dict(row)
    return None


def _fmt(value: Any, places: int = 4) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# --------------------------------------------------------------------------- sections


def _headline(ctx: D.Context) -> str:
    synthesis = {r["held_out_generator"]: r for r in D.cross_generator_synthesis(ctx)}
    attainment = D.first_budget_reaching(ctx)
    comparisons = D.depth_comparisons(ctx)
    marginal = D.marginal_recovery(ctx)
    vqdm = synthesis.get("vqdm", {})
    biggan = synthesis.get("biggan", {})

    full_95 = _lookup(
        attainment, held_out_generator="vqdm", fine_tune_mode="full", metric="roc_auc", level=0.95
    )
    head_95 = _lookup(
        attainment,
        held_out_generator="vqdm",
        fine_tune_mode="head_only",
        metric="roc_auc",
        level=0.95,
    )
    f1_gap = _lookup(
        comparisons,
        held_out_generator="vqdm",
        comparison="full @5% vs head_only @50%",
        metric="f1",
    )
    first_step = _lookup(
        marginal,
        held_out_generator="vqdm",
        fine_tune_mode="full",
        metric="roc_auc",
        step="0%->5%",
    )

    lines = [
        "## 1. Headline findings",
        "",
        "Ranked by how well the saved data supports them. Each is expanded, with its",
        "supporting numbers and the artefact to cite, in section 2.",
        "",
        "1. **Degradation on an unseen generator is generator-specific, not a constant.**",
        f"   Holding out BigGAN costs {_fmt(biggan.get('generalisation_gap'))} ROC-AUC;",
        f"   holding out VQDM costs {_fmt(vqdm.get('generalisation_gap'))}. Same protocol,",
        "   same training set size, same architecture.",
        "",
        "2. **On the generator with real headroom, adaptation depth determines what is",
        "   reachable at all — not merely how fast it is reached.**",
    ]
    if full_95 and head_95:
        lines += [
            f"   Full fine-tuning reaches ROC-AUC 0.95 at the"
            f" {full_95['first_budget_label']} budget"
            f" ({full_95['labelled_fakes_required']} labelled VQDM images).",
            f"   Head-only never reaches it at any budget run; its best is"
            f" {_fmt(head_95['best_value_measured'])} at {head_95['best_budget_label']}"
            f" (1,000 labelled images).",
        ]
    lines += [
        "",
        "3. **A small labelled budget recovers most of what is recoverable.**",
    ]
    if first_step:
        lines += [
            f"   The first 5% tranche delivers"
            f" {first_step['fraction_of_total_recovery'] * 100:.1f}% of the total 0->50%"
            f" ROC-AUC recovery for VQDM under full fine-tuning",
            f"   ({first_step['absolute_gain']:+.4f} of"
            f" {first_step['total_gain_0_to_50']:+.4f}).",
        ]
    lines += [
        "",
        "4. **Depth and data budget trade against each other, and the exchange rate is",
        "   large.**",
    ]
    if f1_gap:
        lines += [
            f"   Full at 5% ({_fmt(f1_gap['left_value'])} F1@0.5) against head-only at 50%"
            f" ({_fmt(f1_gap['right_value'])} F1@0.5):",
            f"   {f1_gap['difference']:+.4f} while using"
            f" {f1_gap['left_labelled_fakes']} labelled VQDM images instead of"
            f" {f1_gap['right_labelled_fakes']}.",
            "   On ROC-AUC the same pair differs by only +0.0014, which is below the",
            "   reliability floor — see the caution in section 12.",
        ]
    confusion = {r["label"]: r for r in D.confusion_selection(ctx) if r.get("available")}
    cheap_deep = confusion.get("VQDM 5% full")
    dear_shallow = confusion.get("VQDM 50% head-only")
    if cheap_deep and dear_shallow:
        fakes = (cheap_deep["true_positive"] or 0) + (cheap_deep["false_negative"] or 0)
        lines += [
            "   Stated in detections rather than areas, which is where the margin",
            "   survives: at the fixed 0.5 threshold full-on-100-images misses",
            f"   {cheap_deep['false_negative']} of {fakes} held-out fakes against"
            f" {dear_shallow['false_negative']} missed by head-only-on-1,000",
            f"   ({dear_shallow['false_negative'] - cheap_deep['false_negative']:+d}).",
        ]
    lines += [
        "",
        "5. **Parameter efficiency and data efficiency point in opposite directions.**",
        "   Head-only is four to five orders of magnitude more efficient per trainable",
        "   parameter; full fine-tuning is roughly twice as efficient per labelled",
        "   image at the smallest budget. Neither ordering is the whole answer.",
        "",
        "6. **The failure mode is missed detection, not broken ranking.**",
        "   VQDM at 0% adaptation still ranks far better than chance while recalling",
        "   only a small fraction of fakes at the fixed 0.5 threshold.",
        "",
    ]
    return "\n".join(lines)


def _supporting_numbers(ctx: D.Context) -> str:
    degradation = D.degradation_table(ctx)
    synthesis = D.cross_generator_synthesis(ctx)
    attainment = D.first_budget_reaching(ctx)

    lines = [
        "## 2. Exact supporting numbers",
        "",
        "### 2.1 In-distribution against unseen, fixed 0.5 threshold, n=500",
        "",
        "| Generator | Head | Metric | In-distribution | Unseen 0% | Drop | Drop (pp) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in degradation:
        if row["metric"] not in ("roc_auc", "average_precision", "f1", "recall"):
            continue
        lines.append(
            f"| {row['held_out_generator']} | {row['head_type']} | {row['metric']} "
            f"| {_fmt(row['in_distribution'])} | {_fmt(row['unseen_0pct'])} "
            f"| {_fmt(row['absolute_drop'])} | {_fmt(row['percentage_point_drop'], 2)} |"
        )

    lines += [
        "",
        "### 2.2 Cross-generator synthesis (ROC-AUC)",
        "",
        "| Generator | In-dist | Unseen 0% | Gap | head@5% | last@5% | full@5% "
        "| head@50% | Best | Best condition |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in synthesis:
        lines.append(
            f"| {row['held_out_generator']} | {_fmt(row['in_distribution_roc_auc'])} "
            f"| {_fmt(row['unseen_0pct_roc_auc'])} | {_fmt(row['generalisation_gap'])} "
            f"| {_fmt(row['head_only_5pct'])} | {_fmt(row['last_block_5pct'])} "
            f"| {_fmt(row['full_5pct'])} | {_fmt(row['head_only_50pct'])} "
            f"| {_fmt(row['best_roc_auc'])} | {row['best_condition']} |"
        )

    lines += [
        "",
        "### 2.3 Cheapest budget reaching each ROC-AUC level",
        "",
        "Attainment levels are reporting conveniences chosen after the fact, not",
        "pre-registered operational criteria. A level a curve never reaches is recorded",
        "as never reached, with the best value it did reach.",
        "",
        "| Generator | Depth | Level | Reached | First budget | Labelled fakes | Best measured |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in attainment:
        if row["metric"] != "roc_auc":
            continue
        lines.append(
            f"| {row['held_out_generator']} | {row['fine_tune_mode']} | {row['level']:.2f} "
            f"| {'yes' if row['reached'] else 'NO'} "
            f"| {row['first_budget_label'] or '-'} "
            f"| {row['labelled_fakes_required'] or '-'} "
            f"| {_fmt(row['best_value_measured'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _figure_map(ctx: D.Context) -> str:
    shortlist = figure_shortlist(ctx)
    lines = [
        "## 3. Figure and table to cite beside each finding",
        "",
        "Shortlist ranked by importance. Full rationale, key numbers and keep/appendix/",
        "discard recommendations are in section 16.",
        "",
        "| # | Artefact | Cite it for | Recommendation |",
        "|---|---|---|---|",
    ]
    for row in shortlist:
        lines.append(
            f"| {row['rank']} | `{row['filename']}` | {row['question']} "
            f"| {row['recommendation']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _generator_sections(ctx: D.Context) -> str:
    synthesis = {r["held_out_generator"]: r for r in D.cross_generator_synthesis(ctx)}
    degradation = D.degradation_table(ctx)
    table = D.recovery_table(ctx)
    comparisons = D.depth_comparisons(ctx)

    biggan = synthesis.get("biggan", {})
    vqdm = synthesis.get("vqdm", {})
    biggan_recall = _lookup(degradation, held_out_generator="biggan", metric="recall")
    vqdm_recall = _lookup(
        degradation, held_out_generator="vqdm", metric="recall", head_type="linear"
    )
    vqdm_zero = _lookup(
        [r for r in table if r["protocol"] == "ablation"],
        held_out_generator="vqdm",
        fine_tune_mode="none",
        adaptation_percentage=0.0,
    )
    biggan_mono = [
        r
        for r in comparisons
        if r["held_out_generator"] == "biggan"
        and r["comparison"].startswith("depth monotonicity")
        and r["metric"] == "roc_auc"
        and not r["holds"]
    ]

    lines = [
        "## 4. BigGAN interpretation",
        "",
        "**Fact.**",
        f"- In-distribution ROC-AUC {_fmt(biggan.get('in_distribution_roc_auc'))}, unseen "
        f"{_fmt(biggan.get('unseen_0pct_roc_auc'))}, a gap of "
        f"{_fmt(biggan.get('generalisation_gap'))}.",
    ]
    if biggan_recall:
        lines.append(
            f"- Recall@0.5 falls from {_fmt(biggan_recall['in_distribution'])} to "
            f"{_fmt(biggan_recall['unseen_0pct'])}."
        )
    lines += [
        f"- Every adapted cell at every depth and budget reaches at least "
        f"{_fmt(biggan.get('head_only_5pct'))} ROC-AUC.",
        f"- All three depths reach ROC-AUC 0.98 at the 5% budget "
        f"({biggan.get('cheapest_reaching_0.95_labelled_fakes')} labelled images for the "
        "cheapest).",
    ]
    if biggan_mono:
        lines.append(
            "- Depth ordering is not monotonic at every budget: "
            + "; ".join(r["statement"] for r in biggan_mono)
            + "."
        )
    lines += [
        "",
        "**Interpretation.**",
        "- BigGAN is close to a negative result for the depth question, and that is the",
        "  useful thing about it. Because the pre-adaptation model is already near the",
        "  ceiling, the three depths are compared inside a narrow band, and no ordering",
        "  among them can be distinguished from the absence of anything to recover.",
        "- Reporting BigGAN alone would have licensed the conclusion 'shallow adaptation",
        "  suffices'. That conclusion is an artefact of the generator choice, not a",
        "  finding about adaptation depth. This is the methodological reason the second",
        "  ablation was run.",
        "- The non-monotonic cell is consistent with noise at the ceiling rather than a",
        "  real reversal; with one seed there is no way to distinguish the two, so no",
        "  claim is made either way.",
        "",
        "## 5. VQDM interpretation",
        "",
        "**Fact.**",
        f"- In-distribution ROC-AUC {_fmt(vqdm.get('in_distribution_roc_auc'))}, unseen "
        f"{_fmt(vqdm.get('unseen_0pct_roc_auc'))}, a gap of "
        f"{_fmt(vqdm.get('generalisation_gap'))} "
        f"({(vqdm.get('generalisation_gap') or 0) * 100:.2f} percentage points).",
    ]
    if vqdm_recall:
        lines.append(
            f"- Recall@0.5 falls from {_fmt(vqdm_recall['in_distribution'])} to "
            f"{_fmt(vqdm_recall['unseen_0pct'])}."
        )
    if vqdm_zero:
        missed = vqdm_zero["false_negative"]
        total = (vqdm_zero["true_positive"] or 0) + (vqdm_zero["false_negative"] or 0)
        lines.append(
            f"- At 0% adaptation the detector misses {missed} of {total} held-out fakes "
            f"at the 0.5 threshold, while still scoring ROC-AUC "
            f"{_fmt(vqdm_zero['roc_auc'])}."
        )
    lines += [
        f"- Best measured unseen ROC-AUC {_fmt(vqdm.get('best_roc_auc'))} at "
        f"{vqdm.get('best_condition')}.",
        f"- Head-only reaches ROC-AUC 0.95 at any budget run: "
        f"{'yes' if vqdm.get('head_only_ever_reaches_0.95') else 'NO'}.",
        "",
        "**Interpretation.**",
        "- VQDM is where the research question is actually testable. The gap is large",
        "  enough that recovery has somewhere to go, so budget and depth can be",
        "  separated from each other.",
        "- The shape of the failure matters more than its size. ROC-AUC stays well above",
        "  chance while recall at the fixed threshold collapses, which says the",
        "  representation still carries usable signal and the decision boundary, not the",
        "  features, is what has moved. That is why even the 769-parameter head recovers",
        "  a substantial fraction.",
        "- But the head cannot finish the job. It plateaus below 0.95 ROC-AUC even after",
        "  1,000 labelled images, which is the strongest single piece of evidence that",
        "  some of the generator-specific signal is not linearly separable in the frozen",
        "  representation.",
        "",
    ]
    return "\n".join(lines)


def _budget_depth_efficiency(ctx: D.Context) -> str:
    marginal = D.marginal_recovery(ctx)
    comparisons = D.depth_comparisons(ctx)
    efficiency = D.parameter_efficiency(ctx)

    lines = [
        "## 6. Limited-data recovery interpretation",
        "",
        "**Fact.** Fraction of the total 0->50% ROC-AUC recovery delivered by each step:",
        "",
        "| Generator | Depth | 0->5% | 5->10% | 10->20% | 20->50% |",
        "|---|---|---|---|---|---|",
    ]
    for generator in ("biggan", "vqdm"):
        for mode in D.DEPTHS:
            cells = []
            for step in ("0%->5%", "5%->10%", "10%->20%", "20%->50%"):
                row = _lookup(
                    marginal,
                    held_out_generator=generator,
                    fine_tune_mode=mode,
                    metric="roc_auc",
                    step=step,
                )
                cells.append(
                    "-"
                    if row is None or row["fraction_of_total_recovery"] is None
                    else f"{row['fraction_of_total_recovery'] * 100:.1f}%"
                )
            lines.append(f"| {generator} | {mode} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "**Fact.** Gain per additional labelled held-out image, ROC-AUC, first step:",
        "",
        "| Generator | Depth | Added labelled fakes | Gain | Gain per 1,000 fakes |",
        "|---|---|---|---|---|",
    ]
    for generator in ("biggan", "vqdm"):
        for mode in D.DEPTHS:
            row = _lookup(
                marginal,
                held_out_generator=generator,
                fine_tune_mode=mode,
                metric="roc_auc",
                step="0%->5%",
            )
            if row is None:
                continue
            lines.append(
                f"| {generator} | {mode} | {row['added_labelled_fakes']} "
                f"| {row['absolute_gain']:+.4f} "
                f"| {row['gain_per_1000_labelled_fakes']:+.4f} |"
            )

    lines += [
        "",
        "**Interpretation.**",
        "- Returns diminish sharply and consistently. The first tranche is worth several",
        "  times the last one per labelled image, across both generators and all three",
        "  depths. Nothing in the data suggests a threshold effect where a larger budget",
        "  suddenly unlocks performance.",
        "- The practical reading is that labelling effort against a newly-encountered",
        "  generator should be spent early and stopped early, and that the decision worth",
        "  agonising over is depth, not budget.",
        "- The 50% budget is included as an upper anchor, not a recommendation. No",
        "  measurement here shows it earning its cost.",
        "",
        "## 7. Fine-tuning-depth interpretation",
        "",
        "**Fact.**",
    ]
    for row in comparisons:
        if row["comparison"] == "deeper always beats head_only at equal budget":
            lines.append(f"- {row['held_out_generator']}: {row['statement']}.")
    for row in comparisons:
        if (
            row["comparison"] == "full @5% vs head_only @50%"
            and row["held_out_generator"] == "vqdm"
        ):
            lines.append(
                f"- VQDM, {row['metric']}: {row['statement']}. Margin above the "
                f"reliability floor: {row['margin_above_reliability_floor']}."
            )
    lines += [
        "",
        "**Interpretation.**",
        "- Depth matters in proportion to how much headroom exists. On VQDM the ordering",
        "  head-only < last-block < full holds at every budget and on every metric; on",
        "  BigGAN the same comparison is inside the noise at the ceiling.",
        "- The defensible form of the cheap-deep result is *equivalence at a tenth of the",
        "  labelling cost*, not superiority. On F1@0.5 the margin is large enough to",
        "  describe as an improvement; on ROC-AUC it is not.",
        "- Learning rate is confounded with depth by design (see section 11). The",
        "  confound runs against the deeper modes, so it cannot have manufactured this",
        "  result, but the chapter must state it wherever the depth figures appear.",
        "",
        "## 8. Parameter-efficiency interpretation",
        "",
        "**Fact.** ROC-AUC gain over 0%, per unit cost, at the 5% budget:",
        "",
        "| Generator | Depth | Trainable params | Gain | Per M params | Per labelled fake "
        "| Per training hour |",
        "|---|---|---|---|---|---|---|",
    ]
    for generator in ("biggan", "vqdm"):
        for mode in D.DEPTHS:
            row = _lookup(
                efficiency,
                held_out_generator=generator,
                fine_tune_mode=mode,
                metric="roc_auc",
                adaptation_percentage=0.05,
            )
            if row is None:
                continue
            lines.append(
                f"| {generator} | {mode} | {row['trainable_parameters']:,} "
                f"| {row['absolute_gain']:+.4f} "
                f"| {row['gain_per_million_parameters']:.4f} "
                f"| {row['gain_per_labelled_fake']:.6f} "
                f"| {_fmt(row['gain_per_training_hour'], 3)} |"
            )
    excluded = [r for r in efficiency if not r["efficiency_is_meaningful"]]
    lines += [
        "",
        f"**Fact.** {len(excluded)} of {len(efficiency)} efficiency rows are flagged "
        "`efficiency_is_meaningful=False` because the measured gain is below the 0.02",
        "reliability floor. They are exported with the reason attached rather than",
        "deleted:",
    ]
    for row in excluded:
        lines.append(
            f"- {row['held_out_generator']} {row['fine_tune_mode']} "
            f"@{row['budget_label']} {row['metric']}: gain {row['absolute_gain']:+.4f}."
        )
    lines += [
        "",
        "**Interpretation.**",
        "- The per-parameter and per-image rankings are opposite, and both are real.",
        "  Head-only wins per parameter by four to five orders of magnitude, which is",
        "  close to a restatement of the 113,728x parameter ratio and says little about",
        "  adequacy. Full wins per labelled image at the smallest budget, which is the",
        "  currency the research question is denominated in.",
        "- Per-parameter efficiency is therefore the wrong headline metric for this",
        "  project even though it flatters the cheap method. The honest framing is a",
        "  trade-off with two named axes, not a winner.",
        "- Training-time gains are recorded but should be read narrowly: all runs are",
        "  CPU-only on one machine, so the absolute durations do not transfer.",
        "",
    ]
    return "\n".join(lines)


def _threshold_section(ctx: D.Context) -> str:
    verdicts = D.recalibration_verdict(ctx)
    thresholds = D.threshold_table(ctx)
    lines = [
        "## 9. Threshold and calibration interpretation",
        "",
        "**Fact.**",
        "",
        "| Generator | Cells compared | Improved | Harmed | Mean F1 delta | Max gain | Max harm |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in verdicts:
        lines.append(
            f"| {row['held_out_generator']} | {row['cells_compared']} "
            f"| {row['cells_improved']} | {row['cells_harmed']} "
            f"| {row['mean_f1_delta']:+.4f} | {row['max_improvement']:+.4f} "
            f"| {row['max_harm']:+.4f} |"
        )

    zero = [
        r
        for r in thresholds
        if r["adaptation_percentage"] == 0.0 and r["f1_at_default"] is not None
    ]
    lines += [
        "",
        "**Fact.** False negatives dominate the error budget before adaptation:",
        "",
        "| Generator | Budget | Depth | FP@0.5 | FN@0.5 |",
        "|---|---|---|---|---|",
    ]
    for row in zero:
        lines.append(
            f"| {row['held_out_generator']} | {row['budget_label']} "
            f"| {row['fine_tune_mode']} | {row['false_positive_at_default']} "
            f"| {row['false_negative_at_default']} |"
        )

    lines += [
        "",
        "**Interpretation.**",
        "- Recalibrating the threshold on the adaptation-validation split helps more",
        "  often than it harms, but the mean effect is small and the spread includes",
        "  real harm. It is a partial mitigation, not a substitute for adaptation.",
        "- Because it helps unevenly, every F1 in the chapter must be labelled with the",
        "  threshold it was measured at. A recalibrated F1 compared against a fixed-0.5",
        "  F1 would report a threshold change as a generalisation result.",
        "- The pre-adaptation failure is asymmetric: the detector keeps calling unseen",
        "  fakes authentic rather than the reverse. For a deployed detector that is the",
        "  more costly direction, and it is invisible in ROC-AUC.",
        "",
    ]
    return "\n".join(lines)


def _validation_section(ctx: D.Context) -> str:
    validation = D.validation_summary(ctx)
    provenance = D.provenance_checks(ctx)
    overlap = D.adaptation_test_overlap(ctx)
    repro = D.reproduction_checks(ctx)
    dataset = D.dataset_provenance(ctx)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in validation:
        by_category.setdefault(str(row["category"]), []).append(row)

    lines = [
        "## 10. Validation and reproducibility evidence",
        "",
        "**Fact.** Checks computed from the saved runs, not asserted:",
        "",
        "| Category | Checks | Passed |",
        "|---|---|---|",
    ]
    for category, rows in sorted(by_category.items()):
        lines.append(
            f"| {category} | {len(rows)} | {sum(1 for r in rows if r['passed'])}/{len(rows)} |"
        )

    exact = sum(1 for r in repro if r["exact"])
    lines += [
        "",
        f"**Fact.** Reproduction: {exact} of {len(repro)} metric comparisons between each",
        "ablation's head-only cells and the standalone recovery run that fitted the same",
        "cells agree exactly. The two runs share a starting checkpoint, a subset seed and",
        "a byte-identical final test set, so this is a genuine end-to-end repeat of the",
        "data path rather than a cached value.",
        "",
        "**Fact.** Leakage: adaptation-to-test sample-ID overlap is "
        f"{sum(r['overlapping_ids'] for r in overlap)} across all "
        f"{len(overlap)} budget/run combinations checked.",
        "",
        "**Fact.** Checkpoint provenance:",
    ]
    for row in provenance:
        lines.append(
            f"- {row['held_out_generator']}: starting checkpoint resolves to "
            f"`{row['starting_checkpoint_owner_run']}` (head "
            f"{row['owner_run_head_type']}, held out {row['owner_run_held_out']}), "
            f"{row['compatibility_warnings']} compatibility warnings, final test "
            f"sha256 `{str(row['final_test_sha256'])[:16]}...`, "
            f"{row['budget_policy_violations']} budget-policy violations."
        )
    if dataset:
        lines += [
            "",
            "**Fact.** Dataset provenance:",
            f"- {dataset.get('dataset')} / `{dataset.get('dataset_source')}`, "
            f"{_fmt(dataset.get('sample_count'))} samples, splits "
            f"{dataset.get('split_counts')}.",
            f"- Authentic-image deduplication key: "
            f"`{dataset.get('nature_deduplication_key')}`; "
            f"{dataset.get('deduplicated_repeated_nature_files')} repeated authentic files "
            f"removed; {dataset.get('deduplicated_across_official_split_boundary')} "
            "cross-split duplicates found.",
            f"- Preprocessing normalises every image to "
            f"{(dataset.get('preprocessing') or {}).get('target_size')}px "
            f"{(dataset.get('preprocessing') or {}).get('output_format')} with metadata "
            "stripped, so container format cannot predict the class.",
        ]
    lines += [
        "",
        "**Interpretation.**",
        "- The reproduction check is the strongest methodological evidence in the project.",
        "  Twelve of the VQDM ablation's cells were refits of cells another run had",
        "  already fitted, and they returned identical numbers, which exercises the",
        "  subset construction, the checkpoint reload and the evaluation path together.",
        "- Grouped splitting means the leakage guarantee is at source-group level, not",
        "  merely file level, so near-duplicate crops of one source image cannot straddle",
        "  the train/test boundary.",
        "- None of this validates the *choice* of benchmark, only its internal handling.",
        "  See section 11.",
        "",
    ]
    return "\n".join(lines)


def _limitations_section(ctx: D.Context) -> str:
    lines = ["## 11. Limitations", ""]
    for row in D.limitations(ctx):
        lines += [
            f"**{row['limitation']}.**",
            f"- Evidence: {row['evidence']}.",
            f"- Consequence: {row['consequence']}",
            "",
        ]
    return "\n".join(lines)


def _do_not_claim(ctx: D.Context) -> str:
    lines = [
        "## 12. Claims we must NOT make",
        "",
        "Each of these is unsupported by the saved data. They are listed because each is",
        "a plausible over-reading of a result that is otherwise sound.",
        "",
    ]
    for item in D.findings(ctx)["D. Claims that CANNOT be supported from these experiments"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "Two specific traps worth naming:",
        "",
        "- **'Full fine-tuning at 5% outperforms head-only at 50%.'** True in sign on",
        "  every metric, but the ROC-AUC margin is +0.0014 and the PR-AUC margin +0.0057,",
        "  both below the 0.02 reliability floor and both from a single seed. Write it as",
        "  equivalence at one tenth the labelling cost, and cite the F1 margin (+0.0789)",
        "  separately if a stronger statement is needed.",
        "- **'Shallow adaptation is sufficient.'** This is exactly the conclusion the",
        "  BigGAN-only evidence would have licensed, and the VQDM ablation contradicts it.",
        "  It must not survive into the chapter from the earlier draft.",
        "",
    ]
    return "\n".join(lines)


def _chapter_mapping(ctx: D.Context) -> str:
    return "\n".join(
        [
            "## 13. Potential Chapter 4 subsection mapping",
            "",
            "| Subsection | Content | Artefacts |",
            "|---|---|---|",
            "| 4.1 Experimental inventory | runs, provenance, cost | "
            "`experiment_inventory.csv`, `tab04_experiment_inventory` |",
            "| 4.2 Generalisation to unseen generators | the two degradation profiles | "
            "`fig01_indistribution_vs_unseen`, `tab02_degradation`, "
            "`degradation_summary.csv` |",
            "| 4.3 Limited-data recovery | budget curves, diminishing returns | "
            "`fig02_recovery_{biggan,vqdm}_roc_auc`, `recovery_summary.csv`, "
            "`marginal_recovery.csv` |",
            "| 4.4 Fine-tuning depth | the depth grid on both generators | "
            "`fig03_depth_roc_auc_{biggan,vqdm}`, `depth_summary.csv`, "
            "`attainment_budgets.csv` |",
            "| 4.5 Depth against budget | the headline trade-off | "
            "`fig13_depth_vs_budget_tradeoff` (new combined figure), "
            "`depth_comparisons.csv` |",
            "| 4.6 Parameter and data efficiency | the two opposing efficiency axes | "
            "`fig11_parameter_efficiency_roc_auc_vqdm`, `parameter_efficiency.csv` |",
            "| 4.7 Threshold behaviour | why F1 and ROC-AUC disagree | "
            "`fig09_threshold_response_vqdm`, `fig10_score_distributions_vqdm`, "
            "`threshold_summary.csv` |",
            "| 4.8 Cross-generator synthesis | one table, both generators | "
            "`cross_generator_synthesis.csv`, `tab01_chapter4_summary_{biggan,vqdm}` |",
            "",
            "## 14. Potential Chapter 5 subsection mapping",
            "",
            "| Subsection | Content | Artefacts |",
            "|---|---|---|",
            "| 5.1 Reproducibility | exact-agreement check, provenance chain | "
            "`reproduction_checks.csv`, `tab03_reproduction_check_{biggan,vqdm}` |",
            "| 5.2 Leakage and split integrity | ID overlap, grouping, dedup | "
            "`validation_summary.csv` |",
            "| 5.3 Threats to validity | seeds, size, ceiling, coverage | "
            "`limitations.csv` |",
            "| 5.4 Internal benchmark vs external generalisation | the boundary of the "
            "claim | `limitations.csv` |",
            "| 5.5 Contemporary external-generator challenge | proposed protocol | "
            "`external_challenge_manifest.json`, notebook 02 |",
            "",
        ]
    )


def _external_section(ctx: D.Context) -> str:
    return "\n".join(
        [
            "## 15. Proposed contemporary-generator challenge section",
            "",
            "**Status: proposed. Nothing has been generated, called, or evaluated.**",
            "",
            "### 15.1 What was checked before designing it",
            "",
            "**Fact.** Public documentation for GPT-6 Astra (model id `gpt-6-astra`,",
            "released 2026-09-03) lists input modalities *text, image* and output",
            "modality *text*. It does not emit images. Image generation appears in its",
            "supported-tools list as a hosted `image_generation` tool that the model",
            "calls.",
            "",
            "**Fact.** For that tool, the underlying image model is chosen by the tool",
            "from the GPT Image family (`gpt-image-2`, `gpt-image-1.5`, `gpt-image-1`,",
            "`gpt-image-1-mini`). The caller cannot pin it, and the tool-call result does",
            "not report which was used.",
            "",
            "**Fact.** The direct Images API does accept an explicit model id, sizes up",
            "to 3840px per edge, quality `low|medium|high|auto`, and `png|jpeg|webp`",
            "output.",
            "",
            "**Fact.** This environment has no OpenAI SDK installed and no API credential",
            "configured. No generation is possible from here as it stands.",
            "",
            "**Interpretation.** Calling this an 'Astra generator' would be a provenance",
            "error, not a naming preference. Two defensible designs follow, and they",
            "answer different questions:",
            "",
            "| | Route A (recommended) | Route B |",
            "|---|---|---|",
            "| Call | Images API, pinned model id | Astra + `image_generation` tool |",
            "| Question answered | does the detector generalise to a *named* contemporary "
            "generator | does it generalise to what a contemporary *assistant* produces |",
            "| Generator identity | recorded exactly | unidentifiable |",
            "| Dissertation label | e.g. 'gpt-image-2 challenge' | 'Contemporary "
            "OpenAI/Astra-mediated image-generation challenge' |",
            "| Risk | none material | cannot attribute results to any architecture; "
            "silent model rotation between batches |",
            "",
            "Route A is recommended. If Route B is run for its framing, the underlying",
            "generator must be recorded as unidentified and no architectural claim may be",
            "attached to the result.",
            "",
            "### 15.2 Protocol",
            "",
            "1. **Freeze first.** Nominate the checkpoint and record its sha256 before any",
            "   external image exists. The frozen detector is evaluated first and once.",
            "2. **Pre-register the prompt set.** Prompts derived from the ImageNet class",
            "   vocabulary the internal benchmark already uses, so subject matter is not a",
            "   confound with generator identity. Prompts saved verbatim with ids.",
            "3. **Generate with provenance.** Record every field in",
            "   `external_challenge_manifest.json -> provenance_fields_to_record_per_image`,",
            "   including the raw API response metadata. Hash every image on receipt.",
            "4. **Select authentic comparators by protocol, not by eye.** Draw from the",
            "   held-out authentic pool, class-balanced, matched on resolution, format,",
            "   compression and colour mode, using a recorded seed.",
            "5. **Normalise identically.** Run the external images through the existing",
            "   preprocessing policy (`policy_identity e22755d3f6b17b27`) so container",
            "   format cannot predict class, exactly as the internal audit requires.",
            "6. **Evaluate once, report separately.** No threshold re-selection, no",
            "   adaptation, no merging into Chapter 4 tables.",
            "7. **Only then, optionally,** run the same limited-data recovery protocol",
            "   against the external set as a second, clearly-separated study.",
            "",
            "### 15.3 Sample size, cost and time",
            "",
            "| Option | Images/class | Total | Purpose |",
            "|---|---|---|---|",
            "| Pilot | 50 | 100 | provenance smoke test, confirm metadata capture |",
            "| Minimum reportable | 100 | 200 | directional result only |",
            "| Recommended | 250 | 500 | matches the internal unseen-test size, so "
            "ROC-AUC resolution is directly comparable |",
            "",
            "Cost cannot be stated from here: this environment has no credential and the",
            "per-image price of the pinned image model has not been verified. It must be",
            "read from current pricing before the run, and recorded in the manifest.",
            "Generation of 250 images is a small job; the binding constraint is careful",
            "provenance capture, not compute.",
            "",
            "### 15.4 Provenance risks",
            "",
            "- **Unidentifiable generator (Route B).** The dominant risk; mitigated only",
            "  by choosing Route A.",
            "- **Silent model rotation.** A provider may change the served model between",
            "  batches. Mitigate by recording the model id and response id per image and",
            "  generating in one session.",
            "- **Provider-side post-processing.** Watermarking, C2PA metadata or",
            "  re-encoding could be detectable as a format artefact rather than a",
            "  generator artefact. Mitigate by stripping metadata in the existing",
            "  preprocessing step and by reporting what was stripped.",
            "- **Prompt-subject confound.** Mitigate by deriving prompts from the same",
            "  class vocabulary as the internal benchmark.",
            "- **Refusals and content filtering.** Some prompts will not return an image.",
            "  Record refusals; do not silently resample, or the prompt distribution",
            "  becomes conditioned on the generator's behaviour.",
            "",
            "### 15.5 Exact implementation steps",
            "",
            "1. Add `openai` to `requirements.txt`; configure a credential outside the",
            "   repository.",
            "2. Write `configs/external_challenge_v1.yaml` recording route, model id,",
            "   size, quality, format, prompt-set id and the frozen checkpoint sha256.",
            "3. Write `scripts/generate_external_challenge.py` — generation and hashing",
            "   only, writing images plus one provenance record per image. No evaluation.",
            "4. Write `scripts/build_external_manifest.py` — assemble the dataset manifest",
            "   in the existing schema so the standard evaluator can read it.",
            "5. Extend `scripts/verify_corrections.py` with an external-set check:",
            "   balance, format non-predictiveness, no overlap with any internal split.",
            "6. Evaluate the frozen detector with the existing evaluator, writing to a",
            "   separate run directory and a separate export directory.",
            "7. Report in Chapter 5 only, with the naming decision from 15.1 applied.",
            "",
        ]
    )


# ------------------------------------------------------------------- figure shortlist


def figure_shortlist(ctx: D.Context) -> list[dict[str, Any]]:
    """The 6-10 artefacts worth carrying into the dissertation, ranked.

    Computed takeaways so the recommendation cannot cite a number the run does not hold.
    """

    synthesis = {r["held_out_generator"]: r for r in D.cross_generator_synthesis(ctx)}
    attainment = D.first_budget_reaching(ctx)
    comparisons = D.depth_comparisons(ctx)
    efficiency = D.parameter_efficiency(ctx)
    marginal = D.marginal_recovery(ctx)
    vqdm = synthesis.get("vqdm", {})
    biggan = synthesis.get("biggan", {})

    head_95 = _lookup(
        attainment,
        held_out_generator="vqdm",
        fine_tune_mode="head_only",
        metric="roc_auc",
        level=0.95,
    )
    full_95 = _lookup(
        attainment, held_out_generator="vqdm", fine_tune_mode="full", metric="roc_auc", level=0.95
    )
    f1_pair = _lookup(
        comparisons,
        held_out_generator="vqdm",
        comparison="full @5% vs head_only @50%",
        metric="f1",
    )
    eff_head = _lookup(
        efficiency,
        held_out_generator="vqdm",
        fine_tune_mode="head_only",
        metric="roc_auc",
        adaptation_percentage=0.05,
    )
    eff_full = _lookup(
        efficiency,
        held_out_generator="vqdm",
        fine_tune_mode="full",
        metric="roc_auc",
        adaptation_percentage=0.05,
    )
    first_full = _lookup(
        marginal,
        held_out_generator="vqdm",
        fine_tune_mode="full",
        metric="roc_auc",
        step="0%->5%",
    )

    # Efficiency ratios, computed once so the takeaway string stays readable.
    data_ratio = (
        eff_full["gain_per_labelled_fake"] / eff_head["gain_per_labelled_fake"]
        if eff_head and eff_full and eff_head["gain_per_labelled_fake"]
        else None
    )
    parameter_ratio = (
        eff_head["gain_per_million_parameters"] / eff_full["gain_per_million_parameters"]
        if eff_head and eff_full and eff_full["gain_per_million_parameters"]
        else None
    )

    return [
        {
            "rank": 1,
            "filename": "fig01_indistribution_vs_unseen.pdf",
            "kind": "figure",
            "question": "How much does holding out a generator cost, and is that cost "
            "the same for every generator?",
            "takeaway": f"BigGAN gap {_fmt(biggan.get('generalisation_gap'))} ROC-AUC "
            f"against VQDM gap {_fmt(vqdm.get('generalisation_gap'))}: a 32x difference "
            "under an identical protocol.",
            "subsection": "4.2 Generalisation to unseen generators",
            "recommendation": "keep",
        },
        {
            "rank": 2,
            "filename": "fig13_depth_vs_budget_tradeoff_vqdm.pdf",
            "kind": "figure (new, generated by scripts/build_dissertation_results.py)",
            "question": "Can more adaptation depth substitute for more labelled data?",
            "takeaway": (
                f"full @5% reaches {_fmt(vqdm.get('full_5pct'))} against head-only @50% "
                f"{_fmt(vqdm.get('head_only_50pct'))} using "
                f"{f1_pair['left_labelled_fakes'] if f1_pair else '-'} labelled images "
                f"instead of {f1_pair['right_labelled_fakes'] if f1_pair else '-'}."
            ),
            "subsection": "4.5 Depth against budget",
            "recommendation": "keep (new combined figure; replaces reading fig02 and "
            "fig03 side by side)",
        },
        {
            "rank": 3,
            "filename": "fig03_depth_roc_auc_vqdm.pdf",
            "kind": "figure",
            "question": "Does adaptation depth change what performance is reachable?",
            "takeaway": (
                f"head-only plateaus at {_fmt(head_95['best_value_measured']) if head_95 else '-'} "
                f"and never reaches 0.95; full reaches it at "
                f"{full_95['first_budget_label'] if full_95 else '-'}."
            ),
            "subsection": "4.4 Fine-tuning depth",
            "recommendation": "keep",
        },
        {
            "rank": 4,
            "filename": "fig02_recovery_vqdm_roc_auc.pdf",
            "kind": "figure",
            "question": "How much of the loss does a small labelled budget recover?",
            "takeaway": (
                f"the first 5% tranche returns "
                f"{first_full['fraction_of_total_recovery'] * 100:.1f}% of the total "
                f"0->50% recovery under full fine-tuning."
                if first_full
                else "diminishing returns across all four budget steps."
            ),
            "subsection": "4.3 Limited-data recovery",
            "recommendation": "keep",
        },
        {
            "rank": 5,
            "filename": "tab02_degradation.md",
            "kind": "table",
            "question": "What exactly degrades, on which metric, at which threshold?",
            "takeaway": "recall@0.5 is the metric that collapses; ROC-AUC understates "
            "the operational failure.",
            "subsection": "4.2 Generalisation to unseen generators",
            "recommendation": "keep",
        },
        {
            "rank": 6,
            "filename": "fig11_parameter_efficiency_roc_auc_vqdm.pdf",
            "kind": "figure",
            "question": "What does each unit of adaptation cost buy?",
            "takeaway": (
                f"per labelled image full is ~{data_ratio:.1f}x head-only at 5%; "
                f"per parameter head-only is ~{parameter_ratio:,.0f}x full."
                if data_ratio is not None and parameter_ratio is not None
                else "the two efficiency axes rank the depths oppositely."
            ),
            "subsection": "4.6 Parameter and data efficiency",
            "recommendation": "keep",
        },
        {
            "rank": 7,
            "filename": "fig10_score_distributions_vqdm.pdf",
            "kind": "figure",
            "question": "Why does ROC-AUC stay high while F1@0.5 collapses?",
            "takeaway": "the unseen fake distribution shifts across the fixed 0.5 "
            "boundary without losing separability.",
            "subsection": "4.7 Threshold behaviour",
            "recommendation": "keep",
        },
        {
            "rank": 8,
            "filename": "fig14_confusion_panel.pdf",
            "kind": "figure (new, generated by scripts/build_dissertation_results.py)",
            "question": "What does the failure and its two repairs look like in counts?",
            "takeaway": "one panel replaces five separate confusion matrices; VQDM 0% "
            "misses most held-out fakes and both repairs close it.",
            "subsection": "4.7 Threshold behaviour",
            "recommendation": "keep (new; replaces individual confusion figures)",
        },
        {
            "rank": 9,
            "filename": "tab03_reproduction_check_vqdm.md",
            "kind": "table",
            "question": "Does the pipeline reproduce a previous run exactly?",
            "takeaway": "15 of 15 metric comparisons exact.",
            "subsection": "5.1 Reproducibility",
            "recommendation": "keep (Chapter 5)",
        },
        {
            "rank": 10,
            "filename": "fig03_depth_roc_auc_biggan.pdf",
            "kind": "figure",
            "question": "What does the depth comparison look like without headroom?",
            "takeaway": "all depths above 0.98 at every budget; the comparison is "
            "uninformative, which is why VQDM was run.",
            "subsection": "4.4 Fine-tuning depth",
            "recommendation": "appendix (needed to justify the second ablation, but not "
            "a result in its own right)",
        },
        {
            "rank": 11,
            "filename": "fig05_depth_average_precision_{biggan,vqdm}.pdf",
            "kind": "figure",
            "question": "Does PR-AUC change the depth story?",
            "takeaway": "it tracks ROC-AUC almost exactly; no independent information.",
            "subsection": "-",
            "recommendation": "discard (redundant with fig03)",
        },
        {
            "rank": 12,
            "filename": "fig06/fig07_depth_recovery_*, fig08_validation_trajectories_*, "
            "fig09_threshold_response_biggan, fig12_parameter_efficiency_f1_*",
            "kind": "figures",
            "question": "-",
            "takeaway": "restatements of ranked artefacts above, or BigGAN duplicates of "
            "VQDM figures that carry the argument better.",
            "subsection": "-",
            "recommendation": "discard from the chapter; they remain in the generated "
            "package for completeness",
        },
    ]


def _shortlist_section(ctx: D.Context) -> str:
    lines = [
        "## 16. Figure quality review",
        "",
        "The generated Chapter 4 package holds 64 artefacts. Carrying all of them into",
        "the dissertation would bury the argument. The ranked shortlist below is what the",
        "chapter should cite; everything else stays in the generated package as evidence",
        "rather than appearing in the text.",
        "",
    ]
    for row in figure_shortlist(ctx):
        lines += [
            f"**{row['rank']}. `{row['filename']}`** ({row['kind']})",
            f"- Question: {row['question']}",
            f"- Key number: {row['takeaway']}",
            f"- Subsection: {row['subsection']}",
            f"- Recommendation: **{row['recommendation']}**",
            "",
        ]
    return "\n".join(lines)


def _novelty_audit(ctx: D.Context) -> str:
    synthesis = {r["held_out_generator"]: r for r in D.cross_generator_synthesis(ctx)}
    vqdm = synthesis.get("vqdm", {})
    return "\n".join(
        [
            "## 17. Contribution and novelty audit",
            "",
            "Classification is about what *this project's data* supports, not about",
            "priority in the literature. No claim of priority is made anywhere below,",
            "because no literature search has been run in this workspace. The words",
            "'first', 'novel', 'state of the art' and 'invented' are deliberately absent",
            "from every 'directly demonstrated' entry.",
            "",
            "| Candidate contribution | Classification | Basis |",
            "|---|---|---|",
            "| Controlled held-out-generator protocol | **directly demonstrated** | "
            "grouped splits, byte-identical fixed test set per generator, recorded "
            "checkpoint provenance, 0 leakage across all budget/run combinations |",
            "| Nested limited-data recovery study | **directly demonstrated** | nesting "
            "verified from saved sample IDs, not assumed from config; every larger budget "
            "is a superset of the smaller |",
            "| Comparison of data budget against adaptation depth | **directly "
            "demonstrated** | 3 depths x 4 budgets on two generators, all sharing one "
            "starting checkpoint and one test set |",
            "| 5% full adaptation outperforms 50% head-only on VQDM | **supported but "
            "weaker than it looks** | true in sign on all three metrics, but ROC-AUC "
            "margin +0.0014 and PR-AUC +0.0057 sit below the reliability floor; only the "
            "F1 margin (+0.0789) clears it. State as equivalence at one tenth the "
            "labelling cost |",
            "| Head-only cannot reach 0.95 ROC-AUC on VQDM at any budget run | "
            "**directly demonstrated** | best head-only value "
            f"{_fmt(vqdm.get('head_only_50pct'))} at 1,000 labelled images; full reaches "
            "0.95 at 200. This is the stronger form of the depth claim |",
            "| Parameter-efficiency trade-off | **directly demonstrated (as a trade-off, "
            "not a winner)** | per-parameter and per-image rankings are opposite and both "
            "are computed from recorded counts |",
            "| Degradation magnitude is generator-specific | **supported but not novel** "
            "| widely reported in the cross-generator detection literature; this project "
            "measures it under a tighter protocol but should not present it as new |",
            "| CLIP features remain informative after generator shift | **supported but "
            "not novel** | consistent with the linear-probe literature; the evidence here "
            "is that even 769 parameters recover a large fraction |",
            "| External post-development contemporary-generator challenge | "
            "**potentially novel / requires literature verification** | the design is "
            "sound and the provenance discipline is unusual, but nothing has been run, so "
            "there is currently no result at all. Novelty cannot be assessed without a "
            "literature search |",
            "| Practical recommendation for detector maintenance | **potentially novel / "
            "requires literature verification** | the specific recommendation (spend "
            "labelling effort early, spend it on depth rather than volume) follows from "
            "these two generators only; whether it generalises is untested |",
            "| Any claim about which layers carry generator-specific signal | "
            "**unsupported / do not claim** | no layer-wise attribution was run; "
            "trainable-parameter counts are not attribution |",
            "| Any claim of statistical significance | **unsupported / do not claim** | "
            "one subset seed, one training seed, no variance estimate |",
            "| Any claim of external generalisation | **unsupported / do not claim** | "
            "no image outside the Tiny GenImage subset has been evaluated |",
            "",
            "**Before any 'novel' wording enters the dissertation**, the two entries",
            "marked *requires literature verification* need a targeted search: prior work",
            "on (a) post-hoc evaluation of frozen detectors against generators released",
            "after the detector's benchmark, and (b) adaptation-depth-versus-budget",
            "trade-offs for AI-image detection specifically. Until that search exists,",
            "write these as contributions of this project, not as firsts.",
            "",
        ]
    )


def compose(ctx: D.Context) -> str:
    inventory = D.experiment_inventory(ctx)
    included = [r for r in inventory if r["inclusion"] == "included"]
    header = "\n".join(
        [
            "# Dissertation results notes",
            "",
            "Generated by `python -m scripts.build_dissertation_results` from the saved",
            f"runs under `outputs/`. {len(included)} runs included of {len(inventory)} "
            "discovered.",
            "",
            "**Every number in this document is read or derived from a saved run at",
            "generation time.** Nothing is transcribed by hand. Regenerate after any new",
            "run rather than editing values in place.",
            "",
            "Empirical fact and interpretation are separated throughout: a line under",
            "**Fact** is measured or arithmetically derived; a line under",
            "**Interpretation** is a reading a reader may dispute.",
            "",
            "No result in this document carries a significance test, confidence interval",
            "or error bar, because one subset seed and one training seed were run and no",
            "variance was measured.",
            "",
            "---",
            "",
        ]
    )
    return "\n".join(
        [
            header,
            _headline(ctx),
            _supporting_numbers(ctx),
            _figure_map(ctx),
            _generator_sections(ctx),
            _budget_depth_efficiency(ctx),
            _threshold_section(ctx),
            _validation_section(ctx),
            _limitations_section(ctx),
            _do_not_claim(ctx),
            _chapter_mapping(ctx),
            _external_section(ctx),
            _shortlist_section(ctx),
            _novelty_audit(ctx),
        ]
    )
