"""Limited-data fine-tuning recovery experiment scaffold.

###############################################################################
RESEARCH QUESTION
###############################################################################

After performance on an unseen generator is measured, how much can the detector
recover using limited labelled data from that generator? The planned budgets are
5%, 10%, 20%, and 50% of a *predefined adaptation pool*, never the final test set.

Why these values:
    5% tests very-low-data adaptation; 10% tests whether modest evidence is enough;
    20% reveals whether improvement continues beyond the smallest budgets; and 50%
    provides a substantial-but-still-limited reference. Actual sample counts must be
    reported because identical percentages can represent very different evidence.

Nested subsets (5% contained in 10%, etc.) reduce one source of comparison noise.
Repeated subset seeds remain important because which examples are labelled can matter
as much as the nominal percentage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AdaptationBudget:
    """One limited-label condition and its reproducible sample identity.

    TODO 1: Store actual count and selected IDs after subset construction.
    TODO 2: Include subset seed separately from model-training seed if both vary.
    """

    fraction: float
    subset_seed: int


def build_nested_adaptation_subsets(
    *,
    adaptation_pool: object,
    fractions: tuple[float, ...],
    seed: int,
) -> dict[float, list[str]]:
    """Return nested sample-ID lists for fair data-budget comparison.

    TODO 1: Validate increasing unique fractions in (0, 1], including configured
        0.05, 0.10, 0.20, and 0.50 for the primary study.
    TODO 2: Operate only on the persisted adaptation pool and assert no final-test ID
        or provenance group is present.
    TODO 3: Randomise groups/records once with a local seed; take increasing prefixes
        so smaller sets are contained in larger sets.
    TODO 4: If stratification is needed, keep both nesting and group integrity. Explain
        any algorithmic trade-off rather than silently losing one guarantee.
    TODO 5: Apply a documented rounding/minimum rule and report actual counts.
    TODO 6: Save IDs for exact reuse across fine-tuning-mode ablations.
    """
    raise NotImplementedError("Implement persisted nested adaptation subsets.")


def run_fine_tuning(config_path: Path) -> None:
    """Fine-tune compatible unseen-generator checkpoints at each data budget.

    TODO 1: Validate unseen generator, percentages, subset seeds, training seeds, and
        starting-checkpoint compatibility (model, known generators, manifest/split).
    TODO 2: Load persisted adaptation/final-test partitions and audit disjoint groups.
    TODO 3: Create/load nested subset ID files once; do not resample opportunistically.
    TODO 4: For each percentage and seed, reload the *same untouched starting
        checkpoint*. Do not continue 5% -> 10% unless continual learning is the
        explicitly declared research question.
    TODO 5: Apply the configured freeze mode and adaptation hyperparameters; create a
        validation strategy from adaptation data without touching final test data.
    TODO 6: Fine-tune/select using adaptation train/validation only. Small subsets may
        require grouped cross-validation or repeated splits; predeclare the method.
    TODO 7: Evaluate every selected model on the exact same final unseen test and real
        comparison samples with the same threshold policy.
    TODO 8: Save predictions/metrics with percentage, actual count, subset seed,
        training seed, freeze mode, and starting checkpoint ID.
    TODO 9: Plot recovery relative to the saved 0% result and report uncertainty—not
        a fabricated or selectively chosen best run.
    """
    raise NotImplementedError("Implement limited-data recovery after 0% unseen evaluation.")


# IMPLEMENTATION CHECKLIST
# [ ] Partition adaptation and final test pools before sampling percentages.
# [ ] Build, save, and test nested 5/10/20/50% ID sets across subset seeds.
# [ ] Reload identical starting weights for every independent condition.
# [ ] Define small-data validation/early-stopping without final-test leakage.
# [ ] Hold final test, real pool, metric, and threshold policy constant.
# [ ] Report actual counts plus subset/training seed variability.
# [ ] Include the measured 0% condition in recovery plots.
