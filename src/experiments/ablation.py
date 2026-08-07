"""Fine-tuning-depth ablation experiment scaffold.

###############################################################################
RESEARCH QUESTION
###############################################################################

Does recovery require changing CLIP broadly, or can a lightweight classifier update
adapt existing features? Compare:

- ``head_only``: tests whether existing CLIP features already separate the new source.
- ``last_block``: tests whether limited high-level representation adaptation suffices.
- ``full``: tests maximum adaptability, with higher compute and overfitting risk.

Only the trainable layers should differ. Reuse the same starting checkpoint, subset
IDs, test samples, seeds, selection metric, and—unless explicitly studying it—training
budget. Learning rates may need mode-specific values, but if so they must be tuned by
a predeclared validation procedure and reported as part of the comparison.
"""

from __future__ import annotations

from pathlib import Path


def run_ablation(config_path: Path) -> None:
    """Compare head-only, last-block, and full adaptation fairly.

    TODO 1: Require all three freeze modes (or document a deliberately smaller study)
        and validate every referenced adaptation subset/checkpoint.
    TODO 2: Loop over mode, percentage, subset seed, and training seed using a stable
        run matrix saved before execution.
    TODO 3: Reload identical starting weights for every cell, apply the freeze policy,
        and log trainable parameter names/counts as an invariant check.
    TODO 4: Construct a fresh optimiser after freezing. Ensure frozen parameters are
        absent and no state leaks between cells.
    TODO 5: Control comparison budget. Decide whether fairness means equal epochs,
        optimiser updates, wall-clock, or validation-selected training; state it.
    TODO 6: Select checkpoints from adaptation validation only, then evaluate the same
        final unseen test samples exactly once per selected model.
    TODO 7: Save sample predictions and a tidy table with mode, fraction, count, seeds,
        trainable parameters, compute/time, and metrics.
    TODO 8: Interpret conclusions carefully: head-only success implies useful frozen
        features; last-block gains imply high-level adaptation; full gains imply wider
        change may help, while full underperformance may reflect small-data overfit or
        optimisation difficulty rather than lack of capacity.
    """
    raise NotImplementedError("Implement controlled fine-tuning-depth ablations.")


# IMPLEMENTATION CHECKLIST
# [ ] Freeze the ablation run matrix and shared subset/checkpoint identities.
# [ ] Verify trainable names/counts for head-only, last-block, and full modes.
# [ ] Recreate optimiser and reload starting weights for every condition.
# [ ] Define fair training/selection budget and any mode-specific hyperparameter policy.
# [ ] Evaluate identical final-test IDs with identical metric semantics.
# [ ] Report compute and variation across both subset and training seeds.
# [ ] Separate representational conclusions from optimisation/overfitting explanations.
