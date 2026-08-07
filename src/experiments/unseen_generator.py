"""Leave-one-generator-out generalisation experiment scaffold.

###############################################################################
RESEARCH QUESTION
###############################################################################

Train on generators A/B/C and evaluate on generator D, which is absent from training
and model-selection data. A performance drop relative to the baseline estimates how
well learned cues transfer to a new generation process. Repeating D across generators
distinguishes a general pattern from one unusually easy or difficult generator.

The unseen test needs real negatives under a declared comparison policy. Keep that
real pool fixed where appropriate, and disclose that generator-wise test sets may
therefore share negatives.
"""

from __future__ import annotations

from pathlib import Path


def run_unseen_generator(config_path: Path) -> None:
    """Train without one generator and evaluate its untouched final partition.

    TODO 1: Validate exactly one ``unseen_generator`` and a non-empty set of known
        training generators that excludes it.
    TODO 2: Load persisted partitions created before experimentation.
    TODO 3: Assert no unseen-generator fake record/group appears in train, validation,
        threshold selection, or early stopping. Print/audit IDs on any violation.
    TODO 4: Train/select the model using only known-generator train/validation data,
        or load a compatible declared known-generator checkpoint.
    TODO 5: Define a validation-selected threshold before accessing unseen test labels.
    TODO 6: Evaluate the unseen generator's final test partition plus the declared real
        comparison pool; never use the adaptation pool in this 0% result.
    TODO 7: Save sample predictions and overall/per-generator metrics with a 0%
        adaptation label. This becomes the recovery curve's starting point.
    TODO 8: Compare with a genuinely comparable baseline (same architecture, seed,
        real pool, sample policy, and metric), not merely any prior headline number.
    TODO 9: Repeat leave-one-generator-out and seeds as specified in the protocol.
    """
    raise NotImplementedError("Implement strict leave-one-generator-out orchestration.")


# IMPLEMENTATION CHECKLIST
# [ ] Assert held-out generator exclusion from every model-development partition.
# [ ] Keep adaptation and final unseen test pools disjoint by provenance group.
# [ ] Document and fix the real-image comparison pool.
# [ ] Reuse identical model/training settings for comparable baseline and unseen runs.
# [ ] Save the zero-shot/0%-adaptation prediction table for each seed/generator.
# [ ] Repeat each declared generator as held out before generalising conclusions.
