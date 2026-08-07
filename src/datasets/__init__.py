"""Dataset contracts, filtering, loading, and leakage-aware splitting.

Implement this package before the model or training packages. If sample identity,
label semantics, or generator partitions are wrong, a technically correct training
loop will still answer the wrong research question.

IMPLEMENTATION CHECKLIST
------------------------
[ ] Finalise the manifest schema.
[ ] Validate labels and generator names.
[ ] Implement image loading and transforms.
[ ] Persist and audit split assignments.
[ ] Test filters, subsets, and batching.
"""

from .schema import DatasetRecord

__all__ = ["DatasetRecord"]
