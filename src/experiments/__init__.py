"""Protocol-level orchestration for the dissertation's four experiment families.

Experiment modules decide *which* data/model/checkpoint is used and in what order.
They should delegate mechanics to datasets, models, training, and evaluation modules.
This keeps research comparisons explicit and prevents copy-pasted training loops.

IMPLEMENTATION CHECKLIST
------------------------
[ ] Implement shared construction only after lower-level contracts are tested.
[ ] Run baseline before unseen-generator evaluation.
[ ] Partition adaptation/test data before recovery experiments.
[ ] Hold all non-ablation conditions constant across fine-tuning modes.
"""
