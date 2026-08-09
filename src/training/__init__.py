"""Training components and epoch engine.

Implement metrics/model shape tests before this package. Training code should
optimise an already-defined contract; it should not decide dataset membership,
generator holdouts, or test-set evaluation policy.

IMPLEMENTATION CHECKLIST
------------------------
[x] Implement loss, optimiser, and scheduler factories.
[x] Overfit a tiny batch.
[x] Add validation, early stopping, mixed precision, and checkpoint resume.
[ ] Verify no test loader enters the training API.
"""
