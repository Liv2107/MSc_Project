"""Training components and epoch engine.

Implement metrics/model shape tests before this package. Training code should
optimise an already-defined contract; it should not decide dataset membership,
generator holdouts, or test-set evaluation policy.

IMPLEMENTATION CHECKLIST
------------------------
[ ] Implement loss, optimiser, and scheduler factories.
[ ] Overfit a tiny batch.
[ ] Add validation, early stopping, mixed precision, and checkpoint resume.
[ ] Verify no test loader enters the training API.
"""
