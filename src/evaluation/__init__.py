"""Reusable binary metrics, generator-wise evaluation, inference, and plots.

Evaluation is implemented independently from training so stored predictions can be
re-analysed without loading a GPU checkpoint. Establish metric semantics on small
hand-calculated cases before trusting experiment outputs.

IMPLEMENTATION CHECKLIST
------------------------
[ ] Test score/label validation and threshold semantics.
[ ] Implement scalar metrics and curve coordinates.
[ ] Handle undefined single-class metrics explicitly.
[ ] Save sample-level predictions and generator-wise tables.
[ ] Build plots only from saved evaluated data.
"""
