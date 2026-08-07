"""CLIP detector components, trainability policies, and checkpoint state.

The model package should know how pixels become logits, but not which generator is
held out or which experiment is running. Keeping those concerns separate makes the
same detector usable across baseline, recovery, and ablation protocols.

IMPLEMENTATION CHECKLIST
------------------------
[ ] Select and record one pretrained CLIP model identifier.
[ ] Verify feature dimensions and logit shapes.
[ ] Implement explicit freeze policies.
[ ] Test checkpoint round trips and device portability.
"""
