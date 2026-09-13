# Checkpoints

Local model weights are excluded from version control. Baseline and
unseen-generator runs save checkpoints in `outputs/<run_id>/`, alongside the
resolved config, model identity and checkpoint digests.

`best_checkpoint.pt` is selected on validation. Interrupted runs can resume from
`last_checkpoint.pt` and `training_state.json`; see the root README. Recovery
and ablation runs retain cell checkpoint digests and predictions, but remove
cell weights after evaluation.
