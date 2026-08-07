# Checkpoints directory

## Purpose

Store local model states during development. A checkpoint is scientifically useful only when paired with model identity, freeze policy, optimiser/scheduler state, epoch, validation criterion, random seed, and resolved configuration.

Do not commit large weight files. Final experiments should record a checksum or artefact-store identifier so a reported model can be recovered unambiguously.

## Implementation checklist

- [ ] Define the checkpoint dictionary schema.
- [ ] Save and restore all state required for a faithful resume.
- [ ] Verify a save/load prediction round trip.
- [ ] Record checksums for dissertation checkpoints.
