# Outputs directory

## Purpose

Store generated run artefacts, never hand-edited results. Each run should have a unique directory containing the resolved config, logs, environment record, split identity, sample-level predictions, aggregate metrics, and plots.

Prefer tidy CSV/Parquet prediction tables with `sample_id`, `label`, `generator`, `score`, `prediction`, `split`, `seed`, and `checkpoint_id`. Sample-level data makes later metric corrections possible without rerunning inference.

## Implementation checklist

- [ ] Define collision-resistant run identifiers.
- [ ] Save resolved configuration and environment metadata.
- [ ] Save sample-level predictions before aggregate metrics.
- [ ] Make notebook outputs traceable to run identifiers.
