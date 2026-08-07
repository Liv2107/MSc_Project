# Test implementation guide

## Purpose

Tests protect the scientific contract, not only Python syntax. A training loop can run successfully while labels are inverted, held-out generators leak into training, batch-size-one logits collapse to scalars, or ROC-AUC is calculated from hard predictions. Those failures should be caught before compute is spent.

The initial files contain skipped specifications. Replace each skip with small deterministic fixtures and assertions as its production TODO is implemented. Prefer synthetic records/arrays and temporary files; unit tests must not require the private dataset, network, or GPU.

## Recommended order

1. Manifest row parsing and label/generator validation.
2. Generator filters, subset reproducibility, and split/group invariants.
3. Metric semantics including edge cases.
4. Model tensor shapes, freeze policies, and checkpoint round trips.
5. One-batch training/validation state changes.
6. Protocol-level leakage assertions using tiny synthetic manifests.

## Implementation checklist

- [ ] Activate each skipped specification alongside the code it tests.
- [ ] Keep tests CPU-only and deterministic unless explicitly marked otherwise.
- [ ] Include malformed and edge cases, not only the happy path.
- [ ] Add a tiny-batch overfit integration test without external downloads.
- [ ] Run the complete suite before every reported experiment.
