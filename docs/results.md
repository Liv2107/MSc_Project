# Results and reporting

Reports read saved run artefacts from `outputs/`. They exclude failed, incomplete
and synthetic smoke runs by default. Rebuilding a report does not retrain a model
or change the source run directories.

## Build reports

```sh
python -m scripts.build_report --output outputs/report
python -m scripts.build_chapter4
python -m scripts.build_dissertation_results
python -m scripts.build_final_results_summary
```

`build_report` selects a run per protocol for detailed plots and also consolidates
all reportable runs. `--include-synthetic-smoke` includes pipeline fixtures for
inspection; those rows are labelled `is_synthetic_smoke`.

| Output | Contents |
|---|---|
| `report/<protocol>/` | Prediction curves, confusion matrices, training history and per-generator tables |
| `report/consolidated/consolidated_results.csv` | One row per run, evaluation set, condition and operating point |
| `report/consolidated/consolidated_results.json` | Consolidated rows, experiment inventory, degradation and recovery summaries |
| `report/consolidated/table_experiment_inventory.*` | All discovered runs, including excluded ones |
| `report/consolidated/table_unseen_degradation.*` | Matched in-distribution and held-out performance |
| `report/consolidated/table_recovery_budget.*` | Budget, depth, seed counts and recovery summaries |
| `report/consolidated/missing_results.md` | Gaps in the available experiments |
| `report/chapter4/` | Chapter 4 figures and tables |
| `report/dissertation_results/` | Analysis tables, findings and validation summaries |
| `report/final_results/` | Internal and external results in one table with explicit dataset labels |

Each export directory includes a README describing its files. The shared analysis
code is in `src/evaluation/aggregation.py` and `src/evaluation/dissertation.py`.

## Reading the tables

Rows retain `run_id`, the source metrics file and its SHA-256 digest. Missing
measurements stay undefined. Classifier head, labelled budget, seed, threshold
and test-set identity distinguish conditions that would otherwise look alike.

Use `row_role == "primary"` in the final-results table for the main reported rows.
`reproduction` rows are repeated conditions retained for agreement checks; do not
count them as extra independent evidence. Use `threshold_role == "default"` for
the fixed 0.5-threshold table. Keep the external and Tiny GenImage datasets separate.

Recovery summaries include:

- `absolute_recovery`: adapted metric minus the measured 0% reference.
- `relative_improvement`: that change divided by the 0% value.
- `gap_closed_fraction`: that change divided by the in-distribution-to-unseen gap.

The in-distribution reference is resolved through the adaptation run's starting
checkpoint. `in_distribution_reference_match` records whether this used checkpoint
provenance or a weaker generator match. A missing checkpoint reference leaves the
ceiling undefined. This prevents linear and cosine runs on the same generator
from sharing the wrong reference.

Threshold-dependent comparisons use matching operating points. A threshold
selected on adaptation-validation data is not directly comparable to an
in-distribution ceiling measured at 0.5.

`gap_closed_is_reliable` flags gaps smaller than 0.02. Dividing by a small gap can
produce large percentages; report absolute changes alongside them. This threshold
is a reporting heuristic, not an estimate of sampling uncertainty.

With one fit, the aggregation records standard deviation as 0.0 and standard error
as undefined. That does not establish zero variability. Recovery plots only draw
an uncertainty band where at least three runs are available.

## External-image evaluation

The external set was generated through an assistant-mediated hosted image tool
whose underlying model was not identified. Its recorded generator is
`astra_mediated_unidentified`. Results apply to this collected set and cannot be
attributed to a named architecture.

With source images, provenance and frozen checkpoints available:

```sh
python -m scripts.build_external_manifest
python main.py --config configs/external_challenge_v1.yaml
python -m scripts.build_external_figures
```

The manifest builder checks decodability, recorded digests, duplicate images and
provenance. It checks exact and perceptual-hash overlap with the internal data,
then applies the same 256×256 RGB JPEG preprocessing to both classes. Authentic
comparators are a seeded subset of the fixed internal real test pool. The builder
writes a manifest and audit, refusing to overwrite an existing build.

The external runner verifies manifest and checkpoint digests and scores every
nominated detector. The config identifies the primary detector. It uses the
existing 0.5 and internal validation-selected thresholds without tuning on the
external images. No optimiser or checkpoint writing is involved.

The collected set has 100 generated and 100 authentic images, below the proposed
250 per class. Its unidentified generator, native resolution and photographic
style limit interpretation. Shared preprocessing does not remove every effect
of downsampling or content style. The shared authentic pool also means external
and internal evaluations are not independent.

External recovery was not run: adaptation cells retained digests but removed
weights after evaluation. Scoring adapted detectors would require refitting them.
The internal aggregation deliberately excludes external runs; the final summary
includes them under `dataset=external_challenge_v1_astra_mediated`.

`python -m scripts.verify_corrections` checks the available on-disk artefacts,
including the external manifest when present. External checks are skipped when
that manifest is absent.
