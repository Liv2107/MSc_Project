# Generalisation of AI-image detectors to unseen generators

This repository implements the first vertical slice of an MSc Computer Science dissertation investigating:

> How well do AI image detectors generalise to unseen image generators, and how effectively can they recover performance through limited fine-tuning?

It contains no research dataset, trained research model, numerical result, or dissertation conclusion. The Week 1 baseline pipeline is implemented and tested; unseen-generator, limited-data recovery, ablation, and reporting code remains staged for later work.

The scaffold targets **Python 3.11 or newer** (it uses modern typing and `StrEnum`). Choose and document one exact Python version for the final environment rather than assuming all 3.11+ environments are numerically identical.

## Why the repository is organised this way

The central research risk is leakage: an experiment can appear to measure unseen-generator generalisation while generator identities or near-duplicate source images leak across splits. The structure therefore separates dataset selection, split creation, model construction, training, evaluation, and experiment orchestration. This makes each research decision inspectable and testable.

```text
MSc_dis/
├── configs/                 # YAML records of experimental intent
├── data/                    # Local raw/processed data; never committed by default
├── notebooks/               # Exploration and reporting, not core experiment logic
├── src/
│   ├── datasets/            # Manifest contract, filters, subsets, and split policy
│   ├── models/              # CLIP backbone, binary head, freezing, checkpoints
│   ├── training/            # Loss/optimiser/scheduler and train/validation loops
│   ├── evaluation/          # Metrics, generator-wise analysis, plots
│   ├── experiments/         # Baseline, unseen-generator, recovery, and ablation flows
│   └── utils/               # Config validation, reproducibility, logging
├── tests/                   # Contract tests activated as TODOs are implemented
├── checkpoints/             # Local model state plus metadata
├── outputs/                 # Predictions, metrics, plots, and run records
├── main.py                  # Thin command-line dispatcher
└── requirements.txt
```

## Research workflow and implementation order

Implement one vertical slice at a time. Do not begin large training runs until the preceding checks pass.

1. **Define the dataset manifest.** Adopt one row per image with at least `image_path`, binary `label`, and `generator`. Decide on a reserved generator value such as `real` for authentic images. Add provenance/group columns needed to prevent related images crossing splits.
2. **Implement and audit loading.** Complete `src/datasets/schema.py` and `detector_dataset.py`. Inspect missing files, corrupt images, label semantics, generator counts, and duplicate hashes in notebook 01.
3. **Implement leakage-aware splitting.** Use grouped or source-aware splitting if images share prompts, originals, identities, or transformations. Save split assignments so every experiment uses identical samples.
4. **Implement the CLIP detector.** Load a pretrained CLIP vision encoder, define an explicit feature-extraction contract, attach a single-logit classifier, and test tensor shapes before training.
5. **Implement evaluation first.** Confirm metric behaviour on small hand-calculated arrays, including single-class edge cases. Persist sample-level predictions; aggregate metrics can always be regenerated from them.
6. **Implement training.** Start with a tiny overfitting test, then validation, mixed precision, checkpointing, and early stopping. Never select a checkpoint using the held-out test set.
7. **Run the baseline.** Train and evaluate on the declared in-distribution split to establish the detector's ordinary behaviour.
8. **Run unseen-generator experiments.** Train on selected generators and evaluate on a generator held entirely out of training and model selection.
9. **Run limited-data recovery.** Fine-tune with 5%, 10%, 20%, and 50% of the held-out generator's designated adaptation split. Repeat sampling across seeds and keep its final test partition untouched.
10. **Run ablations.** Compare head-only, last-block, and full fine-tuning under identical data subsets, seeds, evaluation data, and selection rules.
11. **Analyse, visualise, and report.** Use notebooks 02 and 03 only after outputs are saved. Report variation across seeds and limitations; do not read conclusions into isolated runs.

## Dataset contract

Create a CSV or Parquet manifest rather than inferring research labels from directory names at training time. The minimum schema is:

| Column | Meaning | Example |
|---|---|---|
| `image_path` | Path relative to a configured data root | `images/000123.png` |
| `label` | Binary target with one documented convention | `0` for real, `1` for fake |
| `generator` | Source generator, with a reserved real-image value | `sdxl` or `real` |

Strongly consider `sample_id`, `source_group`, `prompt_id`, `dataset_source`, and `content_hash`. These fields support provenance, duplicate detection, and group-aware splitting. Never silently map unfamiliar labels or generator names.

### GenImage

GenImage is integrated as the initial research dataset. After manually downloading
and extracting the official release, run `python -m scripts.import_genimage` and use
`configs/genimage_baseline.yaml`. Acquisition, licence, folder layout, canonical
generator names, split policy, development caps, and provenance limitations are
documented in `data/GENIMAGE.md`.

To add a dataset or generator:

1. Obtain and document its licence, collection method, preprocessing, and provenance.
2. Convert metadata into the manifest contract without changing the raw images.
3. Add the generator name to the relevant YAML config; do not hard-code it in Python.
4. Run schema, existence, duplication, balance, and leakage checks.
5. Inspect examples manually in notebook 01.
6. Freeze and version the manifest or record a cryptographic hash.

## Configuration and experiment relationships

`configs/base.yaml` holds shared defaults. The other YAML files describe experiment-specific intent. The config loader should merge them explicitly and save the fully resolved config with every run.

- **Baseline:** train on selected generators and evaluate on an in-distribution test split.
- **Unseen generator:** exclude generator D from training and validation, then evaluate once on D's test split. This isolates cross-generator generalisation.
- **Fine-tuning recovery:** adapt the baseline/unseen checkpoint using 5%, 10%, 20%, or 50% of D's adaptation partition. Plot performance against labelled-data budget.
- **Ablation:** repeat adaptation while training only the classifier head, the final vision block plus head, or the full model. This tests whether recovery requires broad representation change.

Fine-tuning percentages must be sampled from a separate adaptation pool, preferably with nested subsets (5% contained within 10%, etc.) and repeated seeds. They must never be percentages of the final test set.

## Week 1 commands

Install the environment, copy and populate `data/manifest_template.csv`, validate it and create persisted splits, then run the baseline:

```powershell
python -m pip install -r requirements.txt
python -m scripts.prepare_dataset --manifest data/manifests/dataset.csv --data-root data --output data/manifests/splits.csv
python main.py --config configs/baseline.yaml
python -m pytest
```

Before running, replace the placeholder generator names in `configs/baseline.yaml` with names that exactly match the validated manifest. The other experiment configurations intentionally remain unavailable until their protocol runners are implemented.

When implemented, each run should create a unique output directory containing the resolved config, seed, environment information, split/manifest identity, sample-level predictions, aggregate metrics, logs, and checkpoint reference. That bundle is the minimum audit trail for a dissertation result.

## Reproducibility and common failure modes

- Fit transforms, sampling rules, and thresholds without consulting the test set.
- Keep the adaptation pool and final unseen-generator test partition disjoint.
- Prevent prompt/original/near-duplicate families from crossing partitions.
- Keep preprocessing comparable across real and fake classes; otherwise the detector may learn file format, resolution, or compression artefacts.
- Decide whether generator-wise metrics include the special `real` group and report the decision.
- Compute ROC-AUC and PR-AUC from continuous scores, never thresholded labels.
- Repeat subset selection and training across seeds; one seed cannot quantify sampling variability.
- Save the exact CLIP model identifier, preprocessing parameters, freeze policy, and decision threshold.
- Treat deterministic GPU execution as a best effort and document remaining nondeterminism.

## What belongs in notebooks

Notebooks are for interactive auditing and presentation. Reusable loading, filtering, metrics, and plotting logic belongs under `src/`; otherwise notebook execution order becomes an undocumented experimental dependency. The included notebooks contain markdown guidance and commented placeholder cells only, and generate no results.

## Repository implementation checklist

- [x] Finalise label and generator naming conventions.
- [ ] Create and validate a provenance-rich manifest.
- [x] Implement image loading, transforms, filtering, subsets, and split persistence.
- [x] Test leakage controls and batch tensor shapes.
- [x] Implement CLIP feature extraction and the single-logit head.
- [x] Unit-test metrics against hand-calculated examples.
- [x] Pass a tiny-batch overfitting test before full training.
- [x] Implement checkpoint save/load and resolved-config logging.
- [ ] Run baseline, unseen-generator, fine-tuning, and ablation protocols in order.
- [ ] Repeat experiments across declared seeds.
- [ ] Populate notebooks only from saved, auditable outputs.
- [ ] Freeze dependencies and document the final compute environment.
