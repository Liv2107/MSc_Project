# Generalisation of AI-image detectors

MSc project investigating how well a CLIP-based image detector handles generators
it did not see during training, and how much labelled data it needs to recover.
The experiments use Tiny GenImage, with a separate external-image evaluation.

The detector uses a CLIP ViT-B/32 vision encoder with a binary classifier. The main
experiments compare in-distribution performance, held-out generators, adaptation
at 5%, 10%, 20% and 50% data budgets, and three fine-tuning depths. A side study
compares linear and cosine classifier heads.

## Setup

Run commands from the repository root. Python 3.11 or newer is required.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
Training uses CUDA when available, otherwise CPU. The first model load downloads
the CLIP weights from Hugging Face; the model revision is pinned in
[`configs/base.yaml`](configs/base.yaml). Dependencies are currently unpinned;
each experiment records its environment in `environment.json`.

## Prepare the data

Images, manifests, model weights and generated results are local files and are not
included in a fresh clone. See [data/GENIMAGE.md](data/GENIMAGE.md) for acquisition
and the expected folder layout.

The `tiny_*` configs use **Tiny GenImage** (`yangsangtai/tiny-genimage` on Kaggle),
a reduced GenImage derivative with seven generators. Stable Diffusion v1.4 is
absent. Extract it under `data/raw/tiny_genimage/`, then import it:

```sh
python -m scripts.import_genimage \
  --genimage-root data/raw/tiny_genimage \
  --tiny-genimage --preprocess \
  --manifest data/manifests/tiny_genimage.csv \
  --splits data/manifests/tiny_genimage_splits.csv \
  --cache-root data/processed/tiny_genimage_cache
```

Preprocessing converts both classes to 256×256 RGB JPEGs at quality 95. This
removes the raw PNG/JPEG and image-size shortcuts, although native-resolution and
compression effects can remain. The importer preserves the originals, checks
files and duplicates, and saves an audit alongside the manifest. It refuses to
overwrite existing manifests or split files.

For another dataset, use [data/manifest_template.csv](data/manifest_template.csv).
Required columns are `sample_id`, `image_path`, `label`, `generator`,
`source_group` and `dataset_source`. Paths are relative to the data root;
labels are `0` for real and `1` for fake. Real images use `generator=real`.

## Run experiments

Configs inherit shared settings from `configs/base.yaml`; Tiny GenImage configs
also inherit `configs/tiny_genimage_base.yaml`. Paths resolve from the repository
root. Generic configs contain placeholder generator names and checkpoint paths
that need setting before use.

For the VQDM study:

```sh
python main.py --config configs/tiny_genimage_baseline.yaml
python main.py --config configs/tiny_unseen_vqdm.yaml
```

Set `fine_tuning.starting_checkpoint` in the recovery and ablation configs to the
unseen run's `outputs/<run_id>/best_checkpoint.pt`, then run:

```sh
python main.py --config configs/tiny_recovery_vqdm.yaml
python main.py --config configs/tiny_ablation_vqdm.yaml
```

The other `tiny_unseen_*` and `tiny_recovery_*` configs cover further generators.
Recovery budgets use a separate adaptation pool; the final test set stays fixed.
Ablations compare head-only, last-block and full fine-tuning. Learning rates differ
by depth, as recorded in the configs. Each adaptation cell starts from the same
checkpoint. Grid runs can be expensive: check the budgets, modes and seed lists.

Baseline and unseen-generator runs can resume from the last completed epoch:

```sh
python main.py --config configs/tiny_unseen_vqdm.yaml --resume outputs/<run_id>
```

Use the original config. Completed runs cannot be resumed; recovery and ablation
grids do not support `--resume`. Those grids retain predictions and checkpoint
digests but remove each cell's weights after evaluation.

## Results and notebooks

Runs save their resolved config, environment, history, predictions, metrics and
checkpoint digests under `outputs/<run_id>/`. To build reports from saved runs:

```sh
python -m scripts.build_report --output outputs/report
python -m scripts.build_chapter4
python -m scripts.build_dissertation_results
python -m scripts.build_final_results_summary
```

The final summary is under `outputs/report/final_results/`. Use `row_role=primary`
and `threshold_role=default` for the main table. Report external and internal
results separately. See [docs/results.md](docs/results.md) for output schemas,
aggregation rules and the external evaluation workflow.

Open the notebooks with `python -m jupyterlab` and run cells from top to bottom:

| Notebook | Purpose |
|---|---|
| [01_dataset_exploration](notebooks/01_dataset_exploration.ipynb) | Validate the manifest, inspect samples and audit saved splits |
| [02_visualisations](notebooks/02_visualisations.ipynb) | Plot the latest completed run of each protocol |
| [03_results_analysis](notebooks/03_results_analysis.ipynb) | Compare metrics, recovery and experiment coverage |
| [04_supervisor_meeting](notebooks/04_supervisor_meeting.ipynb) | Meeting figures and the linear/cosine comparison |
| [Core results](notebooks/dissertation_results/01_core_results.ipynb) | Detailed recovery, depth and threshold analysis |
| [Validation and external challenge](notebooks/dissertation_results/02_validation_and_external_challenge.ipynb) | Reproducibility, leakage checks and external evaluation |

The dataset notebook requires the imported data. The meeting and dissertation
notebooks require the study's saved runs; the meeting notebook also uses
`outputs/analysis/embedding_norms.json` and `figure_embedding_norms.png`, produced
by `python -m scripts.analyse_embedding_norms`. The external tables require
`python -m scripts.build_external_figures`. Saved notebook outputs are cleared
so figures are regenerated from the local inputs.

Most conditions use one training seed. Differences between individual fits do
not establish statistical significance. VQDM informed the cosine-head design;
Wukong was designated as its confirmatory comparison. The external generator's
identity is unknown, and its image style and native resolution introduce further
confounds. These limits remain part of the analysis.

## Checks

The test suite uses small fixtures and mock backbones, so it needs no research
data, downloaded weights or GPU:

```sh
python -m pytest
python -m ruff check .
python -m mypy src scripts main.py
```

To exercise training with synthetic images:

```sh
python -m scripts.make_synthetic_smoke_data
python -m scripts.import_genimage \
  --genimage-root data/raw/genimage_synthetic \
  --manifest data/manifests/smoke_synthetic.csv \
  --splits data/manifests/smoke_synthetic_splits.csv
python main.py --config configs/smoke_synthetic_baseline.yaml
```

This uses CLIP weights and tests the pipeline only. Synthetic smoke runs are
excluded from research reports by default.

## Layout

- `src/datasets/`: manifest validation, preprocessing and splitting.
- `src/models/` and `src/training/`: detector, checkpoints and training loop.
- `src/experiments/`: protocol runners, dispatched by `main.py`.
- `src/evaluation/`: metrics, aggregation and plots.
- `scripts/`: data import, validation and report commands.
- `configs/`: experiment settings; `tests/`: automated checks.
