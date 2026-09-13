# Experiment design notes

These notes summarise the rationale and decision rules previously recorded in the
experiment configs. They describe those design choices, not new acceptance criteria.

## Held-out generators

BigGAN provided an initial recovery experiment, but its small in-distribution gap
left little room to measure recovery. VQDM was selected to examine a larger failure:
the linear detector recorded ROC-AUC 0.9664 in distribution and 0.6746 on VQDM.
The VQDM recovery and ablation runs use that detector's checkpoint.

## Linear and cosine heads

VQDM's observed failure motivated the cosine head. It is therefore a development
condition, even though the generator is excluded from model training and validation.
Wukong was designated as the confirmatory comparison before its paired runs.

The original hypothesis was that embedding magnitude reflected generator-specific
resolution, compression or texture statistics. The cosine head normalises the
embedding and classifier weight, then learns a scale and bias. The embedding-norm
analysis later found little variation between generators, weakening the magnitude
explanation. Effects of weight normalisation and scale remain possible explanations.

The VQDM config recorded these decision rules before the run:

| Outcome | Recorded criterion |
|---|---|
| Convincing | ROC-AUC degradation ≤ 0.15, unseen ROC-AUC ≥ 0.82, in-distribution within about 0.01 of 0.9664 |
| Inconclusive | Unseen ROC-AUC improvement < 0.05, or an in-distribution drop |
| Failure | Degradation ≥ 0.29, or in-distribution degradation |

The original criteria overlap on in-distribution degradation. The meeting notebook's
classifier applies its failure branch before its inconclusive fallback. These are
project decision rules for a single-seed comparison, not significance tests.

The Wukong prediction was a smaller in-distribution-to-unseen ROC-AUC drop for the
cosine head, with in-distribution ROC-AUC within about 0.01 of the linear head.
If the linear head barely degrades, the comparison provides little evidence about
recovery from a large zero-shot failure. Stable Diffusion v1.5 remains in training,
so related-generator exposure limits what this comparison says about novel architectures.

Both head comparisons retain the linear run's generators, manifest, splits, seeds,
epochs, learning rate and threshold policy. Linear and cosine heads have 769 and
770 trainable parameters respectively. Learning rate was not retuned for cosine.

## Fine-tuning depth and learning rate

Each Tiny GenImage ablation uses three depths, four budgets, one subset seed and
one training seed: 12 adaptation fits, plus the unadapted reference. The starting
checkpoint, nested subsets and final test set are shared within a generator.

A three-epoch BigGAN probe at the 5% budget recorded these validation F1 trajectories:

| Depth | Learning rate | Validation F1 by epoch |
|---|---|---|
| Last block | 0.001 | 0.59, 0.87, 0.86 |
| Last block | 0.00001 | 0.81, 0.69, 0.81 |
| Full | 0.001 | 0.00, 0.00, 0.00 |
| Full | 0.00001 | 0.95, 0.97, 1.00 |

The configs consequently keep 0.001 for head-only and use 0.00001 for the deeper
modes. VQDM inherits these rates without a new probe. Learning rate therefore
varies with depth; the comparison does not isolate depth alone. Head-only cells
also reproduce the standalone recovery conditions, providing a consistency check.
