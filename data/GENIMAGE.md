# GenImage integration

GenImage is the selected source dataset for the initial detector experiments.
The official project describes more than one million real/fake pairs covering
Midjourney, Stable Diffusion 1.4 and 1.5, ADM, GLIDE, Wukong, VQDM, and BigGAN.

Official sources:

- Project and dataset terms: https://genimage-dataset.github.io/
- Official repository and folder layout: https://github.com/GenImage-Dataset/GenImage
- Paper: https://arxiv.org/abs/2306.08571
- Google Drive mirror linked by the official repository:
  https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS

## Licence and acquisition

The project website states that the dataset is provided under CC BY-NC-SA 4.0
plus additional Dataset Terms and is restricted to non-commercial purposes such
as academic research, teaching, and scientific publication. Read and retain a
copy of the applicable terms before downloading. Do not commit the images.

Download the required archives manually from an official link and extract them
under `data/raw/genimage/` without renaming the generator folders. The expected
layout is:

```text
data/raw/genimage/
├── Midjourney/{train,val}/{ai,nature}/...
├── Stable Diffusion V1.4/{train,val}/{ai,nature}/...
├── Stable Diffusion V1.5/{train,val}/{ai,nature}/...
├── ADM/{train,val}/{ai,nature}/...
├── GLIDE/{train,val}/{ai,nature}/...
├── Wukong/{train,val}/{ai,nature}/...
├── VQDM/{train,val}/{ai,nature}/...
└── BigGAN/{train,val}/{ai,nature}/...
```

## Import

Create the canonical manifest and persisted split assignments:

```powershell
python -m scripts.import_genimage
```

The importer:

- maps folder names to stable lowercase generator identifiers;
- preserves official `train` images as the training partition;
- splits official `val` images 50/50 into validation and untouched test data;
- keeps provenance and official split columns in the manifest;
- removes repeated logical ImageNet nature files across generator folders only
  after confirming that their contents match;
- refuses source groups that cross the official train/validation boundary; and
- writes `genimage.audit.json` with generator, class, and split counts.

For a quick development-only subset, use a deterministic cap:

```powershell
python -m scripts.import_genimage --max-per-generator-split-class 1000
```

Do not report capped-subset results as full-dataset results. Delete or move the
generated manifest and split files before changing the cap because the importer
refuses accidental overwrite.

Run the in-distribution baseline with:

```powershell
python main.py --config configs/genimage_baseline.yaml
```

The extracted release does not expose a universal prompt/original identifier for
all generated images. The importer therefore uses stable relative file identity as
the generated-image source group while preserving the official split. This
provenance limitation must be stated when discussing near-duplicate leakage.
