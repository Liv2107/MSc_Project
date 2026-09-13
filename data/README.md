# Data

Keep downloaded images in `raw/`, manifests and saved split assignments in
`manifests/`, and derived image caches in `processed/`. These local files are
excluded from version control.

See [GENIMAGE.md](GENIMAGE.md) for the source layout and acquisition details,
and the root README for Tiny GenImage import commands. For other datasets,
start with [manifest_template.csv](manifest_template.csv).

Record licences, acquisition dates, checksums and exclusions. Keep raw images
unchanged. The importer validates files, deduplicates repeated real images and
writes a split audit. Source groups prevent related records crossing partitions;
missing prompt or original-image provenance remains a limitation.
