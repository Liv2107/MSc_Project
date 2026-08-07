# Data directory

## Purpose

Store local manifests and images here while keeping them outside version control. The source code must not assume a particular vendor's folder layout; convert source metadata into the common manifest described in the root README.

## Suggested eventual layout

```text
data/
├── raw/              # Immutable downloaded or collected files
├── manifests/        # Versioned CSV/Parquet metadata and saved split assignments
└── processed/        # Derived files only when preprocessing cannot be done on load
```

Document licences, checksums, acquisition dates, and any excluded/corrupt records. Never modify raw data in place, and never use the unseen-generator test partition to make cleaning decisions that depend on labels or model behaviour.

## Implementation checklist

- [x] Document the selected GenImage licence and allowed uses in `GENIMAGE.md`.
- [x] Define the canonical manifest columns and label convention.
- [x] Require provenance and group identifiers.
- [x] Validate file paths, exact duplicates, and corrupt images.
- [x] Save deterministic split assignments with an audit summary and manifest hash.
