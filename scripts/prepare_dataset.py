"""Validate a manifest, create leakage-aware splits, and save an audit summary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from PIL import Image

from src.datasets.detector_dataset import AIDetectionDataset
from src.datasets.splitting import SplitFractions, create_grouped_splits, save_split_assignments


def _metadata_only_transform(image: Image.Image) -> NoReturn:
    del image
    raise RuntimeError("dataset preparation does not decode samples through a transform")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/dataset.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/splits.csv"))
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = AIDetectionDataset.from_manifest(
        args.manifest,
        data_root=args.data_root,
        transform=_metadata_only_transform,
    )
    fractions = SplitFractions(args.train_fraction, args.validation_fraction, args.test_fraction)
    assignments = create_grouped_splits(dataset.records, fractions, seed=args.seed)
    save_split_assignments(assignments, args.output)
    split_by_id = {item.sample_id: item.split for item in assignments}
    counts = Counter(
        (split_by_id[record.sample_id], str(record.label), record.generator)
        for record in dataset.records
    )
    audit = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": dataset.manifest_sha256,
        "seed": args.seed,
        "fractions": {
            "train": fractions.train,
            "validation": fractions.validation,
            "test": fractions.test,
        },
        "sample_count": len(dataset),
        "counts": [
            {"split": split, "label": int(label), "generator": generator, "count": count}
            for (split, label, generator), count in sorted(counts.items())
        ],
    }
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Validated {len(dataset)} samples and wrote {args.output}")


if __name__ == "__main__":
    main()
