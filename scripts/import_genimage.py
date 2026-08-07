"""Create this project's manifest and splits from an extracted GenImage release."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.datasets.genimage import build_genimage_import, write_genimage_import


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genimage-root", type=Path, default=Path("data/raw/genimage"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/genimage.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/manifests/genimage_splits.csv"))
    parser.add_argument("--generators", nargs="+", default=None)
    parser.add_argument("--validation-fraction-of-official-val", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-generator-split-class",
        type=int,
        default=None,
        help="Optional deterministic development cap; omit for the full research dataset.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_genimage_import(
        genimage_root=args.genimage_root,
        data_root=args.data_root,
        generators=args.generators,
        validation_fraction_of_official_val=args.validation_fraction_of_official_val,
        seed=args.seed,
        max_per_generator_split_class=args.max_per_generator_split_class,
    )
    write_genimage_import(result, manifest_path=args.manifest, split_path=args.splits)
    print(
        f"Indexed {result.audit['sample_count']} GenImage files across "
        f"{len(result.audit['generators'])} generators."
    )
    print(f"Manifest: {args.manifest}")
    print(f"Splits:   {args.splits}")


if __name__ == "__main__":
    main()
