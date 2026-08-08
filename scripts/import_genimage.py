"""Create this project's manifest and splits from an extracted GenImage release.

Supports both the full official release and the reduced Tiny GenImage subset
(``--tiny-genimage``), which ships seven generator folders under their official archive
names and excludes Stable Diffusion v1.4.

With ``--preprocess`` the importer also materialises a deterministic re-encoded cache
(see ``src/datasets/preprocessing.py``) and points the manifest at it. That is required
for the primary experiment on Tiny GenImage, where container format and native
resolution would otherwise separate the classes without any generative evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.datasets.genimage import (
    DATASET_SOURCE,
    TINY_GENIMAGE_DATASET_SOURCE,
    build_genimage_import,
    write_genimage_import,
)
from src.datasets.preprocessing import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_TARGET_SIZE,
    PreprocessingPolicy,
    build_preprocessed_cache,
    rewrite_manifest_to_cache,
)


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
    parser.add_argument(
        "--tiny-genimage",
        action="store_true",
        help="Record provenance as the reduced Tiny GenImage subset rather than full GenImage.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Build the deterministic re-encoded cache and point the manifest at it.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/processed/genimage_cache"),
        help="Destination for the preprocessed cache; must sit beneath --data-root.",
    )
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_source = TINY_GENIMAGE_DATASET_SOURCE if args.tiny_genimage else DATASET_SOURCE
    result = build_genimage_import(
        genimage_root=args.genimage_root,
        data_root=args.data_root,
        generators=args.generators,
        validation_fraction_of_official_val=args.validation_fraction_of_official_val,
        seed=args.seed,
        max_per_generator_split_class=args.max_per_generator_split_class,
        dataset_source=dataset_source,
    )
    rows = result.rows
    if args.preprocess:
        policy = PreprocessingPolicy(
            target_size=args.target_size, jpeg_quality=args.jpeg_quality
        )
        print(
            f"Preprocessing {len(rows)} images to {policy.target_size}x{policy.target_size} "
            f"JPEG q{policy.jpeg_quality} (policy {policy.identity()})..."
        )
        cache = build_preprocessed_cache(
            manifest_rows=rows,
            data_root=args.data_root,
            cache_root=args.cache_root,
            policy=policy,
        )
        print(f"  written={cache.processed} reused={cache.skipped} index={cache.index_path}")
        rows = rewrite_manifest_to_cache(
            manifest_rows=rows, cache_root=args.cache_root, data_root=args.data_root
        )
        result.audit["preprocessing"] = policy.describe()
        result.audit["preprocessing_index"] = str(cache.index_path)
        result.audit["preprocessed_cache_root"] = str(args.cache_root.resolve())

    written = type(result)(rows=rows, assignments=result.assignments, audit=result.audit)
    write_genimage_import(written, manifest_path=args.manifest, split_path=args.splits)
    print(
        f"Indexed {result.audit['sample_count']} GenImage files across "
        f"{len(result.audit['generators'])} generators."
    )
    print(f"Dataset source: {dataset_source}")
    if result.audit.get("excluded_official_generators"):
        excluded = ", ".join(result.audit["excluded_official_generators"])
        print(f"Absent official generators (NOT substituted): {excluded}")
    print(f"Manifest: {args.manifest}")
    print(f"Splits:   {args.splits}")


if __name__ == "__main__":
    main()
