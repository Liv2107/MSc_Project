"""Assemble the external contemporary-generator challenge manifest.

Step 4 of the pre-registered protocol in section 15.5 of ``RESULTS_NOTES.md``. Reads the
generated images and the provenance CSV written at generation time, validates every file,
preprocesses them through the same pinned policy as the internal benchmark, pairs them
with authentic comparators drawn by protocol from the held-out real test pool, and writes
a manifest in the existing dataset schema plus an audit record.

Evaluation only. No split file is written and nothing here can place an external image in
a training, validation, or threshold-selection selection.

Usage:
    python -m scripts.build_external_manifest
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.datasets.external_challenge import (
    build_external_challenge,
    write_external_challenge,
)
from src.datasets.preprocessing import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_TARGET_SIZE,
    PreprocessingPolicy,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dir", type=Path, default=Path("data/external/codex_generated/raw")
    )
    parser.add_argument(
        "--provenance-csv", type=Path, default=Path("data/external/codex_generated/manifest.csv")
    )
    parser.add_argument(
        "--internal-manifest", type=Path, default=Path("data/manifests/tiny_genimage.csv")
    )
    parser.add_argument(
        "--internal-splits", type=Path, default=Path("data/manifests/tiny_genimage_splits.csv")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/processed/external_challenge_v1_cache")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/external_challenge_v1.csv")
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("data/manifests/external_challenge_v1.audit.json")
    )
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument(
        "--near-duplicate-scan",
        choices=("all", "test", "none"),
        default="all",
        help="Scope of the near-duplicate sweep against internal images.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    policy = PreprocessingPolicy(target_size=args.target_size, jpeg_quality=args.jpeg_quality)
    build = build_external_challenge(
        generated_dir=args.generated_dir,
        provenance_csv=args.provenance_csv,
        internal_manifest=args.internal_manifest,
        internal_splits=args.internal_splits,
        data_root=args.data_root,
        cache_root=args.cache_root,
        policy=policy,
        near_duplicate_scan=args.near_duplicate_scan,
    )
    write_external_challenge(build, manifest_path=args.manifest, audit_path=args.audit)
    counts = build.audit["sample_counts"]
    print(
        f"External challenge set: {counts['total']} images "
        f"({counts['external_fake']} generated fake + "
        f"{counts['authentic_comparator']} authentic comparator)"
    )
    print(f"Generator recorded as: {build.audit['generator_recorded_as']} (identity unknown)")
    print(f"Exclusions: {len(build.excluded)}")
    for check in build.audit["leakage_checks"]:
        print(f"  {'PASS' if check['passed'] else 'FAIL'}  {check['check']}")
    print(f"Manifest: {args.manifest}")
    print(f"Audit:    {args.audit}")
    print(
        json.dumps(
            {
                "class_balance": build.audit["class_balance"],
                "sample_size_tier": build.audit["sample_size_tier"],
                "final_evaluation_composition": build.audit["final_evaluation_composition"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
