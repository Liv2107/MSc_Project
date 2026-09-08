"""Verify the methodological corrections against the actual processed dataset.

Read-only. Fails loudly if any invariant is violated, so this can be run as a gate
before committing expensive compute. Each check corresponds to a numbered requirement:

4. container format no longer predicts the class
5. every processed image has the intended spatial dimensions
6. the unseen test set is class balanced for every held-out generator
7. the unseen test membership is identical across every adaptation budget
8. every adaptation budget reloads the original starting checkpoint
9. the external challenge set is balanced, format-neutral, and disjoint from every
   internal split (skipped until the external manifest has been built)

Usage:
    python -m scripts.verify_corrections --manifest data/manifests/tiny_genimage.csv \
        --splits data/manifests/tiny_genimage_splits.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from src.datasets.splitting import load_split_assignments
from src.experiments.unseen_generator import build_balanced_final_test

FIRST_PASS_HELD_OUT = ("biggan", "glide", "stable_diffusion_v1_5", "midjourney")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _record_from_row(row: dict[str, Any], data_root: Path) -> Any:
    from src.datasets.schema import DatasetRecord

    return DatasetRecord(
        row["sample_id"],
        (data_root / row["image_path"]).resolve(),
        int(row["label"]),
        row["generator"],
        row.get("source_group") or None,
        row.get("dataset_source") or None,
    )


def check_format_not_predictive(
    rows: Sequence[dict[str, Any]], data_root: Path, sample_limit: int | None
) -> dict[str, Any]:
    """Container format must be constant across classes, so it cannot predict the label."""

    by_class: dict[int, collections.Counter[str]] = {
        0: collections.Counter(),
        1: collections.Counter(),
    }
    suffixes: dict[int, collections.Counter[str]] = {
        0: collections.Counter(),
        1: collections.Counter(),
    }
    inspected = 0
    for row in rows:
        if sample_limit is not None and inspected >= sample_limit:
            break
        label = int(row["label"])
        path = data_root / row["image_path"]
        with Image.open(path) as image:
            by_class[label][str(image.format)] += 1
        suffixes[label][path.suffix.lower()] += 1
        inspected += 1
    formats_real = set(by_class[0])
    formats_fake = set(by_class[1])
    all_formats = formats_real | formats_fake
    # Predictive iff the two classes use disjoint format sets.
    predictive = bool(formats_real and formats_fake and not (formats_real & formats_fake))
    return {
        "check": "format_not_predictive_of_class",
        "inspected": inspected,
        "real_formats": dict(by_class[0]),
        "fake_formats": dict(by_class[1]),
        "real_suffixes": dict(suffixes[0]),
        "fake_suffixes": dict(suffixes[1]),
        "distinct_formats_overall": sorted(all_formats),
        "format_is_predictive": predictive,
        "passed": (not predictive) and len(all_formats) == 1,
    }


def check_spatial_dimensions(
    rows: Sequence[dict[str, Any]], data_root: Path, expected: int, sample_limit: int | None
) -> dict[str, Any]:
    """Every processed image must carry the intended, generator-independent size."""

    sizes: collections.Counter[str] = collections.Counter()
    by_generator: dict[str, set[str]] = {}
    offenders: list[str] = []
    inspected = 0
    for row in rows:
        if sample_limit is not None and inspected >= sample_limit:
            break
        path = data_root / row["image_path"]
        with Image.open(path) as image:
            size = image.size
        key = f"{size[0]}x{size[1]}"
        sizes[key] += 1
        by_generator.setdefault(row["generator"], set()).add(key)
        if size != (expected, expected):
            offenders.append(f"{row['sample_id']}={key}")
        inspected += 1
    return {
        "check": "spatial_dimensions_normalised",
        "inspected": inspected,
        "expected": f"{expected}x{expected}",
        "observed_sizes": dict(sizes),
        "sizes_per_generator": {name: sorted(v) for name, v in sorted(by_generator.items())},
        "offenders": offenders[:10],
        "passed": not offenders and len(sizes) == 1,
    }


def check_balanced_unseen_tests(
    rows: Sequence[dict[str, Any]], data_root: Path, splits_path: Path
) -> dict[str, Any]:
    """Each held-out generator's final test set must be 50/50 on a shared real pool."""

    records = [_record_from_row(row, data_root) for row in rows]
    split_by_id = {item.sample_id: item.split for item in load_split_assignments(splits_path)}
    per_generator: dict[str, Any] = {}
    real_id_sets: dict[str, frozenset[str]] = {}
    all_passed = True
    for generator in FIRST_PASS_HELD_OUT:
        try:
            selected, metadata = build_balanced_final_test(
                records, split_by_id, unseen_generator=generator
            )
        except ValueError as exc:
            per_generator[generator] = {"error": str(exc), "passed": False}
            all_passed = False
            continue
        fakes = sum(1 for record in selected if record.label == 1)
        reals = sum(1 for record in selected if record.label == 0)
        balanced = fakes == reals
        only_held_out = {record.generator for record in selected} == {generator, "real"}
        real_id_sets[generator] = frozenset(
            record.sample_id for record in selected if record.label == 0
        )
        per_generator[generator] = {
            "total": len(selected),
            "held_out_fakes": fakes,
            "reals": reals,
            "prevalence": metadata["positive_prevalence"],
            "balanced": balanced,
            "only_held_out_generator_fakes": only_held_out,
            "final_test_sha256": metadata["final_test_sha256"],
            "passed": balanced and only_held_out,
        }
        all_passed = all_passed and balanced and only_held_out
    shared_real_pool = len(set(real_id_sets.values())) == 1 if real_id_sets else False
    return {
        "check": "unseen_tests_balanced",
        "per_generator": per_generator,
        "real_pool_identical_across_held_out_generators": shared_real_pool,
        "passed": all_passed,
    }


def check_test_membership_stable_across_budgets(run_dir: Path) -> dict[str, Any]:
    """Every budget in a completed recovery run must score identical test membership."""

    metrics_path = run_dir / "recovery_metrics.json"
    if not metrics_path.is_file():
        return {"check": "test_membership_stable", "skipped": f"no recovery run at {run_dir}"}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    id_sets: dict[str, tuple[str, ...]] = {}
    zero = run_dir / "zero_percent_unseen_test_predictions.csv"
    if zero.is_file():
        id_sets["zero_percent"] = tuple(
            sorted(row["sample_id"] for row in _read_manifest(zero))
        )
    for cell_dir in sorted((run_dir / "cells").glob("*")):
        predictions = cell_dir / "unseen_test_predictions.csv"
        if predictions.is_file():
            id_sets[cell_dir.name] = tuple(
                sorted(row["sample_id"] for row in _read_manifest(predictions))
            )
    distinct = set(id_sets.values())
    return {
        "check": "test_membership_stable",
        "evaluations_compared": len(id_sets),
        "distinct_membership_sets": len(distinct),
        "sample_count": len(next(iter(distinct))) if distinct else 0,
        "declared_composition": metrics.get("final_test_composition"),
        "passed": len(distinct) == 1 and len(id_sets) > 1,
    }


def check_budgets_reload_starting_checkpoint(run_dir: Path) -> dict[str, Any]:
    """Every budget must declare the same original starting checkpoint."""

    metrics_path = run_dir / "recovery_metrics.json"
    if not metrics_path.is_file():
        return {"check": "budgets_reload_starting_checkpoint", "skipped": f"no run at {run_dir}"}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    declared = {
        str(cell.get("starting_checkpoint")) for cell in metrics["cells"]
    }
    per_cell = {
        str(cell["cell_id"]): str(cell.get("starting_checkpoint")) for cell in metrics["cells"]
    }
    return {
        "check": "budgets_reload_starting_checkpoint",
        "distinct_starting_checkpoints": sorted(declared),
        "cells": per_cell,
        "passed": len(declared) == 1,
    }


def check_configs_against_manifest(
    rows: Sequence[dict[str, Any]], data_root: Path, splits_path: Path, config_glob: str
) -> dict[str, Any]:
    """Pre-flight every real config against the manifest before spending compute.

    Catches the failure mode where a config's generator list does not match the manifest
    (a typo, or a shell that did not split a list), which would otherwise waste hours
    training on real images only.
    """

    from src.experiments.unseen_generator import validate_unseen_protocol
    from src.utils.config import load_config

    present = {row["generator"] for row in rows}
    records = [_record_from_row(row, data_root) for row in rows]
    split_by_id = {item.sample_id: item.split for item in load_split_assignments(splits_path)}
    per_config: dict[str, Any] = {}
    all_passed = True
    for path in sorted(Path("configs").glob(config_glob)):
        if path.name.endswith("_base.yaml"):
            # An inheritance base, not a runnable experiment; it has no experiment block.
            continue
        entry: dict[str, Any] = {}
        try:
            loaded = load_config(path)
            generators = loaded.values["generators"]
            declared = set(generators["train"]) | set(generators["validation"])
            if generators.get("unseen"):
                declared.add(generators["unseen"])
            declared.update(generators["test"])
            missing = sorted(declared - present)
            entry["declared_generators"] = sorted(declared)
            entry["missing_from_manifest"] = missing
            experiment_type = loaded.values["experiment"]["type"]
            if experiment_type in {"unseen_generator", "fine_tuning", "ablation"}:
                unseen, known = validate_unseen_protocol(loaded.values)
                selected, metadata = build_balanced_final_test(
                    records, split_by_id, unseen_generator=unseen
                )
                train_fakes = sum(
                    1
                    for record in records
                    if record.label == 1
                    and record.generator in set(known)
                    and split_by_id[record.sample_id] == "train"
                )
                entry["held_out"] = unseen
                entry["known_generator_train_fakes"] = train_fakes
                entry["balanced_test_size"] = len(selected)
                entry["balanced_test_prevalence"] = metadata["positive_prevalence"]
                entry["passed"] = not missing and train_fakes > 0
            else:
                entry["passed"] = not missing
        except Exception as exc:  # surfaced per config, never silently swallowed
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["passed"] = False
        per_config[path.name] = entry
        all_passed = all_passed and bool(entry["passed"])
    return {
        "check": "configs_match_manifest",
        "config_glob": config_glob,
        "per_config": per_config,
        "passed": all_passed,
    }


def check_external_challenge_set(
    internal_rows: Sequence[dict[str, Any]],
    data_root: Path,
    splits_path: Path,
    external_manifest: Path,
    external_audit: Path,
) -> dict[str, Any]:
    """The external evaluation set must be balanced, format-neutral, and fully external.

    This is step 5 of the pre-registered external protocol. It is deliberately separate
    from the internal checks: an external image that turned out to be an internal image,
    or an external set whose container format tracked the class, would make the external
    number measure something other than generator novelty.
    """

    name = "external_challenge_set_balanced_format_neutral_and_disjoint"
    if not external_manifest.is_file():
        return {"check": name, "skipped": f"no external manifest at {external_manifest}"}
    rows = _read_manifest(external_manifest)
    fakes = [row for row in rows if int(row["label"]) == 1]
    reals = [row for row in rows if int(row["label"]) == 0]
    balanced = bool(fakes) and len(fakes) == len(reals)

    formats: dict[int, collections.Counter[str]] = {
        0: collections.Counter(),
        1: collections.Counter(),
    }
    sizes: collections.Counter[str] = collections.Counter()
    for row in rows:
        path = data_root / row["image_path"]
        with Image.open(path) as image:
            formats[int(row["label"])][str(image.format)] += 1
            sizes[f"{image.size[0]}x{image.size[1]}"] += 1
    real_formats, fake_formats = set(formats[0]), set(formats[1])
    format_predictive = bool(
        real_formats and fake_formats and not (real_formats & fake_formats)
    )

    split_by_id = {item.sample_id: item.split for item in load_split_assignments(splits_path)}
    internal_ids = {str(row["sample_id"]) for row in internal_rows}
    fake_ids = {str(row["sample_id"]) for row in fakes}
    fake_id_overlap = sorted(fake_ids & internal_ids)
    fake_paths = {str(row["image_path"]) for row in fakes}
    internal_paths = {str(row["image_path"]) for row in internal_rows}
    fake_path_overlap = sorted(fake_paths & internal_paths)

    # The authentic comparators are internal images by design: the protocol draws them
    # from the held-out real pool. What must hold is that every one of them sits in the
    # test split, so the detectors never trained on any of them.
    comparator_splits = collections.Counter(
        split_by_id.get(str(row["sample_id"]), "not_in_internal_manifest") for row in reals
    )
    comparators_all_held_out = set(comparator_splits) == {"test"}

    audit_checks: list[dict[str, Any]] = []
    audit_flags: dict[str, Any] = {}
    if external_audit.is_file():
        audit = json.loads(external_audit.read_text(encoding="utf-8"))
        audit_checks = list(audit.get("leakage_checks", []))
        audit_flags = {
            key: audit.get(key)
            for key in (
                "used_for_training",
                "used_for_validation",
                "used_for_threshold_selection",
                "generator_identity_known",
            )
        }
    audit_passed = all(bool(item.get("passed")) for item in audit_checks) if audit_checks else False
    never_developed = all(
        audit_flags.get(key) is False
        for key in ("used_for_training", "used_for_validation", "used_for_threshold_selection")
    )

    return {
        "check": name,
        "external_manifest": str(external_manifest),
        "total": len(rows),
        "external_fakes": len(fakes),
        "authentic_comparators": len(reals),
        "balanced": balanced,
        "real_formats": dict(formats[0]),
        "fake_formats": dict(formats[1]),
        "format_is_predictive": format_predictive,
        "observed_sizes": dict(sizes),
        "single_spatial_size": len(sizes) == 1,
        "external_fake_sample_id_overlap_with_internal": fake_id_overlap[:10],
        "external_fake_image_path_overlap_with_internal": fake_path_overlap[:10],
        "comparator_internal_splits": dict(comparator_splits),
        "comparators_all_from_held_out_test_split": comparators_all_held_out,
        "build_audit_leakage_checks_passed": audit_passed,
        "build_audit_development_use_flags": audit_flags,
        "passed": bool(
            balanced
            and not format_predictive
            and len(sizes) == 1
            and not fake_id_overlap
            and not fake_path_overlap
            and comparators_all_held_out
            and audit_passed
            and never_developed
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/tiny_genimage.csv"))
    parser.add_argument(
        "--splits", type=Path, default=Path("data/manifests/tiny_genimage_splits.csv")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--expected-size", type=int, default=256)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Inspect only the first N images for the pixel-level checks (omit for all).",
    )
    parser.add_argument(
        "--config-glob",
        type=str,
        default="tiny_*.yaml",
        help="Pre-flight configs matching this glob under configs/ against the manifest.",
    )
    parser.add_argument(
        "--recovery-run",
        type=Path,
        default=None,
        help="Completed fine_tuning run directory for checks 7 and 8.",
    )
    parser.add_argument(
        "--external-manifest",
        type=Path,
        default=Path("data/manifests/external_challenge_v1.csv"),
        help="External challenge manifest for check 9; skipped when absent.",
    )
    parser.add_argument(
        "--external-audit",
        type=Path,
        default=Path("data/manifests/external_challenge_v1.audit.json"),
        help="Audit record written when the external manifest was built.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = _read_manifest(args.manifest)
    results = [
        check_format_not_predictive(rows, args.data_root, args.sample_limit),
        check_spatial_dimensions(rows, args.data_root, args.expected_size, args.sample_limit),
        check_balanced_unseen_tests(rows, args.data_root, args.splits),
        check_configs_against_manifest(rows, args.data_root, args.splits, args.config_glob),
    ]
    if args.recovery_run is not None:
        results.append(check_test_membership_stable_across_budgets(args.recovery_run))
        results.append(check_budgets_reload_starting_checkpoint(args.recovery_run))
    results.append(
        check_external_challenge_set(
            rows,
            args.data_root,
            args.splits,
            args.external_manifest,
            args.external_audit,
        )
    )

    print(json.dumps(results, indent=2, sort_keys=True, default=str))
    failed = [item["check"] for item in results if item.get("passed") is False]
    skipped = [item["check"] for item in results if "skipped" in item]
    print()
    for item in results:
        state = "SKIP" if "skipped" in item else ("PASS" if item.get("passed") else "FAIL")
        print(f"  {state}  {item['check']}")
    if skipped:
        print(f"\nSkipped (needs a completed run): {', '.join(skipped)}")
    if failed:
        raise SystemExit(f"FAILED checks: {', '.join(failed)}")


if __name__ == "__main__":
    main()
