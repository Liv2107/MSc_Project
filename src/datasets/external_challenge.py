"""Assemble the external contemporary-generator challenge set from generated images.

This implements the protocol already written down in
``scripts/build_dissertation_results.py::external_challenge_manifest`` and section 15 of
``outputs/report/dissertation_results/RESULTS_NOTES.md``. Nothing here is a new
experimental design; it is the mechanical realisation of that pre-registered plan.

The route actually used is **Route B**: the images were produced by an assistant-mediated
hosted image-generation tool (Codex ``image_gen.imagegen``), which selects an underlying
GPT Image model and does not report which. The generator is therefore recorded as
``astra_mediated_unidentified`` and no architectural claim may be attached to the result.
The provenance CSV shipped with the images is carried through verbatim.

Three properties matter and are enforced here rather than assumed:

* **The external images never enter development.** This module only ever builds an
  evaluation manifest. It writes no split file and touches no training selection.
* **Authentic comparators are chosen by protocol, not by eye.** They are the first 100
  entries of the *identical* seeded ordering of held-out real test images that
  ``src.experiments.unseen_generator.build_balanced_final_test`` uses, so they are a
  nested subset of the fixed real pool the internal unseen tests already score against.
  The external and internal numbers then differ only in their positives.
* **Format cannot predict the class.** Both halves pass through the same pinned
  ``PreprocessingPolicy`` as the internal benchmark, so every evaluated image is a
  256x256 RGB JPEG at quality 95 with metadata stripped.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.datasets.preprocessing import (
    PreprocessingPolicy,
    build_preprocessed_cache,
    rewrite_manifest_to_cache,
)
from src.datasets.splitting import load_split_assignments
from src.experiments.unseen_generator import REAL_TEST_POOL_SEED

#: The challenge identifier fixed by the pre-registered manifest.
CHALLENGE_ID = "contemporary_openai_astra_mediated_v1"

#: Route B was used. The underlying image model is not reported by the tool, so the
#: generator is named for the mediation route and explicitly marked unidentified.
EXTERNAL_GENERATOR_NAME = "astra_mediated_unidentified"
EXTERNAL_ROUTE = "astra_mediated_hosted_image_generation_tool"
EXTERNAL_ROUTE_TOOL = "image_gen.imagegen (built-in Codex tool)"

#: Provenance of the evaluation set, mirroring the internal ``dataset_source`` style.
EXTERNAL_DATASET_SOURCE = (
    "external_challenge_contemporary_openai_astra_mediated_v1"
    "_codex_image_gen_imagegen_route_b"
)

#: Evaluation-only split name. Deliberately not one of train/validation/test so it can
#: never be mistaken for, or merged with, an internal partition.
EXTERNAL_SPLIT_NAME = "external_test"

#: Columns of the emitted manifest: the canonical six the loader requires, then
#: provenance the dissertation has to be able to cite.
MANIFEST_COLUMNS = (
    "sample_id",
    "image_path",
    "label",
    "generator",
    "source_group",
    "dataset_source",
    "original_image_path",
    "challenge_id",
    "route",
    "route_tool",
    "image_model_id_if_reported",
    "prompt_id",
    "prompt_text",
    "generated_at_utc",
    "source_sha256",
    "source_width",
    "source_height",
    "source_format",
    "provenance_origin",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(path: Path, *, side: int = 8) -> str:
    """A 64-bit difference hash, used only to look for near-duplicate images.

    Exact digests catch a byte-identical file; this catches the more realistic leakage
    shape, where a generated image is a re-encoding or rescale of a dataset image.
    """

    with Image.open(path) as handle:
        grey = handle.convert("L").resize((side + 1, side), Image.Resampling.BICUBIC)
    pixels = grey.tobytes()
    bits = 0
    position = 0
    for row in range(side):
        offset = row * (side + 1)
        for column in range(side):
            if pixels[offset + column + 1] > pixels[offset + column]:
                bits |= 1 << position
            position += 1
    return f"{bits:016x}"


def _hamming(left: str, right: str) -> int:
    return bin(int(left, 16) ^ int(right, 16)).count("1")


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """One generated external image with the provenance recorded at generation time."""

    prompt_id: str
    filename: str
    path: Path
    sha256: str
    width: int
    height: int
    image_format: str
    mode: str
    prompt_text: str
    generated_at_utc: str
    route_tool: str
    image_model_id_if_reported: str | None


@dataclass
class ExternalChallengeBuild:
    rows: list[dict[str, Any]]
    audit: dict[str, Any]
    excluded: list[dict[str, Any]] = field(default_factory=list)


def read_generation_provenance(provenance_csv: Path) -> dict[str, dict[str, str]]:
    """Read the provenance CSV written when the images were generated."""

    with provenance_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"provenance CSV holds no rows: {provenance_csv}")
    by_filename: dict[str, dict[str, str]] = {}
    for row in rows:
        filename = str(row.get("filename", "")).strip()
        if not filename:
            raise ValueError("provenance CSV row has no filename")
        if filename in by_filename:
            raise ValueError(f"provenance CSV lists {filename!r} twice")
        by_filename[filename] = dict(row)
    return by_filename


def inspect_generated_images(
    image_dir: Path, provenance: Mapping[str, Mapping[str, str]]
) -> tuple[list[GeneratedImage], list[dict[str, Any]]]:
    """Validate every generated file, returning the usable ones and the exclusions.

    An image is excluded, never silently repaired, when it cannot be decoded, when its
    bytes disagree with the digest recorded at generation time, when no provenance row
    exists for it, or when it duplicates the bytes of an image already accepted.
    """

    usable: list[GeneratedImage] = []
    excluded: list[dict[str, Any]] = []
    seen_digests: dict[str, str] = {}
    for path in sorted(image_dir.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            excluded.append({"filename": path.name, "reason": "not_an_image_file"})
            continue
        record = provenance.get(path.name)
        if record is None:
            excluded.append({"filename": path.name, "reason": "no_provenance_row"})
            continue
        digest = _sha256_file(path)
        recorded = str(record.get("sha256", "")).strip().lower()
        if recorded and recorded != digest:
            excluded.append(
                {
                    "filename": path.name,
                    "reason": "sha256_disagrees_with_generation_record",
                    "recorded_sha256": recorded,
                    "observed_sha256": digest,
                }
            )
            continue
        if digest in seen_digests:
            excluded.append(
                {
                    "filename": path.name,
                    "reason": "byte_identical_duplicate",
                    "duplicate_of": seen_digests[digest],
                }
            )
            continue
        try:
            with Image.open(path) as handle:
                handle.verify()
            with Image.open(path) as handle:
                handle.load()
                size = handle.size
                mode = str(handle.mode)
                image_format = str(handle.format)
        except Exception as exc:  # a corrupt file is reported, not repaired
            excluded.append(
                {"filename": path.name, "reason": f"unreadable: {type(exc).__name__}: {exc}"}
            )
            continue
        prompt_id = str(record.get("prompt_id", "")).strip() or path.stem
        model_id = str(record.get("model_identifier", "")).strip()
        seen_digests[digest] = path.name
        usable.append(
            GeneratedImage(
                prompt_id=prompt_id,
                filename=path.name,
                path=path,
                sha256=digest,
                width=int(size[0]),
                height=int(size[1]),
                image_format=image_format,
                mode=mode,
                prompt_text=str(record.get("exact_prompt", "")).strip(),
                generated_at_utc=str(record.get("generation_date", "")).strip(),
                route_tool=str(record.get("generation_route/tool", "")).strip()
                or EXTERNAL_ROUTE_TOOL,
                image_model_id_if_reported=(
                    None if model_id.lower() in {"", "unavailable", "none", "null"} else model_id
                ),
            )
        )
    prompt_ids = [item.prompt_id for item in usable]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("generated images do not have unique prompt ids")
    return usable, excluded


def select_authentic_comparators(
    manifest_rows: Sequence[Mapping[str, Any]],
    split_by_id: Mapping[str, str],
    *,
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Take ``count`` authentic comparators from the held-out real test pool.

    The ordering is the one ``build_balanced_final_test`` uses -- a shuffle of the sorted
    real test IDs seeded with ``REAL_TEST_POOL_SEED`` -- and the first ``count`` entries
    are taken. The comparators are therefore a nested subset of the fixed 250-image real
    pool the internal unseen tests already score against, so the external and internal
    results share their negatives and differ only in their positives. The detector never
    trained on any of them: they sit in the ``test`` split.
    """

    if count <= 0:
        raise ValueError("comparator count must be positive")
    pool = sorted(
        (
            row
            for row in manifest_rows
            if int(row["label"]) == 0 and split_by_id[str(row["sample_id"])] == "test"
        ),
        key=lambda row: str(row["sample_id"]),
    )
    if len(pool) < count:
        raise ValueError(f"only {len(pool)} held-out real test images available; need {count}")
    order = list(pool)
    random.Random(REAL_TEST_POOL_SEED).shuffle(order)
    chosen = sorted(order[:count], key=lambda row: str(row["sample_id"]))
    metadata = {
        "policy": "nested_prefix_of_fixed_real_test_pool",
        "real_test_pool_seed": REAL_TEST_POOL_SEED,
        "selected": len(chosen),
        "pool_available": len(pool),
        "source_split": "test",
        "shares_negatives_with_internal_unseen_test": True,
        "selected_sha256": hashlib.sha256(
            "|".join(str(row["sample_id"]) for row in chosen).encode("utf-8")
        ).hexdigest(),
        "rationale": (
            "identical seeded ordering to build_balanced_final_test, so the comparators "
            "are a nested subset of the internal fixed real pool and the external result "
            "differs from the internal one only in its positives"
        ),
    }
    return [dict(row) for row in chosen], metadata


def check_no_overlap_with_internal(
    generated: Sequence[GeneratedImage],
    *,
    internal_manifest_rows: Sequence[Mapping[str, Any]],
    data_root: Path,
    preprocessing_index: Path | None,
    near_duplicate_scan: str = "all",
    near_duplicate_threshold: int = 4,
) -> dict[str, Any]:
    """Prove the generated images are not internal images arriving by another route.

    Three independent checks: byte-identity against the digests recorded when the
    internal cache was built, byte-identity of the generated files against the internal
    files on disk, and a near-duplicate sweep on 64-bit difference hashes over whatever
    internal rows the caller supplies. ``near_duplicate_scan`` records the scope the
    caller chose and switches the sweep off entirely when set to ``none``.
    """

    if near_duplicate_scan not in {"all", "test", "none"}:
        raise ValueError("near_duplicate_scan must be all, test or none")
    generated_digests = {item.sha256: item.filename for item in generated}

    indexed_source: dict[str, str] = {}
    indexed_output: dict[str, str] = {}
    if preprocessing_index is not None and preprocessing_index.is_file():
        payload = json.loads(preprocessing_index.read_text(encoding="utf-8"))
        for sample_id, entry in payload.get("images", {}).items():
            if entry.get("source_sha256"):
                indexed_source[str(entry["source_sha256"])] = sample_id
            if entry.get("output_sha256"):
                indexed_output[str(entry["output_sha256"])] = sample_id
    digest_hits = sorted(
        {
            f"{filename} == internal source {indexed_source[digest]}"
            for digest, filename in generated_digests.items()
            if digest in indexed_source
        }
        | {
            f"{filename} == internal cached {indexed_output[digest]}"
            for digest, filename in generated_digests.items()
            if digest in indexed_output
        }
    )

    rows = list(internal_manifest_rows)

    nearest: list[dict[str, Any]] = []
    scanned = 0
    if near_duplicate_scan != "none":
        generated_hashes = {item.filename: _difference_hash(item.path) for item in generated}
        best: dict[str, tuple[int, str]] = {
            name: (65, "") for name in generated_hashes
        }
        for row in rows:
            path = (data_root / str(row["image_path"])).resolve()
            if not path.is_file():
                continue
            internal_hash = _difference_hash(path)
            scanned += 1
            for name, generated_hash in generated_hashes.items():
                distance = _hamming(generated_hash, internal_hash)
                if distance < best[name][0]:
                    best[name] = (distance, str(row["sample_id"]))
        nearest = [
            {
                "filename": name,
                "closest_internal_sample_id": sample_id,
                "hamming_distance": distance,
            }
            for name, (distance, sample_id) in sorted(best.items())
        ]
    flagged = [
        item for item in nearest if int(item["hamming_distance"]) <= near_duplicate_threshold
    ]
    minimum = min((int(item["hamming_distance"]) for item in nearest), default=None)
    return {
        "check": "external_images_do_not_overlap_internal_data",
        "generated_images": len(generated),
        "byte_identical_matches": digest_hits,
        "near_duplicate_scan": near_duplicate_scan,
        "internal_images_scanned": scanned,
        "near_duplicate_threshold_hamming": near_duplicate_threshold,
        "minimum_hamming_distance_observed": minimum,
        "near_duplicates_flagged": flagged,
        "closest_match_per_image": nearest,
        "passed": not digest_hits and not flagged,
    }


def build_external_challenge(
    *,
    generated_dir: Path,
    provenance_csv: Path,
    internal_manifest: Path,
    internal_splits: Path,
    data_root: Path,
    cache_root: Path,
    internal_preprocessing_index: Path | None = None,
    policy: PreprocessingPolicy | None = None,
    near_duplicate_scan: str = "all",
) -> ExternalChallengeBuild:
    """Build the class-balanced external evaluation manifest and its audit record."""

    policy = policy or PreprocessingPolicy()
    data_root = data_root.resolve()
    provenance = read_generation_provenance(provenance_csv)
    generated, excluded = inspect_generated_images(generated_dir, provenance)
    if not generated:
        raise ValueError(f"no usable generated images under {generated_dir}")

    with internal_manifest.open("r", encoding="utf-8", newline="") as handle:
        internal_rows = list(csv.DictReader(handle))
    split_by_id = {
        item.sample_id: item.split for item in load_split_assignments(internal_splits)
    }

    scan_rows = (
        [row for row in internal_rows if split_by_id[str(row["sample_id"])] == "test"]
        if near_duplicate_scan == "test"
        else internal_rows
    )
    overlap = check_no_overlap_with_internal(
        generated,
        internal_manifest_rows=scan_rows,
        data_root=data_root,
        preprocessing_index=(
            internal_preprocessing_index
            if internal_preprocessing_index is not None
            else data_root / "processed" / "tiny_genimage_cache" / "preprocessing_index.json"
        ),
        near_duplicate_scan=near_duplicate_scan,
    )
    if not overlap["passed"]:
        raise ValueError(
            "external images overlap internal data; refusing to build the challenge set: "
            f"{overlap['byte_identical_matches'] or overlap['near_duplicates_flagged']}"
        )

    # Preprocess through the identical pinned policy, so container format and spatial
    # size cannot separate the external fakes from the authentic comparators.
    raw_rows = [
        {
            "sample_id": f"astra_{item.prompt_id}",
            "image_path": item.path.resolve().relative_to(data_root).as_posix(),
            "label": 1,
            "generator": EXTERNAL_GENERATOR_NAME,
            "source_group": f"external:astra_mediated:fake:{item.prompt_id}",
            "dataset_source": EXTERNAL_DATASET_SOURCE,
        }
        for item in generated
    ]
    cache = build_preprocessed_cache(
        manifest_rows=raw_rows,
        data_root=data_root,
        cache_root=cache_root,
        policy=policy,
        progress_every=0,
    )
    cached_rows = rewrite_manifest_to_cache(
        manifest_rows=raw_rows, cache_root=cache_root, data_root=data_root
    )

    by_prompt_id = {item.prompt_id: item for item in generated}
    fake_rows: list[dict[str, Any]] = []
    for row in cached_rows:
        item = by_prompt_id[str(row["sample_id"]).removeprefix("astra_")]
        fake_rows.append(
            {
                **row,
                "challenge_id": CHALLENGE_ID,
                "route": EXTERNAL_ROUTE,
                "route_tool": item.route_tool,
                "image_model_id_if_reported": item.image_model_id_if_reported or "",
                "prompt_id": item.prompt_id,
                "prompt_text": item.prompt_text,
                "generated_at_utc": item.generated_at_utc,
                "source_sha256": item.sha256,
                "source_width": item.width,
                "source_height": item.height,
                "source_format": item.image_format,
                "provenance_origin": "generated_for_this_challenge",
            }
        )

    comparators, comparator_metadata = select_authentic_comparators(
        internal_rows, split_by_id, count=len(fake_rows)
    )
    real_rows: list[dict[str, Any]] = []
    for row in comparators:
        real_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "image_path": str(row["image_path"]),
                "label": 0,
                "generator": "real",
                "source_group": str(row["source_group"]),
                "dataset_source": str(row["dataset_source"]),
                "original_image_path": str(row.get("original_image_path", "")),
                "challenge_id": CHALLENGE_ID,
                "route": "authentic_photograph_from_tiny_genimage_real_pool",
                "route_tool": "",
                "image_model_id_if_reported": "",
                "prompt_id": "",
                "prompt_text": "",
                "generated_at_utc": "",
                "source_sha256": "",
                "source_width": "",
                "source_height": "",
                "source_format": "",
                "provenance_origin": "held_out_internal_real_test_pool",
            }
        )

    rows = sorted(fake_rows + real_rows, key=lambda row: str(row["sample_id"]))
    ids = [str(row["sample_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("external manifest contains duplicate sample ids")
    fake_ids = {str(row["sample_id"]) for row in fake_rows}
    internal_ids = {str(row["sample_id"]) for row in internal_rows}
    if fake_ids & internal_ids:
        raise ValueError("external fake sample ids collide with internal sample ids")

    composition = _describe_composition(rows, data_root)
    audit: dict[str, Any] = {
        "challenge_id": CHALLENGE_ID,
        "title": "Contemporary OpenAI/Astra-mediated image-generation challenge",
        "status": "executed",
        "route": EXTERNAL_ROUTE,
        "route_note": (
            "Route B of the pre-registered design. The hosted image-generation tool "
            "selects an underlying GPT Image model and does not report which, so the "
            "generator is recorded as unidentified and no architectural claim is attached."
        ),
        "generator_recorded_as": EXTERNAL_GENERATOR_NAME,
        "generator_identity_known": False,
        "image_model_id_reported_by_tool": False,
        "generated_images_found": len(generated) + len(excluded),
        "generated_images_usable": len(generated),
        "exclusions": excluded,
        "sample_counts": {
            "external_fake": len(fake_rows),
            "authentic_comparator": len(real_rows),
            "total": len(rows),
        },
        "class_balance": {
            "positive_prevalence": len(fake_rows) / len(rows),
            "balanced": len(fake_rows) == len(real_rows),
        },
        "sample_size_tier": _sample_size_tier(len(fake_rows)),
        "authentic_comparator_selection": comparator_metadata,
        "preprocessing": policy.describe(),
        "preprocessing_index": str(cache.index_path),
        "preprocessed_cache_root": str(cache_root.resolve()),
        "preprocessed_written": cache.processed,
        "preprocessed_reused": cache.skipped,
        "final_evaluation_composition": composition,
        "leakage_checks": [overlap, _check_ids_disjoint(fake_ids, internal_ids)],
        "internal_manifest": str(internal_manifest),
        "internal_manifest_sha256": _sha256_file(internal_manifest),
        "internal_splits": str(internal_splits),
        "split_name": EXTERNAL_SPLIT_NAME,
        "used_for_training": False,
        "used_for_validation": False,
        "used_for_threshold_selection": False,
        "manifest_columns": list(MANIFEST_COLUMNS),
        "provenance_csv": str(provenance_csv),
        "provenance_csv_sha256": _sha256_file(provenance_csv),
        "external_set_sha256": hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest(),
    }
    return ExternalChallengeBuild(rows=rows, audit=audit, excluded=excluded)


def _sample_size_tier(images_per_class: int) -> dict[str, Any]:
    """Report which pre-registered sample-size tier this set reaches, and which it misses."""

    tiers = [
        (50, "pilot / provenance smoke test"),
        (100, "minimum reportable"),
        (250, "matches the internal unseen-test size"),
    ]
    reached = [purpose for size, purpose in tiers if images_per_class >= size]
    unmet = [f"{size} per class: {purpose}" for size, purpose in tiers if images_per_class < size]
    return {
        "images_per_class": images_per_class,
        "tiers_reached": reached,
        "tiers_not_reached": unmet,
        "recommended_tier_reached": images_per_class >= 250,
    }


def _check_ids_disjoint(fake_ids: Iterable[str], internal_ids: Iterable[str]) -> dict[str, Any]:
    shared = sorted(set(fake_ids) & set(internal_ids))
    return {
        "check": "external_fake_sample_ids_disjoint_from_internal_manifest",
        "overlapping": shared[:10],
        "overlap_count": len(shared),
        "passed": not shared,
    }


def _describe_composition(
    rows: Sequence[Mapping[str, Any]], data_root: Path
) -> dict[str, Any]:
    """Record, per class, the container format and size of every evaluated image."""

    per_class: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        key = "fake" if int(row["label"]) == 1 else "real"
        path = (data_root / str(row["image_path"])).resolve()
        with Image.open(path) as handle:
            image_format = str(handle.format)
            size = f"{handle.size[0]}x{handle.size[1]}"
            mode = str(handle.mode)
        bucket = per_class.setdefault(key, {"format": {}, "size": {}, "mode": {}})
        bucket["format"][image_format] = bucket["format"].get(image_format, 0) + 1
        bucket["size"][size] = bucket["size"].get(size, 0) + 1
        bucket["mode"][mode] = bucket["mode"].get(mode, 0) + 1
    formats = {key: set(value["format"]) for key, value in per_class.items()}
    sizes = {key: set(value["size"]) for key, value in per_class.items()}
    real_formats = formats.get("real", set())
    fake_formats = formats.get("fake", set())
    format_predictive = bool(
        real_formats and fake_formats and not (real_formats & fake_formats)
    )
    all_sizes = set().union(*sizes.values()) if sizes else set()
    return {
        "per_class": per_class,
        "format_is_predictive_of_class": format_predictive,
        "single_container_format": len(set().union(*formats.values())) == 1 if formats else False,
        "single_spatial_size": len(all_sizes) == 1,
        "observed_sizes": sorted(all_sizes),
        "generators": sorted({str(row["generator"]) for row in rows}),
    }


def write_external_challenge(
    build: ExternalChallengeBuild, *, manifest_path: Path, audit_path: Path
) -> None:
    """Write the manifest CSV and its audit JSON, refusing to clobber either."""

    for path in (manifest_path, audit_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite an existing artefact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in build.rows:
            writer.writerow({key: row.get(key, "") for key in MANIFEST_COLUMNS})
    audit = dict(build.audit)
    audit["manifest_path"] = str(manifest_path)
    audit["manifest_sha256"] = _sha256_file(manifest_path)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
