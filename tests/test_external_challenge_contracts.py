"""Contracts for the external contemporary-generator challenge.

Covers the three properties the external study rests on: the generated images are
validated rather than trusted, the authentic comparators are selected by the same seeded
protocol the internal unseen tests use, and the evaluation-only protocol cannot be
configured into something that trains, adapts, or re-thresholds on external data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.datasets.external_challenge import (
    CHALLENGE_ID,
    EXTERNAL_DATASET_SOURCE,
    EXTERNAL_GENERATOR_NAME,
    EXTERNAL_SPLIT_NAME,
    MANIFEST_COLUMNS,
    _difference_hash,
    _sample_size_tier,
    build_external_challenge,
    check_no_overlap_with_internal,
    inspect_generated_images,
    read_generation_provenance,
    select_authentic_comparators,
    write_external_challenge,
)
from src.datasets.preprocessing import PreprocessingPolicy
from src.datasets.schema import DatasetRecord
from src.experiments.external_challenge import (
    load_external_records,
    validate_external_protocol,
    verify_audited_manifest,
)
from src.experiments.unseen_generator import REAL_TEST_POOL_SEED, build_balanced_final_test
from src.utils.config import validate_config

PROVENANCE_COLUMNS = (
    "prompt_id",
    "filename",
    "exact_prompt",
    "generation_date",
    "generation_route/tool",
    "model_identifier",
    "width",
    "height",
    "file_format",
    "sha256",
    "notes",
)


def _write_image(path: Path, *, colour: tuple[int, int, int], size: int = 64) -> str:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), colour).save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance_row(filename: str, digest: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "prompt_id": Path(filename).stem,
        "filename": filename,
        "exact_prompt": f"a photograph of {Path(filename).stem}",
        "generation_date": "2026-09-06T20:37:44.749Z",
        "generation_route/tool": "image_gen.imagegen (built-in Codex tool)",
        "model_identifier": "unavailable",
        "width": "1254",
        "height": "1254",
        "file_format": "PNG",
        "sha256": digest,
        "notes": "test fixture",
    }
    row.update(overrides)
    return row


def _write_provenance(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PROVENANCE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def generated(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """Four valid generated images with a matching provenance CSV."""

    image_dir = tmp_path / "external" / "gen" / "raw"
    digests = []
    for index in range(4):
        colour = (20 * index, 40, 200 - 20 * index)
        digests.append(_write_image(image_dir / f"gen_{index:03d}.png", colour=colour))
    provenance = tmp_path / "external" / "gen" / "manifest.csv"
    _write_provenance(
        provenance,
        [_provenance_row(f"gen_{index:03d}.png", digests[index]) for index in range(4)],
    )
    return image_dir, provenance, digests


# ------------------------------------------------------- 1. generated-image validation


def test_valid_generated_images_are_all_usable(generated: tuple[Path, Path, list[str]]) -> None:
    image_dir, provenance, _ = generated
    usable, excluded = inspect_generated_images(image_dir, read_generation_provenance(provenance))
    assert len(usable) == 4
    assert excluded == []
    assert {item.image_format for item in usable} == {"PNG"}
    # Route B reports no model id, so the field must come back absent rather than
    # carrying the literal string "unavailable" into the manifest.
    assert {item.image_model_id_if_reported for item in usable} == {None}


def test_image_whose_bytes_disagree_with_its_generation_digest_is_excluded(
    generated: tuple[Path, Path, list[str]],
) -> None:
    image_dir, provenance, digests = generated
    rows = list(read_generation_provenance(provenance).values())
    rows[1]["sha256"] = "0" * 64
    usable, excluded = inspect_generated_images(image_dir, {r["filename"]: r for r in rows})
    assert len(usable) == 3
    assert [item["reason"] for item in excluded] == [
        "sha256_disagrees_with_generation_record"
    ]


def test_byte_identical_duplicate_is_excluded_not_counted_twice(tmp_path: Path) -> None:
    image_dir = tmp_path / "raw"
    first = _write_image(image_dir / "a.png", colour=(10, 20, 30))
    second = _write_image(image_dir / "b.png", colour=(10, 20, 30))
    assert first == second
    provenance = tmp_path / "prov.csv"
    _write_provenance(
        provenance, [_provenance_row("a.png", first), _provenance_row("b.png", second)]
    )
    usable, excluded = inspect_generated_images(image_dir, read_generation_provenance(provenance))
    assert len(usable) == 1
    assert excluded[0]["reason"] == "byte_identical_duplicate"


def test_corrupt_image_is_excluded_rather_than_repaired(tmp_path: Path) -> None:
    import hashlib

    image_dir = tmp_path / "raw"
    image_dir.mkdir(parents=True)
    broken = image_dir / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n not an image")
    provenance = tmp_path / "prov.csv"
    _write_provenance(
        provenance,
        [_provenance_row("broken.png", hashlib.sha256(broken.read_bytes()).hexdigest())],
    )
    usable, excluded = inspect_generated_images(image_dir, read_generation_provenance(provenance))
    assert usable == []
    assert excluded[0]["reason"].startswith("unreadable:")


def test_image_without_a_provenance_row_is_excluded(tmp_path: Path) -> None:
    image_dir = tmp_path / "raw"
    digest = _write_image(image_dir / "known.png", colour=(1, 2, 3))
    _write_image(image_dir / "orphan.png", colour=(9, 9, 9))
    provenance = tmp_path / "prov.csv"
    _write_provenance(provenance, [_provenance_row("known.png", digest)])
    usable, excluded = inspect_generated_images(image_dir, read_generation_provenance(provenance))
    assert [item.filename for item in usable] == ["known.png"]
    assert excluded[0] == {"filename": "orphan.png", "reason": "no_provenance_row"}


def test_provenance_csv_listing_a_filename_twice_is_rejected(tmp_path: Path) -> None:
    provenance = tmp_path / "prov.csv"
    _write_provenance(
        provenance, [_provenance_row("a.png", "0" * 64), _provenance_row("a.png", "1" * 64)]
    )
    with pytest.raises(ValueError, match="twice"):
        read_generation_provenance(provenance)


# --------------------------------------------------- 2. authentic comparator selection


def _internal_rows(count: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "sample_id": f"genimage_real_{index:04d}",
                "image_path": f"processed/cache/real_{index:04d}.jpg",
                "label": 0,
                "generator": "real",
                "source_group": f"genimage:real:{index:04d}",
                "dataset_source": "tiny_genimage+preprocessed",
            }
        )
    for index in range(count):
        rows.append(
            {
                "sample_id": f"genimage_fake_{index:04d}",
                "image_path": f"processed/cache/fake_{index:04d}.jpg",
                "label": 1,
                "generator": "vqdm",
                "source_group": f"genimage:vqdm:{index:04d}",
                "dataset_source": "tiny_genimage+preprocessed",
            }
        )
    return rows


def test_comparators_are_a_nested_prefix_of_the_internal_fixed_real_pool() -> None:
    """The comparators must be a subset of the pool the internal unseen tests score on.

    Both selections shuffle the sorted real test IDs with the same fixed pool seed, so
    the smaller external selection has to be a prefix of the larger internal one. That
    is what makes the external and internal numbers share their negatives.
    """

    rows = _internal_rows(40)
    split_by_id = {str(row["sample_id"]): "test" for row in rows}
    comparators, metadata = select_authentic_comparators(rows, split_by_id, count=10)
    assert len(comparators) == 10
    assert metadata["real_test_pool_seed"] == REAL_TEST_POOL_SEED
    assert metadata["shares_negatives_with_internal_unseen_test"] is True

    records = [
        DatasetRecord(
            sample_id=str(row["sample_id"]),
            image_path=Path("/tmp") / str(row["image_path"]),
            label=int(row["label"]),
            generator=str(row["generator"]),
            source_group=str(row["source_group"]),
            dataset_source=str(row["dataset_source"]),
        )
        for row in rows
    ]
    internal, _ = build_balanced_final_test(records, split_by_id, unseen_generator="vqdm")
    internal_reals = {record.sample_id for record in internal if record.label == 0}
    assert {str(row["sample_id"]) for row in comparators} <= internal_reals


def test_comparator_selection_is_deterministic() -> None:
    rows = _internal_rows(40)
    split_by_id = {str(row["sample_id"]): "test" for row in rows}
    first, _ = select_authentic_comparators(rows, split_by_id, count=10)
    second, _ = select_authentic_comparators(rows, split_by_id, count=10)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]


def test_comparators_are_never_drawn_from_a_development_split() -> None:
    rows = _internal_rows(40)
    split_by_id = {str(row["sample_id"]): "train" for row in rows}
    for row in rows[:12]:
        split_by_id[str(row["sample_id"])] = "test"
    comparators, metadata = select_authentic_comparators(rows, split_by_id, count=5)
    assert metadata["source_split"] == "test"
    assert all(split_by_id[str(row["sample_id"])] == "test" for row in comparators)


def test_comparator_selection_refuses_an_undersized_pool() -> None:
    rows = _internal_rows(3)
    split_by_id = {str(row["sample_id"]): "test" for row in rows}
    with pytest.raises(ValueError, match="need 10"):
        select_authentic_comparators(rows, split_by_id, count=10)


# ------------------------------------------------------------------- 3. leakage checks


def test_overlap_check_flags_a_generated_image_that_is_an_internal_image(
    tmp_path: Path,
) -> None:
    data_root = tmp_path
    internal_path = data_root / "processed" / "cache" / "real_0000.jpg"
    internal_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (12, 34, 56)).save(internal_path, format="JPEG", quality=95)

    image_dir = data_root / "external" / "raw"
    image_dir.mkdir(parents=True, exist_ok=True)
    copy = image_dir / "copied.png"
    copy.write_bytes(internal_path.read_bytes())
    import hashlib

    digest = hashlib.sha256(copy.read_bytes()).hexdigest()
    provenance = tmp_path / "prov.csv"
    _write_provenance(provenance, [_provenance_row("copied.png", digest)])
    usable, _ = inspect_generated_images(image_dir, read_generation_provenance(provenance))

    index = data_root / "index.json"
    index.write_text(
        json.dumps(
            {
                "images": {
                    "genimage_real_0000": {
                        "source_sha256": digest,
                        "output_sha256": "f" * 64,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result = check_no_overlap_with_internal(
        usable,
        internal_manifest_rows=[
            {"sample_id": "genimage_real_0000", "image_path": "processed/cache/real_0000.jpg"}
        ],
        data_root=data_root,
        preprocessing_index=index,
        near_duplicate_scan="none",
    )
    assert result["passed"] is False
    assert result["byte_identical_matches"]


def test_overlap_check_flags_a_near_duplicate_of_an_internal_image(tmp_path: Path) -> None:
    data_root = tmp_path
    internal_path = data_root / "processed" / "cache" / "real_0000.jpg"
    internal_path.parent.mkdir(parents=True, exist_ok=True)
    source = Image.new("RGB", (64, 64), (12, 34, 56))
    for x in range(64):
        for y in range(64):
            source.putpixel((x, y), (x * 4 % 256, y * 4 % 256, 90))
    source.save(internal_path, format="JPEG", quality=95)

    image_dir = data_root / "external" / "raw"
    image_dir.mkdir(parents=True, exist_ok=True)
    rescaled = image_dir / "rescaled.png"
    source.resize((256, 256), Image.Resampling.BICUBIC).save(rescaled, format="PNG")
    import hashlib

    digest = hashlib.sha256(rescaled.read_bytes()).hexdigest()
    provenance = tmp_path / "prov.csv"
    _write_provenance(provenance, [_provenance_row("rescaled.png", digest)])
    usable, _ = inspect_generated_images(image_dir, read_generation_provenance(provenance))

    result = check_no_overlap_with_internal(
        usable,
        internal_manifest_rows=[
            {"sample_id": "genimage_real_0000", "image_path": "processed/cache/real_0000.jpg"}
        ],
        data_root=data_root,
        preprocessing_index=None,
        near_duplicate_scan="all",
    )
    # A rescale is not byte-identical, so only the perceptual sweep can catch it.
    assert result["byte_identical_matches"] == []
    assert result["passed"] is False
    assert result["near_duplicates_flagged"]


def test_difference_hash_is_stable_and_distinguishes_unrelated_images(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(first)
    gradient = Image.new("RGB", (32, 32))
    for x in range(32):
        for y in range(32):
            gradient.putpixel((x, y), (x * 8 % 256, 0, 0))
    gradient.save(second)
    assert _difference_hash(first) == _difference_hash(first)
    assert _difference_hash(first) != _difference_hash(second)


# ------------------------------------------------------------- 4. end-to-end build


def _internal_dataset(tmp_path: Path, *, reals: int = 12, fakes: int = 12) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    rows = []
    assignments = []
    for index in range(reals):
        relative = f"processed/cache/real_{index:04d}.jpg"
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (256, 256))
        for x in range(0, 256, 8):
            for y in range(0, 256, 8):
                image.paste((index * 7 % 256, x % 256, y % 256), (x, y, x + 8, y + 8))
        image.save(path, format="JPEG", quality=95, subsampling=0)
        rows.append(
            {
                "sample_id": f"genimage_real_{index:04d}",
                "image_path": relative,
                "label": 0,
                "generator": "real",
                "source_group": f"genimage:real:{index:04d}",
                "dataset_source": "tiny_genimage+preprocessed",
                "original_image_path": f"raw/real_{index:04d}.jpg",
            }
        )
        assignments.append({"sample_id": f"genimage_real_{index:04d}", "split": "test",
                            "group_id": f"genimage:real:{index:04d}"})
    for index in range(fakes):
        relative = f"processed/cache/fake_{index:04d}.jpg"
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 256), (200, index * 9 % 256, 40)).save(
            path, format="JPEG", quality=95, subsampling=0
        )
        rows.append(
            {
                "sample_id": f"genimage_fake_{index:04d}",
                "image_path": relative,
                "label": 1,
                "generator": "vqdm",
                "source_group": f"genimage:vqdm:{index:04d}",
                "dataset_source": "tiny_genimage+preprocessed",
                "original_image_path": f"raw/fake_{index:04d}.png",
            }
        )
        assignments.append({"sample_id": f"genimage_fake_{index:04d}", "split": "test",
                            "group_id": f"genimage:vqdm:{index:04d}"})
    manifest = data_root / "manifests" / "internal.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    splits = data_root / "manifests" / "internal_splits.csv"
    with splits.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "group_id"])
        writer.writeheader()
        writer.writerows(assignments)
    return manifest, splits


@pytest.fixture()
def built(tmp_path: Path) -> dict[str, Any]:
    manifest, splits = _internal_dataset(tmp_path)
    data_root = tmp_path / "data"
    image_dir = data_root / "external" / "gen" / "raw"
    digests = []
    for index in range(4):
        image = Image.new("RGB", (300, 300))
        for x in range(0, 300, 10):
            image.paste((index * 30 % 256, x % 256, 220), (x, 0, x + 10, 300))
        image_dir.mkdir(parents=True, exist_ok=True)
        image.save(image_dir / f"gen_{index:03d}.png", format="PNG")
        import hashlib

        digests.append(
            hashlib.sha256((image_dir / f"gen_{index:03d}.png").read_bytes()).hexdigest()
        )
    provenance = data_root / "external" / "gen" / "manifest.csv"
    _write_provenance(
        provenance,
        [_provenance_row(f"gen_{index:03d}.png", digests[index]) for index in range(4)],
    )
    build = build_external_challenge(
        generated_dir=image_dir,
        provenance_csv=provenance,
        internal_manifest=manifest,
        internal_splits=splits,
        data_root=data_root,
        cache_root=data_root / "processed" / "external_cache",
        policy=PreprocessingPolicy(),
        near_duplicate_scan="all",
    )
    return {"build": build, "data_root": data_root, "manifest": manifest, "splits": splits}


def test_build_produces_a_class_balanced_set_with_the_canonical_columns(
    built: dict[str, Any],
) -> None:
    build = built["build"]
    assert build.audit["challenge_id"] == CHALLENGE_ID
    assert build.audit["sample_counts"] == {
        "external_fake": 4,
        "authentic_comparator": 4,
        "total": 8,
    }
    assert build.audit["class_balance"] == {"positive_prevalence": 0.5, "balanced": True}
    assert set(MANIFEST_COLUMNS).issubset(set(build.rows[0]))


def test_build_records_the_generator_as_unidentified(built: dict[str, Any]) -> None:
    """Route B forfeits generator attribution, and the audit has to say so."""

    build = built["build"]
    assert build.audit["generator_recorded_as"] == EXTERNAL_GENERATOR_NAME
    assert build.audit["generator_identity_known"] is False
    assert build.audit["image_model_id_reported_by_tool"] is False
    fakes = [row for row in build.rows if int(row["label"]) == 1]
    assert {row["generator"] for row in fakes} == {EXTERNAL_GENERATOR_NAME}
    assert {row["image_model_id_if_reported"] for row in fakes} == {""}
    assert {row["dataset_source"] for row in fakes} == {
        f"{EXTERNAL_DATASET_SOURCE}+preprocessed"
    }


def test_build_asserts_the_set_never_reached_development(built: dict[str, Any]) -> None:
    audit = built["build"].audit
    assert audit["used_for_training"] is False
    assert audit["used_for_validation"] is False
    assert audit["used_for_threshold_selection"] is False
    assert audit["split_name"] == EXTERNAL_SPLIT_NAME
    assert all(check["passed"] for check in audit["leakage_checks"])


def test_build_normalises_both_classes_so_format_cannot_predict_the_label(
    built: dict[str, Any],
) -> None:
    build = built["build"]
    data_root = built["data_root"]
    formats = {0: set(), 1: set()}
    sizes = set()
    for row in build.rows:
        with Image.open(data_root / str(row["image_path"])) as image:
            formats[int(row["label"])].add(str(image.format))
            sizes.add(image.size)
    assert formats[0] == formats[1] == {"JPEG"}
    assert sizes == {(256, 256)}
    composition = build.audit["final_evaluation_composition"]
    assert composition["format_is_predictive_of_class"] is False
    assert composition["single_spatial_size"] is True


def test_build_refuses_when_a_generated_image_duplicates_internal_data(
    tmp_path: Path,
) -> None:
    manifest, splits = _internal_dataset(tmp_path)
    data_root = tmp_path / "data"
    internal = data_root / "processed" / "cache" / "real_0000.jpg"
    image_dir = data_root / "external" / "gen" / "raw"
    image_dir.mkdir(parents=True, exist_ok=True)
    copied = image_dir / "gen_000.png"
    copied.write_bytes(internal.read_bytes())
    import hashlib

    provenance = data_root / "external" / "gen" / "manifest.csv"
    _write_provenance(
        provenance,
        [_provenance_row("gen_000.png", hashlib.sha256(copied.read_bytes()).hexdigest())],
    )
    with pytest.raises(ValueError, match="overlap internal data"):
        build_external_challenge(
            generated_dir=image_dir,
            provenance_csv=provenance,
            internal_manifest=manifest,
            internal_splits=splits,
            data_root=data_root,
            cache_root=data_root / "processed" / "external_cache",
            near_duplicate_scan="all",
        )


def test_write_refuses_to_overwrite_an_existing_manifest(
    built: dict[str, Any], tmp_path: Path
) -> None:
    manifest = tmp_path / "out" / "external.csv"
    audit = tmp_path / "out" / "external.audit.json"
    write_external_challenge(built["build"], manifest_path=manifest, audit_path=audit)
    assert manifest.is_file() and audit.is_file()
    with pytest.raises(FileExistsError):
        write_external_challenge(built["build"], manifest_path=manifest, audit_path=audit)


def test_sample_size_tier_reports_the_unreached_tiers() -> None:
    tier = _sample_size_tier(100)
    assert tier["recommended_tier_reached"] is False
    assert tier["tiers_reached"][-1] == "minimum reportable"
    assert any("250 per class" in item for item in tier["tiers_not_reached"])
    assert _sample_size_tier(250)["recommended_tier_reached"] is True


# ------------------------------------------------------ 5. evaluation-only protocol


def _external_config(**overrides: Any) -> dict[str, Any]:
    section = {
        "challenge_id": CHALLENGE_ID,
        "manifest_path": "data/manifests/external_challenge_v1.csv",
        "audit_path": "data/manifests/external_challenge_v1.audit.json",
        "split_name": EXTERNAL_SPLIT_NAME,
        "generation_route": "astra_mediated_hosted_image_generation_tool",
        "generator_identity_known": False,
        "adaptation_permitted": False,
        "threshold_reselection_permitted": False,
        "primary_checkpoint_run_id": "unseen_generator-A",
        "frozen_checkpoints": [
            {"run_id": "unseen_generator-A", "role": "primary", "checkpoint": "best.pt"},
            {"run_id": "baseline-B", "role": "reference", "checkpoint": "best.pt"},
        ],
    }
    section.update(overrides)
    return {"external_challenge": section}


def test_evaluation_only_protocol_accepts_the_shipped_shape() -> None:
    assert validate_external_protocol(_external_config())["challenge_id"] == CHALLENGE_ID


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"adaptation_permitted": True}, "adaptation_permitted must be false"),
        ({"threshold_reselection_permitted": True}, "threshold_reselection_permitted"),
        ({"split_name": "test"}, "split_name must be"),
        ({"generator_identity_known": True}, "generator_identity_known must be false"),
        ({"frozen_checkpoints": []}, "at least one checkpoint"),
        ({"primary_checkpoint_run_id": "not-nominated"}, "must nominate one of"),
    ],
)
def test_evaluation_only_protocol_rejects_a_config_that_could_adapt_or_retune(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_external_protocol(_external_config(**overrides))


def test_duplicate_nominated_run_is_rejected() -> None:
    section = _external_config()["external_challenge"]
    section["frozen_checkpoints"] = [
        {"run_id": "unseen_generator-A", "role": "primary", "checkpoint": "best.pt"},
        {"run_id": "unseen_generator-A", "role": "again", "checkpoint": "last.pt"},
    ]
    with pytest.raises(ValueError, match="lists a run twice"):
        validate_external_protocol({"external_challenge": section})


def test_shipped_external_config_validates() -> None:
    from src.utils.config import load_config

    path = Path("configs/external_challenge_v1.yaml")
    if not path.is_file():
        pytest.skip("external challenge config is not present")
    loaded = load_config(path)
    assert loaded.values["experiment"]["type"] == "external_challenge"
    validate_external_protocol(loaded.values)


def test_external_challenge_is_an_accepted_experiment_type() -> None:
    from src.utils.config import load_config

    base = load_config(Path("configs/tiny_unseen_vqdm.yaml"))
    config = dict(base.values)
    config["experiment"] = {"type": "external_challenge", "name": "t"}
    config["generators"] = {
        "train": ["vqdm"],
        "validation": ["vqdm"],
        "test": ["astra_mediated_unidentified"],
        "unseen": "astra_mediated_unidentified",
        "include_real_images": True,
    }
    validate_config(config)


# ------------------------------------------------------ 6. audited-manifest gating


def test_verify_refuses_a_manifest_that_changed_since_it_was_audited(tmp_path: Path) -> None:
    manifest = tmp_path / "external.csv"
    manifest.write_text("sample_id\na\n", encoding="utf-8")
    audit = tmp_path / "external.audit.json"
    audit.write_text(
        json.dumps(
            {
                "manifest_sha256": "0" * 64,
                "leakage_checks": [{"check": "x", "passed": True}],
                "used_for_training": False,
                "used_for_validation": False,
                "used_for_threshold_selection": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="changed since it was audited"):
        verify_audited_manifest(manifest, audit)


def test_verify_refuses_a_set_whose_leakage_check_failed(tmp_path: Path) -> None:
    import hashlib

    manifest = tmp_path / "external.csv"
    manifest.write_text("sample_id\na\n", encoding="utf-8")
    audit = tmp_path / "external.audit.json"
    audit.write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "leakage_checks": [{"check": "overlap", "passed": False}],
                "used_for_training": False,
                "used_for_validation": False,
                "used_for_threshold_selection": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="failed its own leakage checks"):
        verify_audited_manifest(manifest, audit)


def test_verify_requires_the_audit_to_assert_no_development_use(tmp_path: Path) -> None:
    import hashlib

    manifest = tmp_path / "external.csv"
    manifest.write_text("sample_id\na\n", encoding="utf-8")
    audit = tmp_path / "external.audit.json"
    audit.write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "leakage_checks": [{"check": "overlap", "passed": True}],
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_threshold_selection": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="used_for_training"):
        verify_audited_manifest(manifest, audit)


def test_loaded_external_records_must_be_balanced_and_all_external(
    built: dict[str, Any], tmp_path: Path
) -> None:
    manifest = tmp_path / "external.csv"
    audit = tmp_path / "external.audit.json"
    write_external_challenge(built["build"], manifest_path=manifest, audit_path=audit)
    records = load_external_records(manifest, data_root=built["data_root"])
    assert len(records) == 8
    fakes = [record for record in records if record.label == 1]
    assert len(fakes) == 4
    assert {record.generator for record in fakes} == {EXTERNAL_GENERATOR_NAME}


def test_loading_an_unbalanced_external_set_is_refused(
    built: dict[str, Any], tmp_path: Path
) -> None:
    manifest = tmp_path / "external.csv"
    audit = tmp_path / "external.audit.json"
    write_external_challenge(built["build"], manifest_path=manifest, audit_path=audit)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    kept = [row for row in rows if int(row["label"]) == 1] + [
        row for row in rows if int(row["label"]) == 0
    ][:1]
    unbalanced = tmp_path / "unbalanced.csv"
    with unbalanced.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(kept)
    with pytest.raises(ValueError, match="class balanced"):
        load_external_records(unbalanced, data_root=built["data_root"])
