"""GenImage-specific indexing without changing or copying the downloaded images."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import DatasetRecord
from .splitting import SplitAssignment, SplitFractions, create_grouped_splits

DATASET_SOURCE = "genimage_arxiv_2306.08571"
GENERATOR_ALIASES = {
    "midjourney": "midjourney",
    "vqdm": "vqdm",
    "wukong": "wukong",
    "stable diffusion v1.4": "stable_diffusion_v1_4",
    "stable diffusion v1.5": "stable_diffusion_v1_5",
    "glide": "glide",
    "biggan": "biggan",
    "adm": "adm",
}
OFFICIAL_GENERATORS = tuple(GENERATOR_ALIASES.values())
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class GenImageImportResult:
    rows: list[dict[str, Any]]
    assignments: list[SplitAssignment]
    audit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _IndexedImage:
    path: Path
    relative_path: str
    label: int
    generator: str
    official_split: str
    source_group: str
    source_folder: str


def _canonical_generator(name: str) -> str:
    normalised = " ".join(name.strip().casefold().split())
    if normalised in GENERATOR_ALIASES:
        return GENERATOR_ALIASES[normalised]
    if normalised in OFFICIAL_GENERATORS:
        return normalised
    raise ValueError(f"unknown GenImage generator name: {name}")


def _discover_generator_folders(root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for child in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        if not child.is_dir():
            continue
        try:
            canonical = _canonical_generator(child.name)
        except ValueError:
            continue
        if canonical in discovered:
            raise ValueError(f"multiple folders map to GenImage generator {canonical!r}")
        discovered[canonical] = child
    return discovered


def _images_below(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"expected GenImage directory is missing: {directory}")
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sample(
    entries: Sequence[_IndexedImage], *, maximum: int | None, seed_key: str
) -> list[_IndexedImage]:
    if maximum is None or len(entries) <= maximum:
        return list(entries)
    order = list(entries)
    random.Random(seed_key).shuffle(order)
    return sorted(order[:maximum], key=lambda item: item.relative_path.casefold())


def build_genimage_import(
    *,
    genimage_root: Path,
    data_root: Path,
    generators: Sequence[str] | None = None,
    validation_fraction_of_official_val: float = 0.5,
    seed: int = 42,
    max_per_generator_split_class: int | None = None,
) -> GenImageImportResult:
    """Index GenImage and create train/validation/test assignments.

    Official ``train`` remains training data. Official ``val`` is divided into
    model-selection validation and untouched test data. Repeated nature images
    across generator folders are retained once using their logical relative path.
    """

    root = genimage_root.resolve()
    data_root = data_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"GenImage root not found: {root}")
    try:
        root.relative_to(data_root)
    except ValueError as exc:
        raise ValueError("genimage_root must be contained beneath data_root") from exc
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0 < validation_fraction_of_official_val < 1:
        raise ValueError("validation fraction of official val must be in (0, 1)")
    if max_per_generator_split_class is not None and max_per_generator_split_class <= 0:
        raise ValueError("maximum per generator/split/class must be positive")

    discovered = _discover_generator_folders(root)
    if not discovered:
        raise ValueError("no recognised GenImage generator folders were found")
    requested = (
        [_canonical_generator(name) for name in generators]
        if generators is not None
        else list(discovered)
    )
    if len(requested) != len(set(requested)):
        raise ValueError("generator selection contains duplicates")
    missing = sorted(set(requested).difference(discovered))
    if missing:
        raise FileNotFoundError(f"requested GenImage folders are missing: {', '.join(missing)}")

    indexed: list[_IndexedImage] = []
    real_by_logical_path: dict[tuple[str, str], _IndexedImage] = {}
    duplicate_real_files = 0
    for generator in requested:
        folder = discovered[generator]
        for official_split in ("train", "val"):
            for class_folder, label in (("ai", 1), ("nature", 0)):
                class_root = folder / official_split / class_folder
                for path in _images_below(class_root):
                    relative_within_class = path.relative_to(class_root).as_posix()
                    if label == 0:
                        logical_key = (official_split, relative_within_class.casefold())
                        existing = real_by_logical_path.get(logical_key)
                        if existing is not None:
                            if (
                                path.stat().st_size != existing.path.stat().st_size
                                or _content_sha256(path) != _content_sha256(existing.path)
                            ):
                                raise ValueError(
                                    "nature files share a logical path but differ in content: "
                                    f"{existing.path} and {path}"
                                )
                            duplicate_real_files += 1
                            continue
                        real_identity = Path(relative_within_class).with_suffix("").as_posix()
                        source_group = f"genimage:real:{real_identity}"
                        item_generator = "real"
                    else:
                        source_group = (
                            f"genimage:{generator}:fake:"
                            f"{Path(relative_within_class).with_suffix('').as_posix()}"
                        )
                        item_generator = generator
                    item = _IndexedImage(
                        path=path,
                        relative_path=path.relative_to(data_root).as_posix(),
                        label=label,
                        generator=item_generator,
                        official_split=official_split,
                        source_group=source_group,
                        source_folder=folder.name,
                    )
                    indexed.append(item)
                    if label == 0:
                        real_by_logical_path[logical_key] = item

    grouped: dict[tuple[str, int, str], list[_IndexedImage]] = defaultdict(list)
    for item in indexed:
        grouped[(item.official_split, item.label, item.generator)].append(item)
    selected: list[_IndexedImage] = []
    for key, entries in sorted(grouped.items()):
        selected.extend(
            _stable_sample(
                entries,
                maximum=max_per_generator_split_class,
                seed_key=f"{seed}:{key}",
            )
        )
    selected.sort(key=lambda item: item.relative_path.casefold())

    group_splits: dict[str, set[str]] = defaultdict(set)
    for item in selected:
        group_splits[item.source_group].add(item.official_split)
    crossing = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if crossing:
        raise ValueError(f"source groups cross official train/val boundaries: {crossing[:10]}")

    rows: list[dict[str, Any]] = []
    records: list[DatasetRecord] = []
    for item in selected:
        identity = f"{item.official_split}|{item.generator}|{item.relative_path}"
        sample_id = f"genimage_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"
        row = {
            "sample_id": sample_id,
            "image_path": item.relative_path,
            "label": item.label,
            "generator": item.generator,
            "source_group": item.source_group,
            "dataset_source": DATASET_SOURCE,
            "official_split": item.official_split,
            "source_generator_folder": item.source_folder,
        }
        rows.append(row)
        records.append(
            DatasetRecord(
                sample_id=sample_id,
                image_path=item.path,
                label=item.label,
                generator=item.generator,
                source_group=item.source_group,
                dataset_source=DATASET_SOURCE,
            )
        )

    official_train_ids = {row["sample_id"] for row in rows if row["official_split"] == "train"}
    official_val_records = [
        record for record, row in zip(records, rows, strict=True) if row["official_split"] == "val"
    ]
    if not official_train_ids or not official_val_records:
        raise ValueError("GenImage import needs non-empty official train and val data")
    val_assignments = create_grouped_splits(
        official_val_records,
        SplitFractions(
            train=0.0,
            validation=validation_fraction_of_official_val,
            test=1.0 - validation_fraction_of_official_val,
        ),
        seed=seed,
    )
    assignments = [
        SplitAssignment(record.sample_id, "train", record.source_group)
        for record in records
        if record.sample_id in official_train_ids
    ] + val_assignments
    assignment_by_id = {item.sample_id: item.split for item in assignments}
    split_counts: dict[str, int] = defaultdict(int)
    distribution: dict[tuple[str, str, int], int] = defaultdict(int)
    for record in records:
        split = assignment_by_id[record.sample_id]
        split_counts[split] += 1
        distribution[(split, record.generator, record.label)] += 1
    audit = {
        "dataset": "GenImage",
        "dataset_source": DATASET_SOURCE,
        "genimage_root": str(root),
        "generators": requested,
        "seed": seed,
        "validation_fraction_of_official_val": validation_fraction_of_official_val,
        "max_per_generator_split_class": max_per_generator_split_class,
        "sample_count": len(rows),
        "deduplicated_repeated_nature_files": duplicate_real_files,
        "split_counts": dict(sorted(split_counts.items())),
        "distribution": [
            {"split": split, "generator": generator, "label": label, "count": count}
            for (split, generator, label), count in sorted(distribution.items())
        ],
    }
    return GenImageImportResult(rows=rows, assignments=assignments, audit=audit)


def write_genimage_import(
    result: GenImageImportResult, *, manifest_path: Path, split_path: Path
) -> None:
    """Atomically write the GenImage manifest, assignments, and audit sidecar."""

    if not result.rows or not result.assignments:
        raise ValueError("cannot write an empty GenImage import")
    for destination in (manifest_path, split_path, manifest_path.with_suffix(".audit.json")):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    def write_csv(destination: Path, fieldnames: list[str], rows: Sequence[dict[str, Any]]) -> None:
        fd, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".tmp", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary_name, destination)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    write_csv(manifest_path, list(result.rows[0]), result.rows)
    split_rows = [
        {"sample_id": item.sample_id, "split": item.split, "group_id": item.group_id}
        for item in result.assignments
    ]
    write_csv(split_path, ["sample_id", "split", "group_id"], split_rows)
    audit_path = manifest_path.with_suffix(".audit.json")
    fd, temporary_name = tempfile.mkstemp(dir=audit_path.parent, suffix=".tmp", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result.audit, handle, indent=2, sort_keys=True)
        os.replace(temporary_name, audit_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
