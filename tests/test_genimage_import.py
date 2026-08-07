"""GenImage layout, deduplication, provenance, and split-policy tests."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image

from src.datasets.detector_dataset import AIDetectionDataset
from src.datasets.genimage import build_genimage_import, write_genimage_import


def _save_image(path: Path, colour: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), colour).save(path)


def _make_genimage_fixture(data_root: Path) -> Path:
    root = data_root / "raw" / "genimage"
    folders = ("Midjourney", "ADM")
    for folder in folders:
        _save_image(root / folder / "train" / "nature" / "class_a" / "real.png", (1, 2, 3))
        _save_image(root / folder / "val" / "nature" / "class_a" / "real_1.png", (4, 5, 6))
        _save_image(root / folder / "val" / "nature" / "class_a" / "real_2.png", (7, 8, 9))
    for generator_index, folder in enumerate(folders):
        _save_image(
            root / folder / "train" / "ai" / "class_a" / "fake.png",
            (40 + generator_index, 10, 10),
        )
        for index in range(2):
            _save_image(
                root / folder / "val" / "ai" / "class_a" / f"fake_{index}.png",
                (80 + generator_index * 10 + index, 20, 20),
            )
    return root


def test_genimage_import_preserves_official_train_and_partitions_val(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    root = _make_genimage_fixture(data_root)
    result = build_genimage_import(
        genimage_root=root,
        data_root=data_root,
        generators=["Midjourney", "adm"],
        validation_fraction_of_official_val=0.5,
        seed=42,
    )
    assert result.audit["generators"] == ["midjourney", "adm"]
    assert result.audit["deduplicated_repeated_nature_files"] == 3
    assert len(result.rows) == 9
    assert {row["generator"] for row in result.rows} == {"real", "midjourney", "adm"}
    split_by_id = {item.sample_id: item.split for item in result.assignments}
    for row in result.rows:
        if row["official_split"] == "train":
            assert split_by_id[row["sample_id"]] == "train"
        else:
            assert split_by_id[row["sample_id"]] in {"validation", "test"}
    assert set(split_by_id.values()) == {"train", "validation", "test"}
    groups: dict[str, set[str]] = {}
    for item in result.assignments:
        assert item.group_id is not None
        groups.setdefault(item.group_id, set()).add(item.split)
    assert all(len(splits) == 1 for splits in groups.values())


def test_written_genimage_manifest_passes_canonical_dataset_audit(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    root = _make_genimage_fixture(data_root)
    result = build_genimage_import(
        genimage_root=root,
        data_root=data_root,
        generators=["midjourney", "adm"],
        seed=9,
        max_per_generator_split_class=2,
    )
    manifest = data_root / "manifests" / "genimage.csv"
    splits = data_root / "manifests" / "genimage_splits.csv"
    write_genimage_import(result, manifest_path=manifest, split_path=splits)

    dataset = AIDetectionDataset.from_manifest(
        manifest,
        data_root=data_root,
        transform=lambda image: torch.zeros(3, image.height, image.width),
    )
    assert len(dataset) == len(result.rows)
    assert manifest.with_suffix(".audit.json").is_file()
    with splits.open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == len(result.assignments)
