"""Dataset validation, loading, filtering, subsets, and split contracts."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.detector_dataset import (
    AIDetectionDataset,
    DatasetFilters,
    filter_records,
    random_subset_indices,
)
from src.datasets.schema import DatasetRecord
from src.datasets.splitting import SplitFractions, create_grouped_splits


def record(tmp_path: Path, sample_id: str, label: int, generator: str, group: str) -> DatasetRecord:
    item = DatasetRecord(
        sample_id, (tmp_path / f"{sample_id}.png").resolve(), label, generator, group, "test"
    )
    return item


@pytest.mark.parametrize("label", [-1, 2, "1", 1.0, True, None])
def test_record_rejects_invalid_binary_label(tmp_path: Path, label: object) -> None:
    item = DatasetRecord("bad", (tmp_path / "bad.png").resolve(), label, "generator_a")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bad.*label"):
        item.validate()


def test_leave_one_generator_out_filter_preserves_real_records(tmp_path: Path) -> None:
    records = [
        record(tmp_path, "real", 0, "real", "r"),
        record(tmp_path, "a", 1, "a", "a"),
        record(tmp_path, "b", 1, "b", "b"),
        record(tmp_path, "d", 1, "d", "d"),
    ]
    selected = filter_records(records, DatasetFilters(exclude_generators=frozenset({"d"})))
    assert [item.sample_id for item in selected] == ["real", "a", "b"]
    with pytest.raises(ValueError, match="unknown generators"):
        filter_records(records, DatasetFilters(exclude_generators=frozenset({"unknown"})))


def test_limited_data_subsets_are_reproducible_and_nested() -> None:
    subsets = [
        set(random_subset_indices(100, fraction, seed=42)) for fraction in (0.05, 0.1, 0.2, 0.5)
    ]
    assert [len(item) for item in subsets] == [5, 10, 20, 50]
    assert subsets[0] < subsets[1] < subsets[2] < subsets[3]
    assert random_subset_indices(100, 0.2, seed=42) == random_subset_indices(100, 0.2, seed=42)
    assert random_subset_indices(100, 0.2, seed=42) != random_subset_indices(100, 0.2, seed=43)


def test_source_groups_never_cross_split_boundaries(tmp_path: Path) -> None:
    records = [
        record(tmp_path, f"{group}-{index}", index % 2, "real" if index % 2 == 0 else "g", group)
        for group in ("one", "two", "three", "four", "five", "six")
        for index in range(2)
    ]
    assignments = create_grouped_splits(records, SplitFractions(0.5, 0.25, 0.25), seed=7)
    assert {item.sample_id for item in assignments} == {item.sample_id for item in records}
    by_group: dict[str, set[str]] = {}
    for item in assignments:
        assert item.group_id is not None
        by_group.setdefault(item.group_id, set()).add(item.split)
    assert all(len(splits) == 1 for splits in by_group.values())


def test_manifest_dataset_and_dataloader_contract(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for index, (label, generator) in enumerate(((0, "real"), (1, "generator_a"))):
        path = image_dir / f"{index}.png"
        Image.new("RGB", (12, 10), color=(index * 255, 10, 20)).save(path)
        rows.append(
            {
                "sample_id": str(index),
                "image_path": f"images/{index}.png",
                "label": label,
                "generator": generator,
                "source_group": str(index),
                "dataset_source": "test",
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def transform(image: Image.Image) -> torch.Tensor:
        return torch.zeros(3, 8, 8, dtype=torch.float32) + image.getpixel((0, 0))[0] / 255

    dataset = AIDetectionDataset.from_manifest(manifest, data_root=tmp_path, transform=transform)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["pixel_values"].shape == (2, 3, 8, 8)
    assert batch["label"].shape == (2,)
    assert batch["label"].dtype == torch.float32
    assert list(batch["sample_id"]) == ["0", "1"]
    assert dataset.manifest_sha256 is not None
