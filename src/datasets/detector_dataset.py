"""PyTorch dataset and deterministic metadata selection helpers."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

from .schema import DatasetRecord

ImageTransform = Callable[[Image.Image], Tensor]


@dataclass(frozen=True, slots=True)
class DatasetFilters:
    include_generators: frozenset[str] | None = None
    exclude_generators: frozenset[str] | None = None
    include_labels: frozenset[int] | None = None

    def validate(self) -> None:
        if self.include_generators is not None and self.exclude_generators is not None:
            raise ValueError("include_generators and exclude_generators are mutually exclusive")
        if self.include_labels is not None and not self.include_labels.issubset({0, 1}):
            raise ValueError("include_labels may contain only 0 and 1")


class AIDetectionDataset(Dataset[dict[str, object]]):
    """Decode validated records into CLIP-normalised tensors plus audit metadata."""

    def __init__(
        self,
        records: Sequence[DatasetRecord],
        *,
        transform: ImageTransform,
    ) -> None:
        if not records:
            raise ValueError("dataset cannot be empty")
        if not callable(transform):
            raise TypeError("transform must be callable")
        immutable_records = tuple(records)
        ids: set[str] = set()
        for record in immutable_records:
            record.validate()
            if record.sample_id in ids:
                raise ValueError(f"duplicate sample_id: {record.sample_id}")
            ids.add(record.sample_id)
        self.records = immutable_records
        self.transform = transform
        self.manifest_path: Path | None = None
        self.manifest_sha256: str | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        try:
            with Image.open(record.image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                pixel_values = self.transform(image)
        except Exception as exc:
            raise RuntimeError(
                f"failed to decode sample {record.sample_id!r} at {record.image_path}"
            ) from exc
        if not isinstance(pixel_values, Tensor):
            raise TypeError(f"transform for {record.sample_id!r} did not return a Tensor")
        if pixel_values.ndim != 3 or pixel_values.shape[0] != 3:
            raise ValueError(
                f"transform for {record.sample_id!r} returned {tuple(pixel_values.shape)}; "
                "expected [3, H, W]"
            )
        if not pixel_values.is_floating_point() or not torch.isfinite(pixel_values).all():
            raise ValueError(f"transform for {record.sample_id!r} returned invalid values")
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(float(record.label), dtype=torch.float32),
            "generator": record.generator,
            "sample_id": record.sample_id,
            "image_path": str(record.image_path),
        }

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        *,
        data_root: Path,
        transform: ImageTransform,
        filters: DatasetFilters | None = None,
    ) -> AIDetectionDataset:
        path = manifest_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"manifest not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path, dtype={"sample_id": "string", "generator": "string"})
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            raise ValueError("manifest format must be CSV or Parquet")
        required = {
            "sample_id",
            "image_path",
            "label",
            "generator",
            "source_group",
            "dataset_source",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(missing)}")
        if frame.empty:
            raise ValueError("manifest contains no rows")
        if frame[list(required)].isnull().any().any():
            raise ValueError("required manifest fields may not be null")
        if frame["sample_id"].astype(str).duplicated().any():
            duplicate_ids = frame.loc[frame["sample_id"].astype(str).duplicated(), "sample_id"]
            raise ValueError(f"duplicate sample IDs: {duplicate_ids.astype(str).tolist()}")

        records = [
            DatasetRecord.from_mapping(row, data_root=data_root)
            for row in frame.to_dict(orient="records")
        ]
        missing_files = [r.sample_id for r in records if not r.image_path.is_file()]
        if missing_files:
            preview = ", ".join(missing_files[:10])
            raise FileNotFoundError(f"{len(missing_files)} manifest images are missing: {preview}")
        content_owners: dict[str, str] = {}
        duplicates: list[tuple[str, str]] = []
        for record in records:
            try:
                with Image.open(record.image_path) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(
                    f"image audit failed for sample {record.sample_id!r}: {record.image_path}"
                ) from exc
            digest = hashlib.sha256(record.image_path.read_bytes()).hexdigest()
            if digest in content_owners:
                duplicates.append((content_owners[digest], record.sample_id))
            else:
                content_owners[digest] = record.sample_id
        if duplicates:
            preview = ", ".join(f"{left}/{right}" for left, right in duplicates[:10])
            raise ValueError(f"exact duplicate image content detected: {preview}")
        if filters is not None:
            records = filter_records(records, filters)
        dataset = cls(records, transform=transform)
        dataset.manifest_path = path
        dataset.manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return dataset


def filter_records(
    records: Iterable[DatasetRecord], filters: DatasetFilters
) -> list[DatasetRecord]:
    filters.validate()
    source = list(records)
    known = {record.generator for record in source}
    requested = (filters.include_generators or frozenset()) | (
        filters.exclude_generators or frozenset()
    )
    unknown = sorted(requested.difference(known))
    if unknown:
        raise ValueError(f"unknown generators requested: {', '.join(unknown)}")
    selected: list[DatasetRecord] = []
    for record in source:
        if (
            filters.include_generators is not None
            and record.generator not in filters.include_generators
        ):
            continue
        if (
            filters.exclude_generators is not None
            and record.generator in filters.exclude_generators
        ):
            continue
        if filters.include_labels is not None and record.label not in filters.include_labels:
            continue
        selected.append(record)
    if not selected:
        raise ValueError("filters selected no records")
    return selected


def random_subset_indices(population_size: int, fraction: float, *, seed: int) -> list[int]:
    if type(population_size) is not int or population_size < 0:
        raise ValueError("population_size must be a non-negative integer")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if population_size == 0:
        return []
    count = max(1, math.ceil(population_size * fraction))
    order = list(range(population_size))
    random.Random(seed).shuffle(order)
    return sorted(order[:count])
