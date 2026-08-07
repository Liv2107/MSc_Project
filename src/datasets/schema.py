"""Canonical, strict metadata contract for image-detection samples."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REAL_LABEL = 0
FAKE_LABEL = 1
REAL_GENERATOR_NAME = "real"


def _optional_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Validated metadata for one image.

    Paths are resolved beneath the configured data root when records are parsed.
    Label semantics are fixed to 0=real and 1=fake.
    """

    sample_id: str
    image_path: Path
    label: int
    generator: str
    source_group: str | None = None
    dataset_source: str | None = None

    @property
    def is_fake(self) -> bool:
        return self.label == FAKE_LABEL

    def validate(self) -> None:
        prefix = f"sample {self.sample_id!r}"
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if self.sample_id != self.sample_id.strip():
            raise ValueError(f"{prefix}: sample_id contains surrounding whitespace")
        if type(self.label) is not int or self.label not in (REAL_LABEL, FAKE_LABEL):
            raise ValueError(f"{prefix}: label must be exactly integer 0 or 1")
        if not isinstance(self.generator, str) or not self.generator.strip():
            raise ValueError(f"{prefix}: generator must be a non-empty string")
        if self.generator != self.generator.strip().lower():
            raise ValueError(f"{prefix}: generator must be stripped lowercase text")
        if self.label == REAL_LABEL and self.generator != REAL_GENERATOR_NAME:
            raise ValueError(f"{prefix}: real images must use generator='real'")
        if self.label == FAKE_LABEL and self.generator == REAL_GENERATOR_NAME:
            raise ValueError(f"{prefix}: fake images require a named non-real generator")
        if not isinstance(self.image_path, Path) or not self.image_path.is_absolute():
            raise ValueError(f"{prefix}: image_path must be a resolved absolute Path")
        for field_name, value in (
            ("source_group", self.source_group),
            ("dataset_source", self.dataset_source),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{prefix}: {field_name} must be a non-empty string")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, data_root: Path) -> DatasetRecord:
        required = {
            "sample_id",
            "image_path",
            "label",
            "generator",
            "source_group",
            "dataset_source",
        }
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"manifest row is missing required fields: {', '.join(missing)}")

        raw_label = row["label"]
        if isinstance(raw_label, bool):
            raise ValueError(f"sample {row.get('sample_id')!r}: boolean labels are invalid")
        if type(raw_label) is int:
            label = raw_label
        elif isinstance(raw_label, float) and raw_label.is_integer():
            label = int(raw_label)
        elif isinstance(raw_label, str) and raw_label.strip() in {"0", "1"}:
            label = int(raw_label.strip())
        else:
            raise ValueError(f"sample {row.get('sample_id')!r}: label must be 0 or 1")

        raw_path = Path(str(row["image_path"]).strip())
        if raw_path.is_absolute():
            raise ValueError(
                f"sample {row.get('sample_id')!r}: manifest image paths must be relative"
            )
        root = data_root.resolve()
        image_path = (root / raw_path).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"sample {row.get('sample_id')!r}: image_path escapes data root"
            ) from exc

        record = cls(
            sample_id=str(row["sample_id"]).strip(),
            image_path=image_path,
            label=label,
            generator=str(row["generator"]).strip().lower(),
            source_group=_optional_text(row.get("source_group")),
            dataset_source=_optional_text(row.get("dataset_source")),
        )
        record.validate()
        return record
