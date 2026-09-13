"""Deterministic, group-aware dataset partitioning and persistence."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .schema import DatasetRecord


@dataclass(frozen=True, slots=True)
class SplitFractions:
    train: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("split fractions must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("split fractions must sum to 1")
        if sum(value > 0 for value in values) < 2:
            raise ValueError("at least two split fractions must be positive")


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    sample_id: str
    split: str
    group_id: str | None


def _validated_groups(records: Sequence[DatasetRecord]) -> dict[str, list[DatasetRecord]]:
    if not records:
        raise ValueError("cannot split an empty record collection")
    groups: dict[str, list[DatasetRecord]] = defaultdict(list)
    ids: set[str] = set()
    for record in records:
        record.validate()
        if record.sample_id in ids:
            raise ValueError(f"duplicate sample_id: {record.sample_id}")
        ids.add(record.sample_id)
        groups[record.source_group or f"sample:{record.sample_id}"].append(record)
    return dict(groups)


def create_grouped_splits(
    records: Sequence[DatasetRecord], fractions: SplitFractions, *, seed: int
) -> list[SplitAssignment]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    groups = _validated_groups(records)
    split_names = ("train", "validation", "test")
    fraction_values = dict(
        zip(
            split_names,
            (fractions.train, fractions.validation, fractions.test),
            strict=True,
        )
    )
    strata: dict[tuple[tuple[int, str], ...], list[tuple[str, list[DatasetRecord]]]] = defaultdict(
        list
    )
    for group_id, members in groups.items():
        signature = tuple(sorted({(member.label, member.generator) for member in members}))
        strata[signature].append((group_id, members))

    group_to_split: dict[str, str] = {}
    rng = random.Random(seed)
    for items in strata.values():
        rng.shuffle(items)
        total = sum(len(members) for _, members in items)
        targets = {name: total * fraction_values[name] for name in split_names}
        achieved = {name: 0 for name in split_names}
        for group_id, members in items:
            eligible = [name for name in split_names if fraction_values[name] > 0]
            chosen = max(
                eligible,
                key=lambda name: (targets[name] - achieved[name], -split_names.index(name)),
            )
            group_to_split[group_id] = chosen
            achieved[chosen] += len(members)

    assignments = [
        SplitAssignment(
            sample_id=record.sample_id,
            split=group_to_split[record.source_group or f"sample:{record.sample_id}"],
            group_id=record.source_group,
        )
        for record in records
    ]
    requested_splits = {name for name, value in fraction_values.items() if value > 0}
    achieved_splits = {item.split for item in assignments}
    missing_splits = sorted(requested_splits.difference(achieved_splits))
    if missing_splits:
        raise ValueError(
            "insufficient independent source groups to populate splits: "
            + ", ".join(missing_splits)
        )
    return assignments


def save_split_assignments(assignments: Sequence[SplitAssignment], destination: Path) -> None:
    if not assignments:
        raise ValueError("cannot save empty split assignments")
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite split assignments: {destination}")
    ids = [item.sample_id for item in assignments]
    if len(ids) != len(set(ids)):
        raise ValueError("split assignments contain duplicate sample IDs")
    legal = {"train", "validation", "test", "development", "adaptation_pool", "unseen_test"}
    if any(item.split not in legal for item in assignments):
        raise ValueError("split assignments contain an unknown split name")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".tmp", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "group_id"])
            writer.writeheader()
            writer.writerows(asdict(item) for item in assignments)
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    sidecar = destination.with_suffix(destination.suffix + ".metadata.json")
    sidecar.write_text(
        json.dumps({"schema_version": 1, "sample_count": len(assignments)}, indent=2),
        encoding="utf-8",
    )


def load_split_assignments(source: Path) -> list[SplitAssignment]:
    if not source.is_file():
        raise FileNotFoundError(f"split file not found: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assignments = [
        SplitAssignment(row["sample_id"], row["split"], row.get("group_id") or None) for row in rows
    ]
    if not assignments or len({a.sample_id for a in assignments}) != len(assignments):
        raise ValueError("split file is empty or contains duplicate sample IDs")
    return assignments
