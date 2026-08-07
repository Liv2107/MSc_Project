"""Versioned, atomic local training checkpoints."""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

SCHEMA_VERSION = 1
REQUIRED_KEYS = {"schema_version", "model_state", "optimizer_state", "metadata", "resolved_config"}


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    epoch: int
    validation_metric_name: str
    validation_metric_value: float
    model_name: str
    fine_tune_mode: str
    seed: int


def build_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    metadata: CheckpointMetadata,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "metadata": asdict(metadata),
        "resolved_config": copy.deepcopy(dict(resolved_config)),
    }
    if not REQUIRED_KEYS.issubset(checkpoint):
        raise RuntimeError("internal checkpoint schema error")
    return checkpoint


def save_checkpoint(checkpoint: Mapping[str, Any], destination: Path) -> None:
    if not REQUIRED_KEYS.issubset(checkpoint):
        raise ValueError("checkpoint is missing required keys")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(dict(checkpoint), temporary_name)
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="ascii"
    )


def load_checkpoint(source: Path, *, map_location: str = "cpu") -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    try:
        checkpoint = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(source, map_location=map_location)
    if not isinstance(checkpoint, dict) or not REQUIRED_KEYS.issubset(checkpoint):
        raise ValueError("invalid checkpoint structure")
    if checkpoint["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {checkpoint['schema_version']}")
    if not isinstance(checkpoint["metadata"], dict):
        raise ValueError("checkpoint metadata must be a mapping")
    return checkpoint
