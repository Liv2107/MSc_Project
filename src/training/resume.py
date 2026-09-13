"""Crash-resumable bookkeeping for long training runs."""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

TRAINING_STATE_FILENAME = "training_state.json"

STATE_SCHEMA_VERSION = 1


def _encode_byte_tensor(tensor: torch.Tensor) -> str:
    """Render a torch generator state as hex so it survives a JSON round trip."""
    return bytes(tensor.to(torch.uint8).cpu().numpy().tobytes()).hex()


def _decode_byte_tensor(value: str) -> torch.Tensor:
    return torch.frombuffer(bytearray.fromhex(value), dtype=torch.uint8).clone()


def capture_rng_state(*, loader_generator: torch.Generator | None) -> dict[str, Any]:
    """Snapshot every generator that influences the next epoch's numbers."""
    python_version, python_state, python_gauss = random.getstate()
    numpy_bit_generator, numpy_keys, numpy_pos, numpy_has_gauss, numpy_gauss = cast(
        tuple[str, Any, int, int, float], np.random.get_state(legacy=True)
    )
    state: dict[str, Any] = {
        "python": {
            "version": python_version,
            "state": list(python_state),
            "gauss_next": python_gauss,
        },
        "numpy": {
            "bit_generator": str(numpy_bit_generator),
            "state": [int(value) for value in numpy_keys],
            "pos": int(numpy_pos),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(numpy_gauss),
        },
        "torch": _encode_byte_tensor(torch.get_rng_state()),
        "loader_generator": (
            None if loader_generator is None else _encode_byte_tensor(loader_generator.get_state())
        ),
        "cuda": (
            [_encode_byte_tensor(item) for item in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }
    return state


def restore_rng_state(state: dict[str, Any], *, loader_generator: torch.Generator | None) -> None:
    """Put every generator back where ``capture_rng_state`` found it."""
    python = state["python"]
    random.setstate(
        (
            int(python["version"]),
            tuple(int(value) for value in python["state"]),
            python["gauss_next"],
        )
    )
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(_decode_byte_tensor(state["torch"]))
    saved_loader = state.get("loader_generator")
    if loader_generator is not None and saved_loader is not None:
        loader_generator.set_state(_decode_byte_tensor(saved_loader))
    cuda_states = state.get("cuda")
    if (
        cuda_states
        and torch.cuda.is_available()
        and len(cuda_states) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all([_decode_byte_tensor(item) for item in cuda_states])


@dataclass(frozen=True, slots=True)
class TrainingState:
    """Everything ``fit`` needs to continue after the last completed epoch."""
    epoch: int
    best_score: float | None
    best_epoch: int | None
    history: list[dict[str, Any]]
    epoch_durations: list[float]
    early_stopping: dict[str, Any] | None
    rng: dict[str, Any]
    resumed_from_epochs: list[int] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "epoch": self.epoch,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "epoch_durations": self.epoch_durations,
            "early_stopping": self.early_stopping,
            "rng": self.rng,
            "resumed_from_epochs": self.resumed_from_epochs,
        }


def save_training_state(state: TrainingState, destination: Path) -> None:
    """Write the sidecar atomically so an interrupted write cannot corrupt it."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".tmp", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state.to_json(), handle, indent=2, sort_keys=True)
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_training_state(source: Path) -> TrainingState:
    if not source.is_file():
        raise FileNotFoundError(f"training state not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"training state must be a JSON object: {source}")
    version = payload.get("schema_version")
    if version != STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported training-state schema: {version}")
    epoch = payload.get("epoch")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("training state epoch must be a positive integer")
    return TrainingState(
        epoch=epoch,
        best_score=None if payload.get("best_score") is None else float(payload["best_score"]),
        best_epoch=None if payload.get("best_epoch") is None else int(payload["best_epoch"]),
        history=list(payload.get("history") or []),
        epoch_durations=[float(value) for value in payload.get("epoch_durations") or []],
        early_stopping=payload.get("early_stopping"),
        rng=dict(payload["rng"]),
        resumed_from_epochs=[int(value) for value in payload.get("resumed_from_epochs") or []],
    )
