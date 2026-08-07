"""Read-only inference that preserves one auditable row per sample."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    sample_id: str
    image_path: str
    label: int
    generator: str
    score: float
    prediction: int
    split: str
    checkpoint_id: str


def _output_logits(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError("model output must expose logits")
    return logits


def collect_predictions(
    *,
    model: nn.Module,
    data_loader: Any,
    device: torch.device | str,
    split_name: str,
    checkpoint_id: str,
    threshold: float,
) -> list[PredictionRecord]:
    if not split_name or not checkpoint_id:
        raise ValueError("split_name and checkpoint_id must be non-empty")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    was_training = model.training
    model.eval()
    predictions: list[PredictionRecord] = []
    try:
        with torch.inference_mode():
            for batch in data_loader:
                if not isinstance(batch, Mapping):
                    raise TypeError("evaluation batches must be mappings")
                pixels = batch["pixel_values"]
                labels = batch["label"]
                if not isinstance(pixels, Tensor) or not isinstance(labels, Tensor):
                    raise TypeError("evaluation tensors are missing")
                logits = _output_logits(model(pixels.to(device, non_blocking=True)))
                scores = torch.sigmoid(logits).cpu()
                decisions = (scores >= threshold).to(torch.int64)
                labels_cpu = labels.to(torch.int64).cpu()
                batch_size = len(scores)
                for index in range(batch_size):
                    score = float(scores[index])
                    label = int(labels_cpu[index])
                    if not math.isfinite(score) or label not in {0, 1}:
                        raise ValueError("model produced an invalid prediction record")
                    predictions.append(
                        PredictionRecord(
                            sample_id=str(batch["sample_id"][index]),
                            image_path=str(batch["image_path"][index]),
                            label=label,
                            generator=str(batch["generator"][index]),
                            score=score,
                            prediction=int(decisions[index]),
                            split=split_name,
                            checkpoint_id=checkpoint_id,
                        )
                    )
    finally:
        model.train(was_training)
    ids = [item.sample_id for item in predictions]
    if not predictions or len(ids) != len(set(ids)):
        raise ValueError("evaluation produced no predictions or duplicate sample IDs")
    expected = len(getattr(data_loader, "dataset", []))
    if expected and len(predictions) != expected:
        raise ValueError(f"evaluation produced {len(predictions)} rows for {expected} samples")
    return predictions


def save_predictions(predictions: Sequence[PredictionRecord], destination: Path) -> None:
    if not predictions:
        raise ValueError("cannot save an empty prediction table")
    ids = [item.sample_id for item in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("prediction table contains duplicate sample IDs")
    if (
        len({item.split for item in predictions}) != 1
        or len({item.checkpoint_id for item in predictions}) != 1
    ):
        raise ValueError("prediction table mixes splits or checkpoints")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".tmp", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(predictions[0])))
            writer.writeheader()
            writer.writerows(asdict(item) for item in predictions)
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
