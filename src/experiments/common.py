"""Shared, transparent builders for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils.config import save_resolved_config, validate_config
from src.utils.logging import configure_logging, make_run_id
from src.utils.reproducibility import collect_environment_metadata, seed_everything


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    run_id: str
    run_dir: Path
    config: Mapping[str, Any]
    device: torch.device
    seed: int


class CLIPImageTransform:
    """Small serialisable CLIP-compatible PIL-to-tensor transform."""

    def __init__(
        self,
        *,
        resize: int,
        center_crop: int,
        mean: list[float],
        std: list[float],
        horizontal_flip_probability: float = 0.0,
    ) -> None:
        if resize <= 0 or center_crop <= 0:
            raise ValueError("resize and center_crop must be positive")
        if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
            raise ValueError("normalisation mean/std must contain three valid values")
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("horizontal flip probability must be in [0, 1]")
        self.resize = resize
        self.center_crop = center_crop
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.horizontal_flip_probability = horizontal_flip_probability

    def __call__(self, image: Image.Image) -> torch.Tensor:
        width, height = image.size
        scale = self.resize / min(width, height)
        resized = image.resize(
            (
                max(self.center_crop, round(width * scale)),
                max(self.center_crop, round(height * scale)),
            ),
            resample=Image.Resampling.BICUBIC,
        )
        left = (resized.width - self.center_crop) // 2
        top = (resized.height - self.center_crop) // 2
        cropped = resized.crop((left, top, left + self.center_crop, top + self.center_crop))
        if self.horizontal_flip_probability and random.random() < self.horizontal_flip_probability:
            cropped = cropped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.asarray(cropped, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


def prepare_experiment(config: Mapping[str, Any]) -> ExperimentContext:
    validate_config(config)
    seed = int(config["reproducibility"]["seed"])
    deterministic = bool(config["reproducibility"].get("deterministic_algorithms", True))
    seed_everything(seed, deterministic=deterministic)
    config_text = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    config_hash = hashlib.sha256(config_text).hexdigest()
    experiment_type = str(config["experiment"]["type"])
    run_id = make_run_id(experiment_type=experiment_type, config_hash=config_hash)
    output_root = Path(str(config["project"]["output_root"])).resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    (run_dir / "environment.json").write_text(
        json.dumps(collect_environment_metadata(), indent=2, sort_keys=True), encoding="utf-8"
    )
    logger = configure_logging(run_id=run_id, log_path=run_dir / "run.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("prepared experiment on device=%s seed=%d", device, seed)
    return ExperimentContext(run_id, run_dir, dict(config), device, seed)


def build_transforms(config: Mapping[str, Any], *, training: bool) -> CLIPImageTransform:
    section = config["transforms"]["train" if training else "evaluation"]
    return CLIPImageTransform(
        resize=int(section["resize"]),
        center_crop=int(section["center_crop"]),
        mean=list(section["normalize_mean"]),
        std=list(section["normalize_std"]),
        horizontal_flip_probability=float(section.get("random_horizontal_flip_probability", 0.0))
        if training
        else 0.0,
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_data_loader(
    dataset: Dataset[Any], config: Mapping[str, Any], *, training: bool
) -> DataLoader[Any]:
    settings = config["training"]
    workers = int(settings.get("num_workers", 0))
    if workers < 0:
        raise ValueError("num_workers must be non-negative")
    generator = torch.Generator()
    generator.manual_seed(int(config["reproducibility"]["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=training,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def finalise_run(context: ExperimentContext, *, status: str) -> None:
    if status not in {"completed", "failed"}:
        raise ValueError("status must be completed or failed")
    (context.run_dir / "status.json").write_text(
        json.dumps({"run_id": context.run_id, "status": status}, indent=2), encoding="utf-8"
    )
    artefacts: list[dict[str, Any]] = []
    for path in sorted(context.run_dir.rglob("*")):
        if path.is_file() and path.name != "artefacts.json":
            artefacts.append(
                {
                    "path": path.relative_to(context.run_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    temporary = context.run_dir / "artefacts.json.tmp"
    temporary.write_text(json.dumps(artefacts, indent=2), encoding="utf-8")
    os.replace(temporary, context.run_dir / "artefacts.json")
