"""Shared, transparent builders for reproducible experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.datasets.detector_dataset import AIDetectionDataset
from src.datasets.schema import DatasetRecord
from src.datasets.splitting import load_split_assignments
from src.evaluation.evaluator import PredictionRecord, collect_predictions
from src.evaluation.metrics import BinaryMetrics, compute_binary_metrics, per_generator_metrics
from src.models.clip_detector import (
    BinaryClassifierHead,
    CLIPBinaryDetector,
    CLIPVisionBackbone,
    configure_trainable_layers,
)
from src.training.components import (
    build_gradient_scaler,
    build_loss,
    build_optimizer,
    build_scheduler,
)
from src.training.early_stopping import EarlyStopping
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


def resolve_runtime_paths(values: Mapping[str, Any], source_path: Path) -> dict[str, Any]:
    """Make configured paths absolute relative to the repository root.

    Mirrors the baseline runner exactly so every protocol resolves data, output, and
    checkpoint locations identically. ``source_path`` is the config file, whose parent
    is ``configs/``; its parent in turn is the documented repository root.
    """

    resolved = copy.deepcopy(dict(values))
    project_root = source_path.parent.parent
    resolved["project"]["output_root"] = str(
        (project_root / values["project"]["output_root"]).resolve()
    )
    resolved["project"]["checkpoint_root"] = str(
        (project_root / values["project"]["checkpoint_root"]).resolve()
    )
    resolved["data"]["root"] = str((project_root / values["data"]["root"]).resolve())
    resolved["data"]["manifest_path"] = str(
        (project_root / values["data"]["manifest_path"]).resolve()
    )
    resolved["data"]["split_path"] = str((project_root / values["data"]["split_path"]).resolve())
    return resolved


@dataclass(frozen=True, slots=True)
class ManifestBundle:
    records: list[DatasetRecord]
    split_by_id: dict[str, str]
    manifest_sha256: str


def _metadata_only_transform(image: Image.Image) -> Any:
    """Guard proving the manifest load reads metadata and never decodes pixels."""

    raise RuntimeError("metadata-only manifest load must not decode images")


def load_manifest_with_splits(config: Mapping[str, Any]) -> ManifestBundle:
    """Load the manifest and persisted splits, auditing identity and group leakage.

    Applies the same two checks the baseline performs before any training: the split
    file must describe exactly the manifest's samples, and no ``source_group`` may
    appear in more than one split.
    """

    dataset = AIDetectionDataset.from_manifest(
        Path(config["data"]["manifest_path"]),
        data_root=Path(config["data"]["root"]),
        transform=_metadata_only_transform,
    )
    split_items = load_split_assignments(Path(config["data"]["split_path"]))
    split_by_id = {item.sample_id: item.split for item in split_items}
    record_ids = {record.sample_id for record in dataset.records}
    if set(split_by_id) != record_ids:
        missing = sorted(record_ids.difference(split_by_id))
        extra = sorted(set(split_by_id).difference(record_ids))
        raise ValueError(f"manifest/split ID mismatch; missing={missing[:5]} extra={extra[:5]}")
    group_splits: dict[str, set[str]] = {}
    for record in dataset.records:
        if record.source_group:
            group_splits.setdefault(record.source_group, set()).add(split_by_id[record.sample_id])
    leaked = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"source groups cross split boundaries: {leaked[:10]}")
    if dataset.manifest_sha256 is None:
        raise RuntimeError("manifest load did not record a manifest digest")
    return ManifestBundle(
        records=list(dataset.records),
        split_by_id=split_by_id,
        manifest_sha256=dataset.manifest_sha256,
    )


def select_records(
    records: Sequence[DatasetRecord],
    split_by_id: Mapping[str, str],
    *,
    split: str,
    fake_generators: Sequence[str],
    include_real: bool,
    require_both_classes: bool = True,
) -> list[DatasetRecord]:
    """Take one persisted split, keeping only the declared fake generators."""

    allowed = set(fake_generators)
    selected = [
        record
        for record in records
        if split_by_id[record.sample_id] == split
        and (
            (record.label == 1 and record.generator in allowed)
            or (record.label == 0 and include_real)
        )
    ]
    if require_both_classes:
        labels = {record.label for record in selected}
        if labels != {0, 1}:
            raise ValueError(
                f"{split} selection must contain both real and fake images; found {labels}"
            )
    return selected


def build_detector(
    config: Mapping[str, Any], *, device: torch.device | str, fine_tune_mode: str | None = None
) -> CLIPBinaryDetector:
    """Construct the pinned CLIP detector and apply a freeze policy."""

    backbone = CLIPVisionBackbone(
        str(config["model"]["clip_model_name"]),
        revision=config["model"].get("clip_revision"),
        feature_source=str(config["model"].get("feature_source", "pooled_output")),
    )
    classifier = BinaryClassifierHead(
        backbone.feature_dim, dropout=float(config["model"].get("classifier_dropout", 0.0))
    )
    model = CLIPBinaryDetector(backbone, classifier).to(device)
    mode = (
        fine_tune_mode
        if fine_tune_mode is not None
        else str(config["training"]["fine_tune_mode"])
    )
    configure_trainable_layers(model, mode)
    return model


@dataclass(frozen=True, slots=True)
class TrainingStack:
    loss_fn: Any
    optimizer: torch.optim.Optimizer
    scheduler: Any | None
    scaler: Any
    early_stopping: EarlyStopping | None


def build_training_stack(
    config: Mapping[str, Any],
    model: torch.nn.Module,
    *,
    device: torch.device | str,
    steps_per_epoch: int,
    epochs: int,
    learning_rate: float | None = None,
) -> TrainingStack:
    """Assemble loss, optimiser, schedule, scaler, and early stopping from config.

    A fresh optimiser is always created here so that a freeze policy applied before
    this call cannot leak momentum state from previously trainable parameters.
    """

    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    training = config["training"]
    loss_fn = build_loss(positive_class_weight=training.get("positive_class_weight")).to(device)
    optimizer = build_optimizer(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        name=str(training["optimizer"]),
        learning_rate=float(training["learning_rate"] if learning_rate is None else learning_rate),
        weight_decay=float(training["weight_decay"]),
    )
    total_steps = int(epochs) * int(steps_per_epoch)
    warmup_steps = min(total_steps - 1, int(total_steps * float(training["warmup_fraction"])))
    scheduler = build_scheduler(
        optimizer,
        name=str(training["scheduler"]),
        total_update_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    scaler = build_gradient_scaler(enabled=bool(training.get("mixed_precision", False)))
    early_config = training["early_stopping"]
    early_stopping = None
    if early_config.get("enabled", False):
        early_stopping = EarlyStopping(
            patience=int(early_config["patience"]),
            mode=str(early_config["mode"]),
            min_delta=float(early_config.get("min_delta", 0.0)),
        )
    return TrainingStack(loss_fn, optimizer, scheduler, scaler, early_stopping)


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    predictions: list[PredictionRecord]
    overall: BinaryMetrics
    per_generator: dict[str, BinaryMetrics]


def evaluate_records(
    *,
    model: torch.nn.Module,
    records: Sequence[DatasetRecord],
    config: Mapping[str, Any],
    device: torch.device | str,
    split_name: str,
    checkpoint_id: str,
) -> EvaluationOutcome:
    """Score one fixed record set and compute overall plus per-generator metrics."""

    transform = build_transforms(config, training=False)
    dataset = AIDetectionDataset(list(records), transform=transform)
    loader = build_data_loader(dataset, config, training=False)
    threshold = float(config["model"]["decision_threshold"])
    predictions = collect_predictions(
        model=model,
        data_loader=loader,
        device=device,
        split_name=split_name,
        checkpoint_id=checkpoint_id,
        threshold=threshold,
    )
    labels = [item.label for item in predictions]
    scores = [item.score for item in predictions]
    generators = [item.generator for item in predictions]
    overall = compute_binary_metrics(labels, scores, threshold=threshold)
    grouped = per_generator_metrics(
        labels,
        scores,
        generators,
        threshold=threshold,
        pair_with_real=config["evaluation"].get("generator_evaluation_policy")
        == "pair_with_fixed_real_pool",
    )
    return EvaluationOutcome(predictions, overall, dict(grouped))


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
