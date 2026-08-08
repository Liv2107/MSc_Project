"""Auditable train/validation loops for a single-logit binary detector."""

from __future__ import annotations

import csv
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.evaluation.metrics import compute_binary_metrics
from src.models.checkpointing import (
    CheckpointMetadata,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.training.early_stopping import EarlyStopping

LOGGER = logging.getLogger(__name__)

# How many progress lines to emit per epoch. Long epochs would otherwise run for many
# minutes in total silence, leaving no way to tell a slow run from a hung one.
PROGRESS_UPDATES_PER_EPOCH = 10


@dataclass(frozen=True, slots=True)
class EpochResult:
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    sample_count: int = 0


def _format_duration(seconds: float) -> str:
    """Render a duration as H:MM:SS or M:SS, whichever is shorter to read."""

    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _batch_total(data_loader: Any) -> int | None:
    """Number of batches, when the loader can report it."""

    try:
        return len(data_loader)
    except TypeError:
        return None


def _log_batch_progress(
    logger: logging.Logger,
    *,
    prefix: str,
    batch_index: int,
    batch_total: int | None,
    samples: int,
    started: float,
    running_loss: float,
) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = samples / elapsed
    if batch_total:
        percent = f"{100 * batch_index / batch_total:3.0f}%"
        position = f"({batch_index}/{batch_total})"
        remaining = (batch_total - batch_index) * (elapsed / batch_index)
        eta = f"eta {_format_duration(remaining)}"
    else:
        percent = "   ?"
        position = f"(batch {batch_index})"
        eta = f"elapsed {_format_duration(elapsed)}"
    logger.info(
        "%-26s %s %-13s loss %.4f   %4.0f img/s   %s",
        prefix,
        percent,
        position,
        running_loss / max(samples, 1),
        rate,
        eta,
    )


def _logits(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError("model output must be a Tensor or expose a Tensor logits field")
    return logits


def _batch(batch: object, device: torch.device | str) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("data loader batches must be mappings")
    pixels = batch.get("pixel_values")
    labels = batch.get("label")
    if not isinstance(pixels, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("batch must contain tensor pixel_values and label fields")
    return pixels.to(device, non_blocking=True), labels.to(
        device, dtype=torch.float32, non_blocking=True
    )


def _metrics(labels: list[float], scores: list[float]) -> dict[str, float]:
    values = compute_binary_metrics(np.asarray(labels), np.asarray(scores))
    result = {
        "accuracy": values.accuracy,
        "precision": values.precision,
        "recall": values.recall,
        "f1": values.f1,
    }
    if values.roc_auc is not None:
        result["roc_auc"] = values.roc_auc
    if values.average_precision is not None:
        result["average_precision"] = values.average_precision
    return result


def train_one_epoch(
    *,
    model: nn.Module,
    data_loader: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device | str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: Any | None = None,
    gradient_clip_norm: float | None = None,
    logger: logging.Logger | None = None,
    progress_prefix: str = "train",
) -> EpochResult:
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    model.train()
    total_loss = 0.0
    sample_count = 0
    all_labels: list[float] = []
    all_scores: list[float] = []
    scaler_enabled = bool(scaler is not None and getattr(scaler, "is_enabled", lambda: False)())
    device_type = torch.device(device).type
    progress_logger = logger or LOGGER
    batch_total = _batch_total(data_loader)
    progress_stride = max(1, (batch_total or 0) // PROGRESS_UPDATES_PER_EPOCH) or 1
    started = time.perf_counter()
    for batch_index, raw_batch in enumerate(data_loader, start=1):
        pixels, labels = _batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, enabled=scaler_enabled):
            logits = _logits(model(pixels))
            if logits.shape != labels.shape:
                raise ValueError(
                    f"logit shape {tuple(logits.shape)} != label shape {tuple(labels.shape)}"
                )
            loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        if scaler is not None and scaler_enabled:
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        count = labels.numel()
        total_loss += float(loss.detach()) * count
        sample_count += count
        all_labels.extend(labels.detach().cpu().tolist())
        all_scores.extend(torch.sigmoid(logits.detach()).cpu().tolist())
        if batch_index % progress_stride == 0 or batch_index == batch_total:
            _log_batch_progress(
                progress_logger,
                prefix=progress_prefix,
                batch_index=batch_index,
                batch_total=batch_total,
                samples=sample_count,
                started=started,
                running_loss=total_loss,
            )
    if sample_count == 0:
        raise ValueError("training loader produced no samples")
    return EpochResult(total_loss / sample_count, _metrics(all_labels, all_scores), sample_count)


def validate_one_epoch(
    *,
    model: nn.Module,
    data_loader: Any,
    loss_fn: nn.Module,
    device: torch.device | str,
    logger: logging.Logger | None = None,
    progress_prefix: str = "validate",
) -> EpochResult:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    sample_count = 0
    all_labels: list[float] = []
    all_scores: list[float] = []
    progress_logger = logger or LOGGER
    batch_total = _batch_total(data_loader)
    progress_stride = max(1, (batch_total or 0) // PROGRESS_UPDATES_PER_EPOCH) or 1
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_index, raw_batch in enumerate(data_loader, start=1):
                pixels, labels = _batch(raw_batch, device)
                logits = _logits(model(pixels))
                if logits.shape != labels.shape:
                    raise ValueError("logit and label shapes differ")
                loss = loss_fn(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite validation loss")
                count = labels.numel()
                total_loss += float(loss) * count
                sample_count += count
                all_labels.extend(labels.cpu().tolist())
                all_scores.extend(torch.sigmoid(logits).cpu().tolist())
                if batch_index % progress_stride == 0 or batch_index == batch_total:
                    _log_batch_progress(
                        progress_logger,
                        prefix=progress_prefix,
                        batch_index=batch_index,
                        batch_total=batch_total,
                        samples=sample_count,
                        started=started,
                        running_loss=total_loss,
                    )
    finally:
        model.train(was_training)
    if sample_count == 0:
        raise ValueError("validation loader produced no samples")
    return EpochResult(total_loss / sample_count, _metrics(all_labels, all_scores), sample_count)


def fit(
    *,
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_fn: nn.Module,
    device: torch.device | str,
    epochs: int,
    output_dir: Path,
    early_stopping: EarlyStopping | None = None,
    scaler: Any | None = None,
    gradient_clip_norm: float | None = None,
    checkpoint_metric: str = "f1",
    resolved_config: Mapping[str, Any] | None = None,
    seed: int = 0,
    logger: logging.Logger | None = None,
    progress_label: str = "",
) -> dict[str, Any]:
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be positive")
    if checkpoint_metric not in {
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    }:
        raise ValueError(f"unsupported checkpoint metric: {checkpoint_metric}")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_checkpoint.pt"
    last_path = output_dir / "last_checkpoint.pt"
    history: list[dict[str, Any]] = []
    best_score: float | None = None
    best_epoch: int | None = None
    selection_mode = "min" if checkpoint_metric == "loss" else "max"
    model_name = str(getattr(getattr(model, "backbone", None), "model_name", type(model).__name__))
    fine_tune_mode = str(getattr(model, "trainability_summary", {}).get("mode", "unknown"))
    progress_logger = logger or LOGGER
    label = f"{progress_label} " if progress_label else ""
    epoch_durations: list[float] = []
    train_batches = _batch_total(train_loader)
    validation_batches = _batch_total(validation_loader)
    progress_logger.info(
        "%s training: up to %d epochs | %s train + %s val batches per epoch | "
        "%s | selecting on val %s",
        label.strip() or "run",
        epochs,
        train_batches if train_batches is not None else "?",
        validation_batches if validation_batches is not None else "?",
        fine_tune_mode,
        checkpoint_metric,
    )

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_result = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
            gradient_clip_norm=gradient_clip_norm,
            logger=progress_logger,
            progress_prefix=f"{label}epoch {epoch}/{epochs} train",
        )
        validation_result = validate_one_epoch(
            model=model,
            data_loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
            logger=progress_logger,
            progress_prefix=f"{label}epoch {epoch}/{epochs} validate",
        )
        for split, result in (("train", train_result), ("validation", validation_result)):
            history.append({"epoch": epoch, "split": split, "loss": result.loss, **result.metrics})
        score = (
            validation_result.loss
            if checkpoint_metric == "loss"
            else validation_result.metrics[checkpoint_metric]
        )
        if not math.isfinite(score):
            raise FloatingPointError("non-finite checkpoint selection metric")
        if early_stopping is not None:
            improved, should_stop = early_stopping.update(score, epoch=epoch)
        else:
            improved = best_score is None or (
                score < best_score if selection_mode == "min" else score > best_score
            )
            should_stop = False
        metadata = CheckpointMetadata(
            epoch=epoch,
            validation_metric_name=checkpoint_metric,
            validation_metric_value=float(score),
            model_name=model_name,
            fine_tune_mode=fine_tune_mode,
            seed=seed,
        )
        checkpoint = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metadata=metadata,
            resolved_config=resolved_config or {},
        )
        save_checkpoint(checkpoint, last_path)
        if improved:
            best_score, best_epoch = float(score), epoch
            save_checkpoint(checkpoint, best_path)

        # Per-epoch summary. The ETA extrapolates from the mean completed epoch and is an
        # upper bound: early stopping can end the run sooner, never later.
        epoch_durations.append(time.perf_counter() - epoch_started)
        mean_epoch = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = epochs - epoch
        progress_logger.info(
            "%-26s EPOCH %d/%d done in %-7s train loss %.4f  val loss %.4f  "
            "val %s %.4f  %-14s best epoch %s (%.4f)%s",
            label.strip() or "run",
            epoch,
            epochs,
            _format_duration(epoch_durations[-1]),
            train_result.loss,
            validation_result.loss,
            checkpoint_metric,
            score,
            "<- IMPROVED" if improved else "no improvement",
            best_epoch,
            best_score if best_score is not None else float("nan"),
            f"  run eta <= {_format_duration(mean_epoch * remaining_epochs)}"
            if remaining_epochs
            else "  (final epoch)",
        )
        if should_stop:
            progress_logger.info(
                "%s early stopping at epoch %d: no improvement for %d epochs",
                label.strip() or "run",
                epoch,
                getattr(early_stopping, "patience", 0),
            )
            break
    if not best_path.exists():
        raise RuntimeError("training completed without selecting a best checkpoint")
    progress_logger.info(
        "%s finished: %d epoch(s) in %s | selected epoch %s | val %s %.4f",
        label.strip() or "run",
        len(epoch_durations),
        _format_duration(sum(epoch_durations)),
        best_epoch,
        checkpoint_metric,
        best_score if best_score is not None else float("nan"),
    )
    best = load_checkpoint(best_path, map_location=str(device))
    model.load_state_dict(best["model_state"], strict=True)
    history_path = output_dir / "train_history.csv"
    fieldnames = sorted(
        {key for row in history for key in row},
        key=lambda key: (key not in {"epoch", "split", "loss"}, key),
    )
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return {
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "history_path": history_path,
        "history": history,
        "best_epoch": best_epoch,
        "best_score": best_score,
    }
