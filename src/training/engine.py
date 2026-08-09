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
from src.training.resume import (
    TRAINING_STATE_FILENAME,
    TrainingState,
    capture_rng_state,
    load_training_state,
    restore_rng_state,
    save_training_state,
)

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


def _restore_for_resume(
    *,
    state_path: Path,
    last_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: Any | None,
    early_stopping: EarlyStopping | None,
    device: torch.device | str,
    loader_generator: torch.Generator | None,
) -> TrainingState:
    """Put an interrupted run back exactly where it stopped, or refuse to continue.

    Every failure here is raised rather than worked around. Silently restarting, or
    continuing from a half-written state, would produce a run whose reported epoch
    count does not describe the optimisation that actually happened -- which is worse
    than losing the interrupted run.
    """

    if not state_path.is_file() or not last_path.is_file():
        raise FileNotFoundError(
            f"cannot resume: {last_path.name} and {state_path.name} must both exist in "
            f"{last_path.parent}. A run that was interrupted before completing its first "
            "epoch has nothing to resume from and must be started again."
        )
    state = load_training_state(state_path)
    checkpoint = load_checkpoint(last_path, map_location=str(device))
    checkpoint_epoch = int(checkpoint["metadata"]["epoch"])
    if checkpoint_epoch != state.epoch:
        raise RuntimeError(
            f"cannot resume: {last_path.name} holds epoch {checkpoint_epoch} but "
            f"{state_path.name} records epoch {state.epoch}. The run was interrupted "
            "between the two writes, so the weights and the bookkeeping describe "
            "different epochs and cannot be recombined."
        )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler_state = checkpoint.get("scheduler_state")
    if (scheduler is None) != (scheduler_state is None):
        raise RuntimeError(
            "cannot resume: the interrupted run's learning-rate schedule does not match "
            "the configured one"
        )
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = checkpoint.get("scaler_state")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    if (early_stopping is None) != (state.early_stopping is None):
        raise RuntimeError(
            "cannot resume: early stopping was "
            f"{'enabled' if state.early_stopping else 'disabled'} for the interrupted "
            "run and is now the opposite"
        )
    if early_stopping is not None and state.early_stopping is not None:
        early_stopping.load_state_dict(state.early_stopping)
    restore_rng_state(state.rng, loader_generator=loader_generator)
    return state


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
    resume: bool = False,
) -> dict[str, Any]:
    """Train for ``epochs`` epochs, selecting on validation only.

    With ``resume=True`` the run continues from the last epoch recorded in
    ``output_dir`` instead of starting at epoch 1: model, optimiser, scheduler, scaler,
    history, best score, early-stopping counters, and generator positions are all
    restored from ``last_checkpoint.pt`` plus the ``training_state.json`` sidecar. A
    ``resume=False`` call behaves exactly as it always has, and any run that reaches the
    end deletes the sidecar, so a completed run directory is unchanged by this feature.
    """

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
    state_path = output_dir / TRAINING_STATE_FILENAME
    history: list[dict[str, Any]] = []
    best_score: float | None = None
    best_epoch: int | None = None
    start_epoch = 1
    resumed_from_epochs: list[int] = []
    # The generator the training loader shuffles with. It advances across epochs, so it
    # is part of "where the run got to" and is snapshotted with the other generators.
    loader_generator = getattr(train_loader, "generator", None)
    selection_mode = "min" if checkpoint_metric == "loss" else "max"
    model_name = str(getattr(getattr(model, "backbone", None), "model_name", type(model).__name__))
    fine_tune_mode = str(getattr(model, "trainability_summary", {}).get("mode", "unknown"))
    progress_logger = logger or LOGGER
    label = f"{progress_label} " if progress_label else ""
    epoch_durations: list[float] = []
    train_batches = _batch_total(train_loader)
    validation_batches = _batch_total(validation_loader)
    if resume:
        state = _restore_for_resume(
            state_path=state_path,
            last_path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            device=device,
            loader_generator=loader_generator,
        )
        history = list(state.history)
        best_score = state.best_score
        best_epoch = state.best_epoch
        epoch_durations = list(state.epoch_durations)
        resumed_from_epochs = [*state.resumed_from_epochs, state.epoch]
        start_epoch = state.epoch + 1
        progress_logger.info(
            "%s RESUMING from completed epoch %d: continuing at epoch %d of %d | "
            "best epoch so far %s (val %s %s)",
            label.strip() or "run",
            state.epoch,
            start_epoch,
            epochs,
            best_epoch,
            checkpoint_metric,
            "none" if best_score is None else f"{best_score:.4f}",
        )
        if start_epoch > epochs:
            progress_logger.info(
                "%s already reached the configured epoch budget; finalising without "
                "further training",
                label.strip() or "run",
            )

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

    for epoch in range(start_epoch, epochs + 1):
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
        # Written after the checkpoint, so the sidecar can never claim an epoch whose
        # weights were not persisted. The reverse gap (checkpoint newer than sidecar) is
        # detected and refused on resume rather than silently mixing two epochs.
        save_training_state(
            TrainingState(
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                history=history,
                epoch_durations=epoch_durations,
                early_stopping=(
                    None if early_stopping is None else early_stopping.state_dict()
                ),
                rng=capture_rng_state(loader_generator=loader_generator),
                resumed_from_epochs=resumed_from_epochs,
            ),
            state_path,
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
    # Training reached its end, so there is nothing left to resume. Removing the sidecar
    # keeps a finished run directory identical to one produced before resume existed,
    # and makes the file's presence an unambiguous "this run stopped part-way" marker.
    state_path.unlink(missing_ok=True)
    return {
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "history_path": history_path,
        "history": history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "resumed_from_epochs": resumed_from_epochs,
    }
