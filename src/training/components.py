"""Explicit factories for the baseline optimisation components."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn


def build_loss(*, positive_class_weight: float | None = None) -> nn.Module:
    if positive_class_weight is None:
        return nn.BCEWithLogitsLoss()
    if not math.isfinite(positive_class_weight) or positive_class_weight <= 0:
        raise ValueError("positive_class_weight must be finite and positive")
    return nn.BCEWithLogitsLoss(pos_weight=Tensor([positive_class_weight]))


def build_optimizer(
    parameters: Iterable[nn.Parameter],
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    selected = list(parameters)
    if not selected:
        raise ValueError("optimizer received no parameters")
    if any(not parameter.requires_grad for parameter in selected):
        raise ValueError("optimizer received a frozen parameter")
    if name.lower() == "adamw":
        return torch.optim.AdamW(selected, lr=learning_rate, weight_decay=weight_decay)
    if name.lower() == "sgd":
        return torch.optim.SGD(selected, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"unknown optimizer: {name}")


def build_scheduler(
    optimizer: object, *, name: str, total_update_steps: int, warmup_steps: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if type(total_update_steps) is not int or total_update_steps <= 0:
        raise ValueError("total_update_steps must be positive")
    if type(warmup_steps) is not int or not 0 <= warmup_steps < total_update_steps:
        raise ValueError("warmup_steps must be in [0, total_update_steps)")
    if name.lower() in {"none", "off"}:
        return None
    if name.lower() != "cosine_with_warmup":
        raise ValueError(f"unknown scheduler: {name}")

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_update_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_gradient_scaler(*, enabled: bool) -> object:
    actually_enabled = bool(enabled and torch.cuda.is_available())
    try:
        return torch.amp.GradScaler("cuda", enabled=actually_enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=actually_enabled)
