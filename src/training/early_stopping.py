"""Validation-only early-stopping state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "max"
    min_delta: float = 0.0
    best_score: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    _last_epoch: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.patience) is not int or self.patience <= 0:
            raise ValueError("patience must be a positive integer")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")

    def update(self, score: float, *, epoch: int) -> tuple[bool, bool]:
        if not math.isfinite(score):
            raise ValueError("early-stopping score must be finite")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if self._last_epoch is not None and epoch <= self._last_epoch:
            raise ValueError("epochs supplied to early stopping must increase")
        self._last_epoch = epoch
        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score = float(score)
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience
