"""Validation-only early-stopping state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


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

    def state_dict(self) -> dict[str, Any]:
        """Serialisable counters, so a resumed run does not forget its bad epochs.

        The patience configuration is included so a resume can refuse a checkpoint whose
        stopping rule no longer matches the config being run.
        """

        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "last_epoch": self._last_epoch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            int(state["patience"]) != self.patience
            or str(state["mode"]) != self.mode
            or float(state["min_delta"]) != self.min_delta
        ):
            raise ValueError(
                "early-stopping configuration changed since the interrupted run: "
                f"saved patience={state['patience']} mode={state['mode']} "
                f"min_delta={state['min_delta']}, configured patience={self.patience} "
                f"mode={self.mode} min_delta={self.min_delta}"
            )
        saved_best = state.get("best_score")
        self.best_score = None if saved_best is None else float(saved_best)
        saved_epoch = state.get("best_epoch")
        self.best_epoch = None if saved_epoch is None else int(saved_epoch)
        self.bad_epochs = int(state["bad_epochs"])
        last_epoch = state.get("last_epoch")
        self._last_epoch = None if last_epoch is None else int(last_epoch)
