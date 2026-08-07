"""Randomness controls and privacy-conscious environment metadata."""

from __future__ import annotations

import importlib.metadata
import platform
import random
import sys
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool) -> None:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def collect_environment_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "pandas",
        "Pillow",
        "PyYAML",
        "scikit-learn",
        "torch",
        "torchvision",
        "transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if cuda_available
        else [],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
