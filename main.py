"""Thin command-line dispatcher for dissertation experiments."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from src.experiments.ablation import run_ablation
from src.experiments.baseline import run_baseline
from src.experiments.fine_tuning import run_fine_tuning
from src.experiments.unseen_generator import run_unseen_generator
from src.utils.config import load_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one configured AI-image detection experiment."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to the experiment YAML file."
    )
    return parser.parse_args(argv)


def run(config_path: Path) -> Path | None:
    loaded = load_config(config_path)
    experiment_type = str(loaded.values["experiment"]["type"])
    runners: dict[str, Callable[[Path], object]] = {
        "baseline": run_baseline,
        "unseen_generator": run_unseen_generator,
        "fine_tuning": run_fine_tuning,
        "ablation": run_ablation,
    }
    result = runners[experiment_type](loaded.source_path)
    return result if isinstance(result, Path) else None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
