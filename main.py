"""Thin command-line dispatcher for dissertation experiments."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from src.experiments.ablation import run_ablation
from src.experiments.baseline import run_baseline
from src.experiments.external_challenge import run_external_challenge
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
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help=(
            "Continue an interrupted run from its last completed epoch, reusing the "
            "given output directory. The config must be unchanged since that run "
            "started. Supported for baseline and unseen_generator, which each train a "
            "single long model."
        ),
    )
    return parser.parse_args(argv)


# Protocols that train one long model and can therefore continue from a saved epoch.
# fine_tuning and ablation are grids of many short fits, where an interruption is
# resumed by re-running the grid rather than by continuing one model.
RESUMABLE_EXPERIMENTS = ("baseline", "unseen_generator")


def run(config_path: Path, *, resume_from: Path | None = None) -> Path | None:
    loaded = load_config(config_path)
    experiment_type = str(loaded.values["experiment"]["type"])
    runners: dict[str, Callable[..., object]] = {
        "baseline": run_baseline,
        "unseen_generator": run_unseen_generator,
        "fine_tuning": run_fine_tuning,
        "ablation": run_ablation,
        "external_challenge": run_external_challenge,
    }
    if resume_from is not None and experiment_type not in RESUMABLE_EXPERIMENTS:
        raise ValueError(
            f"--resume is not supported for experiment.type={experiment_type}; it applies "
            f"to {' and '.join(RESUMABLE_EXPERIMENTS)}"
        )
    if resume_from is not None:
        result = runners[experiment_type](loaded.source_path, resume_from=resume_from)
    else:
        result = runners[experiment_type](loaded.source_path)
    return result if isinstance(result, Path) else None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args.config, resume_from=args.resume)


if __name__ == "__main__":
    main()
