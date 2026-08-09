"""Contract tests for resuming an interrupted training run.

The claim under test is narrow and falsifiable: a run interrupted part-way must
continue from its last completed epoch, and the result must match the run that would
have happened had it never been interrupted. Asserting only that "it finishes" would
pass for a silent restart, so the headline test compares a resumed run's per-epoch
history against an uninterrupted reference run of the same configuration.
"""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

import src.experiments.baseline as baseline_module
import src.training.engine as engine_module
from src.models.checkpointing import load_checkpoint
from src.training.resume import TRAINING_STATE_FILENAME
from src.utils.config import load_config

EPOCHS = 4
INTERRUPT_BEFORE_EPOCH = 3


class OfflineBackbone(nn.Module):
    """Tiny stand-in for CLIP so the test never touches the network or a real dataset."""

    def __init__(self, model_name: str, **_: object) -> None:
        super().__init__()
        self.model_name = model_name
        self.feature_dim = 4
        self.projection = nn.Linear(3, 4)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.projection(pixels.mean(dim=(2, 3)))


def _write_fixture_project(root: Path) -> Path:
    """Build a minimal manifest, split file, and images, and return the config path."""

    config_dir = root / "configs"
    data_dir = root / "data"
    image_dir = data_dir / "images"
    config_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for label, generator in ((0, "real"), (1, "generator_a")):
            for replicate in range(2):
                sample_id = f"{split}-{label}-{replicate}"
                offset = split_index * 10 + replicate * 3
                colour = (20 + offset, 30, 40) if label == 0 else (220 - offset, 210, 200)
                Image.new("RGB", (12, 12), colour).save(image_dir / f"{sample_id}.png")
                rows.append(
                    {
                        "sample_id": sample_id,
                        "image_path": f"images/{sample_id}.png",
                        "label": label,
                        "generator": generator,
                        "source_group": sample_id,
                        "dataset_source": "synthetic_test_fixture",
                    }
                )
                split_rows.append(
                    {"sample_id": sample_id, "split": split, "group_id": sample_id}
                )
    with (data_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (data_dir / "splits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)

    source_config = load_config(Path(__file__).parents[1] / "configs" / "baseline.yaml").values
    config = copy.deepcopy(dict(source_config))
    config["project"]["output_root"] = "outputs"
    config["data"]["manifest_path"] = "data/manifest.csv"
    config["data"]["split_path"] = "data/splits.csv"
    config["generators"]["train"] = ["generator_a"]
    config["generators"]["validation"] = ["generator_a"]
    config["generators"]["test"] = ["generator_a"]
    config["training"].update(
        {
            "epochs": EPOCHS,
            "batch_size": 2,
            "num_workers": 0,
            "mixed_precision": False,
            "scheduler": "cosine_with_warmup",
            "learning_rate": 0.01,
        }
    )
    # Early stopping stays ON so the resumed run must also restore its counters; the
    # patience exceeds the epoch budget so it cannot end the run before epoch 4.
    config["training"]["early_stopping"].update({"enabled": True, "patience": EPOCHS + 1})
    config_path = config_dir / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _count_train_epochs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record every epoch the engine actually trains, to catch a silent restart."""

    calls: list[int] = []
    real_train = engine_module.train_one_epoch

    def counting_train(**kwargs: Any) -> Any:
        calls.append(len(calls) + 1)
        return real_train(**kwargs)

    monkeypatch.setattr(engine_module, "train_one_epoch", counting_train)
    return calls


def _interrupt_at(monkeypatch: pytest.MonkeyPatch, epoch: int) -> None:
    """Simulate a crash at the start of ``epoch`` -- a lost node, not a clean stop."""

    real_train = engine_module.train_one_epoch
    seen = {"count": 0}

    def crashing_train(**kwargs: Any) -> Any:
        seen["count"] += 1
        if seen["count"] >= epoch:
            raise RuntimeError("simulated interruption")
        return real_train(**kwargs)

    monkeypatch.setattr(engine_module, "train_one_epoch", crashing_train)


def _read_history(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "train_history.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_uninterrupted(config_path: Path) -> Path:
    return baseline_module.run_baseline(config_path)


@pytest.fixture
def offline_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(baseline_module, "CLIPVisionBackbone", OfflineBackbone)


def test_interrupted_run_resumes_from_its_last_epoch_and_matches_an_unbroken_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline_backbone: None
) -> None:
    """The whole feature, end to end: crash at epoch 3, resume, land where we would have.

    Two projects with identical fixtures and identical configs are trained. One runs
    straight through. The other dies at the start of epoch 3, is resumed, and must then
    train only epochs 3 and 4 -- and produce the same per-epoch losses and metrics as
    the run that was never interrupted.
    """

    reference_config = _write_fixture_project(tmp_path / "reference")
    reference_dir = _run_uninterrupted(reference_config)

    interrupted_config = _write_fixture_project(tmp_path / "interrupted")
    with monkeypatch.context() as patch:
        patch.setattr(baseline_module, "CLIPVisionBackbone", OfflineBackbone)
        _interrupt_at(patch, INTERRUPT_BEFORE_EPOCH)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            baseline_module.run_baseline(interrupted_config)

    output_root = tmp_path / "interrupted" / "outputs"
    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1, "the interrupted attempt must not fan out into several runs"
    interrupted_dir = run_dirs[0]

    # The failed run is marked failed, and carries exactly the two completed epochs.
    status = json.loads((interrupted_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    state = json.loads((interrupted_dir / TRAINING_STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["epoch"] == INTERRUPT_BEFORE_EPOCH - 1
    checkpoint = load_checkpoint(interrupted_dir / "last_checkpoint.pt")
    assert checkpoint["metadata"]["epoch"] == INTERRUPT_BEFORE_EPOCH - 1
    assert not (interrupted_dir / "train_history.csv").exists()

    trained_epochs = _count_train_epochs(monkeypatch)
    resumed_dir = baseline_module.run_baseline(interrupted_config, resume_from=interrupted_dir)

    # It continued rather than restarted: two more epochs, in the same directory.
    assert resumed_dir == interrupted_dir
    assert len(trained_epochs) == EPOCHS - (INTERRUPT_BEFORE_EPOCH - 1)
    assert [path for path in output_root.iterdir() if path.is_dir()] == [interrupted_dir]

    resumed_history = _read_history(resumed_dir)
    reference_history = _read_history(reference_dir)
    assert [row["epoch"] for row in resumed_history] == [
        str(epoch) for epoch in range(1, EPOCHS + 1) for _ in ("train", "validation")
    ]
    assert resumed_history == reference_history, (
        "a resumed run must reproduce the run that was never interrupted; a difference "
        "means the continuation is not equivalent to the original optimisation"
    )

    resumed_metrics = json.loads((resumed_dir / "test_metrics.json").read_text(encoding="utf-8"))
    reference_metrics = json.loads(
        (reference_dir / "test_metrics.json").read_text(encoding="utf-8")
    )
    assert resumed_metrics == reference_metrics


def test_completed_run_directory_is_unchanged_by_the_resume_feature(
    tmp_path: Path, offline_backbone: None
) -> None:
    """A run that completes leaves no resume scaffolding behind."""

    run_dir = _run_uninterrupted(_write_fixture_project(tmp_path / "project"))
    assert not (run_dir / TRAINING_STATE_FILENAME).exists()
    assert not (run_dir / "resume_events.json").exists()
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == (
        "completed"
    )
    artefacts = json.loads((run_dir / "artefacts.json").read_text(encoding="utf-8"))
    assert TRAINING_STATE_FILENAME not in {item["path"] for item in artefacts}


def test_resuming_a_completed_run_is_refused(tmp_path: Path, offline_backbone: None) -> None:
    """Finished results are not silently overwritten by a stray --resume."""

    config_path = _write_fixture_project(tmp_path / "project")
    run_dir = _run_uninterrupted(config_path)
    with pytest.raises(ValueError, match="already completed"):
        baseline_module.run_baseline(config_path, resume_from=run_dir)


def test_resuming_with_a_changed_config_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline_backbone: None
) -> None:
    """Epochs trained under one configuration are never attributed to another."""

    config_path = _write_fixture_project(tmp_path / "project")
    with monkeypatch.context() as patch:
        _interrupt_at(patch, INTERRUPT_BEFORE_EPOCH)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            baseline_module.run_baseline(config_path)
    run_dir = next(path for path in (tmp_path / "project" / "outputs").iterdir() if path.is_dir())

    changed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    changed["training"]["learning_rate"] = 0.02
    config_path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration has changed"):
        baseline_module.run_baseline(config_path, resume_from=run_dir)


def test_resume_refuses_a_run_with_no_completed_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline_backbone: None
) -> None:
    """Dying inside epoch 1 leaves nothing to continue from, and says so."""

    config_path = _write_fixture_project(tmp_path / "project")
    with monkeypatch.context() as patch:
        _interrupt_at(patch, 1)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            baseline_module.run_baseline(config_path)
    run_dir = next(path for path in (tmp_path / "project" / "outputs").iterdir() if path.is_dir())
    assert not (run_dir / TRAINING_STATE_FILENAME).exists()
    with pytest.raises(FileNotFoundError, match="nothing to resume"):
        baseline_module.run_baseline(config_path, resume_from=run_dir)


def test_resume_refuses_a_checkpoint_and_sidecar_that_disagree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline_backbone: None
) -> None:
    """A crash between the two writes is detected, not silently spliced together."""

    config_path = _write_fixture_project(tmp_path / "project")
    with monkeypatch.context() as patch:
        _interrupt_at(patch, INTERRUPT_BEFORE_EPOCH)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            baseline_module.run_baseline(config_path)
    run_dir = next(path for path in (tmp_path / "project" / "outputs").iterdir() if path.is_dir())

    state_path = run_dir / TRAINING_STATE_FILENAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["epoch"] = state["epoch"] + 1
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="interrupted between the two writes"):
        baseline_module.run_baseline(config_path, resume_from=run_dir)


def test_cli_rejects_resume_for_grid_protocols(tmp_path: Path) -> None:
    """--resume is refused where it has no meaning, rather than quietly ignored."""

    import main as main_module

    config_path = Path(__file__).parents[1] / "configs" / "tiny_recovery_biggan.yaml"
    with pytest.raises(ValueError, match="not supported for experiment.type=fine_tuning"):
        main_module.run(config_path, resume_from=tmp_path)


def test_cli_parses_the_resume_flag() -> None:
    import main as main_module

    args = main_module.parse_args(["--config", "configs/x.yaml", "--resume", "outputs/run-1"])
    assert args.resume == Path("outputs/run-1")
    assert main_module.parse_args(["--config", "configs/x.yaml"]).resume is None


def test_run_baseline_signature_keeps_resume_optional() -> None:
    """Existing call sites keep working untouched."""

    runner: Callable[..., Path] = baseline_module.run_baseline
    import inspect

    parameters = inspect.signature(runner).parameters
    assert parameters["resume_from"].default is None
    assert parameters["resume_from"].kind is inspect.Parameter.KEYWORD_ONLY
