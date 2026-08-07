"""Offline end-to-end baseline orchestration test using a tiny injected backbone."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import nn

import src.experiments.baseline as baseline_module
from src.utils.config import load_config


class OfflineBackbone(nn.Module):
    def __init__(self, model_name: str, **_: object) -> None:
        super().__init__()
        self.model_name = model_name
        self.feature_dim = 4
        self.projection = nn.Linear(3, 4)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.projection(pixels.mean(dim=(2, 3)))


def test_baseline_creates_complete_audit_bundle(tmp_path: Path, monkeypatch: object) -> None:
    project = tmp_path / "project"
    config_dir = project / "configs"
    data_dir = project / "data"
    image_dir = data_dir / "images"
    config_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for label, generator in ((0, "real"), (1, "generator_a")):
            sample_id = f"{split}-{label}"
            offset = split_index * 10
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
            split_rows.append({"sample_id": sample_id, "split": split, "group_id": sample_id})
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
    config["data"]["manifest_path"] = "data/manifest.csv"
    config["data"]["split_path"] = "data/splits.csv"
    config["generators"]["train"] = ["generator_a"]
    config["generators"]["validation"] = ["generator_a"]
    config["generators"]["test"] = ["generator_a"]
    config["training"].update(
        {
            "epochs": 2,
            "batch_size": 2,
            "num_workers": 0,
            "mixed_precision": False,
            "scheduler": "none",
        }
    )
    config["training"]["early_stopping"]["enabled"] = False
    config_path = config_dir / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(baseline_module, "CLIPVisionBackbone", OfflineBackbone)

    run_dir = baseline_module.run_baseline(config_path)
    expected = {
        "resolved_config.yaml",
        "environment.json",
        "run.log",
        "train_history.csv",
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "test_predictions.csv",
        "test_metrics.json",
        "status.json",
        "artefacts.json",
    }
    assert expected.issubset({path.name for path in run_dir.iterdir()})
    assert (
        json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "completed"
    )
    metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall"]["support"] == 2
