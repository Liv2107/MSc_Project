"""Configuration inheritance and typo-rejection contracts."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.utils.config import deep_merge, load_config, validate_config


def test_baseline_config_resolves_without_mutating_base() -> None:
    loaded = load_config(Path(__file__).parents[1] / "configs" / "baseline.yaml")
    assert loaded.values["experiment"]["type"] == "baseline"
    assert loaded.values["training"]["batch_size"] == 32


def test_genimage_baseline_config_uses_all_official_generators() -> None:
    loaded = load_config(Path(__file__).parents[1] / "configs" / "genimage_baseline.yaml")
    assert loaded.values["experiment"]["type"] == "baseline"
    assert len(loaded.values["generators"]["train"]) == 8
    assert loaded.values["data"]["manifest_path"].endswith("genimage.csv")


def test_deep_merge_replaces_lists_and_does_not_mutate_inputs() -> None:
    base = {"nested": {"items": [1, 2], "value": 1}}
    override = {"nested": {"items": [3]}}
    result = deep_merge(base, override)
    assert result == {"nested": {"items": [3], "value": 1}}
    assert base["nested"]["items"] == [1, 2]


def test_config_rejects_nested_typos() -> None:
    values = copy.deepcopy(
        dict(load_config(Path(__file__).parents[1] / "configs" / "baseline.yaml").values)
    )
    values["training"]["learnng_rate"] = 1e-5
    with pytest.raises(ValueError, match="learnng_rate"):
        validate_config(values)
