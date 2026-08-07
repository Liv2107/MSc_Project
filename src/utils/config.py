"""Safe YAML inheritance, validation, and resolved-config persistence."""

from __future__ import annotations

import copy
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    values: Mapping[str, Any]
    source_path: Path


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            if (
                key in result
                and result[key] is not None
                and value is not None
                and isinstance(result[key], Mapping) != isinstance(value, Mapping)
            ):
                raise TypeError(f"configuration type changes are not allowed at {key!r}")
            result[key] = copy.deepcopy(value)
    return result


def _read_config(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"configuration inheritance cycle: {chain}")
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, Mapping):
        raise ValueError(f"configuration must contain a top-level mapping: {resolved}")
    values = dict(values)
    base_name = values.pop("base_config", None)
    if base_name is None:
        return values
    if not isinstance(base_name, str) or not base_name.strip():
        raise ValueError("base_config must be a non-empty relative path")
    base_path = Path(base_name)
    if base_path.is_absolute():
        raise ValueError("base_config must be relative to the child config")
    return deep_merge(_read_config(resolved.parent / base_path, (*stack, resolved)), values)


def load_config(config_path: Path) -> LoadedConfig:
    source = config_path.resolve()
    values = _read_config(source, ())
    validate_config(values)
    return LoadedConfig(values=values, source_path=source)


def _positive_number(value: Any, name: str, errors: list[str]) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        errors.append(f"{name} must be finite and positive")


def validate_config(config: Mapping[str, Any]) -> None:
    allowed_top = {
        "project",
        "data",
        "model",
        "transforms",
        "training",
        "evaluation",
        "reproducibility",
        "experiment",
        "generators",
        "unseen_protocol",
        "fine_tuning",
        "ablation",
        "mode_overrides",
    }
    errors: list[str] = []
    unknown = sorted(set(config).difference(allowed_top))
    if unknown:
        errors.append(f"unknown top-level keys: {', '.join(unknown)}")
    required = {
        "project",
        "data",
        "model",
        "transforms",
        "training",
        "evaluation",
        "reproducibility",
        "experiment",
        "generators",
    }
    missing = sorted(required.difference(config))
    if missing:
        errors.append(f"missing top-level keys: {', '.join(missing)}")
    if errors:
        raise ValueError("invalid configuration:\n- " + "\n- ".join(errors))

    allowed_sections = {
        "project": {"name", "output_root", "checkpoint_root"},
        "data": {
            "root",
            "manifest_path",
            "split_path",
            "image_path_column",
            "sample_id_column",
            "label_column",
            "generator_column",
            "group_column",
            "real_generator_name",
            "real_label",
            "fake_label",
        },
        "model": {
            "clip_model_name",
            "clip_revision",
            "feature_source",
            "classifier_dropout",
            "image_size",
            "initial_checkpoint",
            "decision_threshold",
        },
        "training": {
            "epochs",
            "batch_size",
            "num_workers",
            "learning_rate",
            "weight_decay",
            "optimizer",
            "loss",
            "positive_class_weight",
            "scheduler",
            "warmup_fraction",
            "mixed_precision",
            "gradient_clip_norm",
            "fine_tune_mode",
            "early_stopping",
            "checkpoint_metric",
        },
        "evaluation": {
            "metrics",
            "save_sample_predictions",
            "confusion_matrix",
            "roc_curve",
            "precision_recall_curve",
            "per_generator",
            "generator_evaluation_policy",
        },
        "reproducibility": {"seed", "experiment_seeds", "deterministic_algorithms"},
        "experiment": {"type", "name"},
        "generators": {"train", "validation", "test", "unseen", "include_real_images"},
        "unseen_protocol": {
            "adaptation_fraction",
            "adaptation_split_name",
            "final_test_split_name",
            "fixed_real_test_pool",
        },
        "fine_tuning": {
            "percentages",
            "subset_seeds",
            "nested_subsets",
            "adaptation_split_name",
            "final_test_split_name",
            "starting_checkpoint",
            "reload_starting_checkpoint_each_run",
            "fine_tune_mode",
            "adaptation_validation_fraction",
        },
        "ablation": {
            "modes",
            "control_starting_checkpoint",
            "control_subset_ids",
            "control_final_test_ids",
            "training_budget_policy",
            "record_trainable_parameter_names",
            "record_training_time",
        },
        "mode_overrides": {"head_only", "last_block", "full"},
    }
    for section_name, allowed in allowed_sections.items():
        section = config.get(section_name)
        if section is None:
            continue
        if not isinstance(section, Mapping):
            errors.append(f"{section_name} must be a mapping")
            continue
        nested_unknown = sorted(set(section).difference(allowed))
        if nested_unknown:
            errors.append(f"unknown {section_name} keys: {', '.join(nested_unknown)}")

    training = config["training"]
    model = config["model"]
    experiment = config["experiment"]
    generators = config["generators"]
    if not all(
        isinstance(section, Mapping) for section in (training, model, experiment, generators)
    ):
        raise ValueError("model, training, experiment, and generators must be mappings")
    _positive_number(training.get("epochs"), "training.epochs", errors)
    _positive_number(training.get("batch_size"), "training.batch_size", errors)
    _positive_number(training.get("learning_rate"), "training.learning_rate", errors)
    _positive_number(model.get("image_size"), "model.image_size", errors)
    if type(training.get("epochs")) is not int or type(training.get("batch_size")) is not int:
        errors.append("training.epochs and training.batch_size must be integers")
    workers = training.get("num_workers")
    if type(workers) is not int or workers < 0:
        errors.append("training.num_workers must be a non-negative integer")
    weight_decay = training.get("weight_decay")
    if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        errors.append("training.weight_decay must be non-negative")
    warmup = training.get("warmup_fraction")
    if not isinstance(warmup, (int, float)) or not 0 <= warmup < 1:
        errors.append("training.warmup_fraction must be in [0, 1)")
    dropout = model.get("classifier_dropout")
    if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        errors.append("model.classifier_dropout must be in [0, 1)")
    if training.get("optimizer") not in {"adamw", "sgd"}:
        errors.append("training.optimizer must be adamw or sgd")
    if training.get("scheduler") not in {"cosine_with_warmup", "none"}:
        errors.append("training.scheduler must be cosine_with_warmup or none")
    if training.get("fine_tune_mode") not in {"head_only", "last_block", "full"}:
        errors.append("training.fine_tune_mode is invalid")
    early = training.get("early_stopping")
    if not isinstance(early, Mapping):
        errors.append("training.early_stopping must be a mapping")
    else:
        early_unknown = sorted(
            set(early).difference({"enabled", "metric", "mode", "patience", "min_delta"})
        )
        if early_unknown:
            errors.append(f"unknown training.early_stopping keys: {', '.join(early_unknown)}")
    threshold = model.get("decision_threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        errors.append("model.decision_threshold must be in [0, 1]")
    experiment_type = experiment.get("type")
    if experiment_type not in {"baseline", "unseen_generator", "fine_tuning", "ablation"}:
        errors.append("experiment.type is invalid")
    for split in ("train", "validation", "test"):
        names = generators.get(split)
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
        ):
            errors.append(f"generators.{split} must be a non-empty string list")
        elif "real" in names:
            errors.append(
                f"generators.{split} lists only fake generators; real is controlled separately"
            )
    unseen = generators.get("unseen")
    if unseen is not None and unseen in generators.get("train", []):
        errors.append("the unseen generator cannot occur in training generators")
    if experiment_type == "baseline" and unseen is not None:
        errors.append("baseline generators.unseen must be null")
    if experiment_type in {"fine_tuning", "ablation"}:
        percentages = config.get("fine_tuning", {}).get("percentages")
        if percentages != [0.05, 0.1, 0.2, 0.5]:
            errors.append("fine_tuning.percentages must be [0.05, 0.10, 0.20, 0.50]")
    seed = config["reproducibility"].get("seed")
    if type(seed) is not int or seed < 0:
        errors.append("reproducibility.seed must be a non-negative integer")
    if errors:
        raise ValueError("invalid configuration:\n- " + "\n- ".join(errors))


def save_resolved_config(config: Mapping[str, Any], destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite resolved config: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".tmp", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(copy.deepcopy(dict(config)), handle, sort_keys=False, allow_unicode=True)
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
