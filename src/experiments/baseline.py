"""End-to-end in-distribution CLIP baseline experiment."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.datasets.detector_dataset import AIDetectionDataset
from src.datasets.schema import DatasetRecord
from src.datasets.splitting import load_split_assignments
from src.evaluation.evaluator import collect_predictions, save_predictions
from src.evaluation.metrics import compute_binary_metrics, per_generator_metrics
from src.experiments.common import (
    build_data_loader,
    build_transforms,
    finalise_run,
    prepare_experiment,
)
from src.models.checkpointing import load_checkpoint
from src.models.clip_detector import (
    BinaryClassifierHead,
    CLIPBinaryDetector,
    CLIPVisionBackbone,
    configure_trainable_layers,
)
from src.training.components import (
    build_gradient_scaler,
    build_loss,
    build_optimizer,
    build_scheduler,
)
from src.training.early_stopping import EarlyStopping
from src.training.engine import fit
from src.utils.config import load_config


def _select_records(
    records: Sequence[DatasetRecord],
    assignments: Mapping[str, str],
    *,
    split: str,
    fake_generators: Sequence[str],
    include_real: bool,
) -> list[DatasetRecord]:
    allowed = set(fake_generators)
    selected = [
        record
        for record in records
        if assignments[record.sample_id] == split
        and (
            (record.label == 1 and record.generator in allowed)
            or (record.label == 0 and include_real)
        )
    ]
    labels = {record.label for record in selected}
    if labels != {0, 1}:
        raise ValueError(
            f"{split} selection must contain both real and fake images; found {labels}"
        )
    return selected


def _resolve_runtime_paths(values: Mapping[str, Any], source_path: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(values))
    project_root = source_path.parent.parent
    resolved["project"]["output_root"] = str(
        (project_root / values["project"]["output_root"]).resolve()
    )
    resolved["project"]["checkpoint_root"] = str(
        (project_root / values["project"]["checkpoint_root"]).resolve()
    )
    resolved["data"]["root"] = str((project_root / values["data"]["root"]).resolve())
    resolved["data"]["manifest_path"] = str(
        (project_root / values["data"]["manifest_path"]).resolve()
    )
    resolved["data"]["split_path"] = str((project_root / values["data"]["split_path"]).resolve())
    return resolved


def run_baseline(config_path: Path) -> Path:
    loaded = load_config(config_path)
    if loaded.values["experiment"]["type"] != "baseline":
        raise ValueError("run_baseline requires experiment.type=baseline")
    config = _resolve_runtime_paths(loaded.values, loaded.source_path)
    context = prepare_experiment(config)
    logger = logging.getLogger(f"ai_detector.{context.run_id}")
    try:
        train_transform = build_transforms(config, training=True)
        evaluation_transform = build_transforms(config, training=False)
        full_dataset = AIDetectionDataset.from_manifest(
            Path(config["data"]["manifest_path"]),
            data_root=Path(config["data"]["root"]),
            transform=evaluation_transform,
        )
        split_items = load_split_assignments(Path(config["data"]["split_path"]))
        split_by_id = {item.sample_id: item.split for item in split_items}
        record_ids = {record.sample_id for record in full_dataset.records}
        if set(split_by_id) != record_ids:
            missing = sorted(record_ids.difference(split_by_id))
            extra = sorted(set(split_by_id).difference(record_ids))
            raise ValueError(f"manifest/split ID mismatch; missing={missing[:5]} extra={extra[:5]}")
        group_splits: dict[str, set[str]] = {}
        for record in full_dataset.records:
            if record.source_group:
                group_splits.setdefault(record.source_group, set()).add(
                    split_by_id[record.sample_id]
                )
        leaked = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
        if leaked:
            raise ValueError(f"source groups cross split boundaries: {leaked[:10]}")

        include_real = bool(config["generators"].get("include_real_images", True))
        train_records = _select_records(
            full_dataset.records,
            split_by_id,
            split="train",
            fake_generators=config["generators"]["train"],
            include_real=include_real,
        )
        validation_records = _select_records(
            full_dataset.records,
            split_by_id,
            split="validation",
            fake_generators=config["generators"]["validation"],
            include_real=include_real,
        )
        test_records = _select_records(
            full_dataset.records,
            split_by_id,
            split="test",
            fake_generators=config["generators"]["test"],
            include_real=include_real,
        )
        logger.info(
            "selected samples train=%d validation=%d test=%d manifest_sha256=%s",
            len(train_records),
            len(validation_records),
            len(test_records),
            full_dataset.manifest_sha256,
        )
        train_dataset = AIDetectionDataset(train_records, transform=train_transform)
        validation_dataset = AIDetectionDataset(validation_records, transform=evaluation_transform)
        test_dataset = AIDetectionDataset(test_records, transform=evaluation_transform)
        train_loader = build_data_loader(train_dataset, config, training=True)
        validation_loader = build_data_loader(validation_dataset, config, training=False)
        test_loader = build_data_loader(test_dataset, config, training=False)

        backbone = CLIPVisionBackbone(
            str(config["model"]["clip_model_name"]),
            revision=config["model"].get("clip_revision"),
            feature_source=str(config["model"].get("feature_source", "pooled_output")),
        )
        classifier = BinaryClassifierHead(
            backbone.feature_dim, dropout=float(config["model"].get("classifier_dropout", 0.0))
        )
        model = CLIPBinaryDetector(backbone, classifier).to(context.device)
        configure_trainable_layers(model, str(config["training"]["fine_tune_mode"]))
        loss_fn = build_loss(
            positive_class_weight=config["training"].get("positive_class_weight")
        ).to(context.device)
        optimizer = build_optimizer(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            name=str(config["training"]["optimizer"]),
            learning_rate=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        total_steps = int(config["training"]["epochs"]) * len(train_loader)
        warmup_steps = min(
            total_steps - 1, int(total_steps * float(config["training"]["warmup_fraction"]))
        )
        scheduler = build_scheduler(
            optimizer,
            name=str(config["training"]["scheduler"]),
            total_update_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        scaler = build_gradient_scaler(
            enabled=bool(config["training"].get("mixed_precision", False))
        )
        early_config = config["training"]["early_stopping"]
        early_stopping = None
        if early_config.get("enabled", False):
            early_stopping = EarlyStopping(
                patience=int(early_config["patience"]),
                mode=str(early_config["mode"]),
                min_delta=float(early_config.get("min_delta", 0.0)),
            )
        result = fit(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=context.device,
            epochs=int(config["training"]["epochs"]),
            output_dir=context.run_dir,
            early_stopping=early_stopping,
            scaler=scaler,
            gradient_clip_norm=config["training"].get("gradient_clip_norm"),
            checkpoint_metric=str(config["training"]["checkpoint_metric"]),
            resolved_config=config,
            seed=context.seed,
        )
        best_checkpoint = load_checkpoint(
            Path(result["best_checkpoint"]), map_location=str(context.device)
        )
        model.load_state_dict(best_checkpoint["model_state"], strict=True)
        predictions = collect_predictions(
            model=model,
            data_loader=test_loader,
            device=context.device,
            split_name="test",
            checkpoint_id=Path(result["best_checkpoint"]).name,
            threshold=float(config["model"]["decision_threshold"]),
        )
        save_predictions(predictions, context.run_dir / "test_predictions.csv")
        labels = [item.label for item in predictions]
        scores = [item.score for item in predictions]
        generators = [item.generator for item in predictions]
        overall = compute_binary_metrics(
            labels, scores, threshold=float(config["model"]["decision_threshold"])
        )
        generator_metrics = per_generator_metrics(
            labels,
            scores,
            generators,
            threshold=float(config["model"]["decision_threshold"]),
            pair_with_real=config["evaluation"].get("generator_evaluation_policy")
            == "pair_with_fixed_real_pool",
        )
        metrics_payload = {
            "overall": asdict(overall),
            "per_generator": {name: asdict(metrics) for name, metrics in generator_metrics.items()},
            "best_epoch": result["best_epoch"],
            "best_validation_score": result["best_score"],
        }
        (context.run_dir / "test_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("baseline completed; test_f1=%.4f", overall.f1)
        finalise_run(context, status="completed")
        return context.run_dir
    except Exception:
        logger.exception("baseline failed")
        finalise_run(context, status="failed")
        raise
