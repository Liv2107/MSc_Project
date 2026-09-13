"""End-to-end in-distribution CLIP baseline experiment."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.datasets.detector_dataset import AIDetectionDataset
from src.datasets.splitting import load_split_assignments
from src.evaluation.evaluator import collect_predictions, save_predictions
from src.evaluation.metrics import compute_binary_metrics, per_generator_metrics
from src.experiments.common import (
    build_data_loader,
    build_transforms,
    finalise_run,
    prepare_experiment,
    resolve_runtime_paths,
    select_records,
)
from src.models.checkpointing import load_checkpoint
from src.models.clip_detector import (
    CLIPBinaryDetector,
    CLIPVisionBackbone,
    build_classifier_head,
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


def run_baseline(config_path: Path, *, resume_from: Path | None = None) -> Path:
    """Train and evaluate the in-distribution reference."""
    loaded = load_config(config_path)
    if loaded.values["experiment"]["type"] != "baseline":
        raise ValueError("run_baseline requires experiment.type=baseline")
    config = resolve_runtime_paths(loaded.values, loaded.source_path)
    context = prepare_experiment(config, resume_run_dir=resume_from)
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
        train_records = select_records(
            full_dataset.records,
            split_by_id,
            split="train",
            fake_generators=config["generators"]["train"],
            include_real=include_real,
        )
        validation_records = select_records(
            full_dataset.records,
            split_by_id,
            split="validation",
            fake_generators=config["generators"]["validation"],
            include_real=include_real,
        )
        test_records = select_records(
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
        classifier = build_classifier_head(
            backbone.feature_dim,
            head_type=str(config["model"].get("head_type", "linear")),
            dropout=float(config["model"].get("classifier_dropout", 0.0)),
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
            logger=logger,
            progress_label="baseline",
            resume=resume_from is not None,
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
