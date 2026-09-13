"""Cross-run aggregation of saved experiment artefacts."""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_METRIC_FILES: Mapping[str, str] = {
    "baseline": "test_metrics.json",
    "unseen_generator": "unseen_generator_metrics.json",
    "fine_tuning": "recovery_metrics.json",
    "ablation": "ablation_metrics.json",
}

METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)

COUNT_NAMES: tuple[str, ...] = (
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
)

SUMMARY_METRICS: tuple[str, ...] = (
    "roc_auc",
    "average_precision",
    "f1",
    "accuracy",
    "precision",
    "recall",
)

THRESHOLD_FREE_METRICS: frozenset[str] = frozenset({"roc_auc", "average_precision"})

CONSOLIDATED_COLUMNS: tuple[str, ...] = (
    "run_id",
    "experiment_type",
    "experiment_name",
    "protocol",
    "is_synthetic_smoke",
    "evaluation_set",
    "condition",
    "generator",
    "held_out_generator",
    "fine_tune_mode",
    "head_type",
    "adaptation_percentage",
    "subset_seed",
    "training_seed",
    "operating_point",
    "threshold",
    "threshold_provenance",
    "support",
    "positive_prevalence",
    *METRIC_NAMES,
    *COUNT_NAMES,
    "trainable_parameters",
    "total_parameters",
    "trainable_parameter_fraction",
    "learning_rate",
    "epochs",
    "best_epoch",
    "best_validation_score",
    "training_seconds",
    "labelled_images_consumed",
    "held_out_fake_count",
    "selected_checkpoint",
    "selected_checkpoint_sha256",
    "starting_checkpoint",
    "seed",
    "manifest_sha256",
    "source_file",
    "source_sha256",
)

RUN_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "experiment_type",
    "experiment_name",
    "status",
    "is_synthetic_smoke",
    "started_at",
    "held_out_generator",
    "fine_tune_mode",
    "head_type",
    "seed",
    "epochs",
    "learning_rate",
    "batch_size",
    "model_name",
    "manifest_path",
    "manifest_sha256",
    "evaluated_conditions",
    "cells",
    "python",
    "torch",
    "transformers",
    "cuda_available",
    "platform",
    "run_dir",
)

RECOVERY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "held_out_generator",
    "fine_tune_mode",
    "head_type",
    "adaptation_percentage",
    "operating_point",
    "metric",
    "runs",
    "mean",
    "standard_deviation",
    "standard_error",
    "minimum",
    "maximum",
    "zero_percent_reference",
    "absolute_recovery",
    "relative_improvement",
    "in_distribution_reference",
    "generalisation_gap",
    "gap_closed_fraction",
    "gap_closed_is_reliable",
    "labelled_images_consumed",
    "trainable_parameters",
    "in_distribution_reference_run_id",
    "in_distribution_reference_match",
)

MINIMUM_RELIABLE_GAP: float = 0.02

_RUN_ID_TIMESTAMP = re.compile(r"-(\d{8}T\d{6}\d*Z)-")


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Everything known about one run directory without opening its metrics file."""
    run_id: str
    run_dir: Path
    experiment_type: str
    experiment_name: str
    status: str
    is_synthetic_smoke: bool
    metrics_path: Path | None
    started_at: str | None = None
    held_out_generator: str | None = None
    fine_tune_mode: str | None = None
    head_type: str | None = None
    seed: int | None = None
    epochs: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    model_name: str | None = None
    manifest_path: str | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)
    artefact_digests: Mapping[str, str] = field(default_factory=dict)

    @property
    def has_metrics(self) -> bool:
        return self.metrics_path is not None and self.metrics_path.is_file()

    @property
    def is_reportable(self) -> bool:
        """Completed, carrying a metrics file, and not a synthetic smoke run."""
        return self.status == "completed" and self.has_metrics and not self.is_synthetic_smoke


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_status(run_dir: Path) -> str:
    """``completed``/``failed`` as recorded, or ``incomplete`` when no status was written."""
    status = _read_json(run_dir / "status.json")
    if isinstance(status, dict) and isinstance(status.get("status"), str):
        return str(status["status"])
    return "incomplete"


def _read_resolved_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_artefact_digests(run_dir: Path) -> dict[str, str]:
    entries = _read_json(run_dir / "artefacts.json")
    if not isinstance(entries, list):
        return {}
    digests: dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            digest = entry.get("sha256")
            if isinstance(digest, str):
                digests[entry["path"]] = digest
    return digests


def _started_at(run_id: str) -> str | None:
    match = _RUN_ID_TIMESTAMP.search(run_id)
    return match.group(1) if match else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def describe_run(run_dir: Path) -> RunRecord | None:
    """Build a :class:`RunRecord` for one directory, or ``None`` if it is not a run."""
    if not run_dir.is_dir():
        return None
    experiment_type = run_dir.name.split("-", 1)[0]
    if experiment_type not in EXPERIMENT_METRIC_FILES:
        return None

    metrics_path = run_dir / EXPERIMENT_METRIC_FILES[experiment_type]
    config = _read_resolved_config(run_dir)

    def block(name: str) -> Mapping[str, Any]:
        section = config.get(name)
        return section if isinstance(section, dict) else {}

    experiment = block("experiment")
    training = block("training")
    model = block("model")
    data = block("data")
    generators = block("generators")
    reproducibility = block("reproducibility")

    experiment_name = str(experiment.get("name") or run_dir.name)
    environment = _read_json(run_dir / "environment.json")

    return RunRecord(
        run_id=run_dir.name,
        run_dir=run_dir,
        experiment_type=experiment_type,
        experiment_name=experiment_name,
        status=_read_status(run_dir),
        is_synthetic_smoke=experiment_name.startswith("SMOKE_"),
        metrics_path=metrics_path if metrics_path.is_file() else None,
        started_at=_started_at(run_dir.name),
        held_out_generator=(str(generators["unseen"]) if generators.get("unseen") else None),
        fine_tune_mode=(
            str(training["fine_tune_mode"]) if training.get("fine_tune_mode") else None
        ),
        head_type=(str(model.get("head_type", "linear")) if model else None),
        seed=_as_int(reproducibility.get("seed")),
        epochs=_as_int(training.get("epochs")),
        learning_rate=_as_float(training.get("learning_rate")),
        batch_size=_as_int(training.get("batch_size")),
        model_name=(str(model["clip_model_name"]) if model.get("clip_model_name") else None),
        manifest_path=(str(data["manifest_path"]) if data.get("manifest_path") else None),
        environment=environment if isinstance(environment, dict) else {},
        artefact_digests=_read_artefact_digests(run_dir),
    )


def discover_runs(output_root: Path) -> list[RunRecord]:
    """Describe every run directory under ``output_root``, oldest first."""
    records = [
        record
        for path in sorted(output_root.iterdir() if output_root.is_dir() else [])
        if (record := describe_run(path)) is not None
    ]
    records.sort(key=lambda record: (record.started_at or "", record.run_id))
    return records


def reportable_runs(
    records: Iterable[RunRecord], *, include_synthetic_smoke: bool = False
) -> list[RunRecord]:
    """Completed runs that carry a metrics file, excluding smoke runs by default."""
    return [
        record
        for record in records
        if record.status == "completed"
        and record.has_metrics
        and (include_synthetic_smoke or not record.is_synthetic_smoke)
    ]


def load_metrics(record: RunRecord) -> dict[str, Any]:
    """Read a run's metrics file. Raises if the run has none, rather than returning {}."""
    if record.metrics_path is None:
        raise FileNotFoundError(
            f"run {record.run_id} has no {EXPERIMENT_METRIC_FILES[record.experiment_type]}"
        )
    payload = _read_json(record.metrics_path)
    if not isinstance(payload, dict):
        raise ValueError(f"metrics file is not a JSON object: {record.metrics_path}")
    return payload


def _metric_block(block: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy a saved metric block verbatim, deriving only prevalence from saved counts."""
    row: dict[str, Any] = dict.fromkeys(METRIC_NAMES)
    row.update(dict.fromkeys(COUNT_NAMES))
    row["support"] = None
    row["threshold"] = None
    row["positive_prevalence"] = None
    if not block:
        return row
    for name in METRIC_NAMES:
        row[name] = _as_float(block.get(name))
    for name in COUNT_NAMES:
        row[name] = _as_int(block.get(name))
    row["support"] = _as_int(block.get("support"))
    row["threshold"] = _as_float(block.get("threshold"))
    positives = row["true_positive"], row["false_negative"]
    if all(value is not None for value in positives) and row["support"]:
        row["positive_prevalence"] = (positives[0] + positives[1]) / row["support"]
    return row


def _base_row(record: RunRecord, protocol: str | None) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(CONSOLIDATED_COLUMNS)
    row.update(
        run_id=record.run_id,
        experiment_type=record.experiment_type,
        experiment_name=record.experiment_name,
        protocol=protocol,
        is_synthetic_smoke=record.is_synthetic_smoke,
        head_type=record.head_type,
        seed=record.seed,
        epochs=record.epochs,
        learning_rate=record.learning_rate,
        source_file=(record.metrics_path.name if record.metrics_path else None),
        source_sha256=(
            record.artefact_digests.get(record.metrics_path.name) if record.metrics_path else None
        ),
    )
    return row


def _finalise_row(row: dict[str, Any]) -> dict[str, Any]:
    """Derive the trainable fraction and drop anything outside the declared schema."""
    trainable, total = row.get("trainable_parameters"), row.get("total_parameters")
    if isinstance(trainable, int) and isinstance(total, int) and total > 0:
        row["trainable_parameter_fraction"] = trainable / total
    return {column: row.get(column) for column in CONSOLIDATED_COLUMNS}


def _selected_checkpoint(record: RunRecord) -> tuple[str | None, str | None]:
    """The checkpoint a long-training run's reported numbers came from, and its digest."""
    name = "best_checkpoint.pt"
    if name in record.artefact_digests:
        return str(record.run_dir / name), record.artefact_digests[name]
    return None, None


def _rows_from_baseline(record: RunRecord, metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    checkpoint, digest = _selected_checkpoint(record)
    shared = {
        "evaluation_set": "in_distribution_test",
        "fine_tune_mode": record.fine_tune_mode,
        "adaptation_percentage": 0.0,
        "operating_point": "default",
        "threshold_provenance": "fixed_prior_from_config_model.decision_threshold",
        "best_epoch": _as_int(metrics.get("best_epoch")),
        "best_validation_score": _as_float(metrics.get("best_validation_score")),
        "selected_checkpoint": checkpoint,
        "selected_checkpoint_sha256": digest,
    }
    rows: list[dict[str, Any]] = []

    overall = _base_row(record, "in_distribution_baseline")
    overall.update(shared, condition="overall", generator=None)
    overall.update(_metric_block(metrics.get("overall")))
    rows.append(_finalise_row(overall))

    for name, block in sorted((metrics.get("per_generator") or {}).items()):
        row = _base_row(record, "in_distribution_baseline")
        row.update(shared, condition=f"per_generator:{name}", generator=name)
        row.update(_metric_block(block))
        rows.append(_finalise_row(row))
    return rows


def _rows_from_unseen_generator(
    record: RunRecord, metrics: Mapping[str, Any]
) -> list[dict[str, Any]]:
    held_out = metrics.get("held_out_generator")
    checkpoint, digest = _selected_checkpoint(record)
    thresholds = metrics.get("thresholds") or {}
    shared = {
        "held_out_generator": held_out,
        "fine_tune_mode": "none",
        "adaptation_percentage": 0.0,
        "best_epoch": _as_int(metrics.get("best_epoch")),
        "best_validation_score": _as_float(metrics.get("best_validation_score")),
        "selected_checkpoint": checkpoint,
        "selected_checkpoint_sha256": digest,
        "manifest_sha256": metrics.get("manifest_sha256"),
    }
    default_provenance = ((thresholds.get("default") or {}).get("provenance")) or None
    selected_provenance = (
        (thresholds.get("baseline_validation_selected") or {}).get("provenance")
    ) or None
    rows: list[dict[str, Any]] = []

    for evaluation_set, key in (
        ("in_distribution_test", "in_distribution_test"),
        ("unseen_test", "unseen_test"),
    ):
        block = metrics.get(key) or {}
        for operating_point, sub_key, provenance in (
            ("default", "at_default_threshold", default_provenance),
            ("validation_selected", "at_validation_selected_threshold", selected_provenance),
        ):
            if sub_key not in block:
                continue
            row = _base_row(record, str(metrics.get("protocol") or "leave_one_generator_out"))
            row.update(
                shared,
                evaluation_set=evaluation_set,
                condition="overall",
                generator=None,
                operating_point=operating_point,
                threshold_provenance=provenance,
            )
            row.update(_metric_block(block[sub_key]))
            rows.append(_finalise_row(row))

        for name, per_generator in sorted((block.get("per_generator") or {}).items()):
            row = _base_row(record, str(metrics.get("protocol") or "leave_one_generator_out"))
            row.update(
                shared,
                evaluation_set=evaluation_set,
                condition=f"per_generator:{name}",
                generator=name,
                operating_point="default",
                threshold_provenance=default_provenance,
            )
            row.update(_metric_block(per_generator))
            rows.append(_finalise_row(row))
    return rows


_CELL_OPERATING_POINTS: tuple[tuple[str, str], ...] = (
    ("default", "overall"),
    ("adaptation_selected", "at_adaptation_selected_threshold"),
    ("baseline_unchanged", "at_baseline_threshold"),
)

_CELL_THRESHOLD_KEYS: Mapping[str, str] = {
    "adaptation_selected": "adaptation_validation_selected",
    "baseline_unchanged": "baseline_unchanged",
}


def _rows_from_cells(record: RunRecord, metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the recovery/ablation grid. Both protocols share the cell schema."""
    protocol = str(metrics.get("protocol") or record.experiment_type)
    manifest_sha = metrics.get("manifest_sha256")
    rows: list[dict[str, Any]] = []
    for cell in metrics.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        cell_thresholds = cell.get("thresholds") or {}
        checkpoint_digests = cell.get("cell_checkpoint_sha256") or {}
        shared = {
            "evaluation_set": "unseen_test",
            "held_out_generator": cell.get("held_out_generator")
            or metrics.get("held_out_generator"),
            "fine_tune_mode": cell.get("fine_tune_mode"),
            "adaptation_percentage": _as_float(cell.get("adaptation_percentage")),
            "subset_seed": _as_int(cell.get("subset_seed")),
            "training_seed": _as_int(cell.get("training_seed")),
            "trainable_parameters": _as_int(cell.get("trainable_parameters")),
            "total_parameters": _as_int(cell.get("total_parameters")),
            "learning_rate": _as_float(cell.get("learning_rate")) or record.learning_rate,
            "epochs": _as_int(cell.get("epochs")) or record.epochs,
            "best_epoch": _as_int(cell.get("best_epoch")),
            "best_validation_score": _as_float(cell.get("best_adaptation_validation_score")),
            "training_seconds": _as_float(cell.get("training_seconds")),
            "labelled_images_consumed": _as_int(cell.get("labelled_images_consumed")),
            "held_out_fake_count": _as_int(cell.get("held_out_fake_count")),
            "starting_checkpoint": cell.get("starting_checkpoint"),
            "selected_checkpoint_sha256": checkpoint_digests.get("best_checkpoint.pt"),
            "manifest_sha256": manifest_sha,
        }

        for operating_point, block_key in _CELL_OPERATING_POINTS:
            block = cell.get(block_key)
            if not isinstance(block, dict):
                continue
            threshold_key = _CELL_THRESHOLD_KEYS.get(operating_point)
            provenance = (
                (cell_thresholds.get(threshold_key) or {}).get("provenance")
                if threshold_key
                else "fixed_prior_from_config_model.decision_threshold"
            )
            row = _base_row(record, protocol)
            row.update(
                shared,
                condition=str(cell.get("cell_id") or "cell"),
                generator=None,
                operating_point=operating_point,
                threshold_provenance=provenance,
            )
            row.update(_metric_block(block))
            rows.append(_finalise_row(row))

        for name, per_generator in sorted((cell.get("per_generator") or {}).items()):
            row = _base_row(record, protocol)
            row.update(
                shared,
                condition=f"{cell.get('cell_id') or 'cell'}:per_generator:{name}",
                generator=name,
                operating_point="default",
                threshold_provenance="fixed_prior_from_config_model.decision_threshold",
            )
            row.update(_metric_block(per_generator))
            rows.append(_finalise_row(row))
    return rows


_ROW_BUILDERS = {
    "baseline": _rows_from_baseline,
    "unseen_generator": _rows_from_unseen_generator,
    "fine_tuning": _rows_from_cells,
    "ablation": _rows_from_cells,
}


def consolidate_runs(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    """Flatten every given run into one tidy table of evaluated conditions."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if not record.has_metrics:
            continue
        builder = _ROW_BUILDERS.get(record.experiment_type)
        if builder is None:
            continue
        rows.extend(builder(record, load_metrics(record)))
    return rows


def summarise_runs(
    records: Sequence[RunRecord], consolidated: Sequence[Mapping[str, Any]] = ()
) -> list[dict[str, Any]]:
    """One row per discovered run: what it was, whether it finished, and its environment."""
    condition_counts: dict[str, int] = {}
    for row in consolidated:
        run_id = str(row.get("run_id"))
        condition_counts[run_id] = condition_counts.get(run_id, 0) + 1

    summary: list[dict[str, Any]] = []
    for record in records:
        packages = record.environment.get("packages") or {}
        cells: int | None = None
        if record.has_metrics and record.experiment_type in {"fine_tuning", "ablation"}:
            payload = load_metrics(record)
            cells = len(payload.get("cells") or [])
        summary.append(
            {
                "run_id": record.run_id,
                "experiment_type": record.experiment_type,
                "experiment_name": record.experiment_name,
                "status": record.status,
                "is_synthetic_smoke": record.is_synthetic_smoke,
                "started_at": record.started_at,
                "held_out_generator": record.held_out_generator,
                "fine_tune_mode": record.fine_tune_mode,
                "head_type": record.head_type,
                "seed": record.seed,
                "epochs": record.epochs,
                "learning_rate": record.learning_rate,
                "batch_size": record.batch_size,
                "model_name": record.model_name,
                "manifest_path": record.manifest_path,
                "manifest_sha256": next(
                    (
                        row.get("manifest_sha256")
                        for row in consolidated
                        if row.get("run_id") == record.run_id and row.get("manifest_sha256")
                    ),
                    None,
                ),
                "evaluated_conditions": condition_counts.get(record.run_id, 0),
                "cells": cells,
                "python": record.environment.get("python"),
                "torch": packages.get("torch"),
                "transformers": packages.get("transformers"),
                "cuda_available": record.environment.get("cuda_available"),
                "platform": record.environment.get("platform"),
                "run_dir": str(record.run_dir),
            }
        )
    return summary


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _standard_deviation(values: Sequence[float]) -> float:
    """Sample standard deviation; 0.0 for a single run, where spread is unmeasured."""
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def in_distribution_reference(
    records: Sequence[RunRecord],
    metric: str,
    *,
    held_out_generator: str | None = None,
    starting_checkpoint: str | None = None,
) -> tuple[float | None, str | None, str | None]:
    """The prevalence-matched in-distribution value a recovery curve is measured against."""
    gap_key = {"f1": "f1_at_default_threshold"}.get(metric, metric)

    def reference_of(record: RunRecord) -> float | None:
        if not record.has_metrics:
            return None
        entry = (load_metrics(record).get("generalisation_gap") or {}).get(gap_key)
        return _as_float(entry.get("in_distribution")) if isinstance(entry, dict) else None

    unseen_runs = [record for record in records if record.experiment_type == "unseen_generator"]

    if starting_checkpoint:
        owner = Path(starting_checkpoint).parent.name
        for record in unseen_runs:
            if record.run_id != owner:
                continue
            value = reference_of(record)
            if value is not None:
                return value, record.run_id, "starting_checkpoint"
        return None, None, None

    for record in reversed(unseen_runs):
        if not record.has_metrics:
            continue
        if held_out_generator and load_metrics(record).get("held_out_generator") != (
            held_out_generator
        ):
            continue
        value = reference_of(record)
        if value is not None:
            return value, record.run_id, "held_out_generator"
    return None, None, None


def summarise_recovery(
    consolidated: Sequence[Mapping[str, Any]],
    *,
    records: Sequence[RunRecord] = (),
    metrics: Sequence[str] = SUMMARY_METRICS,
) -> list[dict[str, Any]]:
    """Aggregate adapted cells by budget and depth, with uncertainty and recovery."""
    cell_rows = [
        row
        for row in consolidated
        if row.get("experiment_type") in {"fine_tuning", "ablation"}
        and row.get("generator") is None
        and row.get("adaptation_percentage") is not None
    ]
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in cell_rows:
        key = (
            row.get("run_id"),
            row.get("held_out_generator"),
            row.get("fine_tune_mode"),
            row.get("head_type"),
            float(row["adaptation_percentage"]),
            row.get("operating_point"),
        )
        grouped.setdefault(key, []).append(row)

    zero_by_key: dict[tuple[Any, Any, str], float] = {}
    for (run_id, _held_out, mode, _head, percentage, operating_point), rows in grouped.items():
        if percentage != 0.0:
            continue
        for metric in metrics:
            values = [
                value for row in rows if (value := _as_float(row.get(metric))) is not None
            ]
            if values:
                zero_by_key[(run_id, operating_point, metric)] = _mean(values)
        del mode

    summary: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        run_id, held_out, mode, head_type, percentage, operating_point = key
        starting_checkpoint = next(
            (
                str(row["starting_checkpoint"])
                for row in rows
                if row.get("starting_checkpoint")
            ),
            None,
        )
        if percentage == 0.0:
            continue
        for metric in metrics:
            values = [
                value for row in rows if (value := _as_float(row.get(metric))) is not None
            ]
            if not values:
                continue
            mean = _mean(values)
            deviation = _standard_deviation(values)
            zero_value = zero_by_key.get((run_id, operating_point, metric))

            comparable = metric in THRESHOLD_FREE_METRICS or operating_point == "default"
            reference, reference_run, reference_match = (
                in_distribution_reference(
                    records,
                    metric,
                    held_out_generator=held_out,
                    starting_checkpoint=starting_checkpoint,
                )
                if comparable
                else (None, None, None)
            )

            absolute = None if zero_value is None else mean - zero_value
            relative = (
                absolute / zero_value if absolute is not None and zero_value else None
            )
            gap = (
                reference - zero_value
                if reference is not None and zero_value is not None
                else None
            )
            gap_closed = absolute / gap if absolute is not None and gap not in (None, 0.0) else None

            summary.append(
                {
                    "run_id": run_id,
                    "held_out_generator": held_out,
                    "fine_tune_mode": mode,
                    "head_type": head_type,
                    "adaptation_percentage": percentage,
                    "operating_point": operating_point,
                    "metric": metric,
                    "runs": len(values),
                    "mean": mean,
                    "standard_deviation": deviation,
                    "standard_error": (
                        deviation / math.sqrt(len(values)) if len(values) > 1 else None
                    ),
                    "minimum": min(values),
                    "maximum": max(values),
                    "zero_percent_reference": zero_value,
                    "absolute_recovery": absolute,
                    "relative_improvement": relative,
                    "in_distribution_reference": reference,
                    "generalisation_gap": gap,
                    "gap_closed_fraction": gap_closed,
                    "gap_closed_is_reliable": (
                        None if gap is None else abs(gap) >= MINIMUM_RELIABLE_GAP
                    ),
                    "labelled_images_consumed": next(
                        (
                            row.get("labelled_images_consumed")
                            for row in rows
                            if row.get("labelled_images_consumed") is not None
                        ),
                        None,
                    ),
                    "trainable_parameters": next(
                        (
                            row.get("trainable_parameters")
                            for row in rows
                            if row.get("trainable_parameters") is not None
                        ),
                        None,
                    ),
                    "in_distribution_reference_run_id": reference_run,
                    "in_distribution_reference_match": reference_match,
                }
            )
    return summary


def degradation_rows(consolidated: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """In-distribution versus unseen, per held-out generator, at each operating point."""
    by_key: dict[tuple[Any, Any, Any], dict[str, Mapping[str, Any]]] = {}
    for row in consolidated:
        if row.get("experiment_type") != "unseen_generator" or row.get("condition") != "overall":
            continue
        key = (row.get("run_id"), row.get("held_out_generator"), row.get("operating_point"))
        by_key.setdefault(key, {})[str(row.get("evaluation_set"))] = row

    rows: list[dict[str, Any]] = []
    for (run_id, held_out, operating_point), sides in sorted(
        by_key.items(), key=lambda item: tuple(str(part) for part in item[0])
    ):
        in_distribution = sides.get("in_distribution_test")
        unseen = sides.get("unseen_test")
        if unseen is None:
            continue
        for metric in SUMMARY_METRICS:
            unseen_value = _as_float(unseen.get(metric))
            reference = (
                _as_float(in_distribution.get(metric)) if in_distribution is not None else None
            )
            if unseen_value is None and reference is None:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "held_out_generator": held_out,
                    "head_type": unseen.get("head_type"),
                    "operating_point": operating_point,
                    "metric": metric,
                    "in_distribution": reference,
                    "unseen": unseen_value,
                    "absolute_drop": (
                        reference - unseen_value
                        if reference is not None and unseen_value is not None
                        else None
                    ),
                    "relative_drop": (
                        (reference - unseen_value) / reference
                        if reference not in (None, 0.0) and unseen_value is not None
                        else None
                    ),
                    "in_distribution_prevalence": (
                        in_distribution.get("positive_prevalence")
                        if in_distribution is not None
                        else None
                    ),
                    "unseen_prevalence": unseen.get("positive_prevalence"),
                    "unseen_support": unseen.get("support"),
                }
            )
    return rows


def write_table(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str], destination: Path
) -> Path:
    """Write a tidy CSV with a declared column order. Missing keys become empty cells."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    return destination
