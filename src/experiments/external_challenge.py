"""Score frozen detectors, once, on the external contemporary-generator challenge set.

This is step 6 of the pre-registered protocol in section 15 of ``RESULTS_NOTES.md``:
evaluate the frozen detector on the external set and report separately. It is
inference-only by construction. There is no optimiser, no loss, no checkpoint writing
and no threshold search anywhere in this module, so an external image cannot influence
any model, any operating point, or any internal result.

Three guarantees are asserted at runtime rather than trusted:

* **The set is the audited one.** The manifest digest recorded in the build audit must
  still match the manifest on disk, and every leakage check in that audit must have
  passed, or the run refuses to start.
* **The checkpoints are the reported ones.** Each nominated checkpoint is hashed and
  compared against the digest its own run recorded. A config that pins a digest must
  match it too.
* **No operating point is chosen here.** Metrics are reported at the project-wide default
  threshold and, additionally, at whatever threshold the checkpoint's own run selected on
  *its* development validation data. Both were fixed before any external image existed.

Each detector is also reported alongside the internal reference numbers already saved in
its own run directory, read read-only, so the external drop is read against a number this
project already published rather than a re-derived one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.datasets.detector_dataset import AIDetectionDataset
from src.datasets.external_challenge import (
    EXTERNAL_GENERATOR_NAME,
    EXTERNAL_SPLIT_NAME,
)
from src.datasets.schema import DatasetRecord
from src.evaluation.evaluator import save_predictions
from src.evaluation.metrics import compute_binary_metrics
from src.experiments.common import (
    build_detector,
    evaluate_records,
    finalise_run,
    prepare_experiment,
    resolve_runtime_paths,
)
from src.models.checkpointing import load_checkpoint
from src.utils.config import load_config

LOGGER = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_external_protocol(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Confirm the config describes an evaluation-only external challenge."""

    section = config.get("external_challenge")
    if not isinstance(section, Mapping):
        raise ValueError("experiment.type=external_challenge requires an external_challenge block")
    if section.get("adaptation_permitted") is not False:
        raise ValueError("external_challenge.adaptation_permitted must be false")
    if section.get("threshold_reselection_permitted") is not False:
        raise ValueError("external_challenge.threshold_reselection_permitted must be false")
    if section.get("split_name") != EXTERNAL_SPLIT_NAME:
        raise ValueError(
            f"external_challenge.split_name must be {EXTERNAL_SPLIT_NAME!r} so external "
            "predictions can never be mistaken for an internal partition"
        )
    checkpoints = section.get("frozen_checkpoints")
    if not isinstance(checkpoints, Sequence) or not checkpoints:
        raise ValueError("external_challenge.frozen_checkpoints must list at least one checkpoint")
    run_ids = [str(entry["run_id"]) for entry in checkpoints]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("external_challenge.frozen_checkpoints lists a run twice")
    primary = str(section.get("primary_checkpoint_run_id", ""))
    if primary not in run_ids:
        raise ValueError(
            "external_challenge.primary_checkpoint_run_id must nominate one of the "
            "frozen_checkpoints, so the headline detector is fixed in advance"
        )
    if section.get("generator_identity_known") is not False:
        raise ValueError(
            "this challenge used the assistant-mediated route, whose underlying image "
            "model is not reported; generator_identity_known must be false"
        )
    return section


def verify_audited_manifest(manifest_path: Path, audit_path: Path) -> dict[str, Any]:
    """Refuse to evaluate a manifest that is not the audited, leakage-checked one."""

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"external manifest not found: {manifest_path}. Build it first with "
            "python -m scripts.build_external_manifest"
        )
    if not audit_path.is_file():
        raise FileNotFoundError(f"external audit record not found: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    digest = _sha256_file(manifest_path)
    recorded = str(audit.get("manifest_sha256", ""))
    if recorded and recorded != digest:
        raise ValueError(
            "the external manifest has changed since it was audited "
            f"(audit {recorded[:16]}..., on disk {digest[:16]}...); rebuild it"
        )
    failed = [
        str(check.get("check"))
        for check in audit.get("leakage_checks", [])
        if not check.get("passed")
    ]
    if failed:
        raise ValueError(f"external set failed its own leakage checks: {failed}")
    for flag in ("used_for_training", "used_for_validation", "used_for_threshold_selection"):
        if audit.get(flag) is not False:
            raise ValueError(f"external audit does not assert {flag} is false")
    return {"manifest_sha256": digest, "audit": audit}


def load_external_records(
    manifest_path: Path, *, data_root: Path
) -> list[DatasetRecord]:
    """Load the external evaluation records, metadata only, and check their composition."""

    def _no_decode(image: Any) -> Any:
        raise RuntimeError("external manifest load must not decode images")

    dataset = AIDetectionDataset.from_manifest(
        manifest_path, data_root=data_root, transform=_no_decode
    )
    records = sorted(dataset.records, key=lambda record: record.sample_id)
    fakes = [record for record in records if record.label == 1]
    reals = [record for record in records if record.label == 0]
    if not fakes or not reals:
        raise ValueError("the external evaluation set must contain both classes")
    if len(fakes) != len(reals):
        raise ValueError(
            f"the external evaluation set must be class balanced; found {len(fakes)} fake "
            f"and {len(reals)} real"
        )
    generators = {record.generator for record in fakes}
    if generators != {EXTERNAL_GENERATOR_NAME}:
        raise ValueError(
            f"external fakes must all carry generator {EXTERNAL_GENERATOR_NAME!r}; "
            f"found {sorted(generators)}"
        )
    return records


def _internal_reference(run_dir: Path) -> dict[str, Any]:
    """Read the internal numbers this checkpoint's own run already published.

    Read-only. Nothing internal is recomputed, so the external result is compared with
    the exact figures the internal chapters report.
    """

    unseen_metrics = run_dir / "unseen_generator_metrics.json"
    if unseen_metrics.is_file():
        payload = json.loads(unseen_metrics.read_text(encoding="utf-8"))
        return {
            "source": unseen_metrics.name,
            "protocol": "leave_one_generator_out",
            "held_out_generator": payload.get("held_out_generator"),
            "in_distribution_test": payload.get("in_distribution_test", {}).get(
                "at_default_threshold"
            ),
            "unseen_test": payload.get("unseen_test", {}).get("at_default_threshold"),
            "thresholds": payload.get("thresholds"),
        }
    baseline_metrics = run_dir / "test_metrics.json"
    if baseline_metrics.is_file():
        payload = json.loads(baseline_metrics.read_text(encoding="utf-8"))
        return {
            "source": baseline_metrics.name,
            "protocol": "baseline_all_generators_seen",
            "held_out_generator": None,
            "in_distribution_test": payload.get("overall"),
            "per_generator": payload.get("per_generator"),
            "thresholds": {
                "default": {
                    "value": float(payload.get("overall", {}).get("threshold", 0.5)),
                    "provenance": "fixed_prior_from_config_model.decision_threshold",
                }
            },
        }
    raise FileNotFoundError(f"no saved internal metrics found in {run_dir}")


def _resolve_checkpoint(entry: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """Locate one nominated checkpoint and confirm it is the artefact its run recorded."""

    run_id = str(entry["run_id"])
    run_dir = (output_root / run_id).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"nominated run directory not found: {run_dir}")
    name = str(entry.get("checkpoint", "best_checkpoint.pt"))
    checkpoint_path = run_dir / name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"nominated checkpoint {name} is absent from {run_id}. Recovery and ablation "
            "cells retained digests only, so their adapted weights cannot be scored "
            "without re-running training."
        )
    digest = _sha256_file(checkpoint_path)
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    recorded_sidecar = (
        sidecar.read_text(encoding="ascii").split()[0] if sidecar.is_file() else None
    )
    if recorded_sidecar and recorded_sidecar != digest:
        raise ValueError(f"{run_id}/{name} does not match its recorded digest")
    declared = str(entry.get("sha256", "")).strip().lower()
    if declared and declared != digest:
        raise ValueError(
            f"{run_id}/{name} digest {digest[:16]}... does not match the digest pinned in "
            f"the config ({declared[:16]}...)"
        )
    artefacts_path = run_dir / "artefacts.json"
    recorded_artefact = None
    if artefacts_path.is_file():
        for item in json.loads(artefacts_path.read_text(encoding="utf-8")):
            if str(item.get("path")) == name:
                recorded_artefact = str(item.get("sha256"))
    if recorded_artefact and recorded_artefact != digest:
        raise ValueError(f"{run_id}/{name} does not match the digest in artefacts.json")
    return {
        "run_id": run_id,
        "role": str(entry.get("role", "")),
        "note": str(entry.get("note", "")).strip(),
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": name,
        "sha256": digest,
        "digest_matches_run_sidecar": recorded_sidecar == digest if recorded_sidecar else None,
        "digest_matches_artefacts_json": (
            recorded_artefact == digest if recorded_artefact else None
        ),
    }


def _config_for_checkpoint(
    config: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild the model section from the checkpoint's own resolved config.

    A checkpoint carries the architecture it was trained with -- notably ``head_type``,
    which differs between the linear and cosine variants -- so the detector is
    constructed from the checkpoint's record rather than from this config's defaults.
    The decision threshold stays the project-wide default from this config.
    """

    saved = checkpoint.get("resolved_config")
    if not isinstance(saved, Mapping) or "model" not in saved:
        raise ValueError("checkpoint does not carry the resolved config it was trained under")
    merged = copy.deepcopy(dict(config))
    model = copy.deepcopy(dict(saved["model"]))
    model["decision_threshold"] = float(config["model"]["decision_threshold"])
    merged["model"] = model
    merged["transforms"] = copy.deepcopy(dict(saved.get("transforms", config["transforms"])))
    return merged


def run_external_challenge(config_path: Path) -> Path:
    """Evaluate every nominated frozen detector once on the external challenge set."""

    loaded = load_config(config_path)
    if loaded.values["experiment"]["type"] != "external_challenge":
        raise ValueError("run_external_challenge requires experiment.type=external_challenge")
    config = resolve_runtime_paths(loaded.values, loaded.source_path)
    section = validate_external_protocol(config)
    project_root = loaded.source_path.parent.parent
    manifest_path = (project_root / str(section["manifest_path"])).resolve()
    audit_path = (project_root / str(section["audit_path"])).resolve()
    verified = verify_audited_manifest(manifest_path, audit_path)
    data_root = Path(str(config["data"]["root"]))
    records = load_external_records(manifest_path, data_root=data_root)

    output_root = Path(str(config["project"]["output_root"])).resolve()
    context = prepare_experiment(config)
    logger = logging.getLogger(f"ai_detector.{context.run_id}")
    try:
        resolved = [
            _resolve_checkpoint(entry, output_root)
            for entry in section["frozen_checkpoints"]
        ]
        logger.info(
            "external challenge %s: %d images (%d fake / %d real), %d frozen detectors, "
            "manifest_sha256=%s",
            section["challenge_id"],
            len(records),
            sum(1 for record in records if record.label == 1),
            sum(1 for record in records if record.label == 0),
            len(resolved),
            verified["manifest_sha256"],
        )

        default_threshold = float(config["model"]["decision_threshold"])
        detectors: list[dict[str, Any]] = []
        for item in resolved:
            checkpoint = load_checkpoint(
                item["checkpoint_path"], map_location=str(context.device)
            )
            evaluation_config = _config_for_checkpoint(config, checkpoint)
            model = build_detector(evaluation_config, device=context.device)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            outcome = evaluate_records(
                model=model,
                records=records,
                config=evaluation_config,
                device=context.device,
                split_name=EXTERNAL_SPLIT_NAME,
                checkpoint_id=f"{item['run_id']}/{item['checkpoint_name']}",
            )
            predictions_path = (
                context.run_dir / f"external_test_predictions__{item['role'] or item['run_id']}.csv"
            )
            save_predictions(outcome.predictions, predictions_path)
            labels = [record.label for record in outcome.predictions]
            scores = [record.score for record in outcome.predictions]

            reference = _internal_reference(item["run_dir"])
            selected = (reference.get("thresholds") or {}).get("baseline_validation_selected")
            at_selected: dict[str, Any] | None = None
            if isinstance(selected, Mapping) and selected.get("value") is not None:
                at_selected = {
                    "threshold": float(selected["value"]),
                    "provenance": str(selected.get("provenance", "")),
                    "fixed_before_external_data_existed": True,
                    "metrics": asdict(
                        compute_binary_metrics(
                            labels, scores, threshold=float(selected["value"])
                        )
                    ),
                }

            metadata = checkpoint.get("metadata", {})
            detectors.append(
                {
                    "run_id": item["run_id"],
                    "role": item["role"],
                    "note": item["note"],
                    "is_primary": item["run_id"] == str(section["primary_checkpoint_run_id"]),
                    "checkpoint": item["checkpoint_name"],
                    "checkpoint_sha256": item["sha256"],
                    "checkpoint_digest_matches_run_sidecar": item["digest_matches_run_sidecar"],
                    "checkpoint_digest_matches_artefacts_json": item[
                        "digest_matches_artefacts_json"
                    ],
                    "head_type": str(evaluation_config["model"].get("head_type", "linear")),
                    "checkpoint_metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
                    "seed": (
                        int(metadata["seed"]) if isinstance(metadata, Mapping)
                        and metadata.get("seed") is not None else context.seed
                    ),
                    "predictions_file": predictions_path.name,
                    "external_test": {
                        "at_default_threshold": asdict(outcome.overall),
                        "at_run_selected_threshold": at_selected,
                        "per_generator": {
                            name: asdict(value)
                            for name, value in outcome.per_generator.items()
                        },
                    },
                    "internal_reference": reference,
                }
            )
            logger.info(
                "external %s (%s): roc_auc=%s pr_auc=%s f1=%.4f recall=%.4f",
                item["run_id"],
                item["role"],
                f"{outcome.overall.roc_auc:.4f}"
                if outcome.overall.roc_auc is not None
                else "undefined",
                f"{outcome.overall.average_precision:.4f}"
                if outcome.overall.average_precision is not None
                else "undefined",
                outcome.overall.f1,
                outcome.overall.recall,
            )

        payload = {
            "protocol": "external_contemporary_generator_challenge",
            "challenge_id": str(section["challenge_id"]),
            "status": "executed",
            "generation_route": str(section["generation_route"]),
            "generator_recorded_as": EXTERNAL_GENERATOR_NAME,
            "generator_identity_known": False,
            "architectural_claim_permitted": False,
            "adaptation_permitted": False,
            "threshold_reselection_permitted": False,
            "threshold_provenance": {
                "default": {
                    "value": default_threshold,
                    "provenance": "fixed_prior_from_config_model.decision_threshold",
                },
                "run_selected": (
                    "each detector's own development-validation-selected threshold, "
                    "recorded in its run directory before this external set existed"
                ),
            },
            "seed": context.seed,
            "split_name": EXTERNAL_SPLIT_NAME,
            "manifest_path": str(manifest_path),
            "manifest_sha256": verified["manifest_sha256"],
            "audit_path": str(audit_path),
            "evaluation_composition": {
                "total": len(records),
                "external_fake": sum(1 for record in records if record.label == 1),
                "authentic_comparator": sum(1 for record in records if record.label == 0),
                "positive_prevalence": sum(1 for record in records if record.label == 1)
                / len(records),
                "external_set_sha256": hashlib.sha256(
                    "|".join(record.sample_id for record in records).encode("utf-8")
                ).hexdigest(),
            },
            "build_audit": verified["audit"],
            "detectors": detectors,
            "isolation_rules_observed": [
                "external results are never merged into the internal benchmark tables",
                "no threshold was re-selected on external data",
                "no model was trained, fine-tuned, or selected in this run",
                "the internal benchmark was not re-run or re-reported",
            ],
            "not_done": [
                "step 7 of the protocol (limited-data recovery against the external set) "
                "was not run: the recovery and ablation cells retained checkpoint digests "
                "only, so no adapted detector exists to score without re-training",
                "the recommended 250-per-class sample size was not reached; 100 generated "
                "images were available, giving the minimum-reportable tier",
            ],
        }
        (context.run_dir / "external_challenge_metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        finalise_run(context, status="completed")
        return context.run_dir
    except Exception:
        logger.exception("external challenge evaluation failed")
        finalise_run(context, status="failed")
        raise
