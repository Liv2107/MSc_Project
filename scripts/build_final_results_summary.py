"""Consolidate every reported experimental cell into one machine-readable summary.

Read-only with respect to ``outputs/``. Nothing here trains, infers, or edits a saved
artefact: every number is copied from a completed run's own saved metrics file, so the
summary cannot disagree with the run directories it describes.

Three tables are written:

* ``final_results_summary.csv`` -- one row per (run, evaluation set, cell) at the
  project-wide default 0.5 threshold. This is the table the Results chapter is built
  from.
* ``final_results_by_threshold.csv`` -- the same cells at every operating point a run
  recorded (default, its development-validation-selected threshold, and for adapted
  cells the adaptation-selected threshold). Threshold-free metrics repeat by design.
* ``final_results_summary.json`` -- both tables plus the provenance block: run
  inventory, manifest digests, checkpoint digests, and what is deliberately absent.

Rows carry ``row_role``: ``primary`` for the cell a dissertation table should cite, and
``reproduction`` where a second run scored the identical cell (the head-only budgets
appear in both the standalone recovery run and the depth ablation, and agree exactly).

Usage:
    python -m scripts.build_final_results_summary
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Written alongside the exports, because ``outputs/report/`` is documented as
#: safe to delete and regenerate. Kept as a template file so its markdown tables
#: are not subject to the source line-length rule.
FINAL_RESULTS_README_TEMPLATE = (
    Path(__file__).with_name("report_readmes") / "final_results.md"
)

#: Runs excluded from the summary, with the reason recorded rather than implied.
EXCLUDED_RUNS = {
    "unseen_generator-20260808T144550443491Z-b386ea5c10-bb06": "synthetic smoke run",
    "fine_tuning-20260808T144640138049Z-6ec6cf50a2-f38d": "synthetic smoke run",
    "unseen_generator-20260808T150151025524Z-c74c3e0db3-21d6": "status failed",
    "ablation-20260809T194436211721Z-fc1a22d8e3-5389": "incomplete: no ablation_metrics.json",
    "ablation-20260816T162529520026Z-fc1a22d8e3-f09d": "incomplete: no ablation_metrics.json",
}

COLUMNS = (
    "summary_row_id",
    "run_id",
    "protocol",
    "experiment_name",
    "evaluation_set",
    "dataset",
    "generator",
    "generator_identity_known",
    "recovery_percentage",
    "fine_tune_strategy",
    "head_type",
    "development_training_samples",
    "recovery_labelled_images_total",
    "recovery_labelled_fake_images",
    "evaluation_samples",
    "positive_prevalence",
    "threshold",
    "threshold_role",
    "threshold_provenance",
    "roc_auc",
    "pr_auc",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
    "trainable_parameters",
    "total_parameters",
    "training_seconds",
    "best_epoch",
    "seed",
    "subset_seed",
    "starting_checkpoint",
    "checkpoint_sha256",
    "config_reference",
    "manifest_sha256",
    "predictions_file",
    "row_role",
    "notes",
)

DATASET_INTERNAL = "tiny_genimage"
DATASET_EXTERNAL = "external_challenge_v1_astra_mediated"


def _load_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _status(run_dir: Path) -> str | None:
    path = run_dir / "status.json"
    if not path.is_file():
        return None
    return str(_load_json(path).get("status"))


def _experiment_name(run_dir: Path) -> str:
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        return ""
    match = re.search(r"^\s*name:\s*(\S+)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else ""


def _head_type(run_dir: Path) -> str:
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        return "linear"
    match = re.search(r"^\s*head_type:\s*(\S+)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else "linear"


def _seed(run_dir: Path) -> int | None:
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        return None
    match = re.search(r"^\s*seed:\s*(\d+)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    return int(match.group(1)) if match else None


def _logged_selection_counts(run_dir: Path) -> dict[str, int]:
    """Recover the development selection sizes the run logged when it built them."""

    log = run_dir / "run.log"
    if not log.is_file():
        return {}
    text = log.read_text(encoding="utf-8", errors="replace")
    counts: dict[str, int] = {}
    for key in ("train", "validation"):
        match = re.search(rf"\b{key}=(\d+)", text)
        if match:
            counts[key] = int(match.group(1))
    return counts


def _checkpoint_digest(run_dir: Path, name: str = "best_checkpoint.pt") -> str:
    sidecar = run_dir / f"{name}.sha256"
    if sidecar.is_file():
        return sidecar.read_text(encoding="ascii").split()[0]
    return ""


def _metric_row(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "threshold": metrics.get("threshold"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("average_precision"),
        "f1": metrics.get("f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "accuracy": metrics.get("accuracy"),
        "true_positive": metrics.get("true_positive"),
        "true_negative": metrics.get("true_negative"),
        "false_positive": metrics.get("false_positive"),
        "false_negative": metrics.get("false_negative"),
        "evaluation_samples": metrics.get("support"),
    }


def _base(run_dir: Path, protocol: str) -> dict[str, Any]:
    return {
        "run_id": run_dir.name,
        "protocol": protocol,
        "experiment_name": _experiment_name(run_dir),
        "head_type": _head_type(run_dir),
        "seed": _seed(run_dir),
        "config_reference": (run_dir / "resolved_config.yaml").as_posix(),
        "generator_identity_known": True,
        "dataset": DATASET_INTERNAL,
        "row_role": "primary",
        "notes": "",
    }


def baseline_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics = _load_json(run_dir / "test_metrics.json")
    counts = _logged_selection_counts(run_dir)
    row = {
        **_base(run_dir, "baseline"),
        "evaluation_set": "in_distribution_test_all_seven_generators",
        "generator": "all_seven_known",
        "recovery_percentage": None,
        "fine_tune_strategy": "head_only_initial_training",
        "development_training_samples": counts.get("train"),
        "recovery_labelled_images_total": None,
        "recovery_labelled_fake_images": None,
        "threshold_role": "default",
        "threshold_provenance": "fixed_prior_from_config_model.decision_threshold",
        "best_epoch": metrics.get("best_epoch"),
        "starting_checkpoint": "",
        "checkpoint_sha256": _checkpoint_digest(run_dir),
        "manifest_sha256": "",
        "predictions_file": "test_predictions.csv",
        **_metric_row(metrics["overall"]),
        "notes": (
            "reference detector trained on all seven Tiny GenImage generators; "
            "not class balanced (prevalence 0.5 by construction of the split)"
        ),
    }
    row["positive_prevalence"] = None
    return [row]


def unseen_generator_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics = _load_json(run_dir / "unseen_generator_metrics.json")
    held_out = str(metrics["held_out_generator"])
    counts = metrics.get("sample_counts", {})
    thresholds = metrics.get("thresholds", {})
    selected = thresholds.get("baseline_validation_selected", {})
    shared = {
        **_base(run_dir, "unseen_generator"),
        "recovery_percentage": 0.0,
        "fine_tune_strategy": "none",
        "development_training_samples": counts.get("train"),
        "recovery_labelled_images_total": 0,
        "recovery_labelled_fake_images": 0,
        "best_epoch": metrics.get("best_epoch"),
        "starting_checkpoint": "",
        "checkpoint_sha256": _checkpoint_digest(run_dir),
        "manifest_sha256": metrics.get("manifest_sha256", ""),
    }
    rows: list[dict[str, Any]] = []
    in_distribution = metrics["in_distribution_test"]["at_default_threshold"]
    rows.append(
        {
            **shared,
            "evaluation_set": "in_distribution_test_balanced",
            "generator": ",".join(metrics.get("known_generators", [])),
            "positive_prevalence": metrics["in_distribution_test_composition"].get(
                "positive_prevalence"
            ),
            "threshold_role": "default",
            "threshold_provenance": "fixed_prior_from_config_model.decision_threshold",
            "predictions_file": "in_distribution_test_predictions.csv",
            **_metric_row(in_distribution),
            "notes": (
                "prevalence-matched reference for the unseen row of the same run: same "
                "trained model, same fixed real pool, same number of fakes"
            ),
        }
    )
    unseen = metrics["unseen_test"]["at_default_threshold"]
    unseen_shared = {
        **shared,
        "evaluation_set": "unseen_test_balanced",
        "generator": held_out,
        "positive_prevalence": metrics["final_test_composition"].get("positive_prevalence"),
        "predictions_file": "unseen_test_predictions.csv",
        "notes": "zero-labelled-data reference point of the recovery curve",
    }
    rows.append(
        {
            **unseen_shared,
            "threshold_role": "default",
            "threshold_provenance": "fixed_prior_from_config_model.decision_threshold",
            **_metric_row(unseen),
        }
    )
    at_selected = metrics["unseen_test"].get("at_validation_selected_threshold")
    if at_selected:
        rows.append(
            {
                **unseen_shared,
                "threshold_role": "development_validation_selected",
                "threshold_provenance": str(selected.get("provenance", "")),
                **_metric_row(at_selected),
            }
        )
    return rows


def _cell_rows(
    run_dir: Path,
    protocol: str,
    metrics: Mapping[str, Any],
    cells_csv: Path,
    *,
    manifest_sha256: str,
    baseline_threshold_provenance: str,
) -> list[dict[str, Any]]:
    with cells_csv.open("r", encoding="utf-8", newline="") as handle:
        by_cell = {row["cell_id"]: row for row in csv.DictReader(handle)}
    rows: list[dict[str, Any]] = []
    for cell in metrics["cells"]:
        cell_id = str(cell["cell_id"])
        extra = by_cell.get(cell_id, {})
        is_zero = float(cell["adaptation_percentage"]) == 0.0
        shared = {
            **_base(run_dir, protocol),
            "evaluation_set": "unseen_test_balanced",
            "generator": str(cell["held_out_generator"]),
            "recovery_percentage": float(cell["adaptation_percentage"]),
            "fine_tune_strategy": str(cell["fine_tune_mode"]),
            "development_training_samples": None,
            "recovery_labelled_images_total": int(cell["labelled_images_consumed"]),
            # The earliest recovery run predates the per-class breakdown; it is left
            # blank rather than back-inferred, and the run is flagged in the notes.
            "recovery_labelled_fake_images": (
                int(cell["held_out_fake_count"])
                if cell.get("held_out_fake_count") is not None
                else None
            ),
            "positive_prevalence": metrics.get("final_test_composition", {}).get(
                "positive_prevalence"
            )
            or (metrics.get("controls", {}).get("final_test_composition", {}) or {}).get(
                "positive_prevalence"
            ),
            "trainable_parameters": extra.get("trainable_parameters") or None,
            "total_parameters": extra.get("total_parameters") or None,
            "training_seconds": extra.get("training_seconds") or None,
            "best_epoch": extra.get("best_epoch") or None,
            "subset_seed": cell.get("subset_seed"),
            "starting_checkpoint": str(cell.get("starting_checkpoint", "")),
            "checkpoint_sha256": str(cell.get("cell_checkpoint_sha256") or ""),
            "manifest_sha256": manifest_sha256,
            "predictions_file": (
                "zero_percent_unseen_test_predictions.csv"
                if is_zero
                else f"cells/{cell_id}/unseen_test_predictions.csv"
            ),
            "notes": (
                "identical to the zero-labelled point of the matching unseen_generator run"
                if is_zero
                else ""
            ),
        }
        if cell.get("training_seed") is not None:
            shared["seed"] = int(cell["training_seed"])
        rows.append(
            {
                **shared,
                "threshold_role": "default",
                "threshold_provenance": "fixed_prior_from_config_model.decision_threshold",
                **_metric_row(cell["overall"]),
            }
        )
        thresholds = cell.get("thresholds", {})
        unchanged = cell.get("at_baseline_threshold")
        if unchanged:
            rows.append(
                {
                    **shared,
                    "threshold_role": "development_validation_selected",
                    "threshold_provenance": baseline_threshold_provenance,
                    **_metric_row(unchanged),
                }
            )
        adapted = cell.get("at_adaptation_selected_threshold")
        if adapted and not is_zero:
            rows.append(
                {
                    **shared,
                    "threshold_role": "adaptation_validation_selected",
                    "threshold_provenance": str(
                        (thresholds.get("default") or {}).get("provenance")
                        or "selected_on_adaptation_validation_split"
                    ),
                    **_metric_row(adapted),
                }
            )
    return rows


def fine_tuning_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics = _load_json(run_dir / "recovery_metrics.json")
    return _cell_rows(
        run_dir,
        "fine_tuning",
        metrics,
        run_dir / "recovery_cells.csv",
        manifest_sha256=str(metrics.get("manifest_sha256", "")),
        baseline_threshold_provenance=str(metrics.get("baseline_threshold_provenance", "")),
    )


def ablation_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics = _load_json(run_dir / "ablation_metrics.json")
    controls = metrics.get("controls", {})
    return _cell_rows(
        run_dir,
        "ablation",
        metrics,
        run_dir / "ablation_cells.csv",
        manifest_sha256=str(metrics.get("manifest_sha256", "")),
        baseline_threshold_provenance=str(controls.get("baseline_threshold_provenance", "")),
    )


def external_challenge_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics = _load_json(run_dir / "external_challenge_metrics.json")
    composition = metrics["evaluation_composition"]
    rows: list[dict[str, Any]] = []
    for detector in metrics["detectors"]:
        shared = {
            **_base(run_dir, "external_challenge"),
            "evaluation_set": "external_test_astra_mediated_balanced",
            "dataset": DATASET_EXTERNAL,
            "generator": metrics["generator_recorded_as"],
            "generator_identity_known": False,
            "recovery_percentage": 0.0,
            "fine_tune_strategy": "none_frozen_detector",
            "head_type": str(detector.get("head_type", "linear")),
            "development_training_samples": None,
            "recovery_labelled_images_total": 0,
            "recovery_labelled_fake_images": 0,
            "positive_prevalence": composition["positive_prevalence"],
            "seed": detector.get("seed"),
            "starting_checkpoint": f"{detector['run_id']}/{detector['checkpoint']}",
            "checkpoint_sha256": str(detector["checkpoint_sha256"]),
            "manifest_sha256": str(metrics["manifest_sha256"]),
            "predictions_file": str(detector["predictions_file"]),
            "row_role": "primary" if detector["is_primary"] else "secondary",
            "notes": (
                f"{detector['role']}: {detector['note']} "
                "Generator identity is not reported by the mediation route, so no "
                "architectural claim attaches to this row."
            ).strip(),
        }
        rows.append(
            {
                **shared,
                "threshold_role": "default",
                "threshold_provenance": "fixed_prior_from_config_model.decision_threshold",
                **_metric_row(detector["external_test"]["at_default_threshold"]),
            }
        )
        selected = detector["external_test"].get("at_run_selected_threshold")
        if selected:
            rows.append(
                {
                    **shared,
                    "threshold_role": "development_validation_selected",
                    "threshold_provenance": str(selected.get("provenance", "")),
                    **_metric_row(selected["metrics"]),
                }
            )
    return rows


BUILDERS = {
    "baseline": ("test_metrics.json", baseline_rows),
    "unseen_generator": ("unseen_generator_metrics.json", unseen_generator_rows),
    "fine_tuning": ("recovery_metrics.json", fine_tuning_rows),
    "ablation": ("ablation_metrics.json", ablation_rows),
    "external_challenge": ("external_challenge_metrics.json", external_challenge_rows),
}


def _mark_reproductions(rows: Sequence[dict[str, Any]]) -> None:
    """Flag a cell already reported by another run as a reproduction, not a new result.

    Two conditions are scored by more than one run. The zero-labelled point belongs to
    the leave-one-generator-out run and is copied into the recovery and ablation grids as
    their reference cell. The head-only budgets were run twice -- once standalone, once
    inside the depth ablation -- and agree exactly. Every row is kept so the agreement is
    visible, but only one is marked ``primary`` so a table built from
    ``row_role == "primary"`` cannot double-count a condition.

    Canonical source per condition, matching how the runs are reported: the
    leave-one-generator-out run owns the zero-labelled point, and the depth ablation owns
    every adapted cell because it holds all three depths in one grid. This is the same
    direction ``reproduction_checks.csv`` uses when it checks the standalone recovery run
    against the ablation.
    """

    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    priority = {"unseen_generator": 0, "ablation": 1, "fine_tuning": 2}
    order = sorted(
        (row for row in rows if row["protocol"] in priority),
        key=lambda row: (priority[str(row["protocol"])], str(row["run_id"])),
    )
    for row in order:
        key = (
            row["evaluation_set"],
            row["generator"],
            row["recovery_percentage"],
            row["fine_tune_strategy"],
            row["head_type"],
            row["threshold_role"],
        )
        if key in seen:
            row["row_role"] = "reproduction"
            first = seen[key]
            note = f"reproduces {first['protocol']} run {first['run_id']}"
            row["notes"] = f"{row['notes']} {note}".strip()
        else:
            seen[key] = row


def collect(output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    entry: dict[str, Any]
    for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        protocol = run_dir.name.split("-", 1)[0]
        if protocol not in BUILDERS:
            continue
        status = _status(run_dir)
        entry = {
            "run_id": run_dir.name,
            "protocol": protocol,
            "experiment_name": _experiment_name(run_dir),
            "status": status,
        }
        if run_dir.name in EXCLUDED_RUNS:
            entry["included"] = False
            entry["reason"] = EXCLUDED_RUNS[run_dir.name]
            inventory.append(entry)
            continue
        metrics_name, builder = BUILDERS[protocol]
        if status != "completed" or not (run_dir / metrics_name).is_file():
            entry["included"] = False
            entry["reason"] = (
                f"status={status}, {metrics_name} "
                f"present={(run_dir / metrics_name).is_file()}"
            )
            inventory.append(entry)
            continue
        produced = builder(run_dir)
        entry["included"] = True
        entry["rows"] = len(produced)
        inventory.append(entry)
        rows.extend(produced)
    _mark_reproductions(rows)
    for index, row in enumerate(sorted(rows, key=_sort_key), start=1):
        row["summary_row_id"] = f"R{index:03d}"
    rows = sorted(rows, key=_sort_key)
    return rows, {"run_inventory": inventory}


def _sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    protocol_order = {
        "baseline": 0,
        "unseen_generator": 1,
        "fine_tuning": 2,
        "ablation": 3,
        "external_challenge": 4,
    }
    strategy_order = {"none": 0, "head_only": 1, "last_block": 2, "full": 3}
    return (
        protocol_order.get(str(row["protocol"]), 9),
        str(row.get("generator") or ""),
        str(row.get("head_type") or ""),
        float(row.get("recovery_percentage") or 0.0),
        strategy_order.get(str(row.get("fine_tune_strategy")), 9),
        str(row.get("evaluation_set") or ""),
        str(row.get("threshold_role") or ""),
        str(row.get("run_id") or ""),
    )


def write(
    rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    destination: Path,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    default_rows = [row for row in rows if row["threshold_role"] == "default"]
    for name, subset in (
        ("final_results_summary", default_rows),
        ("final_results_by_threshold", list(rows)),
    ):
        path = destination / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), extrasaction="ignore")
            writer.writeheader()
            for row in subset:
                writer.writerow({key: row.get(key) for key in COLUMNS})
        written.append(path)

    payload = {
        "description": (
            "Consolidated results for the Tiny GenImage unseen-generator degradation and "
            "limited-data recovery study, plus the separately-reported external "
            "contemporary-generator challenge. Every value is copied from a completed "
            "run's saved metrics file; nothing is recomputed, interpolated or smoothed."
        ),
        "columns": list(COLUMNS),
        "threshold_roles": {
            "default": "the project-wide 0.5 prior fixed in config; the table to cite",
            "development_validation_selected": (
                "the operating point each run selected on its own development validation "
                "data, before any held-out or external label was read"
            ),
            "adaptation_validation_selected": (
                "the operating point an adapted cell selected on its own adaptation "
                "validation split; reported for completeness, not as the headline"
            ),
        },
        "row_roles": {
            "primary": "cite this row",
            "reproduction": "a second run scored the identical cell; kept to show agreement",
            "secondary": "a non-nominated frozen detector on the external set",
        },
        "isolation": [
            "external_challenge rows use a different dataset and are never to be merged "
            "into the internal benchmark tables",
            "no threshold in any row was selected on a held-out or external test set",
        ],
        "row_count": len(rows),
        "default_threshold_row_count": len(default_rows),
        **dict(provenance),
        "rows": [dict(row) for row in rows],
    }
    path = destination / "final_results_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    written.append(path)

    readme = destination / "README.md"
    readme.write_text(FINAL_RESULTS_README_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    written.append(readme)
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--destination", type=Path, default=Path("outputs/report/final_results")
    )
    args = parser.parse_args(argv)
    rows, provenance = collect(args.output_root)
    written = write(rows, provenance, args.destination)
    included = [item for item in provenance["run_inventory"] if item.get("included")]
    print(f"Consolidated {len(rows)} rows from {len(included)} runs.")
    for item in provenance["run_inventory"]:
        state = f"{item['rows']} rows" if item.get("included") else f"excluded: {item['reason']}"
        print(f"  {item['run_id']}: {state}")
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
