"""Write the dissertation results-analysis exports from the saved runs.

Read-only with respect to run directories. Nothing here trains, infers, or edits a
saved artefact; it reads ``outputs/`` and writes CSVs plus a drafting note into
``outputs/report/dissertation_results/``.

Usage:
    python -m scripts.build_dissertation_results --output outputs/report/dissertation_results
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation import dissertation as D

#: Status when no external run exists yet. The manifest records it explicitly so a later
#: reader cannot mistake the plan for a result. Once an external run is present under
#: ``outputs/`` the manifest reports ``executed`` instead and carries its numbers.
EXTERNAL_CHALLENGE_STATUS = "proposed_not_executed"
EXTERNAL_CHALLENGE_STATUS_EXECUTED = "executed"


def _executed_external_challenge(output_root: Path | None) -> dict[str, Any] | None:
    """Return the executed external challenge, or ``None`` while it has not been run.

    Read-only. The internal Chapter 4 context deliberately does not discover
    ``external_challenge`` runs, so this is the only place the two meet, and it reports
    the external result beside the plan rather than merging it into any internal table.
    """

    if output_root is None:
        return None
    runs = sorted(
        path
        for path in Path(output_root).glob("external_challenge-*")
        if (path / "external_challenge_metrics.json").is_file()
    )
    if not runs:
        return None
    run_dir = runs[-1]
    metrics = json.loads(
        (run_dir / "external_challenge_metrics.json").read_text(encoding="utf-8")
    )
    audit = metrics.get("build_audit", {})
    primary = next(
        (item for item in metrics["detectors"] if item.get("is_primary")),
        metrics["detectors"][0],
    )
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "route_used": metrics.get("generation_route"),
        "route_option": "astra_responses_api_image_generation_tool (Route B)",
        "generator_recorded_as": metrics.get("generator_recorded_as"),
        "generator_identity_known": False,
        "architectural_claim_permitted": False,
        "evaluation_composition": metrics.get("evaluation_composition"),
        "sample_size_tier": audit.get("sample_size_tier"),
        "authentic_comparator_selection": audit.get("authentic_comparator_selection"),
        "preprocessing_policy_identity": (audit.get("preprocessing") or {}).get(
            "policy_identity"
        ),
        "exclusions": audit.get("exclusions"),
        "leakage_checks": [
            {"check": check.get("check"), "passed": check.get("passed")}
            for check in audit.get("leakage_checks", [])
        ],
        "primary_detector": {
            "run_id": primary["run_id"],
            "role": primary["role"],
            "head_type": primary.get("head_type"),
            "checkpoint_sha256": primary["checkpoint_sha256"],
            "external_at_default_threshold": primary["external_test"][
                "at_default_threshold"
            ],
            "internal_in_distribution": primary["internal_reference"].get(
                "in_distribution_test"
            ),
            "internal_unseen": primary["internal_reference"].get("unseen_test"),
        },
        "all_detectors": [
            {
                "run_id": item["run_id"],
                "role": item["role"],
                "is_primary": item["is_primary"],
                "roc_auc": item["external_test"]["at_default_threshold"]["roc_auc"],
                "average_precision": item["external_test"]["at_default_threshold"][
                    "average_precision"
                ],
                "f1": item["external_test"]["at_default_threshold"]["f1"],
            }
            for item in metrics["detectors"]
        ],
        "isolation_rules_observed": metrics.get("isolation_rules_observed"),
        "not_done": metrics.get("not_done"),
        "exports": "outputs/report/external_challenge/",
    }


def _write_csv(rows: Sequence[Mapping[str, Any]], destination: Path, name: str) -> Path:
    """Write rows to ``name``.csv, unioning keys so a sparse row cannot drop a column."""

    path = destination / f"{name}.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    return path


def external_challenge_manifest(ctx: D.Context) -> dict[str, Any]:
    """The compact machine-readable manifest for the proposed external challenge.

    Provenance fields are ``null`` where the fact is not yet established. They are
    present so the schema is fixed before any image exists, and absent-as-null so the
    manifest cannot be read as claiming something that was never verified.
    """

    provenance = ctx.output_root and D.dataset_provenance(ctx)
    executed = _executed_external_challenge(ctx.output_root)
    return {
        "challenge_id": "contemporary_openai_astra_mediated_v1",
        "title": "Contemporary OpenAI/Astra-mediated image-generation challenge",
        "status": (
            EXTERNAL_CHALLENGE_STATUS_EXECUTED if executed else EXTERNAL_CHALLENGE_STATUS
        ),
        "executed": executed,
        "naming_rationale": (
            "GPT-6 Astra is documented as a text-output model that invokes a hosted "
            "image_generation tool; it is not itself an image-generation architecture. "
            "The tool selects among GPT Image models and the Responses API does not "
            "report which was used, so an Astra-routed image cannot be attributed to a "
            "named generator. The challenge is therefore named for the mediation route, "
            "not for a generator."
        ),
        "generation_route": {
            "decision_required": True,
            "options": [
                {
                    "route": "direct_images_api",
                    "recommended": True,
                    "model_id": "pinned by the caller, e.g. gpt-image-2",
                    "provenance_strength": "exact generator identity is caller-specified "
                    "and recorded",
                    "why": "the dissertation can name the generator it evaluated",
                },
                {
                    "route": "astra_responses_api_image_generation_tool",
                    "recommended": False,
                    "model_id": None,
                    "provenance_strength": "underlying GPT Image model is selected by the "
                    "tool, cannot be pinned, and is not reported in the tool result",
                    "why": "matches the 'what a contemporary assistant produces' framing "
                    "but forfeits generator attribution",
                },
            ],
        },
        "provenance_fields_to_record_per_image": [
            "image_sha256",
            "prompt_text",
            "prompt_id",
            "route",
            "mainline_model_id",
            "image_model_id_if_reported",
            "api_endpoint",
            "api_response_id",
            "requested_size",
            "requested_quality",
            "output_format",
            "generated_at_utc",
            "accessed_by",
            "raw_response_metadata_json",
        ],
        "frozen_detector": {
            "must_be_evaluated_before_any_adaptation": True,
            "candidate_checkpoints": [
                {
                    "run_id": record.run_id,
                    "held_out_generator": ctx.held_out_of(record.run_id),
                    "head_type": record.head_type,
                }
                for record in ctx.by_type("unseen_generator")
            ],
            "note": "The detector is frozen at whichever checkpoint is nominated. No "
            "external image may influence training, validation, threshold selection or "
            "model selection.",
        },
        "authentic_comparison_protocol": {
            "requirement": "class-balanced, matched on the axes the detector could "
            "otherwise exploit",
            "candidate_source": "held-out authentic images from the existing Tiny "
            "GenImage real pool, which the detector never trained on",
            "must_match": ["resolution", "file format", "compression", "colour mode"],
            "rationale": "the preprocessing audit already records that container format "
            "is not predictive of class within Tiny GenImage; the same has to be true of "
            "the external set or the result measures format, not generator",
        },
        "sample_size_options": [
            {"images_per_class": 50, "total": 100, "purpose": "pilot / provenance smoke test"},
            {"images_per_class": 100, "total": 200, "purpose": "minimum reportable"},
            {"images_per_class": 250, "total": 500, "purpose": "matches the internal "
             "unseen-test size, so ROC-AUC is directly comparable in resolution"},
        ],
        "recommended_sample_size": {"images_per_class": 250, "total": 500},
        "isolation_rules": [
            "external results are never merged into the Chapter 4 benchmark tables",
            "external results live in their own export directory and their own manifest",
            "no threshold is re-selected on external data",
            "the internal benchmark is not re-run or re-reported because of this study",
        ],
        "internal_benchmark_reference": {
            "dataset": (provenance or {}).get("dataset"),
            "dataset_source": (provenance or {}).get("dataset_source"),
            "sample_count": (provenance or {}).get("sample_count"),
            "note": "the external set is deliberately outside this benchmark",
        },
        "not_yet_done": (
            executed["not_done"]
            if executed
            else [
                "no image has been generated",
                "no API call has been made",
                "no credential is configured in this environment",
                "no external evaluation has been run",
            ]
        ),
    }


def build(destination: Path, output_root: Path) -> list[Path]:
    ctx = D.load_context(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    written.append(_write_csv(D.core_results(ctx), destination, "core_results"))
    written.append(_write_csv(D.degradation_table(ctx), destination, "degradation_summary"))
    written.append(_write_csv(D.recovery_table(ctx), destination, "recovery_summary"))
    written.append(_write_csv(D.depth_table(ctx), destination, "depth_summary"))
    written.append(_write_csv(D.parameter_efficiency(ctx), destination, "parameter_efficiency"))
    written.append(_write_csv(D.threshold_table(ctx), destination, "threshold_summary"))
    written.append(_write_csv(D.experiment_inventory(ctx), destination, "experiment_inventory"))
    written.append(_write_csv(D.validation_summary(ctx), destination, "validation_summary"))

    # Supporting exports the notebooks also read back.
    written.append(_write_csv(D.marginal_recovery(ctx), destination, "marginal_recovery"))
    written.append(_write_csv(D.first_budget_reaching(ctx), destination, "attainment_budgets"))
    written.append(_write_csv(D.depth_comparisons(ctx), destination, "depth_comparisons"))
    written.append(
        _write_csv(D.cross_generator_synthesis(ctx), destination, "cross_generator_synthesis")
    )
    written.append(_write_csv(D.confusion_selection(ctx), destination, "confusion_selection"))
    written.append(_write_csv(D.reproduction_checks(ctx), destination, "reproduction_checks"))
    written.append(_write_csv(D.limitations(ctx), destination, "limitations"))

    # Combined figures the Chapter 4 package does not provide. Generated here rather
    # than assembled by hand so they regenerate with the data.
    from scripts import _dissertation_figures

    written += _dissertation_figures.build_all(ctx, destination)

    from scripts._dissertation_notes import figure_shortlist

    written.append(_write_csv(figure_shortlist(ctx), destination, "figure_shortlist"))

    manifest_path = destination / "external_challenge_manifest.json"
    manifest_path.write_text(
        json.dumps(external_challenge_manifest(ctx), indent=2) + "\n", encoding="utf-8"
    )
    written.append(manifest_path)

    notes_path = destination / "RESULTS_NOTES.md"
    notes_path.write_text(render_notes(ctx), encoding="utf-8")
    written.append(notes_path)

    return written


def render_notes(ctx: D.Context) -> str:
    """Compose RESULTS_NOTES.md, with every number formatted from a looked-up value."""

    from scripts._dissertation_notes import compose

    return compose(ctx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/report/dissertation_results"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    written = build(args.output, args.output_root)
    print(f"Dissertation results written to {args.output} ({len(written)} files)")


if __name__ == "__main__":
    main()
