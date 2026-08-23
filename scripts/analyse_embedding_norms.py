"""Test the PREMISE of the cosine head, independently of any training outcome.

The cosine head exists because of a specific mechanistic claim:

    CLIP's pooled embedding has a magnitude as well as a direction. An unnormalised
    linear head can use that magnitude. If magnitude varies systematically BETWEEN
    GENERATORS, then a detector trained on six generators can fit a decision boundary
    that partly depends on a quantity which does not transfer to a seventh.

That claim is measurable without training anything, and it should be measured, because
it is what distinguishes a designed modification from a guess. This script encodes a
sample of images with the **frozen, pretrained** CLIP backbone -- no classifier, no
fitting, no labels used -- and reports the distribution of embedding L2 norms per
generator.

Interpretation
--------------
* If the held-out generator's norms sit clearly apart from the training generators',
  the premise holds and the cosine head is addressing a real property of the features.
* If every generator's norms overlap, the premise is weak, and any improvement the
  cosine head shows would need a different explanation. That is a useful negative result
  and must be reported as one rather than quietly dropped.

Sampling policy
---------------
Images are drawn ONLY from the ``train`` split. The final unseen test partitions are
never opened by this script, so a descriptive diagnostic cannot contaminate any reported
result. Sampling is seeded and the selected sample IDs are saved.

Usage
-----
    python -m scripts.analyse_embedding_norms --per-generator 100
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from src.datasets.detector_dataset import AIDetectionDataset
from src.datasets.splitting import load_split_assignments
from src.evaluation.plots import plot_embedding_norms
from src.experiments.common import build_transforms, resolve_runtime_paths
from src.models.clip_detector import CLIPVisionBackbone
from src.utils.config import load_config

SAMPLE_SEED = 20260817


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def analyse(
    config_path: Path, *, per_generator: int, destination: Path, batch_size: int = 16
) -> dict[str, Any]:
    loaded = load_config(config_path)
    config = resolve_runtime_paths(loaded.values, loaded.source_path)
    transform = build_transforms(config, training=False)
    dataset = AIDetectionDataset.from_manifest(
        Path(config["data"]["manifest_path"]),
        data_root=Path(config["data"]["root"]),
        transform=transform,
    )
    split_by_id = {
        item.sample_id: item.split
        for item in load_split_assignments(Path(config["data"]["split_path"]))
    }

    # Train split only: the final unseen test partitions stay untouched.
    by_generator: dict[str, list[Any]] = {}
    for record in dataset.records:
        if split_by_id.get(record.sample_id) != "train":
            continue
        by_generator.setdefault(record.generator, []).append(record)

    rng = random.Random(SAMPLE_SEED)
    backbone = CLIPVisionBackbone(
        str(config["model"]["clip_model_name"]),
        revision=config["model"].get("clip_revision"),
        feature_source=str(config["model"].get("feature_source", "pooled_output")),
    ).eval()

    rows: list[dict[str, Any]] = []
    selected_ids: dict[str, list[str]] = {}
    for generator in sorted(by_generator):
        pool = sorted(by_generator[generator], key=lambda record: record.sample_id)
        chosen = rng.sample(pool, min(per_generator, len(pool)))
        selected_ids[generator] = [record.sample_id for record in chosen]
        subset = AIDetectionDataset(chosen, transform=transform)
        norms: list[float] = []
        for start in range(0, len(subset), batch_size):
            tensors = torch.stack(
                [
                    subset[index]["pixel_values"]  # type: ignore[index,misc]
                    for index in range(start, min(start + batch_size, len(subset)))
                ]
            )
            with torch.no_grad():
                embeddings = backbone(tensors)
            norms.extend(embeddings.norm(dim=1).tolist())
        rows.append(
            {
                "generator": generator,
                "label": "real" if generator == "real" else "fake",
                "n": len(norms),
                "mean_norm": sum(norms) / len(norms),
                "median_norm": _percentile(norms, 0.5),
                "p10_norm": _percentile(norms, 0.10),
                "p90_norm": _percentile(norms, 0.90),
                "min_norm": min(norms),
                "max_norm": max(norms),
                "norms": norms,
            }
        )

    payload = {
        "what_this_measures": (
            "L2 norm of the frozen CLIP pooled embedding, per generator. No classifier "
            "is involved and no training occurs; this describes the feature space the "
            "detector is built on."
        ),
        "sampling": {
            "split": "train",
            "per_generator": per_generator,
            "seed": SAMPLE_SEED,
            "final_test_partitions_opened": False,
        },
        "clip_model_name": config["model"]["clip_model_name"],
        "clip_revision": config["model"].get("clip_revision"),
        "manifest_sha256": dataset.manifest_sha256,
        "selected_sample_ids": selected_ids,
        "per_generator": [{k: v for k, v in row.items() if k != "norms"} for row in rows],
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "embedding_norms.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    raw = {str(row["generator"]): row["norms"] for row in rows}
    (destination / "embedding_norms_raw.json").write_text(json.dumps(raw), encoding="utf-8")

    held_out = (config.get("generators") or {}).get("unseen")
    figure, _ = plot_embedding_norms(
        payload["per_generator"],
        raw_norms=raw,
        highlight=None if held_out is None else str(held_out),
        title=(
            "Does embedding magnitude separate generators? "
            "(frozen CLIP, no classifier, train split only)"
        ),
    )
    for suffix in (".pdf", ".png"):
        figure.savefig(
            (destination / "figure_embedding_norms").with_suffix(suffix),
            bbox_inches="tight",
            dpi=300,
        )
    return {"payload": payload, "rows": rows}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tiny_unseen_vqdm.yaml"))
    parser.add_argument("--per-generator", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = analyse(args.config, per_generator=args.per_generator, destination=args.output)
    print(f"{'generator':24s} {'n':>4s} {'mean':>8s} {'p10':>8s} {'median':>8s} {'p90':>8s}")
    for row in result["rows"]:
        print(
            f"{row['generator']:24s} {row['n']:4d} {row['mean_norm']:8.2f} "
            f"{row['p10_norm']:8.2f} {row['median_norm']:8.2f} {row['p90_norm']:8.2f}"
        )
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
