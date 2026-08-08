"""Generate SYNTHETIC images in the GenImage folder layout for pipeline smoke tests.

WARNING
-------
This script does not download, sample, or approximate GenImage. It writes small
procedurally generated images into the *official GenImage directory layout* so the
end-to-end pipeline (import -> manifest -> group-safe splits -> CLIP detector ->
training -> checkpoint selection -> untouched test evaluation -> metrics) can be
executed and verified without the real 1M-image release.

Numbers produced from this data are PIPELINE EVIDENCE ONLY. They are not research
results and must never appear in the dissertation as detector performance. The
generated tree is written under a ``genimage_synthetic`` root and every config that
consumes it is named ``configs/smoke_synthetic_*.yaml``, precisely so synthetic runs
stay distinguishable from real ones.

The "real" class is low-frequency (smooth gradients plus mild noise). Each "fake"
generator adds its own synthesis artefact, so the task is learnable and the held-out
generator's artefact is genuinely unseen. See ``GENERATOR_FOLDERS`` for why the
held-out generator's artefact differs in kind rather than only in frequency.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Folder names must match the official release so scripts/import_genimage.py maps them.
#
# The three training generators share an artefact FAMILY (an axis-aligned periodic
# grid) and differ only in frequency and phase. BigGAN's artefact is a different kind
# of structure entirely -- a low-amplitude radial ripple -- so a detector that latches
# onto "grid at these frequencies" does not transfer to it for free. That is what makes
# the fixture produce a non-trivial generalisation gap and therefore a recovery curve
# with something to recover, which is the behaviour the protocols need to exercise.
GENERATOR_FOLDERS: dict[str, dict[str, Any]] = {
    "Midjourney": {"kind": "grid", "frequency": 6.0, "phase": 0.0, "amplitude": 0.16},
    "Stable Diffusion V1.4": {"kind": "grid", "frequency": 9.0, "phase": 0.7, "amplitude": 0.15},
    "ADM": {"kind": "grid", "frequency": 13.0, "phase": 1.4, "amplitude": 0.14},
    "BigGAN": {"kind": "radial", "frequency": 7.0, "phase": 2.1, "amplitude": 0.07},
}


def _coordinate_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return np.meshgrid(axis, axis, indexing="xy")


def _real_image(size: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth, low-frequency content standing in for an authentic photograph."""
    x, y = _coordinate_grid(size)
    channels = []
    for _ in range(3):
        direction = rng.uniform(-1.0, 1.0, size=2)
        offset = rng.uniform(0.0, 1.0)
        plane = 0.5 + 0.25 * (direction[0] * (x - 0.5) + direction[1] * (y - 0.5))
        blob = 0.15 * np.sin(2 * np.pi * (0.8 * x + offset)) * np.cos(2 * np.pi * (0.6 * y))
        channels.append(plane + blob)
    image = np.stack(channels, axis=-1)
    image += rng.normal(0.0, 0.012, size=image.shape)
    return image


def _fake_image(size: int, rng: np.random.Generator, parameters: dict[str, Any]) -> np.ndarray:
    """Real-like content plus a generator-specific synthesis artefact."""
    image = _real_image(size, rng)
    x, y = _coordinate_grid(size)
    frequency = float(parameters["frequency"])
    phase = float(parameters["phase"])
    amplitude = float(parameters["amplitude"])
    kind = str(parameters["kind"])
    if kind == "grid":
        artefact = np.sin(2 * np.pi * frequency * x + phase) * np.sin(
            2 * np.pi * frequency * y + phase
        )
    elif kind == "radial":
        radius = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        artefact = np.sin(2 * np.pi * frequency * radius + phase)
    else:
        raise ValueError(f"unknown artefact kind: {kind}")
    image += amplitude * artefact[..., None]
    return image


def _save(image: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0.0, 1.0)
    Image.fromarray((clipped * 255).round().astype(np.uint8), mode="RGB").save(destination)


def build_synthetic_tree(
    *,
    root: Path,
    size: int = 64,
    train_fake_per_generator: int = 48,
    val_fake_per_generator: int = 24,
    train_real: int = 48,
    val_real: int = 40,
    seed: int = 20260808,
    overwrite: bool = False,
) -> dict[str, int]:
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing synthetic tree: {root} (pass --overwrite)"
            )
        shutil.rmtree(root)

    # Real images are generated once and byte-identical across every generator folder.
    # This mirrors GenImage, where the same ImageNet 'nature' files repeat per subset,
    # and exercises the importer's content-verified deduplication path.
    shared_real = root / "_shared_nature"
    real_counts = {"train": train_real, "val": val_real}
    for official_split, count in real_counts.items():
        for index in range(count):
            rng = np.random.default_rng([seed, 0, hash(official_split) % 10_000, index])
            _save(
                _real_image(size, rng),
                shared_real / official_split / f"nature_{official_split}_{index:05d}.png",
            )

    written = {"real_logical": train_real + val_real, "fake": 0}
    for folder_index, (folder_name, parameters) in enumerate(GENERATOR_FOLDERS.items(), start=1):
        generator_root = root / folder_name
        fake_counts = {"train": train_fake_per_generator, "val": val_fake_per_generator}
        for official_split, count in fake_counts.items():
            for index in range(count):
                rng = np.random.default_rng(
                    [seed, folder_index, hash(official_split) % 10_000, index]
                )
                _save(
                    _fake_image(size, rng, parameters),
                    generator_root
                    / official_split
                    / "ai"
                    / f"ai_{official_split}_{index:05d}.png",
                )
                written["fake"] += 1
            destination = generator_root / official_split / "nature"
            destination.mkdir(parents=True, exist_ok=True)
            for source in sorted((shared_real / official_split).iterdir()):
                shutil.copy2(source, destination / source.name)

    shutil.rmtree(shared_real)
    (root / "SYNTHETIC_DATA_DO_NOT_REPORT.txt").write_text(
        "Procedurally generated images for pipeline verification only.\n"
        "These are NOT GenImage images and metrics from them are NOT research results.\n",
        encoding="utf-8",
    )
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/genimage_synthetic"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--train-fake-per-generator", type=int, default=48)
    parser.add_argument("--val-fake-per-generator", type=int, default=24)
    parser.add_argument("--train-real", type=int, default=48)
    parser.add_argument("--val-real", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    written = build_synthetic_tree(
        root=args.root,
        size=args.size,
        train_fake_per_generator=args.train_fake_per_generator,
        val_fake_per_generator=args.val_fake_per_generator,
        train_real=args.train_real,
        val_real=args.val_real,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"SYNTHETIC tree written to {args.root}")
    print(f"  logical real images: {written['real_logical']} (repeated per generator folder)")
    print(f"  fake images:         {written['fake']}")
    print("  These are NOT GenImage images. Do not report metrics from them.")


if __name__ == "__main__":
    main()
