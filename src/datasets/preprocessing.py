"""Deterministic re-encoding cache that removes format and spatial shortcuts.

###############################################################################
WHY THIS EXISTS
###############################################################################

An audit of Tiny GenImage found two shortcuts that let a detector separate the
classes without looking at generative artefacts at all:

1. **Format.** Every ``ai`` image is PNG; every ``nature`` image is JPEG. JPEG
   compression artefacts therefore appear in exactly one class, so file format alone
   is a perfect classifier.
2. **Native resolution.** Fakes are square and generator-specific (BigGAN 128,
   VQDM/ADM/GLIDE 256, SDv1.5/Wukong 512, Midjourney 1024) while reals are variable
   and roughly 500x375. Raw input dimensions therefore identify both the class and,
   among fakes, the generator.

This module writes a preprocessed *cache* in which every image -- real and fake --
has been decoded to RGB, resized under one policy, and re-encoded with identical JPEG
settings. Training and evaluation then read the cache, so neither shortcut survives
into the model's input.

###############################################################################
WHAT THIS DOES AND DOES NOT ACHIEVE
###############################################################################

It prevents *raw input dimensions and container format* from trivially identifying
class or generator. It does **not** remove all native-resolution effects: an image
generated at 128x128 and upscaled to 256x256 still carries different high-frequency
content from one generated at 1024x1024 and downscaled, and a single JPEG pass leaves
different residue on an already-JPEG real than on a never-compressed PNG fake. Those
are genuine properties of the source data, and the dissertation must say so rather
than claim resolution and compression have been neutralised.

###############################################################################
GUARANTEES
###############################################################################

* Original dataset files are never modified; the cache is written elsewhere.
* Deterministic: identical inputs and policy always produce byte-identical outputs.
  Resampling filter, resize policy, JPEG quality, subsampling, and metadata stripping
  are all pinned, and no random state is used.
* Auditable: a JSON sidecar records the policy, the library versions that performed
  the encoding, and per-image source and output digests.
* Idempotent and resumable: an image whose cached output already matches the recorded
  digest is skipped, so an interrupted run can be re-run safely.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import PIL
from PIL import Image, ImageOps

CACHE_SCHEMA_VERSION = 1

# Pinned so the cache is reproducible. Changing any of these changes the data and must
# be treated as a new dataset version, not an in-place edit.
DEFAULT_TARGET_SIZE = 256
DEFAULT_JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0  # 4:4:4, i.e. no chroma subsampling, applied to both classes.
RESAMPLE_FILTER = Image.Resampling.BICUBIC
RESAMPLE_FILTER_NAME = "BICUBIC"


@dataclass(frozen=True, slots=True)
class PreprocessingPolicy:
    """The complete, pinned description of how cached images were produced."""

    target_size: int = DEFAULT_TARGET_SIZE
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    resize_policy: str = "shortest_side_then_center_crop"
    resample_filter: str = RESAMPLE_FILTER_NAME
    jpeg_subsampling: int = JPEG_SUBSAMPLING
    strip_metadata: bool = True
    schema_version: int = CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.target_size) is not int or self.target_size <= 0:
            raise ValueError("target_size must be a positive integer")
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer in [1, 100]")
        if self.resize_policy != "shortest_side_then_center_crop":
            raise ValueError(f"unsupported resize policy: {self.resize_policy}")
        if self.resample_filter != RESAMPLE_FILTER_NAME:
            raise ValueError(f"unsupported resample filter: {self.resample_filter}")

    def identity(self) -> str:
        """Stable digest of the policy, used to detect a stale cache."""

        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def describe(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "policy_identity": self.identity(),
            "pillow_version": PIL.__version__,
            "output_format": "JPEG",
            "output_mode": "RGB",
            "removes": [
                "container-format class shortcut (all images re-encoded as JPEG)",
                "raw input dimension class/generator shortcut (all images same size)",
            ],
            "does_not_remove": [
                "high-frequency content differences from native generation resolution",
                "residue asymmetry between already-JPEG reals and never-compressed fakes",
            ],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_image(image: Image.Image, policy: PreprocessingPolicy) -> Image.Image:
    """Apply the deterministic spatial policy and return an RGB image.

    Shortest side is scaled to ``target_size`` and the centre ``target_size`` square is
    taken. This preserves aspect ratio for the variable-shape real images instead of
    distorting them, while giving every image -- real or fake, upscaled or downscaled --
    exactly the same output dimensions.
    """

    converted = ImageOps.exif_transpose(image).convert("RGB")
    width, height = converted.size
    if width <= 0 or height <= 0:
        raise ValueError("cannot preprocess an image with a zero dimension")
    scale = policy.target_size / min(width, height)
    resized = converted.resize(
        (
            max(policy.target_size, round(width * scale)),
            max(policy.target_size, round(height * scale)),
        ),
        resample=RESAMPLE_FILTER,
    )
    left = (resized.width - policy.target_size) // 2
    top = (resized.height - policy.target_size) // 2
    cropped = resized.crop(
        (left, top, left + policy.target_size, top + policy.target_size)
    )
    if cropped.size != (policy.target_size, policy.target_size):
        raise RuntimeError(f"preprocessing produced {cropped.size}, expected square target")
    return cropped


def write_preprocessed_image(
    source: Path, destination: Path, policy: PreprocessingPolicy
) -> None:
    """Re-encode one image into the cache atomically, stripping metadata."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        processed = preprocess_image(image, policy)
    # A fresh image object carries no EXIF/ICC from the source, so encoder-visible
    # metadata cannot leak class information either. Rebuilding from raw bytes copies the
    # pixels and nothing else.
    clean = Image.frombytes("RGB", processed.size, processed.tobytes())
    handle, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    os.close(handle)
    try:
        clean.save(
            temporary_name,
            format="JPEG",
            quality=policy.jpeg_quality,
            subsampling=JPEG_SUBSAMPLING,
            optimize=False,
            progressive=False,
        )
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class CacheResult:
    cache_root: Path
    manifest_path: Path
    processed: int
    skipped: int
    index_path: Path


def build_preprocessed_cache(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    data_root: Path,
    cache_root: Path,
    policy: PreprocessingPolicy,
    progress_every: int = 2000,
) -> CacheResult:
    """Materialise the cache for every manifest row and write an audit index.

    Returns counts of newly written and skipped-because-current images. Rewrites are
    skipped only when the recorded source digest, policy identity, and output digest all
    still match, so a changed source or policy always forces regeneration.
    """

    data_root = data_root.resolve()
    cache_root = cache_root.resolve()
    try:
        cache_root.relative_to(data_root)
    except ValueError as exc:
        raise ValueError("cache_root must live beneath data_root for portable paths") from exc
    if not manifest_rows:
        raise ValueError("cannot build a cache for an empty manifest")

    index_path = cache_root / "preprocessing_index.json"
    previous: dict[str, Any] = {}
    if index_path.is_file():
        stored = json.loads(index_path.read_text(encoding="utf-8"))
        if stored.get("policy", {}).get("policy_identity") == policy.identity():
            previous = stored.get("images", {})

    entries: dict[str, Any] = {}
    processed = 0
    skipped = 0
    for position, row in enumerate(manifest_rows, start=1):
        sample_id = str(row["sample_id"])
        source = (data_root / str(row["image_path"])).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"manifest image is missing: {source}")
        # Cache layout mirrors the manifest's relative path, with a .jpg suffix, so the
        # provenance of every cached file stays readable.
        relative = Path(str(row["image_path"])).with_suffix(".jpg")
        destination = cache_root / relative
        source_digest = _sha256_file(source)
        record = previous.get(sample_id)
        if (
            record is not None
            and record.get("source_sha256") == source_digest
            and destination.is_file()
            and _sha256_file(destination) == record.get("output_sha256")
        ):
            entries[sample_id] = record
            skipped += 1
        else:
            write_preprocessed_image(source, destination, policy)
            entries[sample_id] = {
                "source_image_path": str(row["image_path"]),
                "cached_image_path": relative.as_posix(),
                "source_sha256": source_digest,
                "output_sha256": _sha256_file(destination),
                "label": int(row["label"]),
                "generator": str(row["generator"]),
            }
            processed += 1
        if progress_every and position % progress_every == 0:
            print(f"  preprocessed {position}/{len(manifest_rows)} images", flush=True)

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "policy": policy.describe(),
        "data_root": str(data_root),
        "cache_root": str(cache_root),
        "image_count": len(entries),
        "images": entries,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return CacheResult(
        cache_root=cache_root,
        manifest_path=index_path,
        processed=processed,
        skipped=skipped,
        index_path=index_path,
    )


def rewrite_manifest_to_cache(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    cache_root: Path,
    data_root: Path,
    dataset_source_suffix: str = "+preprocessed",
) -> list[dict[str, Any]]:
    """Return manifest rows pointing at cached files, with provenance preserved.

    ``image_path`` is repointed at the cache while ``original_image_path`` keeps the
    source location, so a cached run can always be traced back to raw data.
    """

    cache_root = cache_root.resolve()
    data_root = data_root.resolve()
    rewritten: list[dict[str, Any]] = []
    for row in manifest_rows:
        relative = Path(str(row["image_path"])).with_suffix(".jpg")
        cached = (cache_root / relative).resolve()
        if not cached.is_file():
            raise FileNotFoundError(f"cached image missing for {row['sample_id']}: {cached}")
        updated = dict(row)
        updated["original_image_path"] = str(row["image_path"])
        updated["image_path"] = cached.relative_to(data_root).as_posix()
        updated["dataset_source"] = f"{row['dataset_source']}{dataset_source_suffix}"
        rewritten.append(updated)
    return rewritten
