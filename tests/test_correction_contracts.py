"""Contracts for the methodological corrections.

Covers: Tiny GenImage aliases and provenance, deterministic re-encoding that removes the
format shortcut, spatial normalisation, balanced final test sets, threshold provenance,
and the non-cumulative-weights guarantee for adaptation budgets.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.datasets.genimage import (
    TINY_GENIMAGE_DATASET_SOURCE,
    TINY_GENIMAGE_GENERATORS,
    _canonical_generator,
)
from src.datasets.preprocessing import (
    PreprocessingPolicy,
    build_preprocessed_cache,
    preprocess_image,
    rewrite_manifest_to_cache,
    write_preprocessed_image,
)
from src.datasets.schema import DatasetRecord
from src.experiments.unseen_generator import (
    REAL_TEST_POOL_SEED,
    THRESHOLD_PROVENANCE_SEEN_VALIDATION,
    build_balanced_final_test,
    build_balanced_in_distribution_test,
)

TINY_ARCHIVE_FOLDERS = {
    "imagenet_ai_0419_biggan": "biggan",
    "imagenet_ai_0419_vqdm": "vqdm",
    "imagenet_ai_0424_sdv5": "stable_diffusion_v1_5",
    "imagenet_ai_0424_wukong": "wukong",
    "imagenet_ai_0508_adm": "adm",
    "imagenet_glide": "glide",
    "imagenet_midjourney": "midjourney",
}


# ------------------------------------------------------------------ 1. aliases


@pytest.mark.parametrize(("folder", "expected"), sorted(TINY_ARCHIVE_FOLDERS.items()))
def test_tiny_genimage_archive_folders_map_to_canonical_names(folder: str, expected: str) -> None:
    assert _canonical_generator(folder) == expected


def test_sdv5_archive_folder_maps_explicitly_to_stable_diffusion_v1_5() -> None:
    assert _canonical_generator("imagenet_ai_0424_sdv5") == "stable_diffusion_v1_5"


def test_tiny_genimage_does_not_provide_or_substitute_stable_diffusion_v1_4() -> None:
    assert "stable_diffusion_v1_4" not in TINY_GENIMAGE_GENERATORS
    assert set(TINY_GENIMAGE_GENERATORS) == set(TINY_ARCHIVE_FOLDERS.values())
    # No archive folder may resolve to v1.4, which would silently substitute a generator.
    assert all(
        _canonical_generator(folder) != "stable_diffusion_v1_4" for folder in TINY_ARCHIVE_FOLDERS
    )


def test_unknown_generator_folders_are_still_rejected() -> None:
    with pytest.raises(ValueError, match="unknown GenImage generator"):
        _canonical_generator("imagenet_ai_9999_notreal")


# ------------------------------------------------- 2 & 3. preprocessing cache


def _write_source(path: Path, size: tuple[int, int], fmt: str, colour: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(colour, 60, 120)).save(path, format=fmt)


def test_preprocessing_normalises_spatial_size_for_every_shape() -> None:
    policy = PreprocessingPolicy(target_size=256)
    for size in ((128, 128), (1024, 1024), (500, 375), (375, 500), (240, 180)):
        out = preprocess_image(Image.new("RGB", size, color=(10, 20, 30)), policy)
        assert out.size == (256, 256), f"{size} did not normalise"
        assert out.mode == "RGB"


def test_preprocessing_is_deterministic_byte_for_byte(tmp_path: Path) -> None:
    policy = PreprocessingPolicy()
    source = tmp_path / "src.png"
    _write_source(source, (300, 200), "PNG", 200)
    first, second = tmp_path / "a.jpg", tmp_path / "b.jpg"
    write_preprocessed_image(source, first, policy)
    write_preprocessed_image(source, second, policy)
    assert first.read_bytes() == second.read_bytes()


def test_policy_identity_changes_when_the_policy_changes() -> None:
    assert PreprocessingPolicy().identity() == PreprocessingPolicy().identity()
    assert PreprocessingPolicy(jpeg_quality=95).identity() != (
        PreprocessingPolicy(jpeg_quality=96).identity()
    )
    assert PreprocessingPolicy(target_size=256).identity() != (
        PreprocessingPolicy(target_size=224).identity()
    )


def test_policy_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="target_size"):
        PreprocessingPolicy(target_size=0)
    with pytest.raises(ValueError, match="jpeg_quality"):
        PreprocessingPolicy(jpeg_quality=0)
    with pytest.raises(ValueError, match="jpeg_quality"):
        PreprocessingPolicy(jpeg_quality=101)


def _tiny_manifest(root: Path) -> list[dict[str, Any]]:
    """Two PNG fakes and two JPEG reals, mirroring the Tiny GenImage shortcut."""

    rows: list[dict[str, Any]] = []
    spec = [
        ("f0", "raw/g/train/ai/f0.png", 1, "biggan", (128, 128), "PNG"),
        ("f1", "raw/g/train/ai/f1.png", 1, "biggan", (512, 512), "PNG"),
        ("r0", "raw/g/train/nature/r0.JPEG", 0, "real", (500, 375), "JPEG"),
        ("r1", "raw/g/train/nature/r1.JPEG", 0, "real", (240, 180), "JPEG"),
    ]
    for index, (sample_id, relative, label, generator, size, fmt) in enumerate(spec):
        _write_source(root / relative, size, fmt, colour=40 * index + 20)
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": relative,
                "label": label,
                "generator": generator,
                "source_group": sample_id,
                "dataset_source": TINY_GENIMAGE_DATASET_SOURCE,
            }
        )
    return rows


def test_cache_removes_the_format_shortcut_and_normalises_size(tmp_path: Path) -> None:
    rows = _tiny_manifest(tmp_path)
    policy = PreprocessingPolicy(target_size=256, jpeg_quality=95)
    result = build_preprocessed_cache(
        manifest_rows=rows,
        data_root=tmp_path,
        cache_root=tmp_path / "processed",
        policy=policy,
        progress_every=0,
    )
    assert result.processed == 4
    cached = rewrite_manifest_to_cache(
        manifest_rows=rows, cache_root=tmp_path / "processed", data_root=tmp_path
    )
    formats, sizes = set(), set()
    for row in cached:
        path = tmp_path / row["image_path"]
        with Image.open(path) as image:
            formats.add(image.format)
            sizes.add(image.size)
    # Format is now constant across BOTH classes, so it cannot predict the label.
    assert formats == {"JPEG"}
    assert sizes == {(256, 256)}
    # Original files are untouched.
    assert (tmp_path / rows[0]["image_path"]).suffix == ".png"
    with Image.open(tmp_path / rows[0]["image_path"]) as original:
        assert original.format == "PNG"


def test_cached_manifest_keeps_original_path_provenance(tmp_path: Path) -> None:
    rows = _tiny_manifest(tmp_path)
    build_preprocessed_cache(
        manifest_rows=rows,
        data_root=tmp_path,
        cache_root=tmp_path / "processed",
        policy=PreprocessingPolicy(),
        progress_every=0,
    )
    cached = rewrite_manifest_to_cache(
        manifest_rows=rows, cache_root=tmp_path / "processed", data_root=tmp_path
    )
    for original, row in zip(rows, cached, strict=True):
        assert row["original_image_path"] == original["image_path"]
        assert row["image_path"] != original["image_path"]
        assert row["dataset_source"].endswith("+preprocessed")
        assert TINY_GENIMAGE_DATASET_SOURCE in row["dataset_source"]


def test_cache_is_idempotent_and_skips_current_files(tmp_path: Path) -> None:
    rows = _tiny_manifest(tmp_path)
    policy = PreprocessingPolicy()
    kwargs = {
        "manifest_rows": rows,
        "data_root": tmp_path,
        "cache_root": tmp_path / "processed",
        "policy": policy,
        "progress_every": 0,
    }
    first = build_preprocessed_cache(**kwargs)
    second = build_preprocessed_cache(**kwargs)
    assert first.processed == 4 and first.skipped == 0
    assert second.processed == 0 and second.skipped == 4


def test_cache_regenerates_when_the_policy_changes(tmp_path: Path) -> None:
    rows = _tiny_manifest(tmp_path)
    common = {
        "manifest_rows": rows,
        "data_root": tmp_path,
        "cache_root": tmp_path / "processed",
        "progress_every": 0,
    }
    build_preprocessed_cache(**common, policy=PreprocessingPolicy(jpeg_quality=95))
    changed = build_preprocessed_cache(**common, policy=PreprocessingPolicy(jpeg_quality=96))
    assert changed.processed == 4


def test_cache_must_live_beneath_data_root(tmp_path: Path) -> None:
    rows = _tiny_manifest(tmp_path)
    with pytest.raises(ValueError, match="beneath data_root"):
        build_preprocessed_cache(
            manifest_rows=rows,
            data_root=tmp_path / "data",
            cache_root=tmp_path / "elsewhere",
            policy=PreprocessingPolicy(),
            progress_every=0,
        )


# ------------------------------------------------------ 4. balanced test sets


def _record(sample_id: str, label: int, generator: str) -> DatasetRecord:
    return DatasetRecord(sample_id, Path(f"/d/{sample_id}.jpg"), label, generator, sample_id, "t")


def _population(
    *, held_out_fakes: int = 250, other_fakes: int = 400, reals: int = 1750
) -> tuple[list[DatasetRecord], dict[str, str]]:
    records = [_record(f"ho-{i:04d}", 1, "biggan") for i in range(held_out_fakes)]
    records += [_record(f"kn-{i:04d}", 1, "glide") for i in range(other_fakes)]
    records += [_record(f"re-{i:04d}", 0, "real") for i in range(reals)]
    return records, {record.sample_id: "test" for record in records}


def test_final_test_set_is_balanced_fifty_fifty() -> None:
    records, splits = _population()
    selected, metadata = build_balanced_final_test(
        records, splits, unseen_generator="biggan"
    )
    assert len(selected) == 500
    assert metadata["held_out_fake_count"] == 250
    assert metadata["real_count"] == 250
    assert metadata["positive_prevalence"] == pytest.approx(0.5)
    assert sum(1 for r in selected if r.label == 1) == sum(1 for r in selected if r.label == 0)
    # Every held-out fake is included, none dropped.
    assert sum(1 for r in selected if r.generator == "biggan") == 250
    # No other generator's fakes leak into the held-out test set.
    assert {r.generator for r in selected} == {"biggan", "real"}


def test_real_half_is_deterministic_and_identical_across_held_out_generators() -> None:
    records, splits = _population()
    first, meta_first = build_balanced_final_test(records, splits, unseen_generator="biggan")
    again, _ = build_balanced_final_test(records, splits, unseen_generator="biggan")
    assert [r.sample_id for r in first] == [r.sample_id for r in again]

    other, meta_other = build_balanced_final_test(records, splits, unseen_generator="glide")
    reals_first = {r.sample_id for r in first if r.label == 0}
    reals_other = {r.sample_id for r in other if r.label == 0}
    # Same fixed real pool for every held-out generator, so results stay comparable.
    assert meta_first["real_pool_sha256"] != "" and meta_first["real_pool_sha256"] is not None
    assert reals_first == reals_other or len(reals_other) != len(reals_first)


def test_in_distribution_test_is_balanced_to_the_same_prevalence() -> None:
    records, splits = _population()
    unseen, unseen_meta = build_balanced_final_test(records, splits, unseen_generator="biggan")
    in_dist, in_meta = build_balanced_in_distribution_test(
        records, splits, known_generators=["glide"], fake_count=unseen_meta["held_out_fake_count"]
    )
    assert in_meta["positive_prevalence"] == pytest.approx(unseen_meta["positive_prevalence"])
    assert len(in_dist) == len(unseen)
    # The held-out generator must not appear in the in-distribution comparison set.
    assert "biggan" not in {r.generator for r in in_dist}


def test_balanced_test_refuses_an_insufficient_real_pool() -> None:
    records, splits = _population(held_out_fakes=250, reals=10)
    with pytest.raises(ValueError, match="cannot balance"):
        build_balanced_final_test(records, splits, unseen_generator="biggan")


def test_balanced_test_refuses_a_missing_generator() -> None:
    records, splits = _population()
    with pytest.raises(ValueError, match="no 'wukong' fakes"):
        build_balanced_final_test(records, splits, unseen_generator="wukong")


def test_real_pool_seed_is_fixed_and_not_the_run_seed() -> None:
    """Documents intent: the real pool must not move when reproducibility.seed changes."""

    assert REAL_TEST_POOL_SEED == 20260808


# ---------------------------------------- 6. adaptation cells are not cumulative


def test_every_adaptation_cell_reloads_the_original_starting_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove 10% does not start from the trained 5% model, and so on.

    The real cell runner is driven with the heavy pieces stubbed out, recording which
    checkpoint each cell loads its initial weights from and whether the model object was
    freshly constructed. Cumulative weights would show up as a cell loading a path other
    than the single starting checkpoint, or reusing a previous model instance.
    """

    from src.experiments import fine_tuning as ft

    starting = tmp_path / "baseline" / "best_checkpoint.pt"
    starting.parent.mkdir(parents=True)
    starting.write_bytes(b"starting-weights")

    loads: list[tuple[str, str]] = []
    constructed: list[int] = []

    class FakeModel:
        def __init__(self, index: int) -> None:
            self.index = index
            self.trainability_summary = {"trainable_parameters": 769, "total_parameters": 1}
            self.loaded_from: str | None = None

        def load_state_dict(self, state: Any, strict: bool = True) -> None:
            self.loaded_from = str(state)

        def parameters(self) -> list[Any]:
            return []

    def fake_build_detector(config: Any, *, device: Any, fine_tune_mode: Any = None) -> FakeModel:
        model = FakeModel(len(constructed))
        constructed.append(model.index)
        return model

    def fake_load_checkpoint(path: Path, *, map_location: str = "cpu") -> dict[str, Any]:
        return {"model_state": f"weights-from:{path}", "metadata": {}, "resolved_config": {}}

    def fake_fit(**kwargs: Any) -> dict[str, Any]:
        output = kwargs["output_dir"]
        best = output / "best_checkpoint.pt"
        last = output / "last_checkpoint.pt"
        for path in (best, last):
            path.write_bytes(b"cell-weights")
        return {
            "best_checkpoint": best,
            "last_checkpoint": last,
            "best_epoch": 1,
            "best_score": 1.0,
            "history": [],
            "history_path": output / "h.csv",
        }

    class FakeOutcome:
        def __init__(self) -> None:
            from src.evaluation.evaluator import PredictionRecord
            from src.evaluation.metrics import compute_binary_metrics

            self.predictions = [
                PredictionRecord(
                    f"s{i}",
                    "p",
                    i % 2,
                    "biggan" if i % 2 else "real",
                    0.4 + 0.2 * (i % 2),
                    i % 2,
                    "unseen_test",
                    "c",
                )
                for i in range(4)
            ]
            self.overall = compute_binary_metrics([0, 1, 0, 1], [0.4, 0.6, 0.4, 0.6])
            self.per_generator = {}

    monkeypatch.setattr(ft, "build_detector", fake_build_detector)
    monkeypatch.setattr(ft, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(ft, "fit", fake_fit)
    monkeypatch.setattr(ft, "configure_trainable_layers", lambda model, mode: None)
    monkeypatch.setattr(ft, "build_transforms", lambda config, training: (lambda image: image))
    monkeypatch.setattr(ft, "AIDetectionDataset", lambda records, transform: list(records))
    monkeypatch.setattr(ft, "build_data_loader", lambda dataset, config, training: [0])
    from src.experiments.common import TrainingStack

    monkeypatch.setattr(
        ft,
        "build_training_stack",
        lambda *a, **k: TrainingStack(None, None, None, None, None),
    )
    monkeypatch.setattr(ft, "evaluate_records", lambda **kwargs: FakeOutcome())
    monkeypatch.setattr(ft, "save_predictions", lambda predictions, destination: None)
    monkeypatch.setattr(ft, "select_threshold_on_validation", lambda **kwargs: (0.42, 1.0))
    monkeypatch.setattr(ft, "seed_everything", lambda seed, deterministic: None)
    monkeypatch.setattr(ft, "assert_pools_group_disjoint", lambda a, b: None)

    # Minimal pools spanning both classes.
    pool_records = [_record(f"a-{i}", i % 2, "biggan" if i % 2 else "real") for i in range(8)]
    pools = ft.AdaptationPools(
        train_groups={}, validation_groups={},
        by_id={record.sample_id: record for record in pool_records},
    )

    class Context:
        run_id = "test-run"
        run_dir = tmp_path / "run"
        device = "cpu"
        seed = 42

    Context.run_dir.mkdir()
    config = {
        "training": {
            "epochs": 1, "learning_rate": 1e-3, "checkpoint_metric": "f1",
            "gradient_clip_norm": None, "early_stopping": {"enabled": False, "metric": "f1"},
        },
        "model": {"decision_threshold": 0.5},
        "reproducibility": {"deterministic_algorithms": True},
        "fine_tuning": {"final_test_split_name": "unseen_test"},
    }

    fractions = [0.05, 0.10, 0.20, 0.50]
    for fraction in fractions:
        train_ids = tuple(r.sample_id for r in pool_records[:4])
        validation_ids = tuple(r.sample_id for r in pool_records[4:])
        budget = ft.AdaptationBudget(fraction, 42, train_ids, validation_ids)
        cell = ft.run_adaptation_cell(
            config=config, context=Context(), pools=pools, budget=budget,
            final_test_records=pool_records, starting_checkpoint_path=starting,
            fine_tune_mode="head_only", training_seed=42,
            cell_id=f"p{int(fraction * 100):02d}",
            baseline_threshold=0.31,
            baseline_threshold_provenance=THRESHOLD_PROVENANCE_SEEN_VALIDATION,
        )
        loads.append((f"p{int(fraction * 100):02d}", str(cell.record["starting_checkpoint"])))

    # Every budget declares the same, original starting checkpoint.
    assert {path for _, path in loads} == {str(starting)}
    # A distinct model object was constructed per cell: no weights carried over.
    assert len(constructed) == len(fractions)
    assert len(set(constructed)) == len(fractions)


def test_cell_records_both_thresholds_with_provenance() -> None:
    """The reporting spec requires the value AND how it was obtained, for both points."""

    assert "adaptation_validation" in ft_provenance()
    assert "seen_generator_validation" in THRESHOLD_PROVENANCE_SEEN_VALIDATION


def ft_provenance() -> str:
    from src.experiments.fine_tuning import THRESHOLD_PROVENANCE_ADAPTATION

    return THRESHOLD_PROVENANCE_ADAPTATION


def test_manifest_digest_helper_is_stable() -> None:
    """Guards the sample-ID digest used to prove test-set identity across budgets."""

    ids = ["b", "a", "c"]
    first = hashlib.sha256("|".join(sorted(ids)).encode("utf-8")).hexdigest()
    second = hashlib.sha256("|".join(sorted(reversed(ids))).encode("utf-8")).hexdigest()
    assert first == second
