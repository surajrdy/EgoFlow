from __future__ import annotations

import json
import io

import numpy as np
import pytest

from hackathon.egoflow.annotations import AnnotationSpan, assign_annotations
from hackathon.egoflow.cache import REQUIRED_KEYS, load_feature_cache, save_feature_cache
from hackathon.egoflow.config import ExtractionConfig, HardCaps
from hackathon.egoflow.features import (
    deterministic_text_embeddings,
    extract_episode_features,
    mean_pool_embeddings,
)
from hackathon.egoflow.manifest import build_cache_manifest
from hackathon.egoflow.synthetic import (
    make_synthetic_episode,
    write_egoverse_writer_fixture,
    write_synthetic_zarr,
)
from hackathon.egoflow.zarr_io import inspect_episode, safe_source_path


def test_latest_started_overlap_and_short_gap_inheritance() -> None:
    spans = [
        AnnotationSpan("reach", 0, 9, 0),
        AnnotationSpan("grasp", 6, 12, 1),
        AnnotationSpan("place", 16, 20, 2),
    ]
    result = assign_annotations(
        [5, 7, 14, 15, 16, 25],
        spans,
        task_description="generic task",
        max_gap_frames=3,
    )
    assert result[0].primary_text == "reach"
    assert result[1].primary_text == "grasp"
    assert result[1].active_texts == ("reach", "grasp")
    assert result[2].primary_text == "grasp" and result[2].inherited
    assert result[3].primary_text == "grasp" and result[3].inherited
    assert result[4].primary_text == "place" and not result[4].inherited
    assert result[5].primary_text == "generic task"
    assert result[5].primary_span_id == -1


def test_inspection_and_cached_embedding_extraction() -> None:
    group = make_synthetic_episode()
    inspection = inspect_episode(group, source_path="fixture.zarr")
    assert inspection.image_key == "observations/front_rgb"
    assert inspection.dino_key == "features/dino.front"
    assert inspection.qwen_key == "features/qwen.annotations"
    assert inspection.annotation_count == 3

    config = ExtractionConfig(sample_fps=4.0, source_fps=8.0)
    result = extract_episode_features(group, config=config, source_path="fixture.zarr")
    assert result.visual_embeddings.shape == (12, 6)
    assert result.language_embeddings.shape == (12, 7)
    assert result.frame_indices.tolist() == list(range(0, 24, 2))
    expected_first = group["features"]["dino.front"][0].mean(axis=0)
    np.testing.assert_allclose(result.visual_embeddings[0], expected_first)
    # At overlap frame 6, the latest-started "grasp" span is primary.
    overlap_row = result.frame_indices.tolist().index(6)
    assert result.annotation_texts[overlap_row] == "grasp object"
    assert result.stage_ids[overlap_row] == 1


def test_deterministic_text_fallback_and_pooling() -> None:
    group = make_synthetic_episode(with_qwen=False)
    result_a = extract_episode_features(group)
    result_b = extract_episode_features(group)
    assert result_a.language_source == "feature_hash"
    assert result_a.language_embeddings.shape[1] == 128
    np.testing.assert_array_equal(result_a.language_embeddings, result_b.language_embeddings)
    pooled = mean_pool_embeddings(np.ones((2, 3, 4, 5), dtype=np.float32))
    assert pooled.shape == (2, 5)
    np.testing.assert_array_equal(
        deterministic_text_embeddings(["Pick up object"]),
        deterministic_text_embeddings([" pick  UP   object "]),
    )


def test_pickle_free_cache_contract_and_episode_splits(tmp_path) -> None:
    paths = []
    for index in range(7):
        features = extract_episode_features(
            make_synthetic_episode(), episode_id=f"episode-{index}", source_path=f"{index}.zarr"
        )
        path = tmp_path / f"episode-{index}.npz"
        save_feature_cache(features, path)
        paths.append(path)
    cache = load_feature_cache(paths[0])
    assert REQUIRED_KEYS.issubset(cache)
    assert cache["visual_embeddings"].dtype == np.float32
    assert cache["visual_embeddings"].shape[0] == cache["annotation_texts"].shape[0]
    manifest = build_cache_manifest(reversed(paths), tmp_path / "manifest.json")
    splits = [entry["split"] for entry in manifest["episodes"]]
    assert splits.count("train") == 5
    assert splits.count("val") == 1
    assert splits.count("test") == 1
    assert json.loads((tmp_path / "manifest.json").read_text())["episode_count"] == 7


def test_hard_caps_and_remote_source_redaction() -> None:
    assert HardCaps().max_feature_workers == 8
    assert HardCaps().max_scoring_workers == 12
    assert safe_source_path("https://example.invalid/x.zarr?signature=never-store") == "<remote-source>"
    with pytest.raises(ValueError, match="hard cap"):
        build_cache_manifest([f"{index}.npz" for index in range(61)])


def test_real_zarr_round_trip_when_dependency_is_available(tmp_path) -> None:
    pytest.importorskip("zarr")
    path = write_synthetic_zarr(tmp_path / "fixture.zarr")
    inspection = inspect_episode(path)
    assert inspection.annotation_count == 3
    result = extract_episode_features(path)
    assert result.episode_id == "synthetic-episode-001"
    assert result.visual_embeddings.shape[1] == 6


def test_egoverse_jpeg_blob_layout_is_discovered_and_decoded(monkeypatch) -> None:
    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.fromarray(np.full((5, 7, 3), 127, dtype=np.uint8)).save(buffer, format="JPEG")
    blob = buffer.getvalue()
    group = make_synthetic_episode(with_qwen=False)
    del group["features"]["dino.front"]
    group["observations"] = {}
    group["images.front_1"] = np.asarray([blob] * 24, dtype=object)
    captured = {}

    def fake_dino(frames, *, model_name, batch_size):
        captured["frames"] = frames
        return np.ones((len(frames), 9), dtype=np.float32)

    monkeypatch.setattr("hackathon.egoflow.features.infer_dinov2_small", fake_dino)
    inspection = inspect_episode(group, source_path="encoded.zarr")
    assert inspection.image_key == "images.front_1"
    result = extract_episode_features(group, source_path="encoded.zarr")
    assert result.visual_embeddings.shape == (12, 9)
    assert len(captured["frames"]) == 12
    assert captured["frames"][0].shape == (5, 7, 3)


def test_exact_egoverse_zarr_writer_layout(tmp_path) -> None:
    pytest.importorskip("zarr")
    pytest.importorskip("PIL.Image")
    path = write_egoverse_writer_fixture(tmp_path / "writer.zarr")
    inspection = inspect_episode(path)
    assert inspection.image_key == "images.front_1"
    assert inspection.annotation_count == 2
    assert inspection.frame_count == 12
