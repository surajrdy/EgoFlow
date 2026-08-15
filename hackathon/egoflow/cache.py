"""Versioned, pickle-free NPZ contract shared by extraction and model lanes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .config import SCHEMA_VERSION
from .features import EpisodeFeatures


REQUIRED_KEYS = frozenset(
    {
        "visual_embeddings",
        "language_embeddings",
        "timestamps",
        "frame_indices",
        "stage_ids",
        "primary_annotation_ids",
        "episode_id",
        "task",
        "annotation_texts",
        "metadata_json",
    }
)


def save_feature_cache(features: EpisodeFeatures, output_path: str | Path) -> Path:
    features.validate()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_path": features.source_path,
        "image_key": features.image_key,
        "fps": features.fps,
        "sample_fps": features.sample_fps,
        "preprocessing_version": SCHEMA_VERSION,
        "visual_source": features.visual_source,
        "language_source": features.language_source,
    }
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", prefix=f".{output.stem}-", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            visual_embeddings=np.asarray(features.visual_embeddings, dtype=np.float32),
            language_embeddings=np.asarray(features.language_embeddings, dtype=np.float32),
            timestamps=np.asarray(features.timestamps, dtype=np.float32),
            frame_indices=np.asarray(features.frame_indices, dtype=np.int64),
            stage_ids=np.asarray(features.stage_ids, dtype=np.int32),
            primary_annotation_ids=np.asarray(features.primary_annotation_ids, dtype=np.int32),
            episode_id=np.asarray(features.episode_id, dtype=np.str_),
            task=np.asarray(features.task, dtype=np.str_),
            annotation_texts=np.asarray(features.annotation_texts, dtype=np.str_),
            all_active_texts_json=np.asarray(features.all_active_texts_json, dtype=np.str_),
            inherited_annotation=np.asarray(features.inherited_annotation, dtype=np.bool_),
            source_path=np.asarray(features.source_path, dtype=np.str_),
            image_key=np.asarray(features.image_key or "", dtype=np.str_),
            fps=np.asarray(features.fps, dtype=np.float32),
            sample_fps=np.asarray(features.sample_fps, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
    os.replace(temporary, output)
    return output


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"feature cache missing required keys: {sorted(missing)}")
        result = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(result["metadata_json"].item()))
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported feature cache schema: {metadata.get('schema_version')}")
    count = len(result["frame_indices"])
    for key in (
        "visual_embeddings",
        "language_embeddings",
        "timestamps",
        "stage_ids",
        "primary_annotation_ids",
        "annotation_texts",
    ):
        if len(result[key]) != count:
            raise ValueError(f"cache key {key!r} is not aligned to frame_indices")
    return result
