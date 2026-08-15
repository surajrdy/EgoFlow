"""Episode-level sampling and frozen feature extraction."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .annotations import FrameAnnotation, assign_annotations
from .config import ExtractionConfig, SCHEMA_VERSION
from .zarr_io import (
    array_map,
    extract_annotation_spans,
    inspect_episode,
    open_episode,
    safe_source_path,
)


@dataclass
class EpisodeFeatures:
    visual_embeddings: np.ndarray
    language_embeddings: np.ndarray
    timestamps: np.ndarray
    frame_indices: np.ndarray
    stage_ids: np.ndarray
    primary_annotation_ids: np.ndarray
    episode_id: str
    task: str
    annotation_texts: np.ndarray
    source_path: str
    image_key: str | None
    fps: float
    sample_fps: float
    visual_source: str
    language_source: str
    all_active_texts_json: np.ndarray
    inherited_annotation: np.ndarray

    def validate(self) -> None:
        count = len(self.frame_indices)
        per_frame = (
            self.visual_embeddings,
            self.language_embeddings,
            self.timestamps,
            self.stage_ids,
            self.primary_annotation_ids,
            self.annotation_texts,
            self.all_active_texts_json,
            self.inherited_annotation,
        )
        if any(len(value) != count for value in per_frame):
            raise ValueError("all feature-cache arrays must share the same time dimension")
        if self.visual_embeddings.ndim != 2 or self.language_embeddings.ndim != 2:
            raise ValueError("visual and language embeddings must have shape [T, D]")
        if count == 0:
            raise ValueError("cannot cache an empty episode")


def sample_frame_indices(
    frame_count: int, source_fps: float, sample_fps: float, max_frames: int
) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    step = source_fps / sample_fps
    indices = np.unique(np.floor(np.arange(0, frame_count, step)).astype(np.int64))
    indices = indices[indices < frame_count]
    if len(indices) > max_frames:
        # Cover the whole episode instead of silently truncating its ending.
        positions = np.linspace(0, len(indices) - 1, max_frames).round().astype(np.int64)
        indices = indices[positions]
    return indices


def _take(array: Any, indices: np.ndarray) -> np.ndarray:
    try:
        if hasattr(array, "oindex"):
            return np.asarray(array.oindex[indices])
        return np.asarray(array[indices])
    except (IndexError, TypeError, ValueError):
        return np.stack([np.asarray(array[int(index)]) for index in indices])


def mean_pool_embeddings(values: np.ndarray) -> np.ndarray:
    """Mean-pool all patch/view axes while preserving time and embedding axes."""

    values = np.asarray(values)
    if values.ndim < 2:
        raise ValueError(f"expected embedding array with >=2 dimensions, got {values.shape}")
    if values.ndim > 2:
        values = values.mean(axis=tuple(range(1, values.ndim - 1)))
    return np.asarray(values, dtype=np.float32)


def deterministic_text_embeddings(texts: Sequence[str], dim: int = 128) -> np.ndarray:
    """Signed feature hashing fallback; deterministic across machines and processes."""

    if dim < 1:
        raise ValueError("fallback embedding dimension must be positive")
    output = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        normalized = " ".join(str(text).lower().split()) or "<empty>"
        tokens = normalized.split()
        features = tokens + [f"{tokens[i]}::{tokens[i + 1]}" for i in range(len(tokens) - 1)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:8], "little") % dim
            sign = 1.0 if digest[8] & 1 else -1.0
            output[row, bucket] += sign
        norm = float(np.linalg.norm(output[row]))
        if norm:
            output[row] /= norm
    return output


def infer_dinov2_small(
    frames: Sequence[np.ndarray], *, model_name: str, batch_size: int
) -> np.ndarray:
    """Run public frozen DINOv2-small only when no cached DINO array exists."""

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:  # pragma: no cover - GPU deployment path
        raise RuntimeError(
            "No dino.* array was found. Public DINOv2 inference requires torch and "
            "transformers; install the optional packages in requirements.txt."
        ) from exc

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        inputs = processor(images=list(batch), return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            pooled = hidden[:, 1:].mean(dim=1) if hidden.shape[1] > 1 else hidden[:, 0]
        chunks.append(pooled.float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def decode_image_blobs(values: Sequence[Any]) -> list[np.ndarray]:
    """Decode sampled EgoVerse JPEG blobs, preferring simplejpeg when installed."""

    decoded: list[np.ndarray] = []
    try:
        import simplejpeg
    except ImportError:  # Pillow is the public DINO transform's usual dependency.
        simplejpeg = None
    if simplejpeg is None:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - GPU deployment path
            raise RuntimeError(
                "Encoded images.front_1 frames require simplejpeg or Pillow for decoding"
            ) from exc
    for value in values:
        if isinstance(value, np.ndarray):
            value = value.item() if value.ndim == 0 else value.tobytes()
        blob = bytes(value)
        if simplejpeg is not None:
            image = simplejpeg.decode_jpeg(blob, colorspace="RGB")
        else:
            with Image.open(io.BytesIO(blob)) as opened:
                image = np.asarray(opened.convert("RGB"))
        decoded.append(np.asarray(image, dtype=np.uint8))
    return decoded


def _visual_embeddings(
    arrays: dict[str, Any], inspection: Any, indices: np.ndarray, config: ExtractionConfig
) -> tuple[np.ndarray, str]:
    if inspection.dino_key:
        return mean_pool_embeddings(_take(arrays[inspection.dino_key], indices)), inspection.dino_key
    if not inspection.image_key:
        raise ValueError("episode contains neither a dino.* feature array nor RGB frames")
    frames = _take(arrays[inspection.image_key], indices)
    if frames.ndim == 1 and frames.dtype.kind in ("O", "S", "V"):
        decoded = decode_image_blobs(frames)
        return (
            infer_dinov2_small(
                decoded, model_name=config.dino_model, batch_size=config.dino_batch_size
            ),
            config.dino_model,
        )
    # Multi-camera images are reduced to one stable view for the emergency fallback.
    while frames.ndim > 4:
        frames = frames[:, 0]
    if frames.ndim != 4:
        raise ValueError(f"unsupported image shape after sampling: {frames.shape}")
    return (
        infer_dinov2_small(
            frames, model_name=config.dino_model, batch_size=config.dino_batch_size
        ),
        config.dino_model,
    )


def _language_embeddings(
    arrays: dict[str, Any],
    qwen_key: str | None,
    indices: np.ndarray,
    frame_count: int,
    assignments: Sequence[FrameAnnotation],
    span_count: int,
    config: ExtractionConfig,
) -> tuple[np.ndarray, str]:
    texts = [item.primary_text for item in assignments]
    if not qwen_key:
        return deterministic_text_embeddings(texts, config.fallback_text_dim), "feature_hash"
    raw = np.asarray(arrays[qwen_key][...])
    if raw.ndim < 2:
        return deterministic_text_embeddings(texts, config.fallback_text_dim), "feature_hash"
    if raw.shape[0] == frame_count:
        return mean_pool_embeddings(raw[indices]), qwen_key
    pooled = mean_pool_embeddings(raw)
    if raw.shape[0] == span_count:
        fallback = deterministic_text_embeddings(texts, pooled.shape[-1])
        result = np.empty_like(fallback)
        for row, assignment in enumerate(assignments):
            span_id = assignment.primary_span_id
            result[row] = pooled[span_id] if 0 <= span_id < len(pooled) else fallback[row]
        return result.astype(np.float32, copy=False), qwen_key
    return deterministic_text_embeddings(texts, config.fallback_text_dim), "feature_hash"


def extract_episode_features(
    source: str | Path | Any,
    *,
    config: ExtractionConfig = ExtractionConfig(),
    episode_id: str | None = None,
    task: str | None = None,
    source_path: str | None = None,
) -> EpisodeFeatures:
    """Load, sample and featurize one episode without ever training an encoder."""

    is_path = isinstance(source, (str, Path))
    group = open_episode(source) if is_path else source
    provenance = source_path or (safe_source_path(source) if is_path else "<in-memory>")
    inspection = inspect_episode(group, source_path=provenance)
    arrays = array_map(group)
    if inspection.frame_count is None:
        raise ValueError("could not infer episode frame count")
    fps = float(inspection.fps or config.source_fps)
    indices = sample_frame_indices(
        inspection.frame_count, fps, config.sample_fps, config.caps.max_frames_per_episode
    )
    spans = extract_annotation_spans(group, arrays)
    task_value = task or (inspection.task if inspection.task != "unknown" else config.task_description)
    assignments = assign_annotations(
        indices.tolist(),
        spans,
        task_description=config.task_description or task_value,
        max_gap_frames=int(round(config.gap_inherit_seconds * fps)),
    )
    visual, visual_source = _visual_embeddings(arrays, inspection, indices, config)
    language, language_source = _language_embeddings(
        arrays,
        inspection.qwen_key,
        indices,
        inspection.frame_count,
        assignments,
        len(spans),
        config,
    )
    stage_ids = np.asarray([item.primary_span_id for item in assignments], dtype=np.int32)
    result = EpisodeFeatures(
        visual_embeddings=visual,
        language_embeddings=language,
        timestamps=(indices / fps).astype(np.float32),
        frame_indices=indices.astype(np.int64, copy=False),
        stage_ids=stage_ids,
        primary_annotation_ids=stage_ids.copy(),
        episode_id=episode_id or inspection.episode_id,
        task=task_value,
        annotation_texts=np.asarray([item.primary_text for item in assignments], dtype=np.str_),
        source_path=safe_source_path(provenance),
        image_key=inspection.image_key,
        fps=fps,
        sample_fps=config.sample_fps,
        visual_source=visual_source,
        language_source=language_source,
        all_active_texts_json=np.asarray(
            [json.dumps(item.active_texts, ensure_ascii=False) for item in assignments],
            dtype=np.str_,
        ),
        inherited_annotation=np.asarray([item.inherited for item in assignments], dtype=np.bool_),
    )
    result.validate()
    return result


def extract_video_features(
    video_path: str | Path,
    *,
    config: ExtractionConfig = ExtractionConfig(),
    episode_id: str,
    task: str,
    task_description: str | None = None,
) -> EpisodeFeatures:
    """Fallback for public Mecka MP4s when processed Zarr access is unavailable.

    Frames are sampled by ffmpeg. With no dense Zarr annotations, the complete
    episode is one semantic stage conditioned on the supplied task description.
    """

    video_path = Path(video_path)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe_payload = json.loads(probe.stdout)
    rate = str(probe_payload["streams"][0].get("avg_frame_rate") or "30/1")
    numerator, denominator = (rate.split("/", 1) + ["1"])[:2]
    source_fps = float(numerator) / max(float(denominator), 1e-9)
    context = str(task_description or task.replace("_", " "))

    with tempfile.TemporaryDirectory(prefix="egoflow-video-") as directory:
        frame_pattern = str(Path(directory) / "%06d.jpg")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"fps={config.sample_fps},scale='min(518,iw)':-2",
                "-q:v",
                "3",
                frame_pattern,
            ],
            check=True,
        )
        frame_paths = sorted(Path(directory).glob("*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"ffmpeg decoded no frames from {video_path.name}")
        if len(frame_paths) > config.caps.max_frames_per_episode:
            positions = np.linspace(
                0, len(frame_paths) - 1, config.caps.max_frames_per_episode
            ).round().astype(np.int64)
            frame_paths = [frame_paths[int(position)] for position in positions]
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Modal image includes Pillow
            raise RuntimeError("public-video fallback requires Pillow") from exc
        frames = []
        for path in frame_paths:
            with Image.open(path) as image:
                frames.append(np.asarray(image.convert("RGB")))
        visual = infer_dinov2_small(
            frames,
            model_name=config.dino_model,
            batch_size=config.dino_batch_size,
        )

    count = len(visual)
    timestamps = np.arange(count, dtype=np.float32) / config.sample_fps
    frame_indices = np.round(timestamps * source_fps).astype(np.int64)
    stage_ids = np.zeros(count, dtype=np.int32)
    annotation_texts = np.asarray([context] * count, dtype=np.str_)
    result = EpisodeFeatures(
        visual_embeddings=visual,
        language_embeddings=deterministic_text_embeddings(
            annotation_texts.tolist(), config.fallback_text_dim
        ),
        timestamps=timestamps,
        frame_indices=frame_indices,
        stage_ids=stage_ids,
        primary_annotation_ids=stage_ids.copy(),
        episode_id=episode_id,
        task=task,
        annotation_texts=annotation_texts,
        source_path="<public-mecka-video>",
        image_key=None,
        fps=source_fps,
        sample_fps=config.sample_fps,
        visual_source=config.dino_model,
        language_source="task_description_feature_hash",
        all_active_texts_json=np.asarray(
            [json.dumps([context])] * count, dtype=np.str_
        ),
        inherited_annotation=np.zeros(count, dtype=np.bool_),
    )
    result.validate()
    return result
