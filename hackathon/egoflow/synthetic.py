"""Tiny deterministic episode fixtures for smoke tests and pipeline bring-up."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import io
import json

import numpy as np


class SyntheticGroup(dict[str, Any]):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}


def make_synthetic_episode(frame_count: int = 24, *, with_qwen: bool = True) -> SyntheticGroup:
    if frame_count < 12:
        raise ValueError("synthetic episode needs at least 12 frames")
    images = np.zeros((frame_count, 8, 8, 3), dtype=np.uint8)
    images[..., 0] = np.arange(frame_count, dtype=np.uint8)[:, None, None]
    # Patch features have shape [T, P, D] and deliberately expose pooling behavior.
    dino = np.arange(frame_count * 3 * 6, dtype=np.float32).reshape(frame_count, 3, 6)
    annotations: dict[str, Any] = {
        "text": np.asarray(["reach for object", "grasp object", "place object"]),
        "start_frame": np.asarray([0, 6, 16], dtype=np.int64),
        "end_frame": np.asarray([9, 13, frame_count - 2], dtype=np.int64),
    }
    features: dict[str, Any] = {"dino.front": dino}
    if with_qwen:
        features["qwen.annotations"] = np.arange(3 * 2 * 7, dtype=np.float32).reshape(3, 2, 7)
    return SyntheticGroup(
        {
            "observations": {"front_rgb": images},
            "features": features,
            "annotations": annotations,
        },
        attrs={
            "episode_id": "synthetic-episode-001",
            "task": "place object",
            "fps": 8.0,
        },
    )


def write_synthetic_zarr(path: str | Path, frame_count: int = 24) -> Path:
    """Materialize the fixture when zarr is installed (useful for a real path smoke test)."""

    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - optional helper
        raise RuntimeError("write_synthetic_zarr requires zarr") from exc
    destination = Path(path)
    source = make_synthetic_episode(frame_count)
    root = zarr.open_group(str(destination), mode="w")
    root.attrs.update(source.attrs)

    def write(group: Any, values: dict[str, Any]) -> None:
        for name, value in values.items():
            if isinstance(value, dict):
                write(group.require_group(name), value)
            elif hasattr(group, "create_array"):
                group.create_array(name, data=value)
            else:  # zarr v2
                group.create_dataset(name, data=value, shape=value.shape, dtype=value.dtype)

    write(root, source)
    return destination


def write_egoverse_writer_fixture(path: str | Path, frame_count: int = 12) -> Path:
    """Write the flat Zarr v3 bytes layout produced by EgoVerse ZarrWriter."""

    try:
        import zarr
        from PIL import Image
        from zarr.core.dtype import VariableLengthBytes
    except ImportError as exc:  # pragma: no cover - optional helper
        raise RuntimeError("writer-layout fixture requires zarr and Pillow") from exc
    destination = Path(path)
    root = zarr.open_group(str(destination), mode="w", zarr_format=3)
    buffer = io.BytesIO()
    Image.fromarray(np.full((6, 8, 3), 96, dtype=np.uint8)).save(buffer, format="JPEG")
    jpeg = buffer.getvalue()
    padded_frames = frame_count + 4
    image_values = np.asarray([jpeg] * padded_frames, dtype=object)
    image_array = root.create_array(
        "images.front_1", shape=(padded_frames,), dtype=VariableLengthBytes()
    )
    image_array[:] = image_values
    records = [
        json.dumps({"text": "reach", "start_idx": 0, "end_idx": 7}).encode(),
        json.dumps({"text": "grasp", "start_idx": 5, "end_idx": 10}).encode(),
    ]
    annotation_array = root.create_array(
        "annotations", shape=(len(records),), dtype=VariableLengthBytes()
    )
    annotation_array[:] = np.asarray(records, dtype=object)
    root.attrs.update(
        {
            "fps": 8,
            "task_name": "synthetic writer task",
            "total_frames": frame_count,
        }
    )
    return destination
