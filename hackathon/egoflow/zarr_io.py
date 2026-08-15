"""Small, read-only adapters for heterogeneous EgoVerse Zarr episodes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .annotations import AnnotationSpan, spans_from_records


@dataclass(frozen=True)
class ArrayInfo:
    path: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class EpisodeInspection:
    source_path: str
    episode_id: str
    task: str
    fps: float | None
    frame_count: int | None
    image_key: str | None
    dino_key: str | None
    qwen_key: str | None
    annotation_count: int
    arrays: tuple[ArrayInfo, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_source_path(source: Any) -> str:
    """Retain local provenance but never persist URLs, which may be signed."""

    value = str(source)
    if "://" in value:
        return "<remote-source>"
    return value


def open_episode(path: str | Path) -> Any:
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - exercised in deployment env
        raise RuntimeError(
            "Reading a Zarr path requires `pip install -r "
            "hackathon/egoflow/requirements.txt`"
        ) from exc
    return zarr.open_group(str(path), mode="r")


def iter_arrays(group: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Traverse Zarr v2/v3 groups and nested dict fixtures without loading arrays."""

    if isinstance(group, Mapping):
        items = group.items()
    elif hasattr(group, "keys"):
        items = ((key, group[key]) for key in group.keys())
    else:
        return
    for raw_key, value in items:
        key = str(raw_key)
        path = f"{prefix}/{key}" if prefix else key
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            yield path, value
        elif isinstance(value, Mapping) or hasattr(value, "keys"):
            yield from iter_arrays(value, path)


def array_map(group: Any) -> dict[str, Any]:
    return {path: array for path, array in iter_arrays(group)}


def _leaf(path: str) -> str:
    return path.rsplit("/", 1)[-1].lower()


def _dtype_kind(dtype: Any) -> str:
    try:
        return np.dtype(dtype).kind
    except TypeError:
        # Zarr v3's VariableLengthBytes is a Zarr dtype, not a NumPy dtype.
        label = str(dtype).lower()
        return "O" if any(token in label for token in ("bytes", "string", "object")) else "?"


def _choose_array(
    arrays: Mapping[str, Any], *, tokens: tuple[str, ...], image: bool = False
) -> str | None:
    candidates: list[tuple[int, str]] = []
    for path, array in arrays.items():
        low = path.lower()
        shape = tuple(int(v) for v in array.shape)
        if not all(token in low for token in tokens):
            continue
        if image:
            decoded_rgb = len(shape) >= 4 and shape[-1] in (1, 3, 4)
            # EgoVerse ZarrWriter stores one JPEG byte blob per frame in arrays
            # such as images.front_1. Object, bytes and void dtypes cover Zarr
            # v2/v3 representations without loading the blobs during inspection.
            dtype_kind = _dtype_kind(array.dtype)
            encoded_jpeg = len(shape) == 1 and dtype_kind in ("O", "S", "V")
            if not decoded_rgb and not encoded_jpeg:
                continue
        candidates.append((len(path.split("/")), path))
    return min(candidates)[1] if candidates else None


def find_image_key(arrays: Mapping[str, Any]) -> str | None:
    preferences = (
        ("front", "img"),
        ("front", "image"),
        ("rgb",),
        ("image",),
        ("img",),
    )
    for tokens in preferences:
        found = _choose_array(arrays, tokens=tokens, image=True)
        if found:
            return found
    # Last resort: shape-based discovery, preferring shorter paths.
    shaped = []
    for path, array in arrays.items():
        shape = tuple(int(value) for value in array.shape)
        kind = _dtype_kind(array.dtype)
        if len(shape) >= 4 and shape[-1] in (1, 3, 4):
            shaped.append(path)
        elif len(shape) == 1 and kind in ("O", "S", "V") and "image" in path.lower():
            shaped.append(path)
    return min(shaped, key=lambda value: (len(value.split("/")), value)) if shaped else None


def find_dino_key(arrays: Mapping[str, Any]) -> str | None:
    keys = [path for path in arrays if "dino" in path.lower()]
    return min(keys, key=lambda path: ("embed" not in path.lower(), len(path), path)) if keys else None


def find_qwen_key(arrays: Mapping[str, Any]) -> str | None:
    keys = [path for path in arrays if "qwen" in path.lower()]
    return min(keys, key=lambda path: ("embed" not in path.lower(), len(path), path)) if keys else None


def _attrs(group: Any) -> Mapping[str, Any]:
    attrs = getattr(group, "attrs", {})
    return attrs if isinstance(attrs, Mapping) else {}


def _whitelisted_attr(group: Any, names: tuple[str, ...], default: Any = None) -> Any:
    attrs = _attrs(group)
    normalized = {str(key).lower(): value for key, value in attrs.items()}
    for name in names:
        if name in normalized:
            return normalized[name]
    return default


def _to_records(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return [
                {name: item[name].item() if hasattr(item[name], "item") else item[name] for name in value.dtype.names}
                for item in value
            ]
        value = value.tolist()
    if isinstance(value, (bytes, str)):
        try:
            value = json.loads(value.decode() if isinstance(value, bytes) else value)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return []
    if isinstance(value, Mapping):
        for key in ("annotations", "spans", "segments", "data"):
            if key in value:
                return _to_records(value[key])
        return [value]
    if isinstance(value, list):
        records = []
        for item in value:
            if isinstance(item, Mapping):
                records.append(item)
            elif isinstance(item, (str, bytes)):
                records.extend(_to_records(item))
        return records
    return []


def extract_annotation_spans(group: Any, arrays: Mapping[str, Any] | None = None) -> list[AnnotationSpan]:
    """Read annotations from structured arrays, JSON attrs, or parallel arrays."""

    arrays = arrays or array_map(group)
    for attr_name in ("annotations", "annotation_spans", "language_annotations"):
        records = _to_records(_whitelisted_attr(group, (attr_name,)))
        if records:
            return spans_from_records(records)

    # Structured/JSON arrays are common conversion outputs.
    for path, array in arrays.items():
        if "annot" not in path.lower() and "segment" not in path.lower():
            continue
        value = np.asarray(array[...])
        records = _to_records(value)
        if records:
            parsed = spans_from_records(records)
            if parsed:
                return parsed

    # Parallel arrays: annotations/text, annotations/start_frame, ...
    for text_path, text_array in arrays.items():
        leaf = _leaf(text_path)
        if leaf not in ("text", "texts", "annotation", "annotations", "language"):
            continue
        parent = text_path.rsplit("/", 1)[0] if "/" in text_path else ""
        siblings = {
            _leaf(path): array
            for path, array in arrays.items()
            if (path.rsplit("/", 1)[0] if "/" in path else "") == parent
        }
        start = next(
            (siblings[key] for key in ("start_frame", "start_idx", "start", "frame_start") if key in siblings),
            None,
        )
        end = next(
            (siblings[key] for key in ("end_frame", "end_idx", "end", "frame_end") if key in siblings),
            None,
        )
        if start is None or end is None:
            continue
        texts = np.asarray(text_array[...]).reshape(-1)
        starts = np.asarray(start[...]).reshape(-1)
        ends = np.asarray(end[...]).reshape(-1)
        count = min(len(texts), len(starts), len(ends))
        records = [
            {
                "text": value.decode(errors="replace") if isinstance(value, bytes) else str(value),
                "start_frame": int(starts[index]),
                "end_frame": int(ends[index]),
            }
            for index, value in enumerate(texts[:count])
        ]
        return spans_from_records(records)
    return []


def inspect_episode(source: str | Path | Any, *, source_path: str | None = None) -> EpisodeInspection:
    group = open_episode(source) if isinstance(source, (str, Path)) else source
    path = safe_source_path(source_path if source_path is not None else source)
    arrays = array_map(group)
    image_key = find_image_key(arrays)
    dino_key = find_dino_key(arrays)
    qwen_key = find_qwen_key(arrays)
    frame_count = None
    declared_frame_count = _whitelisted_attr(group, ("total_frames", "frame_count", "num_frames"))
    if declared_frame_count is not None and int(declared_frame_count) > 0:
        frame_count = int(declared_frame_count)
    for key in (image_key, dino_key):
        if frame_count is None and key and arrays[key].shape:
            frame_count = int(arrays[key].shape[0])
            break
    episode_id = str(
        _whitelisted_attr(group, ("episode_id", "episode_hash", "id"), Path(path).stem)
    )
    task = str(
        _whitelisted_attr(group, ("task", "task_name", "task_description"), "unknown")
    )
    fps_value = _whitelisted_attr(group, ("fps", "frame_rate", "video_fps"))
    fps = float(fps_value) if fps_value is not None else None
    infos = tuple(
        ArrayInfo(path=key, shape=tuple(int(v) for v in value.shape), dtype=str(value.dtype))
        for key, value in sorted(arrays.items())
    )
    spans = extract_annotation_spans(group, arrays)
    return EpisodeInspection(
        source_path=path,
        episode_id=episode_id,
        task=task,
        fps=fps,
        frame_count=frame_count,
        image_key=image_key,
        dino_key=dino_key,
        qwen_key=qwen_key,
        annotation_count=len(spans),
        arrays=infos,
    )
