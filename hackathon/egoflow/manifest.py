"""Build episode-level train/validation/test manifests from completed caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .cache import load_feature_cache
from .config import HardCaps, SCHEMA_VERSION


def _scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def _split_names(count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["train"]
    if count == 2:
        return ["train", "val"]
    validation = max(1, int(round(count * 0.15)))
    test = max(1, int(round(count * 0.15)))
    while validation + test >= count:
        if test >= validation:
            test -= 1
        else:
            validation -= 1
    return ["train"] * (count - validation - test) + ["val"] * validation + ["test"] * test


def build_cache_manifest(
    cache_paths: Iterable[str | Path],
    output_path: str | Path | None = None,
    *,
    caps: HardCaps = HardCaps(),
) -> dict[str, object]:
    paths = [Path(path) for path in cache_paths]
    if len(paths) > caps.max_episodes:
        raise ValueError(f"refusing {len(paths)} caches; hard cap is {caps.max_episodes}")
    entries: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for path in paths:
        cache = load_feature_cache(path)
        episode_id = str(_scalar(cache["episode_id"]))
        if episode_id in seen_ids:
            raise ValueError(f"duplicate episode_id in manifest: {episode_id}")
        seen_ids.add(episode_id)
        metadata = json.loads(str(_scalar(cache["metadata_json"])))
        entries.append(
            {
                "episode_id": episode_id,
                "task": str(_scalar(cache["task"])),
                "cache_path": str(path),
                "frames": int(len(cache["frame_indices"])),
                "duration_sec": float(cache["timestamps"][-1]) if len(cache["timestamps"]) else 0.0,
                "visual_dim": int(cache["visual_embeddings"].shape[-1]),
                "language_dim": int(cache["language_embeddings"].shape[-1]),
                "source_fps": float(metadata["fps"]),
                "fps": float(metadata.get("sample_fps", metadata["fps"])),
                "source_path": metadata.get("source_path", ""),
            }
        )
    # Stable ordering makes the split independent of worker completion order.
    entries.sort(
        key=lambda entry: hashlib.sha256(str(entry["episode_id"]).encode()).hexdigest()
    )
    for entry, split in zip(entries, _split_names(len(entries)), strict=True):
        entry["split"] = split
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "split_policy": "episode_sha256_70_15_15",
        "episode_count": len(entries),
        "episodes": entries,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def discover_caches(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("*.npz"))
