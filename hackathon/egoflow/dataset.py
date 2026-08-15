"""Episode-level loading for deterministic frozen-feature caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class EpisodeFeatures:
    episode_id: str
    task: str
    visual_embeddings: Any
    language_embeddings: Any
    stage_ids: Any
    timestamps: Any
    frame_indices: Any
    annotation_texts: tuple[str, ...] = ()

    @property
    def length(self) -> int:
        return int(self.visual_embeddings.shape[0])


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("EgoFlow feature loading requires numpy") from exc
    return np


def discover_feature_files(cache_dir: str | Path, max_episodes: int | None = None) -> list[Path]:
    paths = sorted(Path(cache_dir).glob("*.npz"))
    return paths if max_episodes is None else paths[: max(0, max_episodes)]


def read_episode_id(path: str | Path) -> str:
    """Read only the tiny scalar id without decompressing feature matrices."""
    np = _numpy()
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if "episode_id" not in archive:
            return path.stem
        value = archive["episode_id"]
        return str(value.item() if value.ndim == 0 else value[0])


def _first_present(archive: Any, names: Sequence[str]) -> Any:
    for name in names:
        if name in archive:
            return archive[name]
    raise KeyError(f"feature cache is missing one of: {', '.join(names)}")


def load_episode(path: str | Path) -> EpisodeFeatures:
    """Load one cache, accepting the common EgoVerse feature key variants."""
    np = _numpy()
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        visual = np.asarray(_first_present(archive, ("visual_embeddings", "dino_embeddings", "dino_features")))
        # Existing DINO arrays are often [frames, patches, dim]. Mean pool them.
        if visual.ndim == 3:
            visual = visual.mean(axis=1)
        if visual.ndim != 2:
            raise ValueError(f"{path}: visual embeddings must be [T,D] or [T,P,D]")

        stage_ids = np.asarray(
            _first_present(archive, ("stage_ids", "primary_annotation_ids", "annotation_ids")),
            dtype=np.int64,
        ).reshape(-1)
        language = np.asarray(
            _first_present(archive, ("language_embeddings", "qwen_embeddings", "text_embeddings"))
        )
        if language.ndim != 2:
            raise ValueError(f"{path}: language embeddings must be [T,D] or [stages,D]")
        if language.shape[0] != visual.shape[0]:
            nonnegative = stage_ids[stage_ids >= 0]
            if len(nonnegative) and nonnegative.max() >= language.shape[0]:
                raise ValueError(f"{path}: stage id cannot index per-stage language embeddings")
            # stage_id=-1 marks task-description/gap frames. Do not accidentally
            # index the last stage embedding via NumPy's negative-index behavior.
            expanded = np.zeros((visual.shape[0], language.shape[1]), dtype=language.dtype)
            valid_stages = stage_ids >= 0
            expanded[valid_stages] = language[stage_ids[valid_stages]]
            language = expanded

        length = visual.shape[0]
        if len(stage_ids) != length or language.shape[0] != length:
            raise ValueError(f"{path}: feature arrays disagree on frame count")
        timestamps = np.asarray(archive["timestamps"] if "timestamps" in archive else np.arange(length) / 4.0)
        frame_indices = np.asarray(archive["frame_indices"] if "frame_indices" in archive else np.arange(length))
        if len(timestamps) != length or len(frame_indices) != length:
            raise ValueError(f"{path}: timestamps/frame_indices disagree on frame count")
        if not np.isfinite(visual).all() or not np.isfinite(language).all():
            raise ValueError(f"{path}: embeddings contain NaN/Inf")

        def scalar_text(name: str, fallback: str) -> str:
            if name not in archive:
                return fallback
            value = archive[name]
            return str(value.item() if value.ndim == 0 else value[0])

        texts = tuple(str(value) for value in archive["annotation_texts"].tolist()) if "annotation_texts" in archive else ()
        return EpisodeFeatures(
            episode_id=scalar_text("episode_id", path.stem),
            task=scalar_text("task", "unknown"),
            visual_embeddings=visual.astype(np.float32, copy=False),
            language_embeddings=language.astype(np.float32, copy=False),
            stage_ids=stage_ids,
            timestamps=timestamps.astype(np.float64, copy=False),
            frame_indices=frame_indices.astype(np.int64, copy=False),
            annotation_texts=texts,
        )


def split_episode_paths(
    paths: Iterable[str | Path], seed: int = 17
) -> dict[str, list[Path]]:
    """Deterministically split by whole episode, never by frames."""
    items = sorted({Path(path) for path in paths})
    random.Random(seed).shuffle(items)
    count = len(items)
    if count == 0:
        return {"train": [], "val": [], "test": []}
    if count == 1:
        return {"train": items, "val": [], "test": []}
    if count == 2:
        return {"train": items[:1], "val": items[1:], "test": []}
    val_count = max(1, round(count * 0.15))
    test_count = max(1, round(count * 0.15))
    if val_count + test_count >= count:
        val_count = test_count = 1
    train_count = count - val_count - test_count
    return {
        "train": items[:train_count],
        "val": items[train_count : train_count + val_count],
        "test": items[train_count + val_count :],
    }


def collate_episodes(episodes: Sequence[EpisodeFeatures], device: str = "cpu") -> dict[str, Any]:
    """Pad full episodes. Padding stage ids are -1 and are excluded from losses."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("collate_episodes requires PyTorch") from exc
    if not episodes:
        raise ValueError("cannot collate an empty batch")
    lengths = torch.tensor([episode.length for episode in episodes], dtype=torch.long)
    max_length = int(lengths.max().item())
    visual_dim = int(episodes[0].visual_embeddings.shape[1])
    language_dim = int(episodes[0].language_embeddings.shape[1])
    visual = torch.zeros((len(episodes), max_length, visual_dim), dtype=torch.float32)
    language = torch.zeros((len(episodes), max_length, language_dim), dtype=torch.float32)
    stages = torch.full((len(episodes), max_length), -1, dtype=torch.long)
    mask = torch.zeros((len(episodes), max_length), dtype=torch.bool)
    for index, episode in enumerate(episodes):
        if episode.visual_embeddings.shape[1] != visual_dim or episode.language_embeddings.shape[1] != language_dim:
            raise ValueError("all episodes in a run must have matching embedding widths")
        length = episode.length
        visual[index, :length] = torch.from_numpy(episode.visual_embeddings)
        language[index, :length] = torch.from_numpy(episode.language_embeddings)
        stages[index, :length] = torch.from_numpy(episode.stage_ids)
        mask[index, :length] = True
    return {
        "visual_embeddings": visual.to(device),
        "language_embeddings": language.to(device),
        "stage_ids": stages.to(device),
        "mask": mask.to(device),
        "lengths": lengths,
    }


def make_synthetic_cache(directory: str | Path, episodes: int = 4, seed: int = 7) -> list[Path]:
    """Create tiny learnable caches for end-to-end smoke tests."""
    np = _numpy()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    output: list[Path] = []
    for episode_index in range(episodes):
        frames, visual_dim, language_dim, stage_count = 48, 16, 12, 3
        stage_ids = np.repeat(np.arange(stage_count), frames // stage_count)
        phase = np.tile(np.linspace(0.0, 1.0, frames // stage_count), stage_count)
        visual = rng.normal(scale=0.15, size=(frames, visual_dim)).astype(np.float32)
        visual[:, 0] += phase * 3.0
        language_table = rng.normal(size=(stage_count, language_dim)).astype(np.float32)
        path = directory / f"synthetic-{episode_index:02d}.npz"
        np.savez_compressed(
            path,
            episode_id=f"synthetic-{episode_index:02d}",
            task="synthetic_pick_place",
            visual_embeddings=visual,
            language_embeddings=language_table,
            stage_ids=stage_ids,
            timestamps=np.arange(frames, dtype=np.float64) / 4.0,
            frame_indices=np.arange(frames, dtype=np.int64),
            annotation_texts=np.asarray(["reach", "pick", "place"]),
        )
        output.append(path)
    return output
