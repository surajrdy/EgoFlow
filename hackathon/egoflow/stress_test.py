"""Blind-test feature perturbations for the frozen EgoFlow checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

from .dataset import load_episode
from .events.detect_events import detect_events, detect_visual_dynamics
from .models.progress_model import ProgressModel
from .score import _predict_prefix


def run_stress_test(
    checkpoint_path: str | Path,
    cache_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    import numpy as np
    import torch

    checkpoint_path, cache_dir = Path(checkpoint_path), Path(cache_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ProgressModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_ids = list(checkpoint.get("episode_id_split", {}).get("test", []))
    if not test_ids:
        raise ValueError("checkpoint has no declared blind-test episodes")

    episodes: dict[str, Any] = {}
    for episode_id in test_ids:
        episode = load_episode(cache_dir / f"{episode_id}.npz")
        length = episode.length
        start, end = int(0.35 * length), int(0.55 * length)
        times, stages = episode.timestamps.tolist(), episode.stage_ids.tolist()
        clean_progress = np.asarray(_predict_prefix(model, episode, length, "cpu"))

        stall_visual = episode.visual_embeddings.copy()
        stall_language = episode.language_embeddings.copy()
        stall_visual[start:end] = stall_visual[start]
        stall_language[start:end] = stall_language[start]
        stalled = replace(
            episode,
            visual_embeddings=stall_visual,
            language_embeddings=stall_language,
        )
        stall_progress = np.asarray(_predict_prefix(model, stalled, length, "cpu"))

        reverse_visual = episode.visual_embeddings.copy()
        reverse_language = episode.language_embeddings.copy()
        reverse_visual[start:end] = reverse_visual[start:end][::-1]
        reverse_language[start:end] = reverse_language[start:end][::-1]
        reversed_episode = replace(
            episode,
            visual_embeddings=reverse_visual,
            language_embeddings=reverse_language,
        )
        reverse_progress = np.asarray(_predict_prefix(model, reversed_episode, length, "cpu"))

        def learned_labels(progress: Any) -> list[str]:
            result = detect_events(progress.tolist(), times, stages)
            return sorted(set(result["frame_labels"][start:end]))

        stall_aux = sorted(
            set(detect_visual_dynamics(stall_visual, times)["frame_labels"][start:end])
        )
        reverse_aux = sorted(
            set(detect_visual_dynamics(reverse_visual, times)["frame_labels"][start:end])
        )
        episodes[episode_id] = {
            "injected_range_sec": [round(times[start], 3), round(times[end - 1], 3)],
            "clean_learned_labels": learned_labels(clean_progress),
            "stall": {
                "learned_labels": learned_labels(stall_progress),
                "auxiliary_labels": stall_aux,
                "auxiliary_pass": "stall" in stall_aux,
                "mean_learned_velocity": round(float(np.mean(np.diff(stall_progress[start:end]))), 6),
            },
            "reverse": {
                "learned_labels": learned_labels(reverse_progress),
                "auxiliary_labels": reverse_aux,
                "auxiliary_pass": bool({"regress", "recover"} & set(reverse_aux)),
                "negative_learned_steps": int(np.sum(np.diff(reverse_progress[start:end]) < 0)),
            },
        }

    stall_passes = sum(value["stall"]["auxiliary_pass"] for value in episodes.values())
    reverse_passes = sum(value["reverse"]["auxiliary_pass"] for value in episodes.values())
    learned_reverse_passes = sum(
        value["reverse"]["negative_learned_steps"] > 0 for value in episodes.values()
    )
    result = {
        "split": "blind_test",
        "episodes": episodes,
        "summary": {
            "auxiliary_stall_response": f"{stall_passes}/{len(episodes)}",
            "auxiliary_reverse_response": f"{reverse_passes}/{len(episodes)}",
            "learned_reverse_response": f"{learned_reverse_passes}/{len(episodes)}",
        },
        "interpretation": (
            "Frozen visual dynamics reacted to injected stalls/reversals, while the coarse "
            "learned reward remained nearly monotonic. Synthetic event success must not be "
            "presented as learned reward sensitivity."
        ),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_stress_test(args.checkpoint, args.cache_dir, args.output)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
