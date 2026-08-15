"""Score cached episodes with EgoFlow and two intentionally cheap baselines."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

from .dataset import collate_episodes, discover_feature_files, load_episode
from .events.detect_events import (
    EventConfig,
    detect_events,
    detect_visual_dynamics,
    derive_global_progress,
    ordered_stages,
    summarize_episode,
)
from .models.progress_model import ProgressModel


def _dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("scoring requires numpy and PyTorch") from exc
    return np, torch


def _final_frame_cosine(visual: Any, tail_frames: int = 5) -> list[float]:
    np, _ = _dependencies()
    target = visual[-min(tail_frames, len(visual)) :].mean(axis=0)
    denominator = np.linalg.norm(visual, axis=1) * max(float(np.linalg.norm(target)), 1e-12)
    cosine = (visual @ target) / np.maximum(denominator, 1e-12)
    return [round(float(value), 6) for value in cosine]


def _predict_prefix(model: Any, episode: Any, length: int, device: str) -> list[float]:
    _, torch = _dependencies()
    prefix = replace(
        episode,
        visual_embeddings=episode.visual_embeddings[:length],
        language_embeddings=episode.language_embeddings[:length],
        stage_ids=episode.stage_ids[:length],
        timestamps=episode.timestamps[:length],
        frame_indices=episode.frame_indices[:length],
    )
    batch = collate_episodes([prefix], device=device)
    with torch.no_grad():
        output = model(batch["visual_embeddings"], batch["language_embeddings"], batch["lengths"])
    return output["local_progress"][0, :length].cpu().tolist()


def _frame_confidence(detection: dict[str, Any]) -> list[float]:
    """Transparent confidence from distance to the robust velocity boundary."""
    result: list[float] = []
    thresholds = detection.get("frame_velocity_thresholds") or [
        detection["velocity_threshold"]
    ] * len(detection["velocity"])
    for velocity, label, raw_threshold in zip(
        detection["velocity"], detection["frame_labels"], thresholds
    ):
        threshold = max(abs(float(raw_threshold)), 1e-8)
        ratio = min(1.0, abs(float(velocity)) / threshold)
        evidence = (1.0 - ratio) if label in {"stall", "hesitate", "abandon"} else ratio
        result.append(round(0.5 + 0.45 * evidence, 6))
    return result


def score_episode(
    feature_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    np, torch = _dependencies()
    episode = load_episode(feature_path)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model = ProgressModel(**checkpoint["model_config"]).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    local = _predict_prefix(model, episode, episode.length, resolved_device)
    timestamps = episode.timestamps.tolist()
    stages = episode.stage_ids.tolist()
    stage_order = ordered_stages(stages)
    global_progress = derive_global_progress(local, stages, stage_order=stage_order)
    detection = detect_events(
        local,
        timestamps,
        stages,
        annotation_texts=episode.annotation_texts,
        config=EventConfig(enable_higher_order_motifs=len(stage_order) > 1),
    )
    visual_detection = detect_visual_dynamics(
        episode.visual_embeddings,
        timestamps,
        max_hesitations=3,
    )
    # The learned curve owns progress; frozen visual dynamics supplement the
    # event layer when task-level public-video context lacks dense annotations.
    learned_events = list(detection["events"])
    auxiliary_events = list(visual_detection["events"])
    learned_labels = list(detection["frame_labels"])
    for event in auxiliary_events:
        if event.get("label") != "hesitate":
            continue
        indices = [
            index
            for index, timestamp in enumerate(timestamps)
            if float(event["start_sec"]) <= float(timestamp) <= float(event["end_sec"])
        ]
        overlap = (
            sum(learned_labels[index] in {"stall", "regress"} for index in indices)
            / len(indices)
            if indices
            else 0.0
        )
        event["learned_nonproductive_overlap"] = round(overlap, 4)
        if overlap >= 0.5:
            event["detector"] = "hybrid_learned_progress_visual_dynamics"
            event["reason"] = "normalized low progress plus visual slowdown/loop-back"
    detection["events"] = learned_events + auxiliary_events
    auxiliary_labels = list(visual_detection["frame_labels"])
    detection["frame_labels"] = [
        auxiliary if auxiliary != "productive" else learned
        for learned, auxiliary in zip(learned_labels, auxiliary_labels)
    ]
    event_sources = [
        "hybrid_learned_progress_visual_dynamics"
        if auxiliary != "productive" and learned in {"stall", "regress"}
        else "auxiliary_visual_dynamics"
        if auxiliary != "productive"
        else "learned_progress_normalized"
        for learned, auxiliary in zip(learned_labels, auxiliary_labels)
    ]
    confidence = _frame_confidence(detection)
    time_fraction = np.linspace(0.0, 1.0, episode.length).tolist()
    final_cosine = _final_frame_cosine(episode.visual_embeddings)
    frame_aligned_annotations = len(episode.annotation_texts) == episode.length
    frames = [
        {
            "frame_index": int(episode.frame_indices[index]),
            "timestamp_sec": round(float(timestamps[index]), 4),
            "stage_id": int(stages[index]),
            "local_progress": round(float(local[index]), 6),
            "global_progress": round(float(global_progress[index]), 6),
            "velocity": detection["velocity"][index],
            "event": detection["frame_labels"][index],
            "event_source": event_sources[index],
            "confidence": round(float(confidence[index]), 6),
            "annotation": (
                episode.annotation_texts[index]
                if frame_aligned_annotations
                else episode.annotation_texts[stages[index]]
                if 0 <= stages[index] < len(episode.annotation_texts)
                else ""
            ),
        }
        for index in range(episode.length)
    ]
    summary = summarize_episode(
        timestamps,
        local,
        global_progress,
        detection["frame_labels"],
        detection["events"],
    )
    split_name = "unassigned"
    for candidate, episode_ids in checkpoint.get("episode_id_split", {}).items():
        if episode.episode_id in episode_ids:
            split_name = str(candidate)
            break
    truncation_confidence: dict[str, float] = {}
    truncation_drop: dict[str, float] = {}
    # Re-run each true prefix because the bidirectional GRU must not see frames
    # beyond the truncation boundary during this sanity check.
    for fraction in (0.25, 0.5, 0.75, 1.0):
        prefix_length = max(2, min(episode.length, round(episode.length * fraction)))
        prefix_local = _predict_prefix(model, episode, prefix_length, resolved_device)
        prefix_times, prefix_stages = timestamps[:prefix_length], stages[:prefix_length]
        prefix_global = derive_global_progress(prefix_local, prefix_stages, stage_order=stage_order)
        prefix_detection = detect_events(
            prefix_local,
            prefix_times,
            prefix_stages,
            annotation_texts=episode.annotation_texts,
        )
        prefix_summary = summarize_episode(
            prefix_times,
            prefix_local,
            prefix_global,
            prefix_detection["frame_labels"],
            prefix_detection["events"],
        )
        truncation_confidence[f"{round(fraction * 100)}%"] = float(
            prefix_summary["completion_confidence"]
        )
        truncation_drop[f"{round(fraction * 100)}%"] = float(
            prefix_summary["drop_score"]
        )
    result: dict[str, Any] = {
        "episode_id": episode.episode_id,
        "task": episode.task,
        "synthetic": bool(
            episode.episode_id.startswith("synthetic-")
            or episode.task.startswith("synthetic_")
        ),
        "provenance_note": (
            "Synthetic training smoke test; not a real-data result or empirical claim."
            if episode.episode_id.startswith("synthetic-")
            or episode.task.startswith("synthetic_")
            else "Scored from an extracted EgoVerse episode cache."
        ),
        "event_detection": {
            "progress_source": "trained_bigru_stage_normalized_velocity",
            "learned_event_count": len(learned_events),
            "auxiliary_event_count": len(auxiliary_events),
            "hybrid_event_count": sum(
                event.get("detector") == "hybrid_learned_progress_visual_dynamics"
                for event in auxiliary_events
            ),
            "auxiliary_source": "frozen_dinov2_visual_dynamics",
            "auxiliary_proposal_cap_per_type": 3,
            "proposal_mode": "source_attributed_preliminary",
            "note": (
                "Visual event proposals supplement the learned progress curve because "
                "public MP4 inputs do not include dense Zarr annotations."
            ),
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "split": split_name,
        "stage_order": stage_order,
        "annotation_texts": list(episode.annotation_texts),
        "frames": frames,
        "events": detection["events"],
        "velocity_threshold": detection["velocity_threshold"],
        "stage_velocity_thresholds": detection["stage_velocity_thresholds"],
        "baselines": {
            "time_fraction": [round(float(value), 6) for value in time_fraction],
            "final_frame_cosine": final_cosine,
        },
        "truncation_completion_confidence": truncation_confidence,
        "truncation_drop_score": truncation_drop,
        **summary,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _feature_paths(value: Path) -> list[Path]:
    return discover_feature_files(value) if value.is_dir() else [value]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="one .npz or a cache directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--max-episodes", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = _feature_paths(args.features)
    if args.max_episodes is not None:
        paths = paths[: args.max_episodes]
    if not paths:
        raise FileNotFoundError(f"no feature files found at {args.features}")
    for path in paths:
        destination = args.output_dir / f"{path.stem}.json"
        result = score_episode(path, args.checkpoint, destination, args.device)
        print(
            f"RESULT: {result['episode_id']} completion={result['completion_confidence']:.3f} "
            f"events={len(result['events'])}",
            flush=True,
        )
    print(f"NEXT COMMAND: python -m hackathon.egoflow.evaluate --scores {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
