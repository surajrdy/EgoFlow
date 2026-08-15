"""Held-out, interruption, truncation, and small-manual-set evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .events.detect_events import EventConfig, detect_events


def _load_scores(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    paths = sorted(source.glob("*.json")) if source.is_dir() else [source]
    scores: list[dict[str, Any]] = []
    for item in paths:
        try:
            payload = json.loads(item.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "frames" in payload and "episode_id" in payload:
            scores.append(payload)
    if not scores:
        raise FileNotFoundError(f"no scored episode JSON files found at {source}")
    return scores


def _pairwise(values: Sequence[float], stages: Sequence[int], min_separation: int = 2) -> tuple[int, int]:
    correct = total = 0
    for later in range(len(values)):
        for earlier in range(0, later - min_separation + 1):
            if stages[earlier] == stages[later] and stages[later] >= 0:
                total += 1
                correct += float(values[later]) > float(values[earlier])
    return int(correct), total


def ranking_metrics(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    totals = defaultdict(lambda: [0, 0])
    for score in scores:
        frames = score["frames"]
        stages = [int(frame["stage_id"]) for frame in frames]
        series = {
            "egoflow": [float(frame["local_progress"]) for frame in frames],
            "time_fraction": score["baselines"]["time_fraction"],
            "final_frame_cosine": score["baselines"]["final_frame_cosine"],
        }
        for name, values in series.items():
            correct, count = _pairwise(values, stages)
            totals[name][0] += correct
            totals[name][1] += count
    return {
        name: {
            "accuracy": round(correct / count, 6) if count else None,
            "correct": correct,
            "pairs": count,
        }
        for name, (correct, count) in totals.items()
    }


def synthetic_interruption_metrics() -> dict[str, Any]:
    """Exercise event post-processing on transparent semantic corruptions."""
    timestamps = [index * 0.25 for index in range(40)]
    stages = [0] * 40
    clean = [index / 39 for index in range(40)]
    hold = clean[:14] + [clean[13]] * 12 + clean[26:]
    reverse = clean[:22] + list(reversed(clean[12:22])) + clean[32:]
    abort = clean[:22] + [clean[21] - (index + 1) * 0.06 for index in range(18)]
    variants = {"clean": clean, "hold": hold, "reverse": reverse, "abort": abort}
    expected = {
        "clean": {"productive"},
        "hold": {"stall", "hesitate", "abandon"},
        "reverse": {"regress", "recover", "hesitate"},
        "abort": {"regress", "abandon"},
    }
    result: dict[str, Any] = {}
    for name, curve in variants.items():
        detection = detect_events(
            curve,
            timestamps,
            stages,
            config=EventConfig(velocity_floor=0.02, min_event_sec=0.5),
        )
        labels = set(detection["frame_labels"])
        result[name] = {
            "labels": sorted(labels),
            "expected_any": sorted(expected[name]),
            "passed": bool(labels & expected[name]),
        }
    result["passed"] = all(value["passed"] for value in result.values())
    result["scope"] = "event-layer proxy; learned model is evaluated separately by held-out ranking"
    return result


def truncation_metrics(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_fraction: dict[str, list[float]] = defaultdict(list)
    per_episode: dict[str, dict[str, float]] = {}
    for score in scores:
        values = {
            str(name): float(value)
            for name, value in score.get("truncation_completion_confidence", {}).items()
        }
        if not values:
            continue
        per_episode[str(score["episode_id"])] = values
        for fraction, value in values.items():
            by_fraction[fraction].append(value)
    means = {
        fraction: round(sum(values) / len(values), 6)
        for fraction, values in sorted(by_fraction.items(), key=lambda item: int(item[0].rstrip("%")))
    }
    ordered = [means.get(f"{fraction}%") for fraction in (25, 50, 75, 100)]
    comparable = [value for value in ordered if value is not None]
    return {
        "mean_completion_confidence": means,
        "mean_drop_score": {
            fraction: round(1.0 - value, 6) for fraction, value in means.items()
        },
        "nondecreasing_mean": all(
            comparable[index] <= comparable[index + 1] + 1e-9
            for index in range(len(comparable) - 1)
        ),
        "full_higher_than_all_truncations": bool(comparable) and all(
            comparable[-1] > value for value in comparable[:-1]
        ),
        "episodes": per_episode,
        "note": "Each prefix is independently inferred; future frames are hidden from the BiGRU.",
    }


def _manual_labels(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    labels: list[dict[str, Any]] = []
    source = Path(path)
    if not source.exists():
        return []
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
        if value.get("label") != "other":
            labels.append(value)
    return labels


def manual_event_metrics(
    scores: Iterable[dict[str, Any]], labels: Sequence[dict[str, Any]], tolerance_sec: float = 1.5
) -> dict[str, Any]:
    predictions: dict[str, list[dict[str, Any]]] = {
        str(score["episode_id"]): list(score.get("events", [])) for score in scores
    }
    used: set[tuple[str, int]] = set()
    matched: list[dict[str, Any]] = []
    by_label = Counter(str(label["label"]).lower() for label in labels)
    matched_by_label: Counter[str] = Counter()
    for label in labels:
        episode_id, wanted = str(label["episode_id"]), str(label["label"]).lower()
        start, end = float(label["start_sec"]), float(label["end_sec"])
        for index, predicted in enumerate(predictions.get(episode_id, [])):
            if (episode_id, index) in used or str(predicted.get("label", "")).lower() != wanted:
                continue
            pred_start, pred_end = float(predicted["start_sec"]), float(predicted["end_sec"])
            if pred_end >= start - tolerance_sec and pred_start <= end + tolerance_sec:
                used.add((episode_id, index))
                matched_by_label[wanted] += 1
                matched.append(
                    {
                        "episode_id": episode_id,
                        "label": wanted,
                        "manual_range": [start, end],
                        "predicted_range": [pred_start, pred_end],
                    }
                )
                break
    evaluated_episodes = {str(label["episode_id"]) for label in labels}
    false_positives = [
        {"episode_id": episode_id, **event}
        for episode_id, events in predictions.items()
        if episode_id in evaluated_episodes
        for index, event in enumerate(events)
        if (episode_id, index) not in used and str(event.get("label", "")).lower() in by_label
    ]
    recall = len(matched) / len(labels) if labels else None
    reviewed_precision_denominator = len(matched) + len(false_positives)
    reviewed_precision = (
        len(matched) / reviewed_precision_denominator
        if reviewed_precision_denominator
        else None
    )
    return {
        "manual_labels": len(labels),
        "matched": len(matched),
        "false_positives": len(false_positives),
        "recall_on_manual_events": round(recall, 6) if recall is not None else None,
        "precision_on_reviewed_episodes": (
            round(reviewed_precision, 6) if reviewed_precision is not None else None
        ),
        "tolerance_sec": tolerance_sec,
        "raw_by_label": {
            label: {"manual": count, "matched": matched_by_label[label]}
            for label, count in sorted(by_label.items())
        },
        "matches": matched,
        "false_positive_events": false_positives,
        "note": (
            "Raw same-set small-sample counts. Precision counts only predictions with "
            "a manually represented label on reviewed episodes; no held-out or "
            "population-level accuracy claim."
        ),
    }


def _scores_for_detector(
    scores: Iterable[dict[str, Any]], detector_names: set[str]
) -> list[dict[str, Any]]:
    """Return score shells containing only events from declared detector sources."""

    return [
        {
            "episode_id": score["episode_id"],
            "events": [
                event
                for event in score.get("events", [])
                if str(event.get("detector", "")) in detector_names
            ],
        }
        for score in scores
    ]


def evaluate(
    scores_path: str | Path,
    manual_labels_path: str | Path | None = None,
    output_path: str | Path | None = None,
    tolerance_sec: float = 1.5,
) -> dict[str, Any]:
    scores = _load_scores(scores_path)
    test_scores = [score for score in scores if score.get("split") == "test"]
    validation_scores = [score for score in scores if score.get("split") == "val"]
    held_out = test_scores or validation_scores or scores
    evaluation_split = "test" if test_scores else "val" if validation_scores else "all_unassigned"
    manual_labels = _manual_labels(manual_labels_path)
    metrics = {
        "episodes": len(held_out),
        "evaluation_split": evaluation_split,
        "pairwise_ranking": ranking_metrics(held_out),
        "synthetic_interruptions": synthetic_interruption_metrics(),
        "truncation": truncation_metrics(held_out),
        "manual_events": manual_event_metrics(
            scores, manual_labels, tolerance_sec=tolerance_sec
        ),
        "manual_events_by_source": {
            "learned_progress_normalized": manual_event_metrics(
                _scores_for_detector(scores, {"learned_progress_normalized"}),
                manual_labels,
                tolerance_sec=tolerance_sec,
            ),
            "hybrid": manual_event_metrics(
                _scores_for_detector(
                    scores, {"hybrid_learned_progress_visual_dynamics"}
                ),
                manual_labels,
                tolerance_sec=tolerance_sec,
            ),
            "auxiliary_visual_only": manual_event_metrics(
                _scores_for_detector(scores, {"frozen_visual_dynamics"}),
                manual_labels,
                tolerance_sec=tolerance_sec,
            ),
        },
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--manual-labels", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance-sec", type=float, default=1.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output or (args.scores / "metrics.json" if args.scores.is_dir() else args.scores.with_name("metrics.json"))
    metrics = evaluate(args.scores, args.manual_labels, output, args.tolerance_sec)
    ranking = metrics["pairwise_ranking"]["egoflow"]
    manual = metrics["manual_events"]
    print(
        f"RESULT: held-out pairs={ranking['pairs']} accuracy={ranking['accuracy']} "
        f"manual={manual['matched']}/{manual['manual_labels']}",
        flush=True,
    )
    print(f"NEXT COMMAND: inspect {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
