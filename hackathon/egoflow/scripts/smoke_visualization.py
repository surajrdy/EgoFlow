#!/usr/bin/env python3
"""Generate clearly synthetic data and prove the no-video visualization path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from results_schema import load_scores, write_summary  # noqa: E402
from visualize import render_timeline  # noqa: E402


def synthetic_payload() -> dict:
    fps, duration = 4.0, 30.0
    timestamps = [i / fps for i in range(int(duration * fps) + 1)]
    local, global_progress, velocity, labels, annotations = [], [], [], [], []
    for t in timestamps:
        if t < 7:
            p, state, annotation = t / 8, "productive", "synthetic reach"
        elif t < 10:
            p, state, annotation = 0.86 + 0.01 * math.sin(t * 3), "stall", "synthetic reach"
        elif t < 13:
            p, state, annotation = 0.86 - (t - 10) * 0.09, "regress", "synthetic reach"
        elif t < 16:
            p, state, annotation = 0.59 + (t - 13) * 0.03, "hesitate", "synthetic reach"
        elif t < 21:
            p, state, annotation = min(1.0, 0.68 + (t - 16) * 0.075), "recover", "synthetic reach"
        elif t < 28:
            p, state, annotation = min(1.0, (t - 21) / 6.0), "productive", "synthetic place"
        else:
            p, state, annotation = 1.0, "complete", "synthetic place"
        local.append(max(0.0, min(1.0, p)))
        global_progress.append(min(1.0, (0.5 * local[-1] if t < 21 else 0.5 + 0.5 * local[-1])))
        labels.append(state)
        annotations.append(annotation)
    for i, p in enumerate(local):
        before = local[max(0, i - 1)]
        velocity.append((p - before) * fps)
    return {
        "schema_version": "egoflow.score.v1",
        "synthetic": True,
        "episode_id": "synthetic_smoke_episode",
        "task": "synthetic two-stage manipulation",
        "completion_confidence": 0.88,
        "timestamps_sec": timestamps,
        "local_progress": local,
        "global_progress": global_progress,
        "progress_velocity": velocity,
        "event_labels": labels,
        "confidences": [0.8] * len(timestamps),
        "annotations": annotations,
        "provenance": "Programmatically generated visualization smoke data. No real episode/model claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "synthetic_scores.json"
    score_path.write_text(json.dumps(synthetic_payload(), indent=2) + "\n", encoding="utf-8")
    series = load_scores(score_path)
    timeline = render_timeline(series, args.output_dir / "example_timeline.png")
    summary = write_summary(series, args.output_dir / "episode_summary.json")
    assert timeline.exists() and timeline.stat().st_size > 1000
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["synthetic"] is True and "not a model result" in payload["provenance_note"].lower()
    print(f"synthetic score input: {score_path}")
    print(f"timeline: {timeline}")
    print(f"summary: {summary}")
    print("PASS: artifacts are explicitly marked synthetic; no real-data/model claim is made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

