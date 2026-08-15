from __future__ import annotations

import json

from hackathon.egoflow.results_schema import load_scores, write_summary
from hackathon.egoflow.visualize import (
    load_hand_events,
    load_manual_events,
    progress_rate_residual,
    render_timeline,
)
from hackathon.egoflow.scripts.make_review_manifest import load_metadata, write_html
from hackathon.egoflow.scripts.summarize_manual_labels import compare, load_manual


def test_progress_rate_residual_is_centered_and_finite():
    timestamps = [float(index) for index in range(9)]
    velocity = [0.1, 0.1, 0.1, 0.1, -0.2, 0.1, 0.1, 0.1, 0.1]
    residual, baseline, scale, guide = progress_rate_residual(timestamps, velocity)
    assert len(residual) == len(velocity) == len(baseline)
    assert residual[4] < 0
    assert scale > 0
    assert guide > 0


def test_load_hand_events_recomputes_source_attributed_candidates(tmp_path):
    points = [(0.10 + index * 0.025, 0.4) for index in range(9)]
    points += [(0.30 - index * 0.018, 0.4 + index * 0.022) for index in range(1, 10)]
    payload = tmp_path / "hands.json"
    payload.write_text(json.dumps({
        "sample_fps": 8,
        "observations": [
            {"timestamp_sec": index / 8, "hand": "right", "x": x, "y": y, "aperture": 1.0}
            for index, (x, y) in enumerate(points)
        ],
    }))
    events = load_hand_events(payload)
    assert events[0]["label"] == "aborted_reach"
    assert events[0]["detector"] == "video_hand_geometry_v1"


def test_loads_scorer_frame_schema_and_renders_without_optional_dependencies(tmp_path):
    score = tmp_path / "episode.json"
    score.write_text(
        json.dumps(
            {
                "episode_id": "held_out_01",
                "task": "organizing objects",
                "completion_confidence": 0.73,
                "annotation_texts": ["reach", "place"],
                "frames": [
                    {"timestamp_sec": 0.0, "stage_id": 0, "local_progress": 0.1, "global_progress": 0.05, "velocity": 0.2, "event": "productive", "confidence": 0.8},
                    {"timestamp_sec": 1.0, "stage_id": 1, "local_progress": 0.3, "global_progress": 0.65, "velocity": -0.1, "event": "regress", "confidence": 0.7},
                ],
            }
        )
    )
    series = load_scores(score)
    assert series.event_labels == ["productive", "regress"]
    assert series.annotations == ["reach", "place"]
    timeline = render_timeline(series, tmp_path / "timeline.png", width=640, height=420)
    assert timeline.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    summary = write_summary(series, tmp_path / "summary.json")
    assert json.loads(summary.read_text())["completion_confidence"] == 0.73

    labels = tmp_path / "manual.jsonl"
    labels.write_text(json.dumps({"episode_id": "held_out_01", "start_sec": 0.25, "end_sec": 0.75, "label": "hesitate"}) + "\n")
    manual = load_manual_events(labels, "held_out_01", "hesitate")
    annotated = render_timeline(
        series,
        tmp_path / "annotated.png",
        width=800,
        height=600,
        manual_events=manual,
        show_event_intervals=True,
    )
    assert annotated.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_synthetic_provenance_and_manual_raw_count_comparison(tmp_path):
    score = tmp_path / "synthetic.json"
    score.write_text(
        json.dumps(
            {
                "episode_id": "synthetic_only",
                "task": "synthetic",
                "synthetic": True,
                "timestamps": [0, 1],
                "local_progress": [0.2, 0.1],
                "labels": ["hesitate", "hesitate"],
            }
        )
    )
    series = load_scores(score)
    assert series.summary()["synthetic"] is True
    assert "not a model result" in series.summary()["provenance_note"].lower()

    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"episode_id": "synthetic_only", "start_sec": 0.2, "end_sec": 0.8, "label": "hesitate", "note": "test"}) + "\n"
    )
    manual = load_manual(labels)
    # Predictions need an episode id, as load_predictions normally supplies.
    predicted = [{"episode_id": series.episode_id, **event} for event in series.events]
    result = compare(manual, predicted, tolerance=0.25)
    assert result["matched_count"] == 1
    assert result["missed_count"] == 0


def test_review_manifest_does_not_copy_secrets_or_signed_urls(tmp_path):
    metadata = tmp_path / "episodes.json"
    metadata.write_text(
        json.dumps(
            [
                {
                    "episode_id": "episode_1",
                    "task": "organizing",
                    "video_path": "https://example.invalid/e.mp4?X-Amz-Signature=do-not-copy",
                    "aws_secret_access_key": "do-not-copy",
                }
            ]
        )
    )
    episodes = load_metadata(metadata)
    assert episodes[0]["video_path"] == ""
    review = tmp_path / "review.html"
    write_html(episodes, review)
    contents = review.read_text()
    assert "do-not-copy" not in contents
    assert "signed media reference omitted" in contents.lower()
