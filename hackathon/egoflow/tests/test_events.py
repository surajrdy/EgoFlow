from __future__ import annotations

import json

from hackathon.egoflow.evaluate import manual_event_metrics, synthetic_interruption_metrics
from hackathon.egoflow.events.detect_events import (
    EventConfig,
    detect_events,
    detect_visual_dynamics,
    derive_global_progress,
)


def test_global_progress_composes_repeated_stage_instances() -> None:
    assert derive_global_progress([0.1, 0.8, 0.2, 0.9], [4, 4, 7, 7]) == [
        0.05,
        0.4,
        0.6,
        0.95,
    ]


def test_clean_progress_is_productive() -> None:
    timestamps = [index * 0.25 for index in range(40)]
    result = detect_events([index / 39 for index in range(40)], timestamps, [0] * 40)
    assert set(result["frame_labels"]) == {"productive"}
    assert all(event["label"] == "productive" for event in result["events"])


def test_low_magnitude_progress_uses_stage_normalized_threshold() -> None:
    timestamps = [index * 0.25 for index in range(40)]
    result = detect_events([0.1 * index / 39 for index in range(40)], timestamps, [0] * 40)
    assert result["velocity_threshold"] < 0.025
    assert "productive" in result["frame_labels"]
    assert set(result["frame_labels"]) != {"stall"}
    assert result["stage_velocity_thresholds"]["0"]["productive"] < 0.025
    assert all(event["detector"] == "learned_progress_normalized" for event in result["events"])


def test_single_task_fallback_does_not_invent_higher_order_learned_events() -> None:
    timestamps = [index * 0.25 for index in range(40)]
    progress = [min(index, 20) / 20 for index in range(40)]
    result = detect_events(
        progress,
        timestamps,
        [0] * 40,
        config=EventConfig(enable_higher_order_motifs=False),
    )
    assert not ({"hesitate", "abandon", "recover"} & set(result["frame_labels"]))


def test_frame_aligned_annotation_is_used() -> None:
    timestamps = [index * 0.25 for index in range(16)]
    annotations = [f"frame {index}" for index in range(16)]
    result = detect_events([index / 15 for index in range(16)], timestamps, [5] * 16, annotation_texts=annotations)
    assert result["events"][0]["annotation"] == "frame 0"


def test_synthetic_corruptions_trigger_expected_motifs() -> None:
    result = synthetic_interruption_metrics()
    assert result["passed"], json.dumps(result, indent=2)


def test_manual_matching_reports_raw_counts() -> None:
    scores = [
        {
            "episode_id": "episode-a",
            "events": [
                {"label": "hesitate", "start_sec": 9.0, "end_sec": 10.0},
                {"label": "hesitate", "start_sec": 20.0, "end_sec": 21.0},
            ],
        }
    ]
    labels = [
        {
            "episode_id": "episode-a",
            "label": "hesitate",
            "start_sec": 10.5,
            "end_sec": 11.0,
        }
    ]
    result = manual_event_metrics(scores, labels, tolerance_sec=1.5)
    assert result["manual_labels"] == 1
    assert result["matched"] == 1
    assert result["false_positives"] == 1
    assert result["recall_on_manual_events"] == 1.0
    assert result["precision_on_reviewed_episodes"] == 0.5


def test_visual_dynamics_emits_finite_explainable_events() -> None:
    import numpy as np

    # Move away from the initial state, pause, then return toward it.
    positions = np.r_[np.linspace(0, 1, 12), np.ones(8), np.linspace(1, 0, 12)]
    visual = np.stack((np.cos(positions), np.sin(positions), np.ones_like(positions)), axis=1)
    result = detect_visual_dynamics(visual, [index * 0.25 for index in range(len(visual))])
    assert len(result["frame_labels"]) == len(visual)
    assert all(np.isfinite(result["motion"]))
    assert all(event["detector"] == "frozen_visual_dynamics" for event in result["events"])
