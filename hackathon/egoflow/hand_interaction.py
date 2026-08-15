"""Video-derived hand kinematics for EgoFlow v2 aborted-reach candidates.

The detector is deliberately geometric. It does not infer intent from a stationary
hand and does not use stored robot/3D trajectories. MediaPipe is an optional runtime
dependency used only while extracting landmarks from public video.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Sequence


def _norm(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def detect_aborted_reaches(
    observations: Sequence[dict[str, Any]],
    *,
    sample_fps: float,
    min_displacement: float = 0.025,
    max_direction_cosine: float = 0.25,
    max_candidates: int = 5,
    max_gap_factor: float = 2.0,
    reject_clear_grasp: bool = False,
    reject_coupled_motion: bool = True,
) -> list[dict[str, Any]]:
    """Detect approach-redirection motifs without treating waiting as an event."""

    window = max(2, round(0.5 * sample_fps))
    by_hand: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if row.get("visible", True):
            by_hand[str(row.get("hand", "unknown"))].append(dict(row))
    candidates: list[dict[str, Any]] = []
    for hand, rows in by_hand.items():
        rows.sort(key=lambda row: float(row["timestamp_sec"]))
        for center in range(window, len(rows) - window):
            before, current, after = rows[center - window], rows[center], rows[center + window]
            # Do not bridge detector gaps: a geometric turn requires continuous evidence.
            expected_span = 2.0 * window / sample_fps
            actual_span = float(after["timestamp_sec"]) - float(before["timestamp_sec"])
            if actual_span > expected_span * max_gap_factor:
                continue
            p0 = (float(before["x"]), float(before["y"]))
            p1 = (float(current["x"]), float(current["y"]))
            p2 = (float(after["x"]), float(after["y"]))
            incoming, outgoing = (p1[0] - p0[0], p1[1] - p0[1]), (p2[0] - p1[0], p2[1] - p1[1])
            incoming_norm, outgoing_norm = _norm(incoming), _norm(outgoing)
            if min(incoming_norm, outgoing_norm) < min_displacement:
                continue  # stationary waiting or tracking jitter
            direction_cosine = (
                incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
            ) / max(incoming_norm * outgoing_norm, 1e-9)
            if direction_cosine > max_direction_cosine:
                continue
            apertures = [
                float(row.get("aperture", 1.0))
                for row in rows[center - window : center + window + 1]
                if row.get("aperture") is not None
            ]
            aperture_drop = max(apertures, default=1.0) - min(apertures, default=1.0)
            if reject_clear_grasp and apertures and min(apertures) < 0.28 and aperture_drop > 0.25:
                continue  # clear close/open cycle: likely completed interaction
            # Reject camera turns and bimanual transport: if the other visible
            # hand follows the same two motion vectors, this is not an isolated
            # approach-abort-switch motif.
            coupled_motion = False
            for other_hand, other_rows in by_hand.items():
                if other_hand == hand:
                    continue
                nearest = [
                    min(other_rows, key=lambda row: abs(float(row["timestamp_sec"]) - time))
                    for time in (
                        float(before["timestamp_sec"]),
                        float(current["timestamp_sec"]),
                        float(after["timestamp_sec"]),
                    )
                ]
                if any(
                    abs(float(row["timestamp_sec"]) - time) > 1.5 / sample_fps
                    for row, time in zip(
                        nearest,
                        (
                            float(before["timestamp_sec"]),
                            float(current["timestamp_sec"]),
                            float(after["timestamp_sec"]),
                        ),
                    )
                ):
                    continue
                q0, q1, q2 = [
                    (float(row["x"]), float(row["y"])) for row in nearest
                ]
                other_in = (q1[0] - q0[0], q1[1] - q0[1])
                other_out = (q2[0] - q1[0], q2[1] - q1[1])
                if min(_norm(other_in), _norm(other_out)) < min_displacement:
                    continue
                in_cos = (incoming[0] * other_in[0] + incoming[1] * other_in[1]) / max(
                    incoming_norm * _norm(other_in), 1e-9
                )
                out_cos = (outgoing[0] * other_out[0] + outgoing[1] * other_out[1]) / max(
                    outgoing_norm * _norm(other_out), 1e-9
                )
                if in_cos > 0.65 and out_cos > 0.65:
                    coupled_motion = True
                    break
            if reject_coupled_motion and coupled_motion:
                continue
            # A pronounced close/open cycle is more consistent with a completed
            # grasp than an aborted reach. Keep it as evidence, not a hard oracle.
            no_grasp_score = max(0.0, min(1.0, 1.0 - aperture_drop / 0.8))
            turn_score = max(0.0, min(1.0, (max_direction_cosine - direction_cosine) / 1.25))
            motion_score = max(0.0, min(1.0, min(incoming_norm, outgoing_norm) / 0.10))
            confidence = 0.50 * turn_score + 0.30 * motion_score + 0.20 * no_grasp_score
            candidates.append(
                {
                    "start_sec": round(float(before["timestamp_sec"]), 3),
                    "end_sec": round(float(after["timestamp_sec"]), 3),
                    "label": "aborted_reach",
                    "presentation_label": "ABORTED REACH?",
                    "confidence": round(confidence, 4),
                    "hand": hand,
                    "detector": "video_hand_geometry_v1",
                    "direction_change_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, direction_cosine)))), 1),
                    "incoming_displacement": round(incoming_norm, 4),
                    "outgoing_displacement": round(outgoing_norm, 4),
                    "aperture_range": round(aperture_drop, 4),
                    "reason": "moving hand redirected; aperture is soft evidence and stationary waiting is excluded",
                }
            )
    # Non-maximum suppression keeps the review queue sparse.
    candidates.sort(key=lambda event: float(event["confidence"]), reverse=True)
    selected: list[dict[str, Any]] = []
    for event in candidates:
        if any(
            str(event["hand"]) == str(other["hand"])
            and float(event["start_sec"]) <= float(other["end_sec"]) + 0.5
            and float(event["end_sec"]) >= float(other["start_sec"]) - 0.5
            for other in selected
        ):
            continue
        selected.append(event)
        if len(selected) >= max_candidates:
            break
    return sorted(selected, key=lambda event: float(event["start_sec"]))


def extract_hand_observations(
    video_path: str | Path,
    *,
    sample_fps: float = 8.0,
    min_detection_confidence: float = 0.35,
) -> dict[str, Any]:
    """Extract normalized 2D palm centers and grasp-aperture proxies from video."""

    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional Modal/runtime path
        raise RuntimeError("hand extraction requires mediapipe and opencv-python-headless") from exc
    capture = cv2.VideoCapture(str(video_path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, round(source_fps / max(1.0, sample_fps)))
    observations: list[dict[str, Any]] = []
    sampled_frames = detected_frames = frame_index = 0
    previous_tracks: dict[str, tuple[float, float]] = {}
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.35,
    ) as detector:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride:
                frame_index += 1
                continue
            sampled_frames += 1
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = list(result.multi_hand_landmarks or [])
            handedness = list(result.multi_handedness or [])
            detected_frames += bool(landmarks)
            detections: list[dict[str, float]] = []
            for hand_index, hand_landmarks in enumerate(landmarks):
                points = hand_landmarks.landmark
                palm_indices = (0, 5, 9, 13, 17)
                x = sum(points[index].x for index in palm_indices) / len(palm_indices)
                y = sum(points[index].y for index in palm_indices) / len(palm_indices)
                scale = math.hypot(points[0].x - points[9].x, points[0].y - points[9].y)
                aperture = math.hypot(points[4].x - points[8].x, points[4].y - points[8].y) / max(scale, 1e-5)
                score = (
                    float(handedness[hand_index].classification[0].score)
                    if hand_index < len(handedness)
                    else 0.0
                )
                detections.append({"x": float(x), "y": float(y), "aperture": float(aperture), "score": score})
            assignments: list[tuple[str, dict[str, float]]] = []
            if len(detections) == 2 and set(previous_tracks) == {"image_left", "image_right"}:
                first, second = detections
                direct = _norm((first["x"] - previous_tracks["image_left"][0], first["y"] - previous_tracks["image_left"][1])) + _norm((second["x"] - previous_tracks["image_right"][0], second["y"] - previous_tracks["image_right"][1]))
                crossed = _norm((second["x"] - previous_tracks["image_left"][0], second["y"] - previous_tracks["image_left"][1])) + _norm((first["x"] - previous_tracks["image_right"][0], first["y"] - previous_tracks["image_right"][1]))
                assignments = [("image_left", first), ("image_right", second)] if direct <= crossed else [("image_left", second), ("image_right", first)]
            elif len(detections) == 2:
                ordered = sorted(detections, key=lambda item: item["x"])
                assignments = [("image_left", ordered[0]), ("image_right", ordered[1])]
            elif len(detections) == 1:
                detection = detections[0]
                if previous_tracks:
                    label = min(
                        previous_tracks,
                        key=lambda name: _norm((detection["x"] - previous_tracks[name][0], detection["y"] - previous_tracks[name][1])),
                    )
                else:
                    label = "image_left" if detection["x"] < 0.5 else "image_right"
                assignments = [(label, detection)]
            for label, detection in assignments:
                previous_tracks[label] = (detection["x"], detection["y"])
                observations.append(
                    {
                        "timestamp_sec": round(timestamp, 4),
                        "hand": label,
                        "x": round(detection["x"], 6),
                        "y": round(detection["y"], 6),
                        "aperture": round(detection["aperture"], 6),
                        "detection_confidence": round(detection["score"], 4),
                        "visible": True,
                    }
                )
            frame_index += 1
    capture.release()
    return {
        "sample_fps": sample_fps,
        "source_fps": source_fps,
        "sampled_frames": sampled_frames,
        "detected_frames": detected_frames,
        "detection_rate": round(detected_frames / sampled_frames, 4) if sampled_frames else 0.0,
        "observations": observations,
    }
