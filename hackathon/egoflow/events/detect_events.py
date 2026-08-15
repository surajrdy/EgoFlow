"""Conservative, explainable event detection over a learned progress curve."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from statistics import median
from typing import Iterable, Sequence


EVENT_LABELS = (
    "productive",
    "stall",
    "regress",
    "recover",
    "hesitate",
    "abandon",
)


@dataclass(frozen=True)
class EventConfig:
    smoothing_sec: float = 0.75
    slope_window_sec: float = 1.0
    min_event_sec: float = 0.75
    hesitation_lookback_sec: float = 2.0
    recovery_lookahead_sec: float = 4.0
    velocity_mad_k: float = 1.5
    velocity_floor: float = 0.0
    productive_quantile: float = 0.40
    completion_threshold: float = 0.82
    recovery_tolerance: float = 0.04
    enable_higher_order_motifs: bool = True


def _validate(progress: Sequence[float], timestamps: Sequence[float], stage_ids: Sequence[int]) -> None:
    if not progress or len(progress) != len(timestamps) or len(progress) != len(stage_ids):
        raise ValueError("progress, timestamps, and stage_ids must have equal non-zero length")
    if any(not math.isfinite(float(value)) for value in progress):
        raise ValueError("progress contains NaN/Inf")
    if any(float(timestamps[index]) <= float(timestamps[index - 1]) for index in range(1, len(timestamps))):
        raise ValueError("timestamps must be strictly increasing")


def _sample_period(timestamps: Sequence[float]) -> float:
    if len(timestamps) == 1:
        return 0.25
    return max(1e-6, median(float(timestamps[i]) - float(timestamps[i - 1]) for i in range(1, len(timestamps))))


def _quantile(values: Sequence[float], fraction: float) -> float:
    """Dependency-free linear quantile for per-stage robust calibration."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _moving_average(values: Sequence[float], radius: int, stage_ids: Sequence[int]) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start, end = max(0, index - radius), min(len(values), index + radius + 1)
        same_stage = [float(values[j]) for j in range(start, end) if stage_ids[j] == stage_ids[index]]
        result.append(sum(same_stage) / len(same_stage))
    return result


def _local_slopes(
    values: Sequence[float], timestamps: Sequence[float], stage_ids: Sequence[int], radius: int
) -> list[float]:
    """Centered least-squares slope, restricted to the active semantic stage."""
    slopes: list[float] = []
    for index in range(len(values)):
        points = [
            j
            for j in range(max(0, index - radius), min(len(values), index + radius + 1))
            if stage_ids[j] == stage_ids[index]
        ]
        if len(points) < 2:
            slopes.append(0.0)
            continue
        mean_t = sum(float(timestamps[j]) for j in points) / len(points)
        mean_p = sum(float(values[j]) for j in points) / len(points)
        numerator = sum((float(timestamps[j]) - mean_t) * (float(values[j]) - mean_p) for j in points)
        denominator = sum((float(timestamps[j]) - mean_t) ** 2 for j in points)
        slopes.append(numerator / denominator if denominator > 1e-12 else 0.0)
    return slopes


def _runs(labels: Sequence[str], stage_ids: Sequence[int]) -> list[tuple[int, int, str]]:
    runs: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start] or stage_ids[index] != stage_ids[start]:
            runs.append((start, index - 1, labels[start]))
            start = index
    return runs


def _event_confidence(
    label: str, velocities: Sequence[float], threshold: float, progress: Sequence[float]
) -> float:
    magnitude = abs(sum(velocities) / max(1, len(velocities))) / max(threshold, 1e-6)
    motion = min(1.0, magnitude / 2.0)
    change = min(1.0, abs(float(progress[-1]) - float(progress[0])) / 0.25)
    if label in {"stall", "hesitate", "abandon"}:
        motion = 1.0 - min(1.0, magnitude)
    return round(max(0.05, min(0.99, 0.45 + 0.35 * motion + 0.2 * change)), 4)


def detect_events(
    local_progress: Sequence[float],
    timestamps: Sequence[float],
    stage_ids: Sequence[int],
    *,
    annotation_texts: Sequence[str] | None = None,
    config: EventConfig | None = None,
) -> dict[str, object]:
    """Derive velocities, conservative motifs, and contiguous event ranges."""
    _validate(local_progress, timestamps, stage_ids)
    config = config or EventConfig()
    dt = _sample_period(timestamps)
    smoothing_radius = max(1, round(config.smoothing_sec / dt / 2))
    slope_radius = max(1, round(config.slope_window_sec / dt / 2))
    minimum_frames = max(2, math.ceil(config.min_event_sec / dt))
    hesitation_frames = max(minimum_frames, round(config.hesitation_lookback_sec / dt))
    recovery_frames = max(minimum_frames, round(config.recovery_lookahead_sec / dt))

    smoothed = _moving_average(local_progress, smoothing_radius, stage_ids)
    # Estimate slope from the learned curve itself. The separately smoothed
    # curve is retained for motif context; differentiating a truncated moving
    # average creates artificial stalls at episode boundaries.
    velocity = _local_slopes(local_progress, timestamps, stage_ids, slope_radius)
    # Calibrate within each semantic stage. Absolute progress slopes depend on
    # episode duration and model calibration, so a fixed threshold can label an
    # entire valid episode as stalled. The positive threshold uses a stage
    # quantile; negative motion must exceed a stage MAD-derived magnitude.
    stage_thresholds: dict[int, dict[str, float]] = {}
    productive_thresholds = [0.0] * len(velocity)
    regress_thresholds = [0.0] * len(velocity)
    for stage in dict.fromkeys(int(value) for value in stage_ids):
        indices = [index for index, value in enumerate(stage_ids) if int(value) == stage]
        stage_velocity = [velocity[index] for index in indices]
        center = median(stage_velocity)
        mad = median(abs(value - center) for value in stage_velocity)
        robust_sigma = max(1e-8, 1.4826 * mad)
        productive_threshold = max(
            float(config.velocity_floor),
            _quantile(stage_velocity, config.productive_quantile),
        )
        if productive_threshold <= 0.0:
            productive_threshold = max(float(config.velocity_floor), 0.5 * robust_sigma)
        regress_threshold = -max(
            float(config.velocity_floor),
            config.velocity_mad_k * robust_sigma,
            1e-8,
        )
        stage_thresholds[stage] = {
            "median": round(float(center), 6),
            "mad": round(float(mad), 6),
            "productive": round(float(productive_threshold), 6),
            "regress": round(float(regress_threshold), 6),
        }
        for index in indices:
            productive_thresholds[index] = productive_threshold
            regress_thresholds[index] = regress_threshold
    labels = [
        "productive"
        if value >= productive_thresholds[index] - 1e-9
        else "regress"
        if value <= regress_thresholds[index]
        else "stall"
        for index, value in enumerate(velocity)
    ]
    threshold = median(productive_thresholds)

    # Suppress short noisy runs before identifying higher-order motifs.
    for start, end, label in _runs(labels, stage_ids):
        if end - start + 1 < minimum_frames:
            for index in range(start, end + 1):
                labels[index] = "stall"

    # A recovery must follow regression/stall and regain the pre-regression peak.
    recovery_runs = list(_runs(labels, stage_ids)) if config.enable_higher_order_motifs else []
    for start, end, label in recovery_runs:
        if label != "regress":
            continue
        stage = stage_ids[start]
        prior_start = max(0, start - recovery_frames)
        prior_peak = max(smoothed[index] for index in range(prior_start, start + 1) if stage_ids[index] == stage)
        search_end = min(len(labels), end + 1 + recovery_frames)
        candidates = [
            index
            for index in range(end + 1, search_end)
            if stage_ids[index] == stage
            and labels[index] == "productive"
            and smoothed[index] >= prior_peak - config.recovery_tolerance
        ]
        if candidates:
            recovery_start = next(
                (index for index in range(end + 1, candidates[0] + 1) if labels[index] == "productive"),
                candidates[0],
            )
            for index in range(recovery_start, candidates[-1] + 1):
                if stage_ids[index] == stage and labels[index] == "productive":
                    labels[index] = "recover"

    # Hesitation requires earlier productive movement and non-completion. If no
    # recovery occurs before stage end, use the stronger abandonment label.
    hesitation_runs = list(_runs(labels, stage_ids)) if config.enable_higher_order_motifs else []
    for start, end, label in hesitation_runs:
        if label not in {"stall", "regress"} or end - start + 1 < minimum_frames:
            continue
        stage = stage_ids[start]
        prior = range(max(0, start - hesitation_frames), start)
        had_progress = any(stage_ids[index] == stage and labels[index] == "productive" for index in prior)
        if not had_progress or smoothed[end] >= config.completion_threshold:
            continue
        stage_end = end
        while stage_end + 1 < len(stage_ids) and stage_ids[stage_end + 1] == stage:
            stage_end += 1
        recovered = any(labels[index] == "recover" for index in range(end + 1, stage_end + 1))
        new_label = "hesitate" if recovered else "abandon"
        for index in range(start, end + 1):
            labels[index] = new_label

    events: list[dict[str, object]] = []
    for start, end, label in _runs(labels, stage_ids):
        duration = float(timestamps[end]) - float(timestamps[start]) + dt
        if duration + 1e-9 < config.min_event_sec:
            continue
        stage = int(stage_ids[start])
        event: dict[str, object] = {
            "start_sec": round(float(timestamps[start]), 4),
            "end_sec": round(float(timestamps[end]), 4),
            "label": label,
            "confidence": _event_confidence(
                label, velocity[start : end + 1], threshold, smoothed[start : end + 1]
            ),
            "stage_id": stage,
            "detector": "learned_progress_normalized",
            "reason": "stage-normalized progress velocity",
        }
        if annotation_texts:
            # Cache producers may store either one string per sampled frame or
            # one string per stage. Prefer the exact frame-aligned value.
            if len(annotation_texts) == len(local_progress):
                event["annotation"] = str(annotation_texts[start])
            elif 0 <= stage < len(annotation_texts):
                event["annotation"] = str(annotation_texts[stage])
        events.append(event)

    return {
        "smoothed_local_progress": [round(value, 6) for value in smoothed],
        "velocity": [round(value, 6) for value in velocity],
        "frame_labels": labels,
        "velocity_threshold": round(threshold, 6),
        "frame_velocity_thresholds": [round(value, 6) for value in productive_thresholds],
        "stage_velocity_thresholds": {str(key): value for key, value in stage_thresholds.items()},
        "detector": "learned_progress_normalized",
        "events": events,
    }


def ordered_stages(stage_ids: Iterable[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for stage in stage_ids:
        stage = int(stage)
        if stage >= 0 and stage not in seen:
            seen.add(stage)
            result.append(stage)
    return result


def derive_global_progress(
    local_progress: Sequence[float],
    stage_ids: Sequence[int],
    *,
    stage_order: Sequence[int] | None = None,
) -> list[float]:
    """Compose local progress over ordered semantic stage instances."""
    if len(local_progress) != len(stage_ids):
        raise ValueError("local_progress and stage_ids must have equal length")
    order = list(stage_order) if stage_order is not None else ordered_stages(stage_ids)
    if not order:
        return [max(0.0, min(1.0, float(value))) for value in local_progress]
    positions = {int(stage): index for index, stage in enumerate(order)}
    result: list[float] = []
    last_position = 0
    for local, stage in zip(local_progress, stage_ids):
        position = positions.get(int(stage), last_position)
        last_position = position
        result.append(max(0.0, min(1.0, (position + max(0.0, min(1.0, float(local)))) / len(order))))
    return result


def summarize_episode(
    timestamps: Sequence[float],
    local_progress: Sequence[float],
    global_progress: Sequence[float],
    frame_labels: Sequence[str],
    events: Sequence[dict[str, object]],
) -> dict[str, object]:
    if not timestamps or not (
        len(timestamps) == len(local_progress) == len(global_progress) == len(frame_labels)
    ):
        raise ValueError("summary arrays must have equal non-zero length")
    counts = Counter(frame_labels)
    count = len(frame_labels)
    final_abandon = any(
        event.get("label") == "abandon"
        and float(event.get("end_sec", 0.0)) >= float(timestamps[-1]) - 5.0
        for event in events
    )
    base = 0.7 * float(global_progress[-1]) + 0.3 * float(local_progress[-1])
    completion_confidence = max(0.0, min(1.0, base - (0.18 if final_abandon else 0.0)))
    return {
        "duration_sec": round(float(timestamps[-1]) - float(timestamps[0]), 4),
        "completion_confidence": round(completion_confidence, 4),
        "drop_score": round(1.0 - completion_confidence, 4),
        "productive_fraction": round(counts["productive"] / count, 4),
        "stall_fraction": round(counts["stall"] / count, 4),
        "regress_fraction": round(counts["regress"] / count, 4),
        "hesitation_fraction": round(counts["hesitate"] / count, 4),
        "recovery_count": sum(event.get("label") == "recover" for event in events),
        "abandonment_count": sum(event.get("label") == "abandon" for event in events),
        "interesting_timestamp_ranges": [
            event for event in events if event.get("label") in {"hesitate", "abandon", "regress", "recover"}
        ],
    }


def detect_visual_dynamics(
    visual_embeddings: object,
    timestamps: Sequence[float],
    *,
    max_hesitations: int = 4,
    max_stalls: int = 3,
) -> dict[str, object]:
    """Propose events directly from frozen visual-feature dynamics.

    This fallback is used when public MP4s lack dense annotation spans. It is
    intentionally transparent: slowdowns, loop-backs, and movement away from the
    final visual state create a small ranked proposal set per episode.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - scorer already requires numpy
        raise RuntimeError("visual dynamics require numpy") from exc
    values = np.asarray(visual_embeddings, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(timestamps) or len(values) < 8:
        return {"events": [], "frame_labels": ["productive"] * len(timestamps)}
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)
    dt = _sample_period(timestamps)
    one_second = max(2, round(1.0 / dt))
    prior_span = max(one_second + 1, round(3.0 / dt))
    lag = max(one_second + 1, round(3.0 / dt))

    motion = np.r_[0.0, 1.0 - np.sum(values[1:] * values[:-1], axis=1)]
    kernel = np.ones(one_second, dtype=np.float64) / one_second
    smooth = np.convolve(motion, kernel, mode="same")
    prior = np.asarray(
        [
            np.median(smooth[max(0, index - prior_span) : max(1, index - one_second)])
            for index in range(len(values))
        ]
    )
    target = values[-min(5, len(values)) :].mean(axis=0)
    target /= max(float(np.linalg.norm(target)), 1e-9)
    goal = values @ target
    goal_delta = np.r_[np.zeros(one_second), goal[one_second:] - goal[:-one_second]]
    loop = np.zeros(len(values), dtype=np.float64)
    loop[lag:] = np.sum(values[lag:] * values[:-lag], axis=1)

    def percentile_rank(series: object) -> object:
        array = np.asarray(series)
        return np.argsort(np.argsort(array)) / max(1, len(array) - 1)

    smooth_rank = percentile_rank(smooth)
    prior_rank = percentile_rank(prior)
    loop_rank = percentile_rank(loop)
    low_after_motion = (smooth_rank < 0.25) & (prior_rank > 0.75)
    regression = goal_delta < min(-0.06, float(np.quantile(goal_delta, 0.05)))
    loop_back = (loop_rank > 0.97) & (smooth_rank > 0.60)
    hesitation_severity = np.maximum(
        np.clip(prior_rank - smooth_rank, 0.0, 1.0),
        np.where(loop_back, loop_rank, 0.0),
    )
    regression_severity = np.clip(
        -goal_delta / max(float(np.quantile(np.abs(goal_delta), 0.90)), 1e-6),
        0.0,
        1.0,
    )

    def groups(mask: object, gap_frames: int, expand_frames: int) -> list[tuple[int, int]]:
        indices = np.flatnonzero(mask)
        if not len(indices):
            return []
        output: list[tuple[int, int]] = []
        start = previous = int(indices[0])
        for raw_index in indices[1:]:
            index = int(raw_index)
            if index - previous > gap_frames:
                if previous - start >= 1:
                    output.append(
                        (
                            max(0, start - expand_frames),
                            min(len(values) - 1, previous + expand_frames),
                        )
                    )
                start = index
            previous = index
        if previous - start >= 1:
            output.append(
                (
                    max(0, start - expand_frames),
                    min(len(values) - 1, previous + expand_frames),
                )
            )
        return [
            item
            for item in output
            if float(timestamps[item[1]]) - float(timestamps[item[0]]) >= 0.75
        ]

    hesitation_groups = groups(
        low_after_motion | loop_back,
        gap_frames=max(2, round(1.25 / dt)),
        expand_frames=max(2, round(1.0 / dt)),
    )
    hesitation_groups = sorted(
        hesitation_groups,
        key=lambda item: float(hesitation_severity[item[0] : item[1] + 1].max()),
        reverse=True,
    )[:max_hesitations]
    regression_groups = groups(
        regression,
        gap_frames=max(2, round(1.25 / dt)),
        expand_frames=max(2, round(1.0 / dt)),
    )
    regression_groups = sorted(
        regression_groups,
        key=lambda item: float(regression_severity[item[0] : item[1] + 1].max()),
        reverse=True,
    )[:max_hesitations]
    events: list[dict[str, object]] = []
    labels = ["productive"] * len(values)
    # One frame can support both an abandonment decision and its preceding
    # hesitation. Keep hesitation as the primary timeline color while retaining
    # the overlapping abandonment event in the structured event list.
    priority = {"productive": 0, "stall": 1, "regress": 2, "recover": 3, "abandon": 4, "hesitate": 5}

    def add_event(start: int, end: int, label: str, confidence: float, reason: str) -> None:
        events.append(
            {
                "start_sec": round(float(timestamps[start]), 4),
                "end_sec": round(float(timestamps[end]), 4),
                "label": label,
                "confidence": round(max(0.05, min(0.99, confidence)), 4),
                "stage_id": 0,
                "detector": "frozen_visual_dynamics",
                "reason": reason,
            }
        )
        for index in range(start, end + 1):
            if priority[label] >= priority[labels[index]]:
                labels[index] = label

    for start, end in hesitation_groups:
        score = float(hesitation_severity[start : end + 1].max())
        add_event(start, end, "hesitate", 0.5 + 0.45 * score, "slowdown/loop-back")

    # A decrease in final-state similarity is evidence of regression, not by
    # itself evidence of hesitation. Keeping the proposal families separate
    # prevents active but ineffective motion from stealing hesitation slots.
    for start, end in regression_groups:
        score = float(regression_severity[start : end + 1].max())
        add_event(start, end, "regress", 0.5 + 0.45 * score, "final-state similarity decreased")
        pre_start = max(0, start - round(2.0 / dt))
        post_end = min(len(values), end + 1 + round(5.0 / dt))
        pre_peak = float(goal[pre_start : start + 1].max())
        post_values = goal[end:post_end]
        post_peak = float(post_values.max()) if len(post_values) else float(goal[end])
        if post_peak < pre_peak - 0.08:
            add_event(start, end, "abandon", 0.55 + 0.4 * score, "no visual recovery within 5 sec")
        elif len(post_values):
            recovery_end = min(len(values) - 1, end + int(np.argmax(post_values)))
            if recovery_end > end:
                add_event(end, recovery_end, "recover", 0.5 + 0.4 * score, "visual state recovered")

    stall_groups = groups(
        smooth_rank < 0.12,
        gap_frames=max(1, round(0.5 / dt)),
        expand_frames=1,
    )
    stall_groups = sorted(
        stall_groups,
        key=lambda item: float(smooth[item[0] : item[1] + 1].mean()),
    )[:max_stalls]
    for start, end in stall_groups:
        add_event(start, end, "stall", 0.7, "bottom-decile visual motion")
    events.sort(key=lambda event: (float(event["start_sec"]), -priority[str(event["label"])]))
    return {
        "events": events,
        "frame_labels": labels,
        "motion": [round(float(value), 6) for value in smooth],
    }
