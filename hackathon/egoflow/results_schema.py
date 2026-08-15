"""Small, dependency-light schema helpers for EgoFlow score artifacts.

The scorer may write either a JSON mapping of parallel arrays, a JSON ``frames``
list, or an NPZ with equivalent array names.  Keeping this loader permissive lets
the visualization lane stay useful while the training/scoring schema evolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


EVENT_LABELS = (
    "productive",
    "stall",
    "regress",
    "recover",
    "hesitate",
    "abandon",
    "complete",
    "transition",
    "other",
)

_ALIASES = {
    "progress": "local_progress",
    "local": "local_progress",
    "global": "global_progress",
    "time": "timestamps_sec",
    "times": "timestamps_sec",
    "timestamps": "timestamps_sec",
    "timestamp": "timestamps_sec",
    "timestamp_sec": "timestamps_sec",
    "velocity": "progress_velocity",
    "state": "event_labels",
    "states": "event_labels",
    "labels": "event_labels",
    "event_label": "event_labels",
    "event": "event_labels",
    "event_source": "event_sources",
    "annotation": "annotations",
    "confidence": "confidences",
}


def _clean_label(value: Any) -> str:
    label = str(value or "other").strip().lower()
    return label if label in EVENT_LABELS else "other"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _first(mapping: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


@dataclass
class ScoreSeries:
    episode_id: str
    task: str
    timestamps_sec: list[float]
    local_progress: list[float]
    global_progress: list[float]
    progress_velocity: list[float]
    event_labels: list[str]
    confidences: list[float]
    annotations: list[str]
    event_sources: list[str]
    events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    synthetic: bool = False

    @property
    def duration_sec(self) -> float:
        return max(self.timestamps_sec, default=0.0)

    def value_at(self, timestamp_sec: float) -> dict[str, Any]:
        if not self.timestamps_sec:
            raise ValueError("score series contains no samples")
        index = min(
            range(len(self.timestamps_sec)),
            key=lambda i: abs(self.timestamps_sec[i] - timestamp_sec),
        )
        return {
            "index": index,
            "timestamp_sec": self.timestamps_sec[index],
            "local_progress": self.local_progress[index],
            "global_progress": self.global_progress[index],
            "progress_velocity": self.progress_velocity[index],
            "event_label": self.event_labels[index],
            "event_source": self.event_sources[index],
            "confidence": self.confidences[index],
            "annotation": self.annotations[index],
        }

    def summary(self) -> dict[str, Any]:
        weights = _sample_weights(self.timestamps_sec)
        total = sum(weights) or 1.0
        fractions = {
            label: sum(w for w, state in zip(weights, self.event_labels) if state == label) / total
            for label in EVENT_LABELS
        }
        terminal = self.global_progress[-1] if self.global_progress else 0.0
        confidence = self.metadata.get("completion_confidence", terminal)
        event_groups: dict[str, list[dict[str, Any]]] = {}
        for event in self.events:
            event_groups.setdefault(_clean_label(event.get("label")), []).append(event)
        return {
            "schema_version": "egoflow.score.v1",
            "episode_id": self.episode_id,
            "task": self.task,
            "duration_sec": round(self.duration_sec, 3),
            "completion_confidence": round(float(confidence), 4),
            "drop_score": round(1.0 - float(confidence), 4),
            "productive_fraction": round(fractions["productive"], 4),
            "stall_fraction": round(fractions["stall"], 4),
            "regress_fraction": round(fractions["regress"], 4),
            "hesitation_fraction": round(fractions["hesitate"], 4),
            "recovery_count": len(event_groups.get("recover", [])),
            "abandonment_count": len(event_groups.get("abandon", [])),
            "hesitations": event_groups.get("hesitate", []),
            "recoveries": event_groups.get("recover", []),
            "abandonments": event_groups.get("abandon", []),
            "interesting_ranges": [
                event for event in self.events if _clean_label(event.get("label")) not in {"productive", "other"}
            ],
            "synthetic": self.synthetic,
            "provenance_note": (
                "Synthetic smoke-test artifact; not a model result or empirical claim."
                if self.synthetic
                else "Generated from supplied EgoFlow scores."
            ),
        }


def _sample_weights(times: list[float]) -> list[float]:
    if len(times) < 2:
        return [1.0] * len(times)
    diffs = [max(0.0, times[i + 1] - times[i]) for i in range(len(times) - 1)]
    fallback = sorted(d for d in diffs if d > 0)
    last = fallback[len(fallback) // 2] if fallback else 1.0
    return diffs + [last]


def _runs_to_events(times: list[float], labels: list[str], annotations: list[str]) -> list[dict[str, Any]]:
    if not times:
        return []
    events: list[dict[str, Any]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            label = labels[start]
            if label not in {"other"}:
                end_time = times[i] if i < len(times) else times[-1]
                events.append(
                    {
                        "start_sec": round(times[start], 3),
                        "end_sec": round(max(times[start], end_time), 3),
                        "label": label,
                        "annotation": annotations[start],
                    }
                )
            start = i
    return events


def _normalize_payload(payload: dict[str, Any], source: Path) -> ScoreSeries:
    if "scores" in payload and isinstance(payload["scores"], dict):
        outer = payload
        payload = {**payload["scores"], **{k: v for k, v in outer.items() if k != "scores"}}

    frames = payload.get("frames") or payload.get("predictions")
    arrays: dict[str, Any] = {}
    if isinstance(frames, list) and frames:
        for raw_name, canonical in _ALIASES.items():
            values = [frame.get(raw_name) for frame in frames if isinstance(frame, dict)]
            if any(value is not None for value in values):
                arrays.setdefault(canonical, values)
        for canonical in (
            "timestamps_sec", "local_progress", "global_progress", "progress_velocity",
            "event_labels", "event_sources", "confidences", "annotations",
        ):
            values = [frame.get(canonical) for frame in frames if isinstance(frame, dict)]
            if any(value is not None for value in values):
                arrays[canonical] = values
    for key, value in payload.items():
        arrays[_ALIASES.get(key, key)] = value

    if "annotations" not in arrays and isinstance(frames, list):
        annotation_texts = payload.get("annotation_texts")
        if isinstance(annotation_texts, list):
            arrays["annotations"] = [
                annotation_texts[int(frame.get("stage_id", -1))]
                if isinstance(frame, dict)
                and str(frame.get("stage_id", "")).lstrip("-").isdigit()
                and 0 <= int(frame.get("stage_id", -1)) < len(annotation_texts)
                else ""
                for frame in frames
            ]

    local = [float(x or 0.0) for x in _as_list(arrays.get("local_progress"))]
    global_progress = [float(x or 0.0) for x in _as_list(arrays.get("global_progress"))]
    times = [float(x or 0.0) for x in _as_list(arrays.get("timestamps_sec"))]
    n = max(len(local), len(global_progress), len(times))
    if not n:
        raise ValueError(f"{source} has no progress/timestamp samples")
    if not times:
        fps = float(_first(payload, ("feature_fps", "fps"), 1.0) or 1.0)
        times = [i / fps for i in range(n)]
    if not local:
        local = list(global_progress)
    if not global_progress:
        global_progress = list(local)
    if len(times) != n or len(local) != n or len(global_progress) != n:
        raise ValueError(
            f"parallel array length mismatch in {source}: time={len(times)}, "
            f"local={len(local)}, global={len(global_progress)}"
        )

    def sized(name: str, default: Any) -> list[Any]:
        values = _as_list(arrays.get(name))
        if not values:
            return [default] * n
        if len(values) == 1 and n > 1:
            return values * n
        if len(values) != n:
            raise ValueError(f"{name} has {len(values)} values, expected {n}")
        return values

    velocity = [float(x or 0.0) for x in sized("progress_velocity", 0.0)]
    labels = [_clean_label(x) for x in sized("event_labels", "other")]
    confidences = [max(0.0, min(1.0, float(x or 0.0))) for x in sized("confidences", 1.0)]
    annotations = [str(x or "") for x in sized("annotations", "")]
    event_sources = [str(x or "unknown") for x in sized("event_sources", "unknown")]

    order = sorted(range(n), key=times.__getitem__)
    times = [times[i] for i in order]
    local = [max(0.0, min(1.0, local[i])) for i in order]
    global_progress = [max(0.0, min(1.0, global_progress[i])) for i in order]
    velocity = [velocity[i] for i in order]
    labels = [labels[i] for i in order]
    confidences = [confidences[i] for i in order]
    annotations = [annotations[i] for i in order]
    event_sources = [event_sources[i] for i in order]

    raw_events = payload.get("events") or payload.get("interesting_ranges") or []
    events = []
    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        event = dict(raw)
        event["label"] = _clean_label(event.get("label") or event.get("state"))
        event["start_sec"] = float(event.get("start_sec", event.get("start", 0.0)))
        event["end_sec"] = float(event.get("end_sec", event.get("end", event["start_sec"])))
        events.append(event)
    if not events:
        events = _runs_to_events(times, labels, annotations)

    known = {
        "episode_id", "task", "frames", "predictions", "events", "interesting_ranges",
        "scores", "synthetic", "timestamps_sec", "timestamps", "times", "time",
        "local_progress", "global_progress", "progress_velocity", "velocity",
        "event_labels", "event_sources", "event_source", "labels", "states", "confidences", "confidence", "annotations",
    }
    metadata = {k: v for k, v in payload.items() if k not in known}
    return ScoreSeries(
        episode_id=str(payload.get("episode_id") or source.stem),
        task=str(payload.get("task") or payload.get("task_description") or "unknown task"),
        timestamps_sec=times,
        local_progress=local,
        global_progress=global_progress,
        progress_velocity=velocity,
        event_labels=labels,
        confidences=confidences,
        annotations=annotations,
        event_sources=event_sources,
        events=events,
        metadata=metadata,
        synthetic=bool(payload.get("synthetic", False)),
    )


def load_scores(path: str | Path) -> ScoreSeries:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    elif suffix == ".npz":
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError("NPZ input requires numpy; JSON input has no third-party dependencies") from exc
        with np.load(source, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
    else:
        raise ValueError(f"unsupported score file {source}; expected .json or .npz")
    if not isinstance(payload, dict):
        raise ValueError(f"score artifact {source} must contain a mapping")
    return _normalize_payload(payload, source)


def write_summary(series: ScoreSeries, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(series.summary(), indent=2) + "\n", encoding="utf-8")
    return output
