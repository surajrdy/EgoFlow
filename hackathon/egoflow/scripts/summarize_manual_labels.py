#!/usr/bin/env python3
"""Validate/summarize manual event JSONL and optionally compare predictions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from results_schema import EVENT_LABELS, load_scores  # noqa: E402


MANUAL_LABELS = set(EVENT_LABELS) - {"transition"}


def load_manual(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        missing = [name for name in ("episode_id", "start_sec", "end_sec", "label") if name not in record]
        if missing:
            raise ValueError(f"{path}:{line_number}: missing {', '.join(missing)}")
        record["label"] = str(record["label"]).lower()
        if record["label"] not in MANUAL_LABELS:
            raise ValueError(f"{path}:{line_number}: unsupported label {record['label']!r}")
        record["start_sec"], record["end_sec"] = float(record["start_sec"]), float(record["end_sec"])
        if record["start_sec"] < 0 or record["end_sec"] < record["start_sec"]:
            raise ValueError(f"{path}:{line_number}: invalid timestamp range")
        records.append(record)
    return records


def load_predictions(paths: list[Path]) -> list[dict[str, Any]]:
    predictions = []
    for path in paths:
        series = load_scores(path)
        for event in series.events:
            predictions.append({"episode_id": series.episode_id, **event})
    return predictions


def matches(one: dict[str, Any], two: dict[str, Any], tolerance: float) -> bool:
    if str(one["episode_id"]) != str(two["episode_id"]) or one["label"] != two["label"]:
        return False
    a0, a1 = float(one["start_sec"]), float(one["end_sec"])
    b0, b1 = float(two["start_sec"]), float(two["end_sec"])
    return a0 <= b1 + tolerance and b0 <= a1 + tolerance


def compare(manual: list[dict[str, Any]], predicted: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    used: set[int] = set()
    true_positive = []
    false_negative = []
    for actual in manual:
        candidate = next((i for i, guess in enumerate(predicted) if i not in used and matches(actual, guess, tolerance)), None)
        if candidate is None:
            false_negative.append(actual)
        else:
            used.add(candidate)
            true_positive.append({"manual": actual, "prediction": predicted[candidate]})
    false_positive = [guess for i, guess in enumerate(predicted) if i not in used]
    by_label = {}
    for label in sorted({x["label"] for x in manual + predicted}):
        tp = sum(item["manual"]["label"] == label for item in true_positive)
        fn = sum(item["label"] == label for item in false_negative)
        fp = sum(item["label"] == label for item in false_positive)
        by_label[label] = {"manual_count": tp + fn, "detected": tp, "missed": fn, "false_positives": fp}
    return {
        "matching_rule": f"same episode + same label + overlap within +/- {tolerance:g} sec",
        "manual_event_count": len(manual),
        "predicted_event_count": len(predicted),
        "matched_count": len(true_positive),
        "missed_count": len(false_negative),
        "false_positive_count": len(false_positive),
        "by_label": by_label,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, nargs="?", default=Path(__file__).resolve().parents[1] / "manual_labels.jsonl")
    parser.add_argument("--predictions", type=Path, nargs="*", default=[], help="score JSON/NPZ artifacts")
    parser.add_argument("--tolerance-sec", type=float, default=1.5)
    parser.add_argument("--output", type=Path, help="optional machine-readable summary JSON")
    args = parser.parse_args()
    manual = load_manual(args.labels)
    label_counts = Counter(record["label"] for record in manual)
    episode_counts = Counter(str(record["episode_id"]) for record in manual)
    task_counts = Counter(str(record.get("task", "unknown")) for record in manual)
    duration = sum(record["end_sec"] - record["start_sec"] for record in manual)
    summary: dict[str, Any] = {
        "schema_version": "egoflow.manual_summary.v1",
        "manual_event_count": len(manual),
        "episode_count": len(episode_counts),
        "labeled_duration_sec": round(duration, 3),
        "counts_by_label": dict(sorted(label_counts.items())),
        "counts_by_episode": dict(sorted(episode_counts.items())),
        "counts_by_task": dict(sorted(task_counts.items())),
    }
    if args.predictions:
        summary["prediction_comparison"] = compare(manual, load_predictions(args.predictions), args.tolerance_sec)
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
