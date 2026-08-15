"""Portable SQLite review catalog and clean-span extraction for EgoFlow."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
from typing import Iterable

from .results_schema import load_scores


REVIEW_LABELS = {"hesitate", "abandon", "recover", "regress", "aborted_reach", "interaction_deviation"}


def event_source(detector: str) -> str:
    return {
        "hybrid_learned_progress_visual_dynamics": "HYBRID",
        "learned_progress_normalized": "LEARNED",
        "frozen_visual_dynamics": "AUX",
        "video_hand_geometry_v1": "HAND EXPERIMENTAL",
        "learned_hand_dynamics_v2": "LEARNED V2",
    }.get(detector, "UNKNOWN")


def clean_spans(
    duration_sec: float,
    events: Iterable[dict[str, object]],
    *,
    min_duration_sec: float = 10.0,
    guard_sec: float = 0.5,
) -> list[tuple[float, float]]:
    """Return complements of suspicious intervals, with a safety guard."""

    blocked = []
    for event in events:
        if str(event.get("label", "")).lower() not in REVIEW_LABELS:
            continue
        start = max(0.0, float(event.get("start_sec", 0.0)) - guard_sec)
        end = min(duration_sec, float(event.get("end_sec", start)) + guard_sec)
        if end > start:
            blocked.append((start, end))
    blocked.sort()
    merged: list[list[float]] = []
    for start, end in blocked:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start - cursor >= min_duration_sec:
            spans.append((round(cursor, 3), round(start, 3)))
        cursor = max(cursor, end)
    if duration_sec - cursor >= min_duration_sec:
        spans.append((round(cursor, 3), round(duration_sec, 3)))
    return spans


def _portable_path(path: str | Path | None, root: Path) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return candidate.name


def build_review_database(
    score_paths: Iterable[str | Path],
    database_path: str | Path,
    *,
    video_paths: dict[str, str | Path] | None = None,
    hand_event_paths: dict[str, str | Path | list[str | Path]] | None = None,
    manual_labels_path: str | Path | None = None,
    metric_paths: Iterable[str | Path] = (),
    min_clean_sec: float = 10.0,
    guard_sec: float = 0.5,
    portable_root: str | Path = ".",
) -> dict[str, object]:
    """Index score files, review candidates, and clean spans in SQLite."""

    database = Path(database_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    root = Path(portable_root)
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS episodes (
          episode_id TEXT PRIMARY KEY, task TEXT NOT NULL, duration_sec REAL NOT NULL,
          completion_score REAL, score_path TEXT NOT NULL, video_path TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
          start_sec REAL NOT NULL, end_sec REAL NOT NULL, label TEXT NOT NULL,
          source TEXT NOT NULL, confidence REAL NOT NULL, reason TEXT
        );
        CREATE TABLE IF NOT EXISTS clean_spans (
          id INTEGER PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
          start_sec REAL NOT NULL, end_sec REAL NOT NULL, duration_sec REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS progress_points (
          episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
          timestamp_sec REAL NOT NULL, progress REAL NOT NULL, rate REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS manual_events (
          id INTEGER PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
          start_sec REAL NOT NULL, end_sec REAL NOT NULL, label TEXT NOT NULL, note TEXT
        );
        CREATE TABLE IF NOT EXISTS run_documents (
          name TEXT PRIMARY KEY, payload_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS events_episode_time ON events(episode_id, start_sec);
        CREATE INDEX IF NOT EXISTS clean_episode_time ON clean_spans(episode_id, start_sec);
        """
    )
    episode_count = event_count = span_count = 0
    clean_seconds = 0.0
    for score_path in score_paths:
        score_path = Path(score_path)
        series = load_scores(score_path)
        events = [dict(event) for event in series.events if str(event.get("label", "")).lower() in REVIEW_LABELS]
        if hand_event_paths and series.episode_id in hand_event_paths:
            from .visualize import load_hand_events
            sources = hand_event_paths[series.episode_id]
            sources = sources if isinstance(sources, list) else [sources]
            for source in sources:
                events.extend(load_hand_events(source))
        video = video_paths.get(series.episode_id) if video_paths else None
        connection.execute(
            "INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?)",
            (
                series.episode_id,
                series.task,
                series.duration_sec,
                    series.summary()["completion_confidence"],
                _portable_path(score_path, root),
                _portable_path(video, root),
            ),
        )
        connection.execute("DELETE FROM events WHERE episode_id = ?", (series.episode_id,))
        connection.execute("DELETE FROM clean_spans WHERE episode_id = ?", (series.episode_id,))
        connection.execute("DELETE FROM progress_points WHERE episode_id = ?", (series.episode_id,))
        for event in events:
            connection.execute(
                "INSERT INTO events(episode_id,start_sec,end_sec,label,source,confidence,reason) VALUES (?,?,?,?,?,?,?)",
                (
                    series.episode_id,
                    float(event.get("start_sec", 0.0)),
                    float(event.get("end_sec", 0.0)),
                    str(event.get("label", "review")),
                    event_source(str(event.get("detector", ""))),
                    float(event.get("confidence", 0.0)),
                    str(event.get("reason", "")),
                ),
            )
        spans = clean_spans(series.duration_sec, events, min_duration_sec=min_clean_sec, guard_sec=guard_sec)
        for start, end in spans:
            connection.execute(
                "INSERT INTO clean_spans(episode_id,start_sec,end_sec,duration_sec) VALUES (?,?,?,?)",
                (series.episode_id, start, end, end - start),
            )
            clean_seconds += end - start
        stride = max(1, len(series.timestamps_sec) // 120)
        for index in range(0, len(series.timestamps_sec), stride):
            connection.execute(
                "INSERT INTO progress_points VALUES (?,?,?,?)",
                (
                    series.episode_id,
                    float(series.timestamps_sec[index]),
                    float(series.global_progress[index]),
                    float(series.progress_velocity[index]),
                ),
            )
        episode_count += 1
        event_count += len(events)
        span_count += len(spans)
    if manual_labels_path:
        connection.execute("DELETE FROM manual_events")
        for line in Path(manual_labels_path).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            event = json.loads(line)
            if connection.execute("SELECT 1 FROM episodes WHERE episode_id=?", (str(event.get("episode_id", "")),)).fetchone():
                connection.execute(
                    "INSERT INTO manual_events(episode_id,start_sec,end_sec,label,note) VALUES (?,?,?,?,?)",
                    (
                        str(event["episode_id"]),
                        float(event["start_sec"]),
                        float(event["end_sec"]),
                        str(event["label"]),
                        str(event.get("note", "")),
                    ),
                )
    for metric_path in metric_paths:
        metric_path = Path(metric_path)
        connection.execute(
            "INSERT OR REPLACE INTO run_documents VALUES (?,?)",
            (metric_path.stem, json.dumps(json.loads(metric_path.read_text()), separators=(",", ":"))),
        )
    connection.commit()
    connection.close()
    return {
        "database": str(database),
        "episodes": episode_count,
        "events": event_count,
        "clean_spans": span_count,
        "clean_seconds": round(clean_seconds, 3),
        "min_clean_sec": min_clean_sec,
        "guard_sec": guard_sec,
    }


def export_clean_slices(
    database_path: str | Path,
    output_dir: str | Path,
    *,
    ffmpeg_path: str | Path | None = None,
) -> dict[str, object]:
    """Materialize cataloged clean spans without modifying source videos."""

    ffmpeg = str(ffmpeg_path) if ffmpeg_path else shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to export clean video slices")
    database, output = Path(database_path), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    rows = connection.execute(
        """SELECT c.episode_id,c.start_sec,c.end_sec,e.video_path
           FROM clean_spans c JOIN episodes e USING(episode_id)
           WHERE e.video_path IS NOT NULL ORDER BY c.episode_id,c.start_sec"""
    ).fetchall()
    connection.close()
    written = []
    for episode_id, start, end, video_path in rows:
        source = Path(video_path)
        if not source.is_absolute():
            source = Path.cwd() / source
        if not source.exists():
            continue
        target = output / f"{episode_id}__{start:08.3f}-{end:08.3f}.mp4"
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-to", str(end), "-i", str(source), "-c", "copy", str(target)],
            check=True,
        )
        written.append(str(target))
    return {"database": str(database), "slice_count": len(written), "slices": written}


def write_report(result: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    return target
