#!/usr/bin/env python3
"""Build an EgoFlow SQLite review database and optionally export clean slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hackathon.egoflow.review_database import build_review_database, export_clean_slices
from hackathon.egoflow.dashboard import render_dashboard


def _score_files(inputs: list[Path]) -> list[Path]:
    files = []
    for source in inputs:
        files.extend(sorted(source.glob("*.json")) if source.is_dir() else [source])
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", type=Path, help="score JSON files or directories")
    parser.add_argument("--database", type=Path, default=Path("results/egoflow-review.sqlite"))
    parser.add_argument("--video-dir", type=Path, help="optional directory containing EPISODE_ID.mp4")
    parser.add_argument(
        "--prefer-scored-video",
        action="store_true",
        help="prefer EPISODE_ID-scored.mp4 over a raw source video",
    )
    parser.add_argument("--hand-events-dir", type=Path, help="optional directory containing EPISODE_ID.json")
    parser.add_argument("--interaction-events-dir", type=Path, help="optional learned-v2 EPISODE_ID.json directory")
    parser.add_argument("--manual-labels", type=Path)
    parser.add_argument("--metrics", nargs="*", type=Path, default=[])
    parser.add_argument("--dashboard", type=Path, help="write a self-contained presentation HTML")
    parser.add_argument("--hero-image", type=Path)
    parser.add_argument("--hero-video", type=Path)
    parser.add_argument(
        "--featured-episode",
        action="append",
        default=[],
        help="episode ID to include in the public dashboard (repeatable, ordered)",
    )
    parser.add_argument(
        "--include-research-events",
        action="store_true",
        help="show all sparse detector sources in a private/research dashboard",
    )
    parser.add_argument("--min-clean-sec", type=float, default=10.0)
    parser.add_argument("--guard-sec", type=float, default=0.5)
    parser.add_argument("--slice-dir", type=Path, help="optionally export clean MP4 slices now")
    parser.add_argument("--ffmpeg-path", type=Path)
    args = parser.parse_args()
    score_files = _score_files(args.scores)
    episode_ids = [path.stem for path in score_files]
    videos = {}
    if args.video_dir:
        for episode_id in episode_ids:
            candidates = sorted(
                args.video_dir.rglob(f"{episode_id}*.mp4"),
                key=(
                    (lambda path: ("scored" not in path.stem, "source" in path.stem, len(str(path))))
                    if args.prefer_scored_video
                    else (lambda path: ("source" not in path.stem, "scored" in path.stem, len(str(path))))
                ),
            )
            if candidates:
                videos[episode_id] = candidates[0]
    hands = {}
    for episode_id in episode_ids:
        sources = []
        if args.hand_events_dir and (args.hand_events_dir / f"{episode_id}.json").exists():
            sources.append(args.hand_events_dir / f"{episode_id}.json")
        if args.interaction_events_dir and (args.interaction_events_dir / f"{episode_id}.json").exists():
            sources.append(args.interaction_events_dir / f"{episode_id}.json")
        if sources:
            hands[episode_id] = sources
    result = build_review_database(
        score_files,
        args.database,
        video_paths=videos,
        hand_event_paths=hands,
        manual_labels_path=args.manual_labels,
        metric_paths=args.metrics,
        min_clean_sec=args.min_clean_sec,
        guard_sec=args.guard_sec,
    )
    if args.slice_dir:
        result["export"] = export_clean_slices(args.database, args.slice_dir, ffmpeg_path=args.ffmpeg_path)
    if args.dashboard:
        result["dashboard"] = str(render_dashboard(
            args.database,
            args.dashboard,
            hero_image=args.hero_image,
            hero_video=args.hero_video,
            featured_episode_ids=args.featured_episode or None,
            include_research_events=args.include_research_events,
        ))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
