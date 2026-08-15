#!/usr/bin/env python3
"""Inspect the curated EgoFlow hero result or render a supplied score JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hackathon.egoflow.results_schema import load_scores, write_summary  # noqa: E402
from hackathon.egoflow.visualize import render_timeline  # noqa: E402


HERO_ID = "69bb1239efeadec2abedad96"
HERO_DIR = ROOT / "hackathon" / "egoflow" / "results" / "hero"
HERO_FIGURE = HERO_DIR / "angry-bird-normalized-human-vs-model.png"
HERO_SUMMARY = HERO_DIR / "angry-bird-normalized-summary.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        default=HERO_ID,
        help=f"the bundled hero ID ({HERO_ID}) or a score JSON path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/demo"),
        help="destination for portable demo output",
    )
    return parser


def _print_summary(summary: dict[str, object], timeline: Path) -> None:
    print(f"episode={summary.get('episode_id')}")
    print(f"task={summary.get('task')}")
    print(f"completion={float(summary.get('completion_confidence', 0.0)):.4f}")
    print(f"timeline={timeline}")
    hesitations = summary.get("hesitations", [])
    if isinstance(hesitations, list):
        for event in hesitations:
            if isinstance(event, dict):
                print(
                    "hesitate="
                    f"{float(event.get('start_sec', 0.0)):.2f}–"
                    f"{float(event.get('end_sec', 0.0)):.2f}s "
                    f"source={event.get('detector', 'unknown')}"
                )


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(args.episode)

    if args.episode == HERO_ID:
        summary = json.loads(HERO_SUMMARY.read_text())
        timeline = output_dir / "hero_timeline.png"
        shutil.copyfile(HERO_FIGURE, timeline)
        (output_dir / "hero_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        _print_summary(summary, timeline)
        print("human_ground_truth=8.00–13.00s hesitate")
        print("note: bundled real-data result; no network or credentials used")
        return 0

    if not candidate.is_file() or candidate.suffix.lower() != ".json":
        raise SystemExit(
            f"--episode must be {HERO_ID} or an existing EgoFlow score JSON"
        )
    series = load_scores(candidate)
    timeline = output_dir / f"{series.episode_id}_timeline.png"
    summary_path = output_dir / f"{series.episode_id}_summary.json"
    render_timeline(series, timeline)
    write_summary(series, summary_path)
    summary = json.loads(summary_path.read_text())
    _print_summary(summary, timeline)
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
