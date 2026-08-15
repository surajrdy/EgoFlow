from __future__ import annotations

import json
import sqlite3

from hackathon.egoflow.dashboard import render_dashboard
from hackathon.egoflow.review_database import build_review_database, clean_spans


def test_clean_spans_exclude_review_events_and_short_fragments():
    events = [
        {"start_sec": 12.0, "end_sec": 15.0, "label": "hesitate"},
        {"start_sec": 28.0, "end_sec": 30.0, "label": "aborted_reach"},
        {"start_sec": 40.0, "end_sec": 50.0, "label": "productive"},
    ]
    assert clean_spans(45.0, events, min_duration_sec=10.0, guard_sec=1.0) == [
        (0.0, 11.0),
        (16.0, 27.0),
        (31.0, 45.0),
    ]


def test_build_review_database_indexes_sources_and_clean_spans(tmp_path):
    score = tmp_path / "episode.json"
    score.write_text(json.dumps({
        "episode_id": "episode",
        "task": "organizing",
        "duration_sec": 30.0,
        "completion_confidence": 0.8,
        "timestamps": [0.0, 15.0, 30.0],
        "local_progress": [0.0, 0.5, 1.0],
        "events": [{
            "start_sec": 10.0,
            "end_sec": 12.0,
            "label": "hesitate",
            "confidence": 0.7,
            "detector": "hybrid_learned_progress_visual_dynamics",
        }],
    }))
    database = tmp_path / "review.sqlite"
    result = build_review_database([score], database, min_clean_sec=10.0, guard_sec=0.0, portable_root=tmp_path)
    assert result["episodes"] == 1
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT source FROM events").fetchone()[0] == "HYBRID"
    assert connection.execute("SELECT start_sec,end_sec FROM clean_spans").fetchall() == [(0.0, 10.0), (12.0, 30.0)]
    connection.close()
    dashboard = render_dashboard(database, tmp_path / "dashboard.html")
    contents = dashboard.read_text()
    assert "Episode database" in contents
    assert "episode-card" in contents
    assert "HYBRID" in contents
    assert 'data-testid="shortlist-toggle"' in contents
    assert 'data-action="watch"' in contents
    assert '<dialog id="viewer"' in contents
    assert "localStorage" in contents
