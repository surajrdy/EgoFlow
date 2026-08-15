# EgoFlow visualization and manual review

The timeline and summary path needs only Python 3. It accepts either a JSON
mapping of parallel arrays, JSON with a `frames` list, or (with NumPy installed)
an NPZ containing the same fields:

- `timestamps_sec`
- `local_progress`
- `global_progress`
- `progress_velocity`
- `event_labels`
- `confidences`
- `annotations`

Field aliases such as `timestamps`, `velocity`, `labels`, and `states` are also
accepted. Missing labels/annotations/confidences get safe defaults. Array length
mismatches fail loudly rather than producing a misleading graph.

```bash
python hackathon/egoflow/visualize.py scores.json \
  --timeline results/example_timeline.png \
  --summary results/episode_summary.json
```

Add `--video episode.mp4 --mp4 results/example_scored.mp4` to create an MP4
with an animated timeline beneath the source video. This optional path needs
`ffmpeg`; missing video or ffmpeg emits a warning while preserving the PNG and
JSON outputs. `--zarr episode.zarr` is a best-effort alternative that locates an
RGB `[T,H,W,3/4]` array and additionally needs `zarr`.

Scored videos deliberately show numerical signed progress rate on ordinary
frames. Low rate is not labeled as a semantic stall, and positive rate is not
labeled as proof of productive work. Only sparse higher-order intervals appear as
review candidates with explicit `LEARNED`, `HYBRID`, or `AUX` provenance.

## Fast manual review

Pass an episode JSON/JSONL manifest or a directory of local videos:

```bash
python hackathon/egoflow/scripts/make_review_manifest.py episodes.json \
  --manifest results/review_manifest.json \
  --html results/review.html
```

Open `review.html` directly. It works offline, supports video-time capture, and
downloads `manual_labels.jsonl`; it never uploads data. If videos are not local,
the page still shows episode IDs/paths and accepts timestamps entered from the
Mecka/EgoVerse browser.

Summarize labels, optionally comparing event predictions within a transparent
time tolerance:

```bash
python hackathon/egoflow/scripts/summarize_manual_labels.py manual_labels.jsonl
python hackathon/egoflow/scripts/summarize_manual_labels.py manual_labels.jsonl \
  --predictions results/episode_a.json results/episode_b.json \
  --tolerance-sec 1.5 --output results/manual_comparison.json
```

The comparison reports raw matches, misses, and false positives because the
manual validation set is intentionally small.

## Dependency-free smoke demo

```bash
python hackathon/egoflow/scripts/smoke_visualization.py
```

This creates `results/example_timeline.png`, `results/episode_summary.json`, and
the score input. All three identify themselves as **synthetic smoke-test data**;
they are not a model result or empirical claim and must not be presented as one.
