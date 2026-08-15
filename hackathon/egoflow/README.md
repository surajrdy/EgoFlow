# EgoFlow — dense progress and hesitation reward model

EgoFlow is the Track 3 hackathon deliverable. It predicts local and global progress
and signed progress velocity, then proposes sparse `recover`, `hesitate`, and
`abandon` review candidates from long-horizon EgoVerse demos. The JSON schema keeps
legacy `productive`/`stall`/`regress` velocity-bin names for compatibility, but the
UI presents them as positive/low/negative rate rather than behavioral truth.
The DINO/Qwen encoders are frozen; only projections and a two-layer BiGRU temporal
head are trained. Frame index, timestamps, and normalized episode time are never
model inputs.

The implementation is self-contained here and does not modify EgoVerse internals.
It understands the actual EgoVerse ZarrWriter layout, including JPEG-byte
`images.front_1`, JSON-byte annotations, padded arrays, existing `dino.*` patch
features, and existing Qwen features. Missing gated DINOv3 features fall back to
public frozen `facebook/dinov2-small`; missing language features use deterministic
cached text hashing.

## Human workflow

The reviewer only needs to fill [episode_selection.csv](episode_selection.csv) with
20–30 episode IDs and add roughly 10–15 timestamp events to
[manual_labels.jsonl](manual_labels.jsonl). Record the Angry Bird hesitation episode
and timestamp first. Valid labels are `productive`, `stall`, `regress`, `recover`,
`hesitate`, `abandon`, `complete`, and `other`.

Create an offline review page from the CSV:

```bash
uv run python -m hackathon.egoflow.scripts.make_review_manifest \
  hackathon/egoflow/episode_selection.csv \
  --manifest hackathon/egoflow/results/review_manifest.json \
  --html hackathon/egoflow/results/review.html
```

The page never uploads data and omits signed or token-like media references. Validate
the downloaded labels with:

```bash
uv run python -m hackathon.egoflow.scripts.summarize_manual_labels \
  hackathon/egoflow/manual_labels.jsonl
```

## Modal pipeline

The checked-in selection uses public episode videos from the
[Mecka EgoVerse explorer](https://partners.mecka.ai/egoverse). It bypasses the
private SQL/R2 catalog and requires no API keys or Modal Secrets:

```bash
make egoflow-secret
```

That target only confirms the credential-free configuration; it does not publish
anything. The private Zarr reader remains available as a local library API for teams
that already have authorized files, but it is not used by this run.

Run the required two-episode, 20-step smoke path:

```bash
uv run modal run hackathon/egoflow/modal_app.py \
  --action smoke --max-episodes 2 --max-steps 20
```

Then process the exact human-selected IDs. Adjust `--max-episodes` to the number of
rows in the CSV:

```bash
uv run modal run hackathon/egoflow/modal_app.py \
  --action full \
  --episode-ids-file hackathon/egoflow/episode_selection.csv \
  --max-episodes 20 \
  --max-steps 750 \
  --training-runs 1
```

The explicit CSV is required by the credential-free Modal entrypoint. For the
bounded A/B run, set `--training-runs 2`; this launches hidden sizes 128 and 256
simultaneously and selects held-out ranking accuracy. It does not launch a
Transformer or DDP.

The full coordinator starts training as soon as 15 feature caches exist while the
remaining extractors continue, scores all completed caches, and writes timelines and
summaries to the `egoverse-data` Volume. Fetch them with:

```bash
uv run modal volume get egoverse-data /egoflow/results \
  hackathon/egoflow/results/remote
```

## Local stages and evaluation

One local episode can be inspected and cached independently:

```bash
uv run python -m hackathon.egoflow.scripts.extract_episode \
  EPISODE.zarr cache/EPISODE.npz
```

Train, score, evaluate, and render from existing caches:

```bash
uv run python -m hackathon.egoflow.train \
  --cache-dir cache --output-dir results/run-a --hidden-size 128 --max-steps 750
uv run python -m hackathon.egoflow.score \
  --features cache --checkpoint results/run-a/best.pt --output-dir results/run-a/scores
uv run python -m hackathon.egoflow.evaluate \
  --scores results/run-a/scores \
  --manual-labels hackathon/egoflow/manual_labels.jsonl \
  --output results/run-a/metrics.json
uv run python -m hackathon.egoflow.visualize results/run-a/scores/EPISODE.json \
  --timeline results/run-a/EPISODE-timeline.png \
  --summary results/run-a/EPISODE-summary.json
```

Evaluation uses episode-disjoint splits, within-stage pairwise ranking, true prefix
re-inference at 25/50/75/100%, a final-frame DINO-cosine baseline, an intentionally
trivial time-fraction chronology baseline, synthetic event corruptions, and raw
manual-event matches within a declared tolerance. The tiny manual set is reported as
counts, not population-level accuracy.

## Real run and safety

Hard caps are enforced in code: 60 episodes, 4 FPS, eight L40S extractors, 12 scoring
workers, two simultaneous H100 trainers, three runs total, 2,500 steps, and 25 minutes
per trainer. Conservative timeout-bound cost for the largest permitted request is
$112.50 under the $250 scheduling guard; normal expected cost is much lower.

The completed real run processed all 18 selected public videos at 4 FPS. The final
protocol freezes an explicit 12/3/3 episode split, puts the human-tuned Angry Bird
episode in validation, and reserves three previously unreviewed episodes for blind
test. Training stopped early at step 400 after 15.35 seconds. The selected step-100
checkpoint has 594,561 trainable parameters. On 183,639 blind-test pairs, its
within-episode ranking accuracy is 0.9982; the intentionally trivial time-fraction
baseline is 1.0 and final-frame DINO cosine is 0.5759, so chronology alone remains
a serious baseline. True-prefix completion rises 0.9132/0.9426/0.9560/0.9608 at
25/50/75/100%.

Public MP4s do not contain dense Zarr annotation spans, and probes of the public
upload API expose only video; Zarr/annotation/Qwen routes return 404. The learned
BiGRU therefore owns coarse single-stage progress only. Its rate thresholds are
episode/stage-normalized (positive-rate q40, negative-rate MAD) rather than the
former fixed 0.025 floor. The presentation does not equate low rate with a semantic
stall or positive rate with productive work. Higher-order learned
hesitation/abandonment is disabled when semantic stages are absent.

A transparent frozen-DINO visual-dynamics layer supplies preliminary proposals,
with every event marked `learned_progress_normalized`,
`hybrid_learned_progress_visual_dynamics`, or `frozen_visual_dynamics`. After one
global correction and a three-proposal cap, the same small manual set has 5/8
overlaps with 16 same-label false positives (23.8% reviewed-set precision). This is
a same-set calibration result, not held-out event accuracy. The 8.0–11.0s hero
hesitation is hybrid: it overlaps normalized learned low-rate evidence plus visual
slowdown, versus the human 8–13s span.

The real checkpoint, scores, metrics, and hero artifacts are under `results/remote`
and `results/hero`. The retained `results/synthetic_checkpoint/` only proves the
wiring and remains explicitly marked synthetic.

## Experimental hand-interaction v2

`hand_interaction.py` extracts video-only 2D palm centers and a normalized
thumb/index aperture proxy. Its bounded detector proposes an `ABORTED REACH?` only
for moving approach/redirection motifs; stationary waiting, obvious close/open
grasp cycles, tracking gaps, and camera-coupled bimanual motion are rejected.
`hand_modal.py` runs at most eight CPU workers and stores source-attributed JSON.

This is an exploratory lane, not part of the frozen evaluation. A parallel pass
over all 18 selected clips achieved 94.6% frame-level hand-detection coverage and
returned 90 candidates (five per clip). A four-variant validation comparison chose
soft aperture evidence plus brief-gap tolerance; it proposes 8.267–10.133 s inside
the known 8–13 s Angry Bird span. This is validation tuning, not held-out accuracy.
Without object identity/persistence, hand turns during successful transport remain
confounded with approach→abort→switch.
See [HAND_V2.md](HAND_V2.md) for the exact claim boundary and command.

The follow-on learned v2 replaces the hand-turn threshold with a two-layer GRU
trained self-supervised to predict the next 2D hand state. Two variants trained in
parallel on the 12 frozen training episodes; the 32-unit, eight-frame model won on
training loss (0.0367 versus 0.0448). With a frozen 6-MAD surprise threshold it
flags 8.433–9.833 s in Angry Bird validation at 14.42 MAD. It is labeled
`LEARNED V2 / INTERACTION DEVIATION`, not “hesitation,” because object identity and
persistent displacement are still absent.

Small, commit-safe protocol artifacts live at `results/frozen_split.json`,
`results/blind_test_metrics.json`, and `results/blind_test_review_queue.json`.
Injected feature stalls and reversals trigger the auxiliary visual event layer on
3/3 blind episodes, but reversal does not reverse the coarse learned reward (0/3);
that failure is reported rather than folded into a learned robustness claim.
