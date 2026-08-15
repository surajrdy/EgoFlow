"""Bounded Modal pipeline for EgoFlow feature extraction, training, and scoring.

The local entrypoint intentionally defaults to a two-episode smoke test. Data and
features live on the existing ``egoverse-data`` Volume. The explicit episode CSV
uses public Mecka video URLs and requires no API secret.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any
import uuid

import modal

from egoverse_modal.modal_resources import (
    data_volume,
    egoverse_image,
)


APP_NAME = "egoflow-hackathon"
VOLUME_ROOT = Path("/vol")
RUN_ROOT = VOLUME_ROOT / "egoflow"
DATASET_ROOT = VOLUME_ROOT / "datasets"
FEATURE_ROOT = RUN_ROOT / "cache" / "features"
MANIFEST_ROOT = RUN_ROOT / "manifests"
CHECKPOINT_ROOT = RUN_ROOT / "checkpoints"
RESULT_ROOT = RUN_ROOT / "results"

MAX_EPISODES = 60
FEATURE_FPS = 4.0
MAX_FEATURE_WORKERS = 8
MAX_SCORING_WORKERS = 12
MAX_TRAIN_RUNS = 3
MAX_SIMULTANEOUS_TRAINING_GPUS = 2
TRAIN_MAX_STEPS = 1_000
FEATURE_GPU = "L40S"
TRAIN_GPU = "H100"
FEATURE_TIMEOUT_SECONDS = 20 * 60
TRAIN_TIMEOUT_SECONDS = 25 * 60
ESTIMATED_COST_GUARD_USD = 250.0

# Deliberately conservative configurable estimates, not a claim about Modal's
# current list prices.  They make accidental job-size changes fail closed.
FEATURE_GPU_HOURLY_ESTIMATE_USD = 5.0
TRAIN_GPU_HOURLY_ESTIMATE_USD = 10.0
EXPECTED_FEATURE_MINUTES_PER_EPISODE = 5.0


def estimate_pipeline_cost(max_episodes: int, training_runs: int) -> dict[str, float]:
    """Return expected and timeout-bound GPU cost estimates."""

    feature_expected = (
        max_episodes
        * EXPECTED_FEATURE_MINUTES_PER_EPISODE
        / 60.0
        * FEATURE_GPU_HOURLY_ESTIMATE_USD
    )
    feature_timeout_bound = (
        max_episodes
        * FEATURE_TIMEOUT_SECONDS
        / 3600.0
        * FEATURE_GPU_HOURLY_ESTIMATE_USD
    )
    training_bound = (
        training_runs
        * TRAIN_TIMEOUT_SECONDS
        / 3600.0
        * TRAIN_GPU_HOURLY_ESTIMATE_USD
    )
    return {
        "expected_usd": round(feature_expected + training_bound, 2),
        "timeout_bound_usd": round(feature_timeout_bound + training_bound, 2),
    }


def validate_run_request(max_episodes: int, training_runs: int, max_steps: int) -> None:
    if not 1 <= int(max_episodes) <= MAX_EPISODES:
        raise ValueError(f"max_episodes must be in [1, {MAX_EPISODES}]")
    if not 0 <= int(training_runs) <= MAX_TRAIN_RUNS:
        raise ValueError(f"training_runs must be in [0, {MAX_TRAIN_RUNS}]")
    if not 1 <= int(max_steps) <= 2_500:
        raise ValueError("max_steps must be in [1, 2500]")
    estimate = estimate_pipeline_cost(int(max_episodes), int(training_runs))
    if estimate["timeout_bound_usd"] >= ESTIMATED_COST_GUARD_USD:
        raise ValueError(
            f"timeout-bound estimate ${estimate['timeout_bound_usd']:.2f} exceeds "
            f"the ${ESTIMATED_COST_GUARD_USD:.2f} guard"
        )


FUNCTION_ENV = {
    "HF_HOME": "/vol/egoflow/cache/huggingface",
    "TORCH_HOME": "/vol/egoflow/cache/torch",
    "MPLCONFIGDIR": "/vol/egoflow/cache/matplotlib",
    "PYTHONPATH": "/opt",
}

egoflow_image = egoverse_image.add_local_dir(
    "hackathon",
    remote_path="/opt/hackathon",
    copy=True,
)

app = modal.App(APP_NAME)


def _json_records(frame: Any) -> list[dict[str, Any]]:
    """Convert pandas records to Modal-safe JSON primitives."""

    return json.loads(frame.to_json(orient="records"))


@app.function(
    image=egoflow_image,
    env=FUNCTION_ENV,
    cpu=8,
    memory=32768,
    timeout=12 * 60,
    max_containers=1,
    volumes={str(VOLUME_ROOT): data_volume},
)
def prepare_dataset(
    max_episodes: int = 2,
    tasks: list[str] | None = None,
    episode_ids: list[str] | None = None,
    episode_metadata: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Resolve explicit human-demo IDs to public videos and persist a manifest."""

    validate_run_request(max_episodes, 0, 1)
    if episode_metadata:
        records: list[dict[str, Any]] = []
        for raw in episode_metadata[: int(max_episodes)]:
            episode_id = str(raw.get("episode_id") or "").strip()
            if len(episode_id) != 24 or any(
                character not in "0123456789abcdefABCDEF" for character in episode_id
            ):
                raise ValueError(f"invalid Mecka episode ID: {episode_id!r}")
            task = str(raw.get("task") or "unknown")
            records.append(
                {
                    "episode_hash": episode_id,
                    "task": task,
                    "task_description": task.replace("_", " "),
                    "embodiment": "human_bimanual",
                    "num_frames": None,
                    "zarr_mp4_path": "",
                    "review_status": str(raw.get("review_status") or ""),
                    "notes": str(raw.get("notes") or ""),
                    "source_path": str(DATASET_ROOT / episode_id),
                    "public_video_url": (
                        "https://partners.mecka.ai/api/egoverse/uploads/"
                        f"{episode_id}/video?redirect=1"
                    ),
                }
            )
        MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
        manifest_path = MANIFEST_ROOT / "selected_episodes.json"
        manifest_path.write_text(json.dumps({"episodes": records}, indent=2) + "\n")
        data_volume.commit()
        return {
            "episode_count": len(records),
            "tasks": sorted({record["task"] for record in records}),
            "manifest_path": str(manifest_path),
            "selection_source": "explicit_public_mecka_videos",
            "episodes": records,
        }
    raise ValueError(
        "EgoFlow's credential-free path requires --episode-ids-file; "
        "use hackathon/egoflow/episode_selection.csv."
    )


@app.function(
    image=egoflow_image,
    env=FUNCTION_ENV,
    gpu=FEATURE_GPU,
    cpu=8,
    memory=32768,
    timeout=FEATURE_TIMEOUT_SECONDS,
    max_containers=MAX_FEATURE_WORKERS,
    volumes={str(VOLUME_ROOT): data_volume},
)
def extract_episode(record: dict[str, Any], fps: float = FEATURE_FPS) -> dict[str, Any]:
    """Extract or reuse one episode's deterministic feature cache."""

    from hackathon.egoflow.cache import load_feature_cache, save_feature_cache
    from hackathon.egoflow.config import ExtractionConfig, SCHEMA_VERSION
    from hackathon.egoflow.features import (
        extract_episode_features,
        extract_video_features,
    )

    episode_id = str(record["episode_hash"])
    output_path = FEATURE_ROOT / f"{episode_id}.npz"
    config = ExtractionConfig(
        sample_fps=min(float(fps), FEATURE_FPS),
        task_description=str(record.get("task_description") or record.get("task") or ""),
    )
    if output_path.is_file():
        cached = load_feature_cache(output_path)
        metadata = json.loads(str(cached["metadata_json"].item()))
        same_sampling = abs(
            float(metadata.get("sample_fps", -1.0)) - config.sample_fps
        ) < 1e-6
        same_preprocessing = (
            int(metadata.get("preprocessing_version", -1)) == SCHEMA_VERSION
            and int(metadata.get("schema_version", -1)) == SCHEMA_VERSION
        )
        cached_backbone = str(metadata.get("visual_source", ""))
        same_fallback = not cached_backbone.startswith("facebook/") or (
            cached_backbone == config.dino_model
        )
        if same_sampling and same_preprocessing and same_fallback:
            return {
                "episode_id": episode_id,
                "feature_path": str(output_path),
                "frames": int(len(cached["frame_indices"])),
                "cache_hit": True,
            }
    source = Path(str(record["source_path"]))
    if source.exists():
        features = extract_episode_features(
            source,
            config=config,
            episode_id=episode_id,
            task=str(record.get("task") or "unknown"),
            source_path=str(source),
        )
    else:
        import shutil
        import urllib.request

        public_url = str(record.get("public_video_url") or "")
        if not public_url.startswith(
            "https://partners.mecka.ai/api/egoverse/uploads/"
        ):
            raise FileNotFoundError(f"episode source is unavailable: {source}")
        with tempfile.TemporaryDirectory(prefix="egoflow-public-") as directory:
            video_path = Path(directory) / f"{episode_id}.mp4"
            with urllib.request.urlopen(public_url, timeout=120) as response:
                with video_path.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            features = extract_video_features(
                video_path,
                config=config,
                episode_id=episode_id,
                task=str(record.get("task") or "unknown"),
                task_description=str(record.get("task_description") or ""),
            )
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    save_feature_cache(features, output_path)
    data_volume.commit()
    return {
        "episode_id": episode_id,
        "feature_path": str(output_path),
        "frames": int(len(features.frame_indices)),
        "visual_shape": [int(value) for value in features.visual_embeddings.shape],
        "language_shape": [int(value) for value in features.language_embeddings.shape],
        "cache_hit": False,
    }


@app.function(
    image=egoflow_image,
    env=FUNCTION_ENV,
    gpu=TRAIN_GPU,
    cpu=12,
    memory=65536,
    timeout=TRAIN_TIMEOUT_SECONDS,
    max_containers=MAX_SIMULTANEOUS_TRAINING_GPUS,
    volumes={str(VOLUME_ROOT): data_volume},
)
def train_progress_head(
    feature_paths: list[str],
    *,
    run_name: str,
    hidden_size: int = 128,
    max_steps: int = 750,
    learning_rate: float = 3e-4,
    seed: int = 17,
    episode_id_split: dict[str, list[str]] | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train one small temporal head on an immutable episode subset."""

    from hackathon.egoflow.train import train

    validate_run_request(len(feature_paths), 1, max_steps)
    if not feature_paths:
        raise ValueError("feature_paths cannot be empty")
    # train() intentionally consumes a directory. Symlinks give each run a stable
    # subset while remaining on the same machine and avoiding feature copies.
    with tempfile.TemporaryDirectory(prefix="egoflow-train-") as temporary:
        cache_dir = Path(temporary)
        for index, raw_path in enumerate(feature_paths):
            source = Path(raw_path)
            if not source.is_file():
                raise FileNotFoundError(f"feature cache not found: {source}")
            (cache_dir / f"{index:04d}-{source.name}").symlink_to(source)
        output_dir = CHECKPOINT_ROOT / run_name
        summary = train(
            cache_dir,
            output_dir,
            hidden_size=hidden_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
            seed=seed,
            episode_id_split=episode_id_split,
            smoke=smoke,
            device="cuda",
        )
    data_volume.commit()
    return {"run_name": run_name, "hidden_size": hidden_size, **summary}


@app.function(
    image=egoflow_image,
    env=FUNCTION_ENV,
    cpu=4,
    memory=8192,
    timeout=8 * 60,
    max_containers=MAX_SCORING_WORKERS,
    volumes={str(VOLUME_ROOT): data_volume},
)
def score_cached_episode(
    feature_path: str,
    checkpoint_path: str,
    *,
    run_name: str,
) -> dict[str, Any]:
    """Score one cached episode with a trained head and persist JSON."""

    from hackathon.egoflow.score import score_episode

    output_path = RESULT_ROOT / run_name / "scores" / f"{Path(feature_path).stem}.json"
    result = score_episode(feature_path, checkpoint_path, output_path, device="cpu")
    data_volume.commit()
    return {
        "episode_id": str(result["episode_id"]),
        "output_path": str(output_path),
        "completion_confidence": float(result["completion_confidence"]),
        "event_count": int(len(result["events"])),
    }


@app.function(
    image=egoflow_image,
    env=FUNCTION_ENV,
    cpu=2,
    memory=4096,
    timeout=5 * 60,
    max_containers=MAX_SCORING_WORKERS,
    volumes={str(VOLUME_ROOT): data_volume},
)
def render_score(score_record: dict[str, Any]) -> dict[str, str]:
    """Render one held-out/demo timeline without requiring a GPU."""

    from hackathon.egoflow.results_schema import load_scores, write_summary
    from hackathon.egoflow.visualize import render_timeline

    score_path = Path(str(score_record["output_path"]))
    series = load_scores(score_path)
    timeline_path = score_path.with_name(f"{score_path.stem}-timeline.png")
    summary_path = score_path.with_name(f"{score_path.stem}-summary.json")
    render_timeline(series, timeline_path)
    write_summary(series, summary_path)
    data_volume.commit()
    return {
        "episode_id": str(score_record["episode_id"]),
        "timeline_path": str(timeline_path),
        "summary_path": str(summary_path),
    }


@app.function(
    image=egoflow_image,
    env=FUNCTION_ENV,
    cpu=4,
    memory=8192,
    timeout=10 * 60,
    max_containers=1,
    volumes={str(VOLUME_ROOT): data_volume},
)
def render_public_scored_video(
    score_json: str,
    episode_id: str,
    *,
    fps: float = 5.0,
) -> dict[str, str]:
    """Render a public Mecka source video above its animated score timeline."""

    from pathlib import Path as RuntimePath
    import tempfile as runtime_tempfile
    from urllib.request import urlopen

    from hackathon.egoflow.results_schema import load_scores
    from hackathon.egoflow.visualize import render_scored_mp4

    if len(episode_id) != 24 or any(
        character not in "0123456789abcdefABCDEF" for character in episode_id
    ):
        raise ValueError("episode_id must be a 24-character hexadecimal Mecka ID")
    output_dir = RESULT_ROOT / "hero"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{episode_id}-scored.mp4"
    with runtime_tempfile.TemporaryDirectory(prefix="egoflow-hero-") as temp_dir:
        temporary = RuntimePath(temp_dir)
        score_path = temporary / "score.json"
        video_path = temporary / "source.mp4"
        score_path.write_text(score_json, encoding="utf-8")
        video_url = (
            "https://partners.mecka.ai/api/egoverse/uploads/"
            f"{episode_id}/video?redirect=1"
        )
        with urlopen(video_url, timeout=90) as response, video_path.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        rendered = render_scored_mp4(
            load_scores(score_path),
            video_path,
            output_path,
            fps=max(1.0, min(10.0, float(fps))),
        )
        if rendered is None:
            raise RuntimeError("ffmpeg did not produce the scored hero video")
    data_volume.commit()
    return {"episode_id": episode_id, "scored_video_path": str(output_path)}


def _finished_calls(
    pending: dict[int, Any],
    completed: list[dict[str, Any]],
) -> None:
    """Move currently finished Modal calls into ``completed`` without blocking."""

    from modal.exception import TimeoutError as ModalTimeoutError

    for index, call in list(pending.items()):
        try:
            result = call.get(timeout=0)
        except (TimeoutError, ModalTimeoutError):
            continue
        completed.append(result)
        del pending[index]


def _read_episode_ids(path: str) -> list[str]:
    """Read a local CSV, text, JSON, or JSONL selection without uploading it."""

    import csv

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"episode selection file not found: {source}")
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        values = [row.get("episode_id") or row.get("episode_hash") or "" for row in rows]
    elif source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload.get("episodes", []) if isinstance(payload, dict) else payload
        values = [
            row.get("episode_id") or row.get("episode_hash") or ""
            if isinstance(row, dict)
            else row
            for row in rows
        ]
    elif source.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        values = [
            row.get("episode_id") or row.get("episode_hash") or ""
            if isinstance(row, dict)
            else row
            for row in rows
        ]
    else:
        values = [line.strip().split(",", 1)[0] for line in source.read_text().splitlines()]
    result = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if not result:
        raise ValueError(f"no episode IDs found in {source}")
    return result


def _read_episode_metadata(path: str) -> list[dict[str, str]]:
    """Read the human selection while retaining task/review metadata."""

    import csv

    source = Path(path)
    if source.suffix.lower() != ".csv":
        return [{"episode_id": value} for value in _read_episode_ids(path)]
    with source.open(newline="", encoding="utf-8") as handle:
        records = [
            {str(key): str(value or "") for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    records = [
        {
            **record,
            "episode_id": record.get("episode_id") or record.get("episode_hash") or "",
        }
        for record in records
        if record.get("episode_id") or record.get("episode_hash")
    ]
    if not records:
        raise ValueError(f"no episode rows found in {source}")
    return records


@app.local_entrypoint()
def hero_video(
    score_path: str,
    episode_id: str = "69bb1239efeadec2abedad96",
    fps: float = 5.0,
) -> None:
    """Render one local score JSON against its public video on Modal."""

    source = Path(score_path)
    if not source.is_file():
        raise FileNotFoundError(f"score JSON not found: {source}")
    print(
        json.dumps(
            render_public_scored_video.remote(
                source.read_text(encoding="utf-8"),
                episode_id,
                fps=fps,
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def frozen_eval(
    episode_ids_file: str,
    seed: int = 8,
    max_steps: int = 750,
) -> None:
    """Retrain from cached features with a declared immutable episode split."""

    selection = _read_episode_metadata(episode_ids_file)
    episode_ids = [record["episode_id"] for record in selection]
    validate_run_request(len(episode_ids), 1, max_steps)
    run_name = f"frozen-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    feature_paths = [str(FEATURE_ROOT / f"{episode_id}.npz") for episode_id in episode_ids]
    hero_id = "69bb1239efeadec2abedad96"
    if hero_id not in episode_ids:
        raise ValueError(f"frozen evaluation requires declared validation hero {hero_id}")
    unreviewed = sorted(
        record["episode_id"]
        for record in selection
        if record.get("review_status", "").strip().lower() == "unreviewed"
    )
    lightly_reviewed = sorted(
        record["episode_id"]
        for record in selection
        if record.get("review_status", "").strip().lower() == "light_review"
    )
    test_ids = (unreviewed + lightly_reviewed)[:3]
    if len(test_ids) < 3:
        raise ValueError("frozen evaluation requires three unreviewed/light-review test episodes")
    remaining = sorted(
        episode_id for episode_id in episode_ids
        if episode_id not in {hero_id, *test_ids}
    )
    explicit_split = {
        "test": test_ids,
        "val": [hero_id, *remaining[:2]],
        "train": remaining[2:],
    }
    print(
        json.dumps(
            {
                "event": "split_frozen",
                "episode_id_split": explicit_split,
                "test_prior_exposure": {
                    record["episode_id"]: record.get("review_status", "unknown")
                    for record in selection if record["episode_id"] in test_ids
                },
            }
        ),
        flush=True,
    )
    trained = train_progress_head.remote(
        feature_paths,
        run_name=run_name,
        hidden_size=128,
        max_steps=max_steps,
        learning_rate=3e-4,
        seed=seed,
        episode_id_split=explicit_split,
        smoke=False,
    )
    checkpoint = str(trained["checkpoint"])
    scores = list(
        score_cached_episode.map(
            feature_paths,
            kwargs={"checkpoint_path": checkpoint, "run_name": run_name},
        )
    )
    print(
        json.dumps(
            {
                "event": "frozen_evaluation_complete",
                "split_seed": seed,
                "winner": trained,
                "scores": scores,
                "next_command": (
                    "uv run modal volume get egoverse-data /egoflow "
                    "hackathon/egoflow/results/frozen-volume"
                ),
            }
        )
    )


@app.local_entrypoint()
def main(
    action: str = "smoke",
    max_episodes: int = 2,
    tasks: str = "",
    episode_ids_file: str = "",
    fps: float = FEATURE_FPS,
    training_runs: int = 1,
    max_steps: int = TRAIN_MAX_STEPS,
) -> None:
    """Run the smoke, extraction, or complete bounded EgoFlow pipeline."""

    if action == "smoke":
        max_episodes = min(max_episodes, 2)
        max_steps = min(max_steps, 20)
        training_runs = 1
    validate_run_request(max_episodes, training_runs, max_steps)
    estimate = estimate_pipeline_cost(max_episodes, training_runs)
    print(json.dumps({"event": "cost_guard", **estimate}))
    selected_tasks = [value.strip() for value in tasks.split(",") if value.strip()]
    selected_episode_metadata = (
        _read_episode_metadata(episode_ids_file) if episode_ids_file else []
    )
    selected_episode_ids = [
        record["episode_id"] for record in selected_episode_metadata
    ]
    if action not in {"smoke", "extract", "full"}:
        raise ValueError("action must be smoke, extract, or full")
    if action == "full" and training_runs not in {1, 2}:
        raise ValueError(
            "full runs support one trainer or the bounded A/B pair; the third "
            "allowed run is reserved for an explicitly requested final retry"
        )
    manifest = prepare_dataset.remote(
        max_episodes,
        selected_tasks or None,
        selected_episode_ids or None,
        selected_episode_metadata or None,
    )
    records = list(manifest["episodes"])
    if action in {"smoke", "extract"}:
        results = list(extract_episode.map(records, kwargs={"fps": fps}))
        print(
            json.dumps(
                {
                    "event": "features_ready",
                    "manifest_path": manifest["manifest_path"],
                    "episode_count": len(results),
                    "features": results,
                }
            )
        )
        if action == "extract":
            return
        run_name = f"smoke-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        feature_paths = [str(item["feature_path"]) for item in results]
        winner = train_progress_head.remote(
            feature_paths,
            run_name=run_name,
            hidden_size=128,
            max_steps=max_steps,
            smoke=True,
        )
        checkpoint = str(winner["checkpoint"])
        score_records = list(
            score_cached_episode.map(
                feature_paths,
                kwargs={"checkpoint_path": checkpoint, "run_name": run_name},
            )
        )
        timelines = list(render_score.map(score_records))
        print(
            json.dumps(
                {
                    "event": "smoke_complete",
                    "winner": winner,
                    "scores": score_records,
                    "timelines": timelines,
                    "next_command": (
                        "uv run modal run hackathon/egoflow/modal_app.py "
                        "--action full --max-episodes 20 --max-steps 750"
                    ),
                }
            )
        )
        return
    else:
        pending = {
            index: extract_episode.spawn(record, fps=fps)
            for index, record in enumerate(records)
        }
        results: list[dict[str, Any]] = []
        threshold = min(len(records), 10)
        last_report = 0.0
        training_calls: list[Any] = []
        run_prefix = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        while pending:
            _finished_calls(pending, results)
            now = time.monotonic()
            if now - last_report >= 20:
                print(
                    json.dumps(
                        {
                            "event": "feature_progress",
                            "ready": len(results),
                            "total": len(records),
                            "training_started": bool(training_calls),
                        }
                    ),
                    flush=True,
                )
                last_report = now
            if len(results) >= threshold and not training_calls:
                initial_paths = [str(item["feature_path"]) for item in results]
                variants = [(128, "a")]
                if training_runs == 2:
                    variants.append((256, "b"))
                training_calls = [
                    train_progress_head.spawn(
                        initial_paths,
                        run_name=f"{run_prefix}-{suffix}",
                        hidden_size=hidden,
                        max_steps=max_steps,
                        smoke=False,
                    )
                    for hidden, suffix in variants
                ]
                print(
                    json.dumps(
                        {
                            "event": "training_started",
                            "episodes": len(initial_paths),
                            "variants": [hidden for hidden, _ in variants],
                        }
                    ),
                    flush=True,
                )
            if pending:
                time.sleep(2)
        # A very small full request reaches this point before the threshold branch.
        if not training_calls:
            feature_paths = [str(item["feature_path"]) for item in results]
            training_calls = [
                train_progress_head.spawn(
                    feature_paths,
                    run_name=f"{run_prefix}-a",
                    hidden_size=128,
                    max_steps=max_steps,
                    smoke=False,
                )
            ]

        training_results = [call.get() for call in training_calls]
        winner = max(
            training_results,
            key=lambda item: float(item.get("best_val_ranking_accuracy", -1.0)),
        )
        checkpoint = str(winner["checkpoint"])
        score_inputs = [str(item["feature_path"]) for item in results]
        score_records = list(
            score_cached_episode.map(
                score_inputs,
                kwargs={"checkpoint_path": checkpoint, "run_name": winner["run_name"]},
            )
        )
        timelines = list(render_score.map(score_records))
        print(
            json.dumps(
                {
                    "event": "pipeline_complete",
                    "winner": winner,
                    "scores": score_records,
                    "timelines": timelines,
                    "next_command": (
                        "uv run modal volume get egoverse-data /egoflow/results "
                        "hackathon/egoflow/results/remote"
                    ),
                }
            )
        )
