"""Fast training entrypoint for the cached-feature EgoFlow head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

from .dataset import (
    collate_episodes,
    discover_feature_files,
    load_episode,
    make_synthetic_cache,
    read_episode_id,
    split_episode_paths,
)
from .models.losses import endpoint_anchor_loss, stage_smoothness_loss, temporal_ranking_loss
from .models.progress_model import ProgressModel


HARD_MAX_STEPS = 2_500
HARD_MAX_SECONDS = 24 * 60
CHECKPOINT_INTERVAL = 250


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("training requires PyTorch (the Modal training image includes it)") from exc
    return torch


def _evaluate(model: Any, paths: Sequence[Path], device: str) -> dict[str, float]:
    torch = _torch()
    if not paths:
        return {"loss": float("nan"), "ranking_accuracy": float("nan"), "ranking_pairs": 0.0}
    model.eval()
    losses: list[float] = []
    correct_weighted = 0.0
    pair_count = 0.0
    with torch.no_grad():
        for path in paths:
            batch = collate_episodes([load_episode(path)], device=device)
            output = model(batch["visual_embeddings"], batch["language_embeddings"], batch["lengths"])
            loss, metrics = temporal_ranking_loss(
                output["local_progress"], batch["stage_ids"], batch["mask"]
            )
            losses.append(float(loss.item()))
            correct_weighted += metrics["ranking_accuracy"] * metrics["ranking_pairs"]
            pair_count += metrics["ranking_pairs"]
    return {
        "loss": sum(losses) / len(losses),
        "ranking_accuracy": correct_weighted / pair_count if pair_count else 0.0,
        "ranking_pairs": pair_count,
    }


def _save_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    step: int,
    model_config: dict[str, Any],
    split: dict[str, list[Path]],
    episode_id_split: dict[str, list[str]],
    metrics: dict[str, Any],
) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "model_config": model_config,
            "episode_split": {name: [item.name for item in paths] for name, paths in split.items()},
            "episode_id_split": episode_id_split,
            "metrics": metrics,
        },
        temporary,
    )
    temporary.replace(path)


def train(
    cache_dir: str | Path,
    output_dir: str | Path,
    hidden_size: int = 128,
    max_steps: int = 750,
    learning_rate: float = 3e-4,
    seed: int = 17,
    smoke: bool = False,
    device: str | None = None,
    episode_id_split: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Train only the projection/GRU heads and return JSON-safe run metadata."""
    torch = _torch()
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if max_steps > HARD_MAX_STEPS:
        raise ValueError(f"max_steps={max_steps} exceeds the hard cap of {HARD_MAX_STEPS}")
    if hidden_size not in ProgressModel.ALLOWED_HIDDEN_SIZES:
        raise ValueError("hidden_size must be 128 or 256")
    if smoke:
        max_steps = min(max_steps, 20)

    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_feature_files(cache_dir, max_episodes=4 if smoke else None)
    if not paths and smoke:
        paths = make_synthetic_cache(output_dir / "synthetic_cache", episodes=4, seed=seed)
    if not paths:
        raise FileNotFoundError(f"no .npz feature caches found in {cache_dir}")

    if episode_id_split is None:
        split = split_episode_paths(paths, seed=seed)
        resolved_episode_split = {
            name: [read_episode_id(item) for item in items] for name, items in split.items()
        }
    else:
        by_id = {read_episode_id(path): path for path in paths}
        declared = [item for name in ("train", "val", "test") for item in episode_id_split.get(name, [])]
        if len(declared) != len(set(declared)):
            raise ValueError("explicit episode split contains duplicate episode IDs")
        if set(declared) != set(by_id):
            missing = sorted(set(by_id) - set(declared))
            unknown = sorted(set(declared) - set(by_id))
            raise ValueError(f"explicit episode split mismatch: missing={missing}, unknown={unknown}")
        if not episode_id_split.get("train"):
            raise ValueError("explicit episode split requires at least one training episode")
        resolved_episode_split = {
            name: list(episode_id_split.get(name, [])) for name in ("train", "val", "test")
        }
        split = {
            name: [by_id[episode_id] for episode_id in resolved_episode_split[name]]
            for name in ("train", "val", "test")
        }
    first = load_episode(split["train"][0])
    model_config = {
        "visual_dim": int(first.visual_embeddings.shape[1]),
        "language_dim": int(first.language_embeddings.shape[1]),
        "hidden_size": int(hidden_size),
        "projection_dim": 128,
    }
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = ProgressModel(**model_config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    started = time.monotonic()
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_step = 0
    stale_evaluations = 0
    completed_steps = 0
    stop_reason = "max_steps"

    print(
        f"MODEL: device={resolved_device} episodes={len(paths)} visual={model_config['visual_dim']} "
        f"language={model_config['language_dim']} params={model.trainable_parameter_count:,}",
        flush=True,
    )
    for step in range(1, max_steps + 1):
        if time.monotonic() - started >= HARD_MAX_SECONDS:
            stop_reason = "hard_wallclock_limit"
            break
        model.train()
        path = random.choice(split["train"])
        batch = collate_episodes([load_episode(path)], device=resolved_device)
        output = model(batch["visual_embeddings"], batch["language_embeddings"], batch["lengths"])
        rank_loss, train_metrics = temporal_ranking_loss(
            output["local_progress"], batch["stage_ids"], batch["mask"]
        )
        # Ranking dominates; anchors only break the translation ambiguity.
        anchor_loss = endpoint_anchor_loss(output["local_progress"], batch["stage_ids"], batch["mask"])
        smooth_loss = stage_smoothness_loss(output["local_progress"], batch["stage_ids"], batch["mask"])
        loss = rank_loss + 0.10 * anchor_loss + 0.02 * smooth_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed_steps = step

        should_validate = step == 1 or step % 50 == 0 or step == max_steps
        if should_validate:
            validation_paths = split["val"] or split["train"][:1]
            validation = _evaluate(model, validation_paths, resolved_device)
            row = {
                "step": step,
                "train_loss": round(float(loss.item()), 6),
                "train_ranking_accuracy": round(train_metrics["ranking_accuracy"], 6),
                "val_loss": round(validation["loss"], 6),
                "val_ranking_accuracy": round(validation["ranking_accuracy"], 6),
            }
            history.append(row)
            print(f"RESULT: {json.dumps(row, sort_keys=True)}", flush=True)
            accuracy = validation["ranking_accuracy"]
            if accuracy > best_accuracy + 0.002:
                best_accuracy, best_step, stale_evaluations = accuracy, step, 0
                _save_checkpoint(
                    output_dir / "best.pt", model, optimizer, step, model_config, split, resolved_episode_split, row
                )
            else:
                stale_evaluations += 1
            # Six checks = 300 steps without material held-out improvement.
            if step >= 250 and stale_evaluations >= 6:
                stop_reason = "early_stopping"
                break

        if step % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(
                output_dir / f"step-{step:04d}.pt",
                model,
                optimizer,
                step,
                model_config,
                split,
                resolved_episode_split,
                history[-1] if history else {},
            )

    elapsed = time.monotonic() - started
    final_metrics = _evaluate(model, split["val"] or split["train"][:1], resolved_device)
    _save_checkpoint(
        output_dir / "last.pt", model, optimizer, completed_steps, model_config, split, resolved_episode_split, final_metrics
    )
    summary: dict[str, Any] = {
        "checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "cache_dir": str(paths[0].parent),
        "synthetic": all(path.stem.startswith("synthetic-") for path in paths),
        "steps": completed_steps,
        "best_step": best_step,
        "best_val_ranking_accuracy": round(best_accuracy, 6),
        "elapsed_sec": round(elapsed, 3),
        "stop_reason": stop_reason,
        "trainable_parameters": model.trainable_parameter_count,
        "split_counts": {name: len(items) for name, items in split.items()},
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = train(
        args.cache_dir,
        args.output_dir,
        hidden_size=args.hidden_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        smoke=args.smoke,
        device=args.device,
    )
    print(f"NEXT COMMAND: python -m hackathon.egoflow.score --features {summary['cache_dir']} "
          f"--checkpoint {summary['checkpoint']} --output-dir {args.output_dir / 'scores'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
