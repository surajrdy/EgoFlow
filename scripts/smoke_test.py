#!/usr/bin/env python3
"""Run a disposable, credential-free EgoFlow end-to-end smoke test."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hackathon.egoflow.dataset import make_synthetic_cache  # noqa: E402
from hackathon.egoflow.score import score_episode  # noqa: E402
from hackathon.egoflow.train import train  # noqa: E402


def main() -> int:
    with TemporaryDirectory(prefix="egoflow-smoke-") as scratch:
        root = Path(scratch)
        cache_dir = root / "features"
        run_dir = root / "run"
        paths = make_synthetic_cache(cache_dir, episodes=4, seed=7)
        summary = train(
            cache_dir,
            run_dir,
            hidden_size=128,
            max_steps=3,
            smoke=True,
            device="cpu",
        )
        result = score_episode(paths[0], run_dir / "best.pt", device="cpu")

        frames = result["frames"]
        progress = [float(frame["local_progress"]) for frame in frames]
        assert len(frames) == 48, "scorer must preserve the synthetic frame count"
        assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in progress)
        assert result["synthetic"] is True
        assert summary["steps"] == 3

        print("PASS: disposable synthetic train → score pipeline")
        print(
            f"frames={len(frames)} progress_range="
            f"[{min(progress):.3f}, {max(progress):.3f}] "
            f"completion={result['completion_confidence']:.3f}"
        )
        print("note: synthetic smoke output is not an empirical result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
