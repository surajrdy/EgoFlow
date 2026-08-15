#!/usr/bin/env python3
"""Extract one local Zarr episode into the model-lane NPZ contract."""

from __future__ import annotations

import argparse

from hackathon.egoflow.cache import save_feature_cache
from hackathon.egoflow.config import ExtractionConfig
from hackathon.egoflow.features import extract_episode_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="local Zarr episode directory")
    parser.add_argument("output", help="output .npz cache")
    parser.add_argument("--episode-id")
    parser.add_argument("--task")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    args = parser.parse_args()
    config = ExtractionConfig(source_fps=args.source_fps, sample_fps=args.sample_fps)
    features = extract_episode_features(
        args.source, config=config, episode_id=args.episode_id, task=args.task
    )
    output = save_feature_cache(features, args.output)
    print(f"cached {len(features.frame_indices)} frames -> {output}")


if __name__ == "__main__":
    main()
