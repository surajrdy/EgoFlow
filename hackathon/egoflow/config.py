"""Central defaults and non-negotiable hackathon resource caps."""

from __future__ import annotations

from dataclasses import dataclass


SCHEMA_VERSION = 1
FEATURE_GPU = "L40S"
TRAINING_GPU = "H100"
ALLOWED_LABELS = frozenset(
    {
        "productive",
        "stall",
        "regress",
        "recover",
        "hesitate",
        "abandon",
        "complete",
        "other",
    }
)


@dataclass(frozen=True)
class HardCaps:
    """Caps are checked in code so a malformed manifest cannot create a huge job."""

    max_episodes: int = 60
    max_feature_workers: int = 8
    max_scoring_workers: int = 12
    max_training_jobs: int = 2
    max_frames_per_episode: int = 1_200  # five minutes at 4 Hz
    feature_timeout_seconds: int = 20 * 60
    training_timeout_seconds: int = 25 * 60
    max_training_steps: int = 2_500
    max_estimated_cost_usd: float = 250.0
    absolute_cost_ceiling_usd: float = 275.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_feature_workers <= 8:
            raise ValueError("max_feature_workers must be in [1, 8]")
        if self.training_timeout_seconds > 25 * 60:
            raise ValueError("training jobs may not exceed 25 minutes")
        if self.max_estimated_cost_usd >= self.absolute_cost_ceiling_usd:
            raise ValueError("estimated-cost guard must be below the absolute ceiling")


@dataclass(frozen=True)
class ExtractionConfig:
    sample_fps: float = 4.0
    source_fps: float = 30.0
    gap_inherit_seconds: float = 1.5
    dino_model: str = "facebook/dinov2-small"
    dino_batch_size: int = 32
    fallback_text_dim: int = 128
    task_description: str = "Perform the demonstrated task"
    caps: HardCaps = HardCaps()

    def __post_init__(self) -> None:
        if not 0 < self.sample_fps <= 4.0:
            raise ValueError("sample_fps must be in (0, 4]")
        if self.source_fps <= 0:
            raise ValueError("source_fps must be positive")
        if self.gap_inherit_seconds < 0:
            raise ValueError("gap_inherit_seconds cannot be negative")
        if not 1 <= self.dino_batch_size <= 128:
            raise ValueError("dino_batch_size must be in [1, 128]")
