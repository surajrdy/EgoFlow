"""The deliberately small EgoFlow temporal progress head.

Only cached visual and annotation embeddings enter this module.  In particular,
timestamps, frame indices, and normalized episode time are not model inputs.
The upstream DINO/Qwen encoders remain frozen because their outputs are loaded
from disk rather than instantiated here.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import Tensor, nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
except ImportError:  # Allows event/evaluation utilities to work without torch.
    torch = None
    Tensor = Any
    nn = None


if nn is not None:

    class ProgressModel(nn.Module):
        """Frozen-feature projections + a two-layer bidirectional GRU.

        Args:
            visual_dim: Width of a mean-pooled DINO feature.
            language_dim: Width of a cached annotation embedding.
            hidden_size: GRU width. The hackathon sweep is intentionally limited
                to 128 and 256.
            projection_dim: Width of each modality projection.
        """

        ALLOWED_HIDDEN_SIZES = (128, 256)

        def __init__(
            self,
            visual_dim: int,
            language_dim: int,
            hidden_size: int = 128,
            projection_dim: int = 128,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if hidden_size not in self.ALLOWED_HIDDEN_SIZES:
                raise ValueError(f"hidden_size must be one of {self.ALLOWED_HIDDEN_SIZES}")
            if min(visual_dim, language_dim, projection_dim) < 1:
                raise ValueError("embedding dimensions must be positive")

            self.visual_dim = int(visual_dim)
            self.language_dim = int(language_dim)
            self.hidden_size = int(hidden_size)
            self.projection_dim = int(projection_dim)

            self.visual_projection = nn.Sequential(
                nn.LayerNorm(visual_dim),
                nn.Linear(visual_dim, projection_dim),
                nn.GELU(),
            )
            self.language_projection = nn.Sequential(
                nn.LayerNorm(language_dim),
                nn.Linear(language_dim, projection_dim),
                nn.GELU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(projection_dim * 2, projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.temporal = nn.GRU(
                input_size=projection_dim,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            output_dim = hidden_size * 2
            self.progress_head = nn.Linear(output_dim, 1)

        def forward(
            self,
            visual_embeddings: Tensor,
            language_embeddings: Tensor,
            lengths: Tensor | None = None,
        ) -> dict[str, Tensor]:
            """Return frame-level progress and commitment probabilities.

            Shapes are ``[batch, frames, dim]``. ``lengths`` only describes
            padding; it is never concatenated into the learned representation.
            """
            if visual_embeddings.ndim != 3 or language_embeddings.ndim != 3:
                raise ValueError("embeddings must have shape [batch, frames, dim]")
            if visual_embeddings.shape[:2] != language_embeddings.shape[:2]:
                raise ValueError("visual and language batch/frame axes must match")

            visual = self.visual_projection(visual_embeddings.float())
            language = self.language_projection(language_embeddings.float())
            fused = self.fusion(torch.cat((visual, language), dim=-1))

            if lengths is not None:
                packed = pack_padded_sequence(
                    fused,
                    lengths.detach().cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed_output, _ = self.temporal(packed)
                temporal, _ = pad_packed_sequence(
                    packed_output,
                    batch_first=True,
                    total_length=fused.shape[1],
                )
            else:
                temporal, _ = self.temporal(fused)

            progress_logits = self.progress_head(temporal).squeeze(-1)
            return {
                "local_progress": torch.sigmoid(progress_logits),
                "progress_logits": progress_logits,
            }

        def config(self) -> dict[str, int | float]:
            return {
                "visual_dim": self.visual_dim,
                "language_dim": self.language_dim,
                "hidden_size": self.hidden_size,
                "projection_dim": self.projection_dim,
            }

        @property
        def trainable_parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

else:

    class ProgressModel:  # type: ignore[no-redef]
        """Helpful error when only the dependency-light utilities are installed."""

        ALLOWED_HIDDEN_SIZES = (128, 256)

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("ProgressModel requires PyTorch; install torch before training")
