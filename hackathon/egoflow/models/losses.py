"""Objectives for local progress without using absolute time as an input."""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except ImportError:
    torch = None
    Tensor = Any


def temporal_ranking_loss(
    progress: Tensor,
    stage_ids: Tensor,
    mask: Tensor | None = None,
    *,
    min_separation: int = 2,
    temperature: float = 0.1,
    max_pairs_per_stage: int = 256,
) -> tuple[Tensor, dict[str, float]]:
    """Rank meaningfully separated frames within the same semantic stage.

    Pair construction uses position only to define the self-supervised target.
    Position is never passed into the model. Stage id ``-1`` denotes padding.
    """
    if torch is None:
        raise RuntimeError("temporal_ranking_loss requires PyTorch")
    if progress.ndim != 2 or stage_ids.shape != progress.shape:
        raise ValueError("progress and stage_ids must both be [batch, frames]")
    if min_separation < 1 or temperature <= 0:
        raise ValueError("min_separation and temperature must be positive")
    valid = stage_ids.ge(0) if mask is None else mask.bool() & stage_ids.ge(0)
    pair_losses: list[Tensor] = []
    correct = 0
    pair_count = 0

    for batch_index in range(progress.shape[0]):
        for stage in torch.unique(stage_ids[batch_index][valid[batch_index]]):
            positions = torch.nonzero(
                valid[batch_index] & stage_ids[batch_index].eq(stage), as_tuple=False
            ).flatten()
            if positions.numel() < 2:
                continue
            earlier, later = torch.triu_indices(
                positions.numel(), positions.numel(), offset=min_separation,
                device=positions.device,
            )
            if earlier.numel() == 0:
                continue
            if earlier.numel() > max_pairs_per_stage:
                # torch.randperm makes sampling reproducible under the run seed.
                chosen = torch.randperm(earlier.numel(), device=positions.device)[:max_pairs_per_stage]
                earlier, later = earlier[chosen], later[chosen]
            differences = (
                progress[batch_index, positions[later]]
                - progress[batch_index, positions[earlier]]
            )
            pair_losses.append(F.softplus(-differences / temperature))
            correct += int(differences.detach().gt(0).sum().item())
            pair_count += int(differences.numel())

    if not pair_losses:
        # Preserve a valid autograd graph for tiny/degenerate smoke batches.
        loss = progress.sum() * 0.0
    else:
        loss = torch.cat(pair_losses).mean()
    metrics = {
        "ranking_accuracy": correct / pair_count if pair_count else 0.0,
        "ranking_pairs": float(pair_count),
    }
    return loss, metrics


def stage_smoothness_loss(progress: Tensor, stage_ids: Tensor, mask: Tensor | None = None) -> Tensor:
    """Small regularizer over adjacent predictions in the same stage."""
    if torch is None:
        raise RuntimeError("stage_smoothness_loss requires PyTorch")
    valid = stage_ids.ge(0) if mask is None else mask.bool() & stage_ids.ge(0)
    adjacent = valid[:, 1:] & valid[:, :-1] & stage_ids[:, 1:].eq(stage_ids[:, :-1])
    if not bool(adjacent.any()):
        return progress.sum() * 0.0
    differences = progress[:, 1:] - progress[:, :-1]
    return differences[adjacent].square().mean()


def endpoint_anchor_loss(progress: Tensor, stage_ids: Tensor, mask: Tensor | None = None) -> Tensor:
    """Weakly anchor stage starts/ends while ranking remains the main objective."""
    if torch is None:
        raise RuntimeError("endpoint_anchor_loss requires PyTorch")
    valid = stage_ids.ge(0) if mask is None else mask.bool() & stage_ids.ge(0)
    anchors: list[Tensor] = []
    targets: list[Tensor] = []
    for batch_index in range(progress.shape[0]):
        for stage in torch.unique(stage_ids[batch_index][valid[batch_index]]):
            positions = torch.nonzero(
                valid[batch_index] & stage_ids[batch_index].eq(stage), as_tuple=False
            ).flatten()
            if positions.numel() >= 2:
                anchors.extend((progress[batch_index, positions[0]], progress[batch_index, positions[-1]]))
                targets.extend((progress.new_tensor(0.1), progress.new_tensor(0.9)))
    if not anchors:
        return progress.sum() * 0.0
    return F.binary_cross_entropy(torch.stack(anchors), torch.stack(targets))
