"""Small self-supervised GRU for expected hand-manipulation dynamics."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


def _tracks(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in payload["observations"]:
        result.setdefault(str(row["hand"]), []).append(row)
    for rows in result.values():
        rows.sort(key=lambda row: float(row["timestamp_sec"]))
    return result


def _examples(payloads: list[dict[str, Any]], window: int):
    xs, ys, meta = [], [], []
    for payload in payloads:
        for hand, rows in _tracks(payload).items():
            features = []
            for index, row in enumerate(rows):
                previous = rows[max(0, index - 1)]
                dt = max(1e-3, float(row["timestamp_sec"]) - float(previous["timestamp_sec"]))
                features.append([
                    float(row["x"]), float(row["y"]), min(3.0, float(row.get("aperture", 1.0))),
                    (float(row["x"]) - float(previous["x"])) / dt,
                    (float(row["y"]) - float(previous["y"])) / dt,
                ])
            for index in range(window, len(rows)):
                span = float(rows[index]["timestamp_sec"]) - float(rows[index - window]["timestamp_sec"])
                if span > (window / float(payload.get("sample_fps", 8.0))) * 1.8:
                    continue
                xs.append(features[index - window:index])
                ys.append(features[index][:3])
                meta.append((payload["episode_id"], hand, float(rows[index]["timestamp_sec"])))
    return xs, ys, meta


def train_expected_dynamics(
    hand_dir: str | Path,
    train_episode_ids: list[str],
    output_dir: str | Path,
    *,
    hidden_size: int = 32,
    window: int = 8,
    max_steps: int = 400,
    seed: int = 17,
    device: str | None = None,
) -> dict[str, Any]:
    """Train a next-state GRU; frozen progress is context, not a target label."""

    import torch
    from torch import nn

    random.seed(seed)
    torch.manual_seed(seed)
    directory = Path(hand_dir)
    payloads = [json.loads((directory / f"{episode_id}.json").read_text()) for episode_id in train_episode_ids]
    xs, ys, _ = _examples(payloads, window)
    x, y = torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)
    mean, std = x.reshape(-1, 5).mean(0), x.reshape(-1, 5).std(0).clamp_min(1e-4)
    target_mean, target_std = y.mean(0), y.std(0).clamp_min(1e-4)
    x, y = (x - mean) / std, (y - target_mean) / target_std
    run_device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(5, hidden_size, num_layers=2, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 3))
        def forward(self, values):
            return self.head(self.gru(values)[0][:, -1])

    model = Predictor().to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch_size = min(512, len(x))
    losses = []
    model.train()
    for step in range(max_steps):
        indices = torch.randint(len(x), (batch_size,))
        prediction = model(x[indices].to(run_device))
        loss = nn.functional.smooth_l1_loss(prediction, y[indices].to(run_device))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 25 == 0 or step == max_steps - 1:
            losses.append(round(float(loss.detach().cpu()), 6))
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "interaction-gru.pt"
    torch.save({
        "state_dict": model.state_dict(), "hidden_size": hidden_size, "window": window,
        "mean": mean, "std": std, "target_mean": target_mean, "target_std": target_std,
    }, checkpoint)
    result = {"checkpoint": str(checkpoint), "hidden_size": hidden_size, "window": window, "max_steps": max_steps, "examples": len(x), "loss_curve": losses, "final_loss": losses[-1]}
    (output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def score_expected_dynamics(hand_json: str | Path, checkpoint: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    """Score unexpected short-horizon hand transitions and return sparse peaks."""

    import torch
    from torch import nn
    payload = json.loads(Path(hand_json).read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    hidden_size, window = int(saved["hidden_size"]), int(saved["window"])

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__(); self.gru = nn.GRU(5, hidden_size, num_layers=2, batch_first=True); self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 3))
        def forward(self, values): return self.head(self.gru(values)[0][:, -1])

    model = Predictor(); model.load_state_dict(saved["state_dict"]); model.eval()
    xs, ys, meta = _examples([payload], window)
    x = (torch.tensor(xs, dtype=torch.float32) - saved["mean"]) / saved["std"]
    y = (torch.tensor(ys, dtype=torch.float32) - saved["target_mean"]) / saved["target_std"]
    with torch.no_grad(): errors = ((model(x) - y) ** 2).mean(1).sqrt().tolist()
    median = statistics.median(errors); mad = max(statistics.median(abs(value - median) for value in errors), 1e-6)
    rows = [{"timestamp_sec": round(t, 3), "hand": hand, "surprise": round((error - median) / mad, 4)} for (_, hand, t), error in zip(meta, errors)]
    peaks = []
    for index in range(1, len(rows) - 1):
        value = rows[index]["surprise"]
        if value >= 6.0 and value >= rows[index - 1]["surprise"] and value >= rows[index + 1]["surprise"]:
            candidate = {"start_sec": round(rows[index]["timestamp_sec"] - 0.5, 3), "end_sec": round(rows[index]["timestamp_sec"] + 0.5, 3), "label": "interaction_deviation", "confidence": round(1 - math.exp(-value / 8), 4), "detector": "learned_hand_dynamics_v2", "surprise_mad": value, "hand": rows[index]["hand"]}
            if not peaks or candidate["start_sec"] > peaks[-1]["end_sec"] + 0.5: peaks.append(candidate)
            elif candidate["confidence"] > peaks[-1]["confidence"]: peaks[-1] = candidate
    merged = []
    for event in sorted(peaks, key=lambda item: item["start_sec"]):
        if merged and event["start_sec"] <= merged[-1]["end_sec"] + 0.25:
            previous = merged[-1]
            previous["end_sec"] = max(previous["end_sec"], event["end_sec"])
            previous["confidence"] = max(previous["confidence"], event["confidence"])
            previous["surprise_mad"] = max(previous["surprise_mad"], event["surprise_mad"])
            previous["hand"] = previous["hand"] if previous["hand"] == event["hand"] else "both"
        else:
            merged.append(dict(event))
    merged = sorted(merged, key=lambda event: event["confidence"], reverse=True)[:8]
    result = {"episode_id": payload["episode_id"], "schema": "egoflow.interaction_dynamics.v2", "events": sorted(merged, key=lambda event: event["start_sec"]), "trace": rows}
    if output:
        target = Path(output); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(result, indent=2) + "\n")
    return result
