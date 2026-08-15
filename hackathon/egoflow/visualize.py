#!/usr/bin/env python3
"""Render EgoFlow score timelines and an optional video+timeline MP4.

The timeline renderer intentionally uses only Python's standard library.  MP4
composition requires an input video plus ``ffmpeg``.  NPZ and Zarr inputs are
optional and require numpy/zarr respectively; JSON score visualization does not.
"""

from __future__ import annotations

import argparse
import binascii
import json
import math
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import tempfile
import zlib

try:
    from .results_schema import EVENT_LABELS, ScoreSeries, load_scores, write_summary
except ImportError:  # Direct ``python hackathon/egoflow/visualize.py`` invocation.
    from results_schema import EVENT_LABELS, ScoreSeries, load_scores, write_summary


COLORS = {
    "background": (13, 20, 32),
    "panel": (22, 31, 46),
    "grid": (59, 72, 91),
    "text": (235, 240, 248),
    "muted": (156, 168, 187),
    "global": (66, 211, 189),
    "local": (119, 221, 119),
    "velocity": (255, 184, 92),
    "residual": (184, 133, 255),
    "faster_bg": (24, 52, 56),
    "slower_bg": (51, 31, 58),
    "guide": (112, 105, 137),
    "productive": (34, 105, 82),
    "stall": (80, 90, 108),
    "regress": (132, 55, 65),
    "recover": (37, 104, 145),
    "hesitate": (133, 100, 38),
    "abandon": (126, 47, 87),
    "aborted_reach": (155, 91, 214),
    "interaction_deviation": (236, 70, 82),
    "complete": (55, 119, 61),
    "transition": (76, 65, 130),
    "other": (48, 56, 69),
    "human_gt": (244, 86, 190),
}


def progress_rate_residual(
    timestamps_sec: list[float],
    progress_velocity: list[float],
    *,
    window_sec: float = 5.0,
) -> tuple[list[float], list[float], float, float]:
    """Return ``(actual - expected) / MAD(actual)`` and plot metadata.

    Expected rate is a centered rolling median. The guide is the 85th
    percentile of absolute residuals and is only a visual reference—not a
    behavioral classification threshold.
    """

    values = [float(value) for value in progress_velocity]
    if not values:
        return [], [], 1.0, 1.0
    positive_steps = [
        later - earlier
        for earlier, later in zip(timestamps_sec, timestamps_sec[1:])
        if later > earlier
    ]
    dt = statistics.median(positive_steps) if positive_steps else 1.0
    radius = max(1, round(max(window_sec, dt) / (2.0 * dt)))
    baseline = []
    for index, value in enumerate(values):
        # A centered expectation is undefined at the boundary. Preserve the
        # observed endpoint rather than manufacturing start/end spikes.
        if index < radius or index >= len(values) - radius:
            baseline.append(value)
        else:
            baseline.append(statistics.median(values[index - radius:index + radius + 1]))
    center = statistics.median(values)
    mad = statistics.median(abs(value - center) for value in values)
    scale = max(mad, 1e-9)
    residual = [(value - expected) / scale for value, expected in zip(values, baseline)]
    ordered = sorted(abs(value) for value in residual)
    guide_index = min(len(ordered) - 1, max(0, math.ceil(0.85 * len(ordered)) - 1))
    guide = max(ordered[guide_index], 1e-3)
    return residual, baseline, scale, guide

DISPLAY_LABELS = {
    "productive": "POSITIVE RATE",
    "stall": "LOW RATE",
    "regress": "NEGATIVE RATE",
    "recover": "RECOVERY CANDIDATE",
    "hesitate": "HESITATION CANDIDATE",
    "abandon": "ABANDON CANDIDATE",
    "aborted_reach": "ABORTED REACH CANDIDATE",
    "interaction_deviation": "LEARNED MANIPULATION DEVIATION",
}

# Compact 5x7 glyphs. Unknown characters become a readable box.
_FONT_ROWS = {
    " ": ("00000",) * 7,
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ",": ("00000", "00000", "00000", "00000", "01100", "01100", "01000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "/": ("00001", "00010", "00100", "00100", "01000", "10000", "00000"),
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
}
_ENCODED = {
    "0": "01110100011001110101110011000101110", "1": "00100011000010000100001000010001110",
    "2": "01110100010000100010001000100011111", "3": "11110000010000101110000010000111110",
    "4": "00010001100101010010111110001000010", "5": "11111100001111000001000011000101110",
    "6": "00110010001000011110100011000101110", "7": "11111000010001000100010000100001000",
    "8": "01110100011000101110100011000101110", "9": "01110100011000101111000010001001100",
    "A": "01110100011000111111100011000110001", "B": "11110100011000111110100011000111110",
    "C": "01111100001000010000100001000001111", "D": "11110100011000110001100011000111110",
    "E": "11111100001000011110100001000011111", "F": "11111100001000011110100001000010000",
    "G": "01111100001000010111100011000101111", "H": "10001100011000111111100011000110001",
    "I": "01110001000010000100001000010001110", "J": "00111000100001000010100101001001100",
    "K": "10001100101010011000101001001010001", "L": "10000100001000010000100001000011111",
    "M": "10001110111010110101100011000110001", "N": "10001110011010110011100011000110001",
    "O": "01110100011000110001100011000101110", "P": "11110100011000111110100001000010000",
    "Q": "01110100011000110001101011001001101", "R": "11110100011000111110101001001010001",
    "S": "01111100001000001110000010000111110", "T": "11111001000010000100001000010000100",
    "U": "10001100011000110001100011000101110", "V": "10001100011000110001100010101000100",
    "W": "10001100011000110101101011101110001", "X": "10001100010101000100010101000110001",
    "Y": "10001100010101000100001000010000100", "Z": "11111000010001000100010001000011111",
}
for _char, _bits in _ENCODED.items():
    _FONT_ROWS[_char] = tuple(_bits[i:i + 5] for i in range(0, 35, 5))


class Canvas:
    def __init__(self, width: int, height: int, color: tuple[int, int, int]):
        self.width, self.height = width, height
        self.data = bytearray(color * (width * height))

    def copy(self) -> "Canvas":
        other = Canvas(self.width, self.height, (0, 0, 0))
        other.data[:] = self.data
        return other

    def pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.data[offset:offset + 3] = bytes(color)

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        x0, x1 = sorted((max(0, x0), min(self.width, x1)))
        y0, y1 = sorted((max(0, y0), min(self.height, y1)))
        row = bytes(color) * max(0, x1 - x0)
        for y in range(y0, y1):
            offset = (y * self.width + x0) * 3
            self.data[offset:offset + len(row)] = row

    def line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], width: int = 1) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            radius = max(0, width // 2)
            self.rect(x0 - radius, y0 - radius, x0 + radius + 1, y0 + radius + 1, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def text(self, x: int, y: int, value: str, color: tuple[int, int, int], scale: int = 2, max_chars: int | None = None) -> None:
        value = str(value).upper().replace("\n", " ")
        if max_chars and len(value) > max_chars:
            value = value[:max(1, max_chars - 3)] + "..."
        cursor = x
        for char in value:
            rows = _FONT_ROWS.get(char, _FONT_ROWS["?"])
            for row_index, row in enumerate(rows):
                for col_index, bit in enumerate(row):
                    if bit == "1":
                        self.rect(
                            cursor + col_index * scale,
                            y + row_index * scale,
                            cursor + (col_index + 1) * scale,
                            y + (row_index + 1) * scale,
                            color,
                        )
            cursor += 6 * scale

    def write_png(self, path: Path) -> None:
        def chunk(name: bytes, payload: bytes) -> bytes:
            return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", binascii.crc32(name + payload) & 0xFFFFFFFF)
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            raw.extend(self.data[y * stride:(y + 1) * stride])
        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(bytes(raw), 7))
        png += chunk(b"IEND", b"")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)


def _event_runs(series: ScoreSeries) -> list[tuple[float, float, str]]:
    runs = []
    if not series.timestamps_sec:
        return runs
    start = 0
    for i in range(1, len(series.event_labels) + 1):
        if i == len(series.event_labels) or series.event_labels[i] != series.event_labels[start]:
            end = series.timestamps_sec[i] if i < len(series.timestamps_sec) else series.duration_sec
            runs.append((series.timestamps_sec[start], max(end, series.timestamps_sec[start]), series.event_labels[start]))
            start = i
    return runs


def _top_event_runs(series: ScoreSeries, limit: int) -> list[tuple[float, float, str]]:
    """Select a small, source-aware set of intervals for presentation figures."""

    source_rank = {
        "learned_hand_dynamics_v2": 5,
        "video_hand_geometry_v1": 4,
        "hybrid_learned_progress_visual_dynamics": 3,
        "learned_progress_normalized": 2,
        "frozen_visual_dynamics": 1,
    }
    # Presentation figures omit ``productive`` and ``stall``: those names are
    # merely velocity bins and should not be shown as behavioral judgments.
    label_rank = {"interaction_deviation": 7, "aborted_reach": 6, "hesitate": 5, "abandon": 4, "recover": 3, "regress": 2}
    candidates = [
        event for event in series.events
        if str(event.get("label", "")).lower() in label_rank
    ]
    hand_candidates = [
        event for event in candidates
        if str(event.get("detector", "")) == "video_hand_geometry_v1"
    ]
    model_candidates = [event for event in candidates if event not in hand_candidates]
    interaction_candidates = [
        event for event in model_candidates
        if str(event.get("detector", "")) == "learned_hand_dynamics_v2"
    ]
    model_candidates = [event for event in model_candidates if event not in interaction_candidates]
    if interaction_candidates and limit > 0:
        model_candidates.append(max(interaction_candidates, key=lambda event: float(event.get("confidence", 0.0))))
    # Keep the experimental hand layer sparse. Prefer a hand redirection that
    # independently overlaps an existing review candidate, then confidence.
    if hand_candidates and limit > 0:
        def hand_key(event: dict[str, object]) -> tuple[int, float]:
            start, end = float(event.get("start_sec", 0.0)), float(event.get("end_sec", 0.0))
            corroborated = any(
                start <= float(other.get("end_sec", 0.0))
                and end >= float(other.get("start_sec", 0.0))
                for other in model_candidates
            )
            return int(corroborated), float(event.get("confidence", 0.0))
        selected_hand = max(hand_candidates, key=hand_key)
        model_candidates.append(selected_hand)
    candidates = model_candidates
    candidates.sort(
        key=lambda event: (
            label_rank.get(str(event.get("label", "")).lower(), 0),
            source_rank.get(str(event.get("detector", "")), 0),
            float(event.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    chosen: list[tuple[float, float, str]] = []
    for event in candidates:
        item = (
            float(event.get("start_sec", 0.0)),
            float(event.get("end_sec", 0.0)),
            str(event.get("label", "other")).lower(),
        )
        if item not in chosen:
            chosen.append(item)
        if len(chosen) >= max(0, limit):
            break
    return sorted(chosen)


def _timeline_canvas(
    series: ScoreSeries,
    width: int = 1280,
    height: int = 720,
    compact: bool = False,
    manual_events: list[dict[str, object]] | None = None,
    show_event_intervals: bool = False,
    max_display_events: int | None = None,
) -> Canvas:
    canvas = Canvas(width, height, COLORS["background"])
    left, right = 82, width - 35
    manual_events = list(manual_events or [])
    top = 86 if compact else 145 if manual_events else 112
    progress_bottom = int(height * 0.59)
    velocity_top = progress_bottom + 48
    velocity_bottom = max(
        velocity_top + 60,
        height - (220 if show_event_intervals and not compact else 72),
    )
    duration = max(series.duration_sec, 1e-6)

    canvas.text(35, 25, f"EGOFLOW / {series.episode_id}", COLORS["text"], 3 if width >= 1000 else 2, 55)
    if not compact:
        canvas.text(35, 55, series.task, COLORS["muted"], 2, max(20, (width - 70) // 12))
    same_progress_curve = bool(series.local_progress) and max(
        abs(local - global_value)
        for local, global_value in zip(series.local_progress, series.global_progress)
    ) < 1e-9
    if same_progress_curve:
        canvas.text(
            max(470, width - 455),
            57,
            "GLOBAL MATCHES LOCAL / ONE SEMANTIC STAGE",
            COLORS["human_gt"],
            1,
        )
    if series.synthetic:
        canvas.rect(width - 260, 20, width - 35, 57, (107, 61, 31))
        canvas.text(width - 247, 31, "SYNTHETIC DEMO", (255, 220, 164), 2)

    canvas.rect(left, top, right, progress_bottom, COLORS["panel"])
    residual_zero_y = (velocity_top + velocity_bottom) // 2
    canvas.rect(left, velocity_top, right, residual_zero_y, COLORS["slower_bg"])
    canvas.rect(left, residual_zero_y, right, velocity_bottom, COLORS["faster_bg"])

    def x_for(t: float) -> int:
        return left + round((right - left) * max(0.0, min(duration, t)) / duration)

    if manual_events and not compact:
        band_top, band_bottom = top - 36, top - 17
        canvas.text(left, band_top - 15, "HUMAN GROUND TRUTH", COLORS["human_gt"], 1)
        for event in manual_events:
            start = float(event.get("start_sec", 0.0))
            end = float(event.get("end_sec", start))
            canvas.rect(x_for(start), band_top, max(x_for(start) + 2, x_for(end)), band_bottom, COLORS["human_gt"])
            label = str(event.get("label", "event"))
            canvas.text(
                max(left, x_for(start)),
                band_top + 5,
                f"{start:.2f}-{end:.2f} {label}",
                COLORS["text"],
                1,
            )
    elif manual_events and compact:
        # A dedicated magenta strip keeps human labels visually separate from
        # model event colors in the animated demo.
        for event in manual_events:
            start = float(event.get("start_sec", 0.0))
            end = float(event.get("end_sec", start))
            canvas.rect(x_for(start), top, max(x_for(start) + 2, x_for(end)), top + 9, COLORS["human_gt"])

    displayed_runs = (
        _top_event_runs(series, max_display_events)
        if max_display_events is not None
        else _event_runs(series)
    )
    for start, end, label in displayed_runs:
        if label == "interaction_deviation":
            canvas.rect(
                x_for(start), velocity_top,
                max(x_for(start) + 2, x_for(end)), velocity_bottom,
                (72, 29, 38),
            )
    for start, end, label in displayed_runs:
        color = COLORS.get(label, COLORS["other"])
        canvas.rect(x_for(start), progress_bottom - 7, max(x_for(start) + 1, x_for(end)), progress_bottom, color)

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = progress_bottom - round((progress_bottom - top) * fraction)
        canvas.line(left, y, right, y, COLORS["grid"])
        canvas.text(10, y - 7, f"{fraction:.2f}", COLORS["muted"], 1)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + round((right - left) * fraction)
        canvas.line(x, top, x, progress_bottom, COLORS["grid"])
        canvas.line(x, velocity_top, x, velocity_bottom, COLORS["grid"])
        canvas.text(max(0, x - 16), height - 49, f"{duration * fraction:.1f}s", COLORS["muted"], 1)

    def plot(values: list[float], color: tuple[int, int, int], lo: float, hi: float, y0: int, y1: int, width_px: int = 3) -> None:
        if len(values) < 2:
            return
        points = []
        span = max(hi - lo, 1e-9)
        for t, value in zip(series.timestamps_sec, values):
            y = y1 - round((y1 - y0) * (max(lo, min(hi, value)) - lo) / span)
            points.append((x_for(t), y))
        for one, two in zip(points, points[1:]):
            canvas.line(*one, *two, color, width_px)

    plot(series.global_progress, COLORS["global"], 0, 1, top, progress_bottom)
    if not same_progress_curve:
        plot(series.local_progress, COLORS["local"], 0, 1, top, progress_bottom)
    raw_residual, _, residual_scale, residual_guide = progress_rate_residual(
        series.timestamps_sec,
        series.progress_velocity,
    )
    # Presentation convention: upward means slower/more suspicious.  The
    # underlying stored residual remains actual-minus-expected.
    residual = [-value for value in raw_residual]
    absolute_residual = sorted(abs(value) for value in residual)
    viewport_index = min(
        len(absolute_residual) - 1,
        max(0, math.ceil(0.98 * len(absolute_residual)) - 1),
    ) if absolute_residual else 0
    observed = absolute_residual[viewport_index] if absolute_residual else 1.0
    vmax = max(residual_guide * 1.25, observed, 0.25)
    zero_y = (velocity_top + velocity_bottom) // 2
    canvas.line(left, zero_y, right, zero_y, COLORS["grid"], 2)
    for sign in (-1.0, 1.0):
        guide_y = velocity_bottom - round(
            (velocity_bottom - velocity_top) * ((sign * residual_guide + vmax) / (2.0 * vmax))
        )
        for x in range(left, right, 10):
            canvas.line(x, guide_y, min(x + 5, right), guide_y, COLORS["guide"])
    if residual:
        for timestamp, value in zip(series.timestamps_sec, residual):
            y = velocity_bottom - round(
                (velocity_bottom - velocity_top) * ((max(-vmax, min(vmax, value)) + vmax) / (2.0 * vmax))
            )
            shade = (104, 54, 111) if value >= 0 else (48, 111, 103)
            canvas.line(x_for(timestamp), zero_y, x_for(timestamp), y, shade)
    plot(residual, COLORS["residual"], -vmax, vmax, velocity_top, velocity_bottom, 2)
    for start, end, label in displayed_runs:
        if label != "interaction_deviation":
            continue
        points = []
        for timestamp, value in zip(series.timestamps_sec, residual):
            if start <= timestamp <= end:
                y = velocity_bottom - round(
                    (velocity_bottom - velocity_top)
                    * ((max(-vmax, min(vmax, value)) + vmax) / (2.0 * vmax))
                )
                points.append((x_for(timestamp), y))
        for one, two in zip(points, points[1:]):
            canvas.line(*one, *two, COLORS["interaction_deviation"], 3)
    canvas.text(10, velocity_top - 3, f"+{vmax:.2f}", COLORS["muted"], 1)
    canvas.text(22, zero_y - 4, "0", COLORS["muted"], 1)
    canvas.text(10, velocity_bottom - 8, f"-{vmax:.2f}", COLORS["muted"], 1)

    for start, end, label in displayed_runs:
        marker_t = (start + end) / 2.0
        nearest = min(
            range(len(series.timestamps_sec)),
            key=lambda index: abs(series.timestamps_sec[index] - marker_t),
            default=0,
        )
        value = residual[nearest] if residual else 0.0
        marker_y = velocity_bottom - round(
            (velocity_bottom - velocity_top) * ((max(-vmax, min(vmax, value)) + vmax) / (2.0 * vmax))
        )
        marker_x = x_for(marker_t)
        color = COLORS.get(label, COLORS["human_gt"])
        canvas.line(marker_x, zero_y, marker_x, marker_y, color)
        canvas.line(marker_x - 5, marker_y - 5, marker_x + 5, marker_y + 5, color, 2)
        canvas.line(marker_x - 5, marker_y + 5, marker_x + 5, marker_y - 5, color, 2)

    canvas.rect(left, top - 25, left + 18, top - 8, COLORS["global"])
    canvas.text(left + 25, top - 24, "PROGRESS" if same_progress_curve else "GLOBAL", COLORS["text"], 1)
    legend_offset = 105
    if not same_progress_curve:
        canvas.rect(left + 95, top - 25, left + 113, top - 8, COLORS["local"])
        canvas.text(left + 120, top - 24, "LOCAL", COLORS["text"], 1)
        legend_offset = 180
    canvas.text(left, velocity_top - 25, "SLOWDOWN DEVIATION / EXPECTED - ACTUAL RATE", COLORS["text"], 1)
    canvas.text(right - 226, velocity_top + 8, "SLOWER THAN EXPECTED / REVIEW", COLORS["muted"], 1)
    canvas.text(right - 171, zero_y + 11, "FASTER THAN EXPECTED", COLORS["muted"], 1)
    if not compact:
        canvas.text(left + 8, velocity_bottom - 18, f"ROBUST SCALE / GUIDES +/-{residual_guide:.2f} MAD", COLORS["muted"], 1)

    legend_x = left + 300
    shown_labels = {label for _, _, label in displayed_runs}
    shown = [label for label in EVENT_LABELS if label in shown_labels and label != "other"]
    shown.extend(label for label in sorted(shown_labels) if label not in shown and label != "other")
    for index, label in enumerate(shown[:7]):
        x = legend_x + (index % 4) * 145
        y = top - 26 + (index // 4) * 19
        canvas.rect(x, y, x + 12, y + 12, COLORS[label])
        canvas.text(x + 17, y + 2, DISPLAY_LABELS.get(label, label), COLORS["muted"], 1)
    if series.synthetic and not compact:
        canvas.text(left, height - 24, "SYNTHETIC SMOKE TEST - NOT A MODEL RESULT OR EMPIRICAL CLAIM", (255, 190, 107), 1)
    if show_event_intervals and not compact:
        grouped: dict[tuple[float, float], dict[str, object]] = {}
        displayed_keys = {
            (round(start, 2), round(end, 2)) for start, end, _ in displayed_runs
        }
        for event in series.events:
            label = str(event.get("label", "other")).lower()
            if label in {"productive", "other"}:
                continue
            key = (round(float(event.get("start_sec", 0.0)), 2), round(float(event.get("end_sec", 0.0)), 2))
            if max_display_events is not None and key not in displayed_keys:
                continue
            values = grouped.setdefault(key, {"labels": [], "sources": [], "confidence": 0.0})
            labels = values["labels"]
            assert isinstance(labels, list)
            if label not in labels:
                labels.append(label)
            detector = str(event.get("detector", "unknown"))
            source = (
                "HYBRID"
                if detector.startswith("hybrid_")
                else "LEARNED"
                if detector == "learned_progress_normalized"
                else "AUX"
                if detector == "frozen_visual_dynamics"
                else "HAND EXPERIMENTAL"
                if detector == "video_hand_geometry_v1"
                else "LEARNED V2"
                if detector == "learned_hand_dynamics_v2"
                else "UNKNOWN"
            )
            if source not in values["sources"]:
                values["sources"].append(source)
            values["confidence"] = max(float(values["confidence"]), float(event.get("confidence", 0.0)))
        canvas.text(left, velocity_bottom + 25, "MODEL PREDICTED EVENT INTERVALS", COLORS["text"], 1)
        lines = [
            f"{start:.2f}-{end:.2f}  {'/'.join(DISPLAY_LABELS.get(label, label).upper() for label in values['labels'])}  {'/'.join(values['sources'])}  CONF {float(values['confidence']):.2f}"
            for (start, end), values in sorted(grouped.items())
        ]
        rows = max(1, math.ceil(len(lines) / 2))
        for index, line in enumerate(lines):
            column, row = divmod(index, rows)
            canvas.text(left + column * max(400, (right - left) // 2), velocity_bottom + 48 + row * 18, line, COLORS["muted"], 1, 58)
    return canvas


def render_timeline(
    series: ScoreSeries,
    output: str | Path,
    width: int = 1280,
    height: int = 720,
    *,
    manual_events: list[dict[str, object]] | None = None,
    show_event_intervals: bool = False,
    max_display_events: int | None = None,
) -> Path:
    target = Path(output)
    _timeline_canvas(
        series,
        width,
        height,
        manual_events=manual_events,
        show_event_intervals=show_event_intervals,
        max_display_events=max_display_events,
    ).write_png(target)
    return target


def load_manual_events(
    source: str | Path,
    episode_id: str,
    label_filter: str | None = None,
) -> list[dict[str, object]]:
    """Load the selected episode's compact JSONL ground-truth spans."""

    events: list[dict[str, object]] = []
    for line_number, line in enumerate(Path(source).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
        if str(event.get("episode_id", "")) != episode_id:
            continue
        if label_filter and str(event.get("label", "")).lower() != label_filter.lower():
            continue
        events.append(event)
    return events


def load_hand_events(source: str | Path) -> list[dict[str, object]]:
    """Load or recompute source-attributed experimental hand candidates."""

    payload = json.loads(Path(source).read_text())
    observations = payload.get("observations")
    if isinstance(observations, list):
        try:
            from .hand_interaction import detect_aborted_reaches
        except ImportError:
            from hand_interaction import detect_aborted_reaches
        return detect_aborted_reaches(
            observations,
            sample_fps=float(payload.get("sample_fps", 8.0)),
        )
    return [dict(event) for event in payload.get("candidates", payload.get("events", []))]


def render_scored_mp4(
    series: ScoreSeries,
    video: str | Path,
    output: str | Path,
    fps: float = 10.0,
    *,
    manual_events: list[dict[str, object]] | None = None,
    ffmpeg_path: str | Path | None = None,
) -> Path | None:
    """Stack the source video over an animated timeline using ffmpeg.

    Returns ``None`` rather than failing the PNG/summary workflow when video or
    ffmpeg is unavailable.
    """
    video_path, output_path = Path(video), Path(output)
    ffmpeg = str(ffmpeg_path) if ffmpeg_path else shutil.which("ffmpeg")
    if not video_path.exists():
        print(f"warning: video not found ({video_path}); skipping scored MP4")
        return None
    if not ffmpeg:
        print("warning: ffmpeg not found; skipping scored MP4")
        return None
    fps = max(1.0, min(30.0, fps))
    frame_count = max(1, math.ceil(series.duration_sec * fps))
    highlighted_runs = _top_event_runs(series, 5)
    highlighted_keys = {
        (round(start, 3), round(end, 3), label)
        for start, end, label in highlighted_runs
    }
    manual_events = list(manual_events or [])
    base = _timeline_canvas(
        series,
        width=960,
        height=300,
        compact=True,
        manual_events=manual_events,
        max_display_events=5,
    )
    left, right = 82, 925
    top, bottom = 86, 285
    raw_rate_residual, _, _, _ = progress_rate_residual(
        series.timestamps_sec,
        series.progress_velocity,
    )
    slowdown_deviation = [-value for value in raw_rate_residual]
    with tempfile.TemporaryDirectory(prefix="egoflow_overlay_") as temp_dir:
        temp = Path(temp_dir)
        for frame_index in range(frame_count):
            timestamp = min(series.duration_sec, frame_index / fps)
            state = series.value_at(timestamp)
            frame = base.copy()
            playhead_x = left + round((right - left) * timestamp / max(series.duration_sec, 1e-6))
            frame.line(playhead_x, top, playhead_x, bottom, (255, 255, 255), 2)
            progress = float(state["global_progress"])
            source_rank = {
                "learned_hand_dynamics_v2": 5,
                "video_hand_geometry_v1": 4,
                "hybrid_learned_progress_visual_dynamics": 3,
                "learned_progress_normalized": 2,
                "frozen_visual_dynamics": 1,
            }
            active = [
                event for event in series.events
                if str(event.get("label", "")) in {"interaction_deviation", "aborted_reach", "hesitate", "abandon", "recover", "regress"}
                and (
                    round(float(event.get("start_sec", 0.0)), 3),
                    round(float(event.get("end_sec", 0.0)), 3),
                    str(event.get("label", "")),
                ) in highlighted_keys
                and float(event.get("start_sec", 0.0)) <= timestamp <= float(event.get("end_sec", 0.0))
            ]
            active.sort(
                key=lambda event: (
                    source_rank.get(str(event.get("detector", "")), 0),
                    float(event.get("confidence", 0.0)),
                ),
                reverse=True,
            )
            active_event = active[0] if active else None
            detector = str(active_event.get("detector", "")) if active_event else "learned_progress_normalized"
            source = (
                "HYBRID" if detector == "hybrid_learned_progress_visual_dynamics"
                else "AUX" if detector == "frozen_visual_dynamics"
                else "HAND EXPERIMENTAL" if detector == "video_hand_geometry_v1"
                else "LEARNED V2" if detector == "learned_hand_dynamics_v2"
                else "LEARNED"
            )
            frame.rect(35, 62, 430, 79, COLORS["grid"])
            frame.rect(37, 64, 37 + round(391 * progress), 77, COLORS["global"])
            frame.text(35, 44, f"PROGRESS {progress * 100:05.1f}%", COLORS["text"], 1)
            in_human = next(
                (
                    event for event in manual_events
                    if float(event.get("start_sec", 0.0)) <= timestamp <= float(event.get("end_sec", 0.0))
                ),
                None,
            )
            if active_event:
                raw_label = str(active_event.get("label", "review"))
                status_label = (
                    "NEGATIVE RATE" if raw_label == "regress" else f"{raw_label.upper()}?"
                )
            else:
                state_index = int(state["index"])
                current_deviation = slowdown_deviation[state_index] if slowdown_deviation else 0.0
                status_label = f"DEVIATION {current_deviation:+.2f}"
            status_color = (
                COLORS.get(str(active_event.get("label")), COLORS["other"])
                if active_event
                else COLORS["panel"]
            )
            frame.rect(500, 18, 925, 80, status_color)
            frame.text(514, 25, f"{timestamp:05.1f}s {status_label}", COLORS["text"], 2, 31)
            detail = (
                f"{source} / CONF {float(active_event.get('confidence', 0.0)):.2f}"
                if active_event
                else "+ SLOWER / 0 EXPECTED / - FASTER"
            )
            frame.text(514, 51, detail, COLORS["text"], 1, 45)
            if in_human:
                frame.rect(500, 61, 925, 80, COLORS["human_gt"])
                frame.text(514, 65, f"HUMAN GT / {in_human.get('label', 'event')}", COLORS["text"], 1, 45)
            frame.write_png(temp / f"frame_{frame_index:06d}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
            "-framerate", str(fps), "-i", str(temp / "frame_%06d.png"),
            "-filter_complex", "[0:v]scale=960:-2:flags=lanczos[v];[v][1:v]vstack=inputs=2[out]",
            "-map", "[out]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "21", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(output_path),
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"warning: ffmpeg failed ({exc}); scored MP4 was not created")
            return None
    return output_path


def _video_from_zarr(zarr_path: Path, output: Path, fps: float, array_name: str | None) -> Path | None:
    """Best-effort conversion of an RGB Zarr array to MP4 for visualization."""
    try:
        import zarr  # type: ignore
    except ImportError:
        print("warning: Zarr video input requires the optional 'zarr' package")
        return None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("warning: ffmpeg not found; cannot convert Zarr frames")
        return None
    root = zarr.open(str(zarr_path), mode="r")
    candidates: list[tuple[str, object]] = []
    def visit(name: str, obj: object) -> None:
        shape = getattr(obj, "shape", ())
        if len(shape) == 4 and shape[-1] in (3, 4):
            candidates.append((name, obj))
    if hasattr(root, "visititems"):
        root.visititems(visit)
    if array_name:
        array = root[array_name]
    elif candidates:
        preferred = [item for item in candidates if any(token in item[0].lower() for token in ("rgb", "image", "camera"))]
        array_name, array = (preferred or candidates)[0]
        print(f"using Zarr RGB array: {array_name}")
    else:
        print(f"warning: no [T,H,W,3/4] image array found in {zarr_path}")
        return None
    shape = array.shape
    height, width, channels = int(shape[1]), int(shape[2]), int(shape[3])
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
        "-pixel_format", "rgba" if channels == 4 else "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index in range(shape[0]):
            frame = array[index]
            process.stdin.write(frame.tobytes(order="C"))
        process.stdin.close()
        if process.wait() != 0:
            print("warning: ffmpeg failed while converting Zarr frames")
            return None
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help="EgoFlow score JSON or NPZ")
    parser.add_argument("--timeline", type=Path, default=Path("results/example_timeline.png"))
    parser.add_argument("--summary", type=Path, default=Path("results/episode_summary.json"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="optional source video for scored MP4")
    source.add_argument("--zarr", type=Path, help="optional episode Zarr containing an RGB frame array")
    parser.add_argument("--zarr-array", help="explicit Zarr image array path")
    parser.add_argument("--mp4", type=Path, default=Path("results/example_scored.mp4"))
    parser.add_argument("--fps", type=float, default=10.0, help="overlay/Zarr video frame rate")
    parser.add_argument("--ffmpeg-path", type=Path, help="optional explicit ffmpeg executable")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--manual-labels", type=Path, help="optional human-event JSONL overlay")
    parser.add_argument("--manual-event-label", help="only overlay this human label")
    parser.add_argument("--hand-events", type=Path, help="experimental hand-geometry JSON overlay")
    parser.add_argument("--show-event-intervals", action="store_true", help="print model intervals below the plot")
    parser.add_argument("--max-display-events", type=int, help="show only the top N source-aware model intervals")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    series = load_scores(args.scores)
    if args.hand_events:
        series.events.extend(load_hand_events(args.hand_events))
    manual_events = (
        load_manual_events(args.manual_labels, series.episode_id, args.manual_event_label)
        if args.manual_labels
        else []
    )
    timeline = render_timeline(
        series,
        args.timeline,
        args.width,
        args.height,
        manual_events=manual_events,
        show_event_intervals=args.show_event_intervals,
        max_display_events=args.max_display_events,
    )
    summary = write_summary(series, args.summary)
    print(f"timeline: {timeline}")
    print(f"summary: {summary}")
    video = args.video
    temp_context = None
    if args.zarr:
        temp_context = tempfile.TemporaryDirectory(prefix="egoflow_zarr_")
        video = _video_from_zarr(args.zarr, Path(temp_context.name) / "source.mp4", args.fps, args.zarr_array)
    try:
        if video:
            result = render_scored_mp4(
                series,
                video,
                args.mp4,
                args.fps,
                manual_events=manual_events,
                ffmpeg_path=args.ffmpeg_path,
            )
            if result:
                print(f"scored video: {result}")
        else:
            print("video: not supplied; PNG and JSON outputs are complete")
    finally:
        if temp_context:
            temp_context.cleanup()
    if series.synthetic:
        print("note: outputs are explicitly marked synthetic and make no empirical/model claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
