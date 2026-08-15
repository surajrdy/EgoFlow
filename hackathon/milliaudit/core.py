"""Solver-agnostic 1--3 mm perturbation audit.

The code deliberately separates the metric harness from EgoVerse's solver so it
can be unit-tested without MuJoCo.  A solver must provide ``ik`` and ``fk`` with
the same interface as ``EvaMinkKinematicsSolver``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np


SCHEMA_VERSION = "milliaudit.kinematic_fragility.v1"
PERTURBATION_MM = (1, 2, 3)
AXES = ("x", "y", "z")


class Solver(Protocol):
    def ik(
        self, pos_xyz: np.ndarray, rot_mat: np.ndarray, cur_jnts: np.ndarray
    ) -> np.ndarray | None: ...

    def fk(self, jnts: np.ndarray) -> tuple[np.ndarray, Any]: ...


@dataclass(frozen=True)
class Trajectory:
    """Cartesian targets and warm-start joints for a sampled trajectory."""

    timestamps: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    seed_joints: np.ndarray
    trajectory_id: str
    provenance: str
    empirical: bool

    def validate(self) -> None:
        n = len(self.timestamps)
        if n < 2:
            raise ValueError("trajectory must contain at least two samples")
        if self.positions.shape != (n, 3):
            raise ValueError(f"positions must have shape ({n}, 3)")
        if self.rotations.shape != (n, 3, 3):
            raise ValueError(f"rotations must have shape ({n}, 3, 3)")
        if self.seed_joints.ndim != 2 or self.seed_joints.shape[0] != n:
            raise ValueError("seed_joints must have shape (samples, joints)")
        arrays = (self.timestamps, self.positions, self.rotations, self.seed_joints)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("trajectory arrays must be finite")
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        should_be_identity = self.rotations @ np.swapaxes(self.rotations, 1, 2)
        identity = np.broadcast_to(np.eye(3), should_be_identity.shape)
        if not np.allclose(should_be_identity, identity, atol=1e-4):
            raise ValueError("rotations must be orthonormal")


def load_trajectory(path: Path) -> Trajectory:
    """Load the documented NPZ contract without enabling pickle."""

    with np.load(path, allow_pickle=False) as data:
        required = {"timestamps", "positions", "rotations", "seed_joints"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"missing trajectory arrays: {sorted(missing)}")
        metadata: dict[str, Any] = {}
        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"].item()))
        trajectory = Trajectory(
            timestamps=np.asarray(data["timestamps"], dtype=np.float64),
            positions=np.asarray(data["positions"], dtype=np.float64),
            rotations=np.asarray(data["rotations"], dtype=np.float64),
            seed_joints=np.asarray(data["seed_joints"], dtype=np.float64),
            trajectory_id=str(metadata.get("trajectory_id", path.stem)),
            provenance=str(metadata.get("provenance", str(path))),
            empirical=bool(metadata.get("empirical", False)),
        )
    trajectory.validate()
    return trajectory


def _rotation_matrix(rotation: Any) -> np.ndarray:
    if hasattr(rotation, "as_matrix"):
        return np.asarray(rotation.as_matrix(), dtype=np.float64)
    return np.asarray(rotation, dtype=np.float64)


def _joint_margin(
    joints: np.ndarray,
    joint_lower: np.ndarray | None,
    joint_upper: np.ndarray | None,
) -> float | None:
    if joint_lower is None or joint_upper is None:
        return None
    return float(np.min(np.minimum(joints - joint_lower, joint_upper - joints)))


def _trial(
    solver: Solver,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    seed: np.ndarray,
    *,
    success_tolerance_m: float,
    nominal_solution: np.ndarray | None,
    nominal_margin: float | None,
    prior_branch_solution: np.ndarray | None,
    nominal_jump: float,
    joint_lower: np.ndarray | None,
    joint_upper: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    solution_raw = solver.ik(target_pos, target_rot, seed)
    if solution_raw is None:
        return {
            "success": False,
            "residual_m": None,
            "joint_delta_rad": None,
            "joint_limit_margin_rad": None,
            "joint_limit_margin_change_rad": None,
            "temporal_jump_rad": None,
            "fragility_score": 1.0,
        }, None

    solution = np.asarray(solution_raw, dtype=np.float64)
    if not np.isfinite(solution).all():
        return {
            "success": False,
            "residual_m": None,
            "joint_delta_rad": None,
            "joint_limit_margin_rad": None,
            "joint_limit_margin_change_rad": None,
            "temporal_jump_rad": None,
            "fragility_score": 1.0,
        }, None

    achieved_pos, _ = solver.fk(solution)
    residual = float(np.linalg.norm(np.asarray(achieved_pos) - target_pos))
    success = bool(residual <= success_tolerance_m)
    joint_delta = (
        float(np.linalg.norm(solution - nominal_solution))
        if nominal_solution is not None
        else None
    )
    margin = _joint_margin(solution, joint_lower, joint_upper)
    margin_change = (
        float(margin - nominal_margin)
        if margin is not None and nominal_margin is not None
        else None
    )
    jump = (
        float(np.linalg.norm(solution - prior_branch_solution))
        if prior_branch_solution is not None
        else None
    )

    if not success:
        fragility = 1.0
    else:
        residual_component = min(residual / success_tolerance_m, 1.0) * 0.30
        joint_component = min((joint_delta or 0.0) / 0.10, 1.0) * 0.35
        margin_component = min(max(-(margin_change or 0.0), 0.0) / 0.10, 1.0) * 0.15
        excess_jump = max((jump or 0.0) - nominal_jump, 0.0)
        jump_component = min(excess_jump / 0.15, 1.0) * 0.20
        fragility = residual_component + joint_component + margin_component + jump_component
    return {
        "success": success,
        "residual_m": residual,
        "joint_delta_rad": joint_delta,
        "joint_limit_margin_rad": margin,
        "joint_limit_margin_change_rad": margin_change,
        "temporal_jump_rad": jump,
        "fragility_score": float(min(max(fragility, 0.0), 1.0)),
    }, solution


def audit_trajectory(
    solver: Solver,
    trajectory: Trajectory,
    *,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
    success_tolerance_m: float = 0.002,
) -> dict[str, Any]:
    """Audit nominal and +/- axis perturbations at 1, 2, and 3 mm."""

    trajectory.validate()
    if success_tolerance_m <= 0:
        raise ValueError("success_tolerance_m must be positive")
    joint_count = trajectory.seed_joints.shape[1]
    if joint_lower is not None:
        joint_lower = np.asarray(joint_lower, dtype=np.float64)
        joint_upper = np.asarray(joint_upper, dtype=np.float64)
        if joint_lower.shape != (joint_count,) or joint_upper.shape != (joint_count,):
            raise ValueError("joint limits must match seed_joints width")

    prior_nominal: np.ndarray | None = None
    prior_branches: dict[str, np.ndarray] = {}
    frames: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []

    for frame_index, timestamp in enumerate(trajectory.timestamps):
        target_pos = trajectory.positions[frame_index]
        target_rot = trajectory.rotations[frame_index]
        seed = trajectory.seed_joints[frame_index]
        nominal, nominal_solution = _trial(
            solver,
            target_pos,
            target_rot,
            seed,
            success_tolerance_m=success_tolerance_m,
            nominal_solution=None,
            nominal_margin=None,
            prior_branch_solution=prior_nominal,
            nominal_jump=0.0,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        )
        nominal_margin = nominal["joint_limit_margin_rad"]
        nominal_jump = float(nominal["temporal_jump_rad"] or 0.0)
        if nominal_solution is not None:
            prior_nominal = nominal_solution

        perturbations: list[dict[str, Any]] = []
        magnitude_scores: dict[str, list[float]] = {
            str(mm): [] for mm in PERTURBATION_MM
        }
        for mm in PERTURBATION_MM:
            for axis_index, axis in enumerate(AXES):
                for sign in (-1, 1):
                    branch = f"{sign * mm:+d}mm_{axis}"
                    offset = np.zeros(3, dtype=np.float64)
                    offset[axis_index] = sign * mm / 1000.0
                    trial, solution = _trial(
                        solver,
                        target_pos + offset,
                        target_rot,
                        nominal_solution if nominal_solution is not None else seed,
                        success_tolerance_m=success_tolerance_m,
                        nominal_solution=nominal_solution,
                        nominal_margin=nominal_margin,
                        prior_branch_solution=prior_branches.get(branch),
                        nominal_jump=nominal_jump,
                        joint_lower=joint_lower,
                        joint_upper=joint_upper,
                    )
                    trial.update(
                        {
                            "magnitude_mm": mm,
                            "axis": axis,
                            "sign": sign,
                            "branch": branch,
                        }
                    )
                    if solution is not None:
                        prior_branches[branch] = solution
                    perturbations.append(trial)
                    all_trials.append(trial)
                    magnitude_scores[str(mm)].append(float(trial["fragility_score"]))

        frames.append(
            {
                "frame_index": frame_index,
                "time_sec": float(timestamp),
                "nominal": nominal,
                "fragility_score": max(
                    (float(trial["fragility_score"]) for trial in perturbations),
                    default=0.0,
                ),
                "fragility_by_magnitude": {
                    mm: max(scores, default=0.0)
                    for mm, scores in magnitude_scores.items()
                },
                "perturbations": perturbations,
            }
        )

    critical = sorted(frames, key=lambda row: row["fragility_score"], reverse=True)[:5]
    summary_by_mm: dict[str, Any] = {}
    for mm in PERTURBATION_MM:
        subset = [trial for trial in all_trials if trial["magnitude_mm"] == mm]
        success_count = sum(bool(trial["success"]) for trial in subset)
        residuals = [
            float(trial["residual_m"])
            for trial in subset
            if trial["residual_m"] is not None
        ]
        deltas = [
            float(trial["joint_delta_rad"])
            for trial in subset
            if trial["joint_delta_rad"] is not None
        ]
        summary_by_mm[str(mm)] = {
            "trials": len(subset),
            "successes": success_count,
            "success_rate": success_count / len(subset) if subset else math.nan,
            "max_residual_m": max(residuals, default=None),
            "max_joint_delta_rad": max(deltas, default=None),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "result_type": "kinematic_fragility",
        "trajectory_id": trajectory.trajectory_id,
        "trajectory_provenance": trajectory.provenance,
        "empirical_trajectory": trajectory.empirical,
        "claim_boundary": (
            "IK feasibility sensitivity only; this does not demonstrate contact, "
            "task, or physical robot failure."
        ),
        "sample_count": len(frames),
        "success_tolerance_m": success_tolerance_m,
        "perturbation_protocol": "original and +/-{1,2,3} mm along x/y/z",
        "summary_by_magnitude_mm": summary_by_mm,
        "critical_regions": [
            {
                "frame_index": row["frame_index"],
                "time_sec": row["time_sec"],
                "fragility_score": row["fragility_score"],
                "fragility_by_magnitude": row["fragility_by_magnitude"],
            }
            for row in critical
        ],
        "frames": frames,
    }


def write_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def render_fragility_svg(report: dict[str, Any], path: Path) -> Path:
    """Render a dependency-free temporal plot suitable for a quick audit."""

    frames = report.get("frames", [])
    if not frames:
        raise ValueError("report has no frames")
    width, height = 1100, 520
    left, right, top, bottom = 80, 30, 72, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    times = [float(row["time_sec"]) for row in frames]
    t0, t1 = min(times), max(times)
    span = max(t1 - t0, 1e-9)

    def x(time: float) -> float:
        return left + (time - t0) / span * plot_w

    def y(score: float) -> float:
        return top + (1.0 - score) * plot_h

    colors = {"1": "#22c55e", "2": "#f59e0b", "3": "#ef4444"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<text x="80" y="34" fill="#f8fafc" font-family="sans-serif" font-size="23" font-weight="700">MilliAudit — Eva IK perturbation fragility</text>',
        f'<text x="80" y="57" fill="#94a3b8" font-family="sans-serif" font-size="13">{_escape(str(report.get("trajectory_id")))} · empirical trajectory: {str(bool(report.get("empirical_trajectory"))).lower()}</text>',
    ]
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y(level)
        elements.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#263248" stroke-width="1"/>')
        elements.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" fill="#94a3b8" font-family="monospace" font-size="12">{level:.2f}</text>')
    for mm in ("1", "2", "3"):
        points = " ".join(
            f'{x(float(row["time_sec"])):.1f},{y(float(row["fragility_by_magnitude"][mm])):.1f}'
            for row in frames
        )
        elements.append(f'<polyline points="{points}" fill="none" stroke="{colors[mm]}" stroke-width="3" stroke-linejoin="round"/>')
    for index, mm in enumerate(("1", "2", "3")):
        legend_x = 760 + index * 100
        elements.append(f'<line x1="{legend_x}" y1="45" x2="{legend_x+24}" y2="45" stroke="{colors[mm]}" stroke-width="4"/>')
        elements.append(f'<text x="{legend_x+31}" y="50" fill="#e2e8f0" font-family="sans-serif" font-size="13">{mm} mm</text>')
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        timestamp = t0 + fraction * span
        xx = x(timestamp)
        elements.append(f'<text x="{xx:.1f}" y="{height-bottom+28}" text-anchor="middle" fill="#94a3b8" font-family="monospace" font-size="12">{timestamp:.2f}s</text>')
    elements.extend(
        [
            f'<text x="{left + plot_w/2:.1f}" y="{height-24}" text-anchor="middle" fill="#cbd5e1" font-family="sans-serif" font-size="14">trajectory time</text>',
            f'<text x="22" y="{top + plot_h/2:.1f}" transform="rotate(-90 22 {top + plot_h/2:.1f})" text-anchor="middle" fill="#cbd5e1" font-family="sans-serif" font-size="14">kinematic fragility score</text>',
            f'<text x="80" y="{height-50}" fill="#94a3b8" font-family="sans-serif" font-size="11">{_escape(str(report.get("claim_boundary", "")))}</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")
    return path


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
