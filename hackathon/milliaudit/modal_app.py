"""Cheap CPU-only Modal smoke run against EgoVerse's real Eva Mink solver."""

from __future__ import annotations

from pathlib import Path

import modal

from egoverse_modal.modal_resources import data_volume, egoverse_image


APP_NAME = "milliaudit-smoke"
VOLUME_ROOT = Path("/vol")
RESULT_ROOT = VOLUME_ROOT / "milliaudit" / "solver-smoke"

image = egoverse_image.add_local_dir(
    "hackathon", remote_path="/opt/hackathon", copy=True
).env({"PYTHONPATH": "/opt", "MPLCONFIGDIR": "/tmp/matplotlib"})
app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=10 * 60,
    max_containers=1,
    volumes={str(VOLUME_ROOT): data_volume},
)
def eva_solver_smoke(sample_count: int = 12) -> dict:
    """Generated joint sweep: a method smoke test, never empirical evidence."""

    import numpy as np

    from hackathon.milliaudit.core import (
        Trajectory,
        audit_trajectory,
        render_fragility_svg,
        write_report,
    )
    from hackathon.milliaudit.eva_adapter import create_eva_solver

    if not 4 <= int(sample_count) <= 40:
        raise ValueError("sample_count must be in [4, 40]")
    solver, lower, upper = create_eva_solver()

    # A conservative, deterministic path through the official Eva kinematic
    # model. It validates the perturbation machinery but is not a recorded or
    # retargeted trajectory.
    alpha = np.linspace(0.0, 1.0, int(sample_count))
    q_start = np.array([-0.25, 0.75, 1.15, -0.45, -0.45, -0.35])
    q_end = np.array([0.35, 1.45, 0.45, 0.55, 0.55, 0.45])
    joints = q_start[None, :] * (1.0 - alpha[:, None]) + q_end[None, :] * alpha[:, None]
    poses = [solver.fk(joint) for joint in joints]
    positions = np.stack([pose[0] for pose in poses])
    rotations = np.stack(
        [pose[1].as_matrix() if hasattr(pose[1], "as_matrix") else pose[1] for pose in poses]
    )
    trajectory = Trajectory(
        timestamps=np.linspace(0.0, 5.5, int(sample_count)),
        positions=positions,
        rotations=rotations,
        seed_joints=joints,
        trajectory_id="eva-generated-joint-sweep-smoke",
        provenance=(
            "Generated FK targets from GaTech-RL2/EgoVerse Eva model at pinned "
            "commit 1d405f5c595b767b7d6ea9d3126d4453a403a6d2; not recorded data"
        ),
        empirical=False,
    )
    report = audit_trajectory(
        solver,
        trajectory,
        joint_lower=lower,
        joint_upper=upper,
        success_tolerance_m=0.002,
    )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = write_report(report, RESULT_ROOT / "fragility.json")
    plot_path = render_fragility_svg(report, RESULT_ROOT / "fragility.svg")
    data_volume.commit()
    return {
        "report_path": str(report_path),
        "plot_path": str(plot_path),
        "empirical_trajectory": False,
        "summary_by_magnitude_mm": report["summary_by_magnitude_mm"],
        "critical_regions": report["critical_regions"],
    }


@app.local_entrypoint()
def main(sample_count: int = 12) -> None:
    import json

    print(json.dumps(eva_solver_smoke.remote(sample_count), indent=2))
