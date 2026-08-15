from __future__ import annotations

import json

import numpy as np
import pytest

from hackathon.milliaudit.core import (
    Trajectory,
    audit_trajectory,
    load_trajectory,
    render_fragility_svg,
)


class LinearSolver:
    """Three translational joints with a deliberate +x reach boundary."""

    def ik(self, pos_xyz, rot_mat, cur_jnts):
        del rot_mat, cur_jnts
        if pos_xyz[0] > 0.502:
            return None
        return np.asarray(pos_xyz, dtype=float)

    def fk(self, joints):
        return np.asarray(joints[:3], dtype=float), np.eye(3)


def trajectory(*, empirical=True):
    return Trajectory(
        timestamps=np.array([0.0, 1.0, 2.0]),
        positions=np.array([[0.49, 0.0, 0.0], [0.50, 0.0, 0.0], [0.501, 0.0, 0.0]]),
        rotations=np.repeat(np.eye(3)[None, :, :], 3, axis=0),
        seed_joints=np.zeros((3, 3)),
        trajectory_id="boundary",
        provenance="unit-test",
        empirical=empirical,
    )


def test_audit_runs_all_axis_sign_magnitude_perturbations():
    report = audit_trajectory(
        LinearSolver(),
        trajectory(),
        joint_lower=np.array([-1.0, -1.0, -1.0]),
        joint_upper=np.array([1.0, 1.0, 1.0]),
    )
    assert report["sample_count"] == 3
    assert len(report["frames"][0]["perturbations"]) == 18
    branches = {trial["branch"] for trial in report["frames"][0]["perturbations"]}
    assert {"-1mm_x", "+1mm_x", "-3mm_z", "+3mm_z"}.issubset(branches)
    assert report["summary_by_magnitude_mm"]["3"]["trials"] == 18
    assert report["summary_by_magnitude_mm"]["3"]["success_rate"] < 1.0
    assert report["critical_regions"][0]["fragility_score"] == 1.0
    assert report["empirical_trajectory"] is True


def test_npz_contract_defaults_to_non_empirical(tmp_path):
    path = tmp_path / "trajectory.npz"
    source = trajectory(empirical=False)
    np.savez_compressed(
        path,
        timestamps=source.timestamps,
        positions=source.positions,
        rotations=source.rotations,
        seed_joints=source.seed_joints,
        metadata_json=np.asarray(json.dumps({"trajectory_id": "test"})),
    )
    loaded = load_trajectory(path)
    assert loaded.trajectory_id == "test"
    assert loaded.empirical is False


def test_rejects_non_orthonormal_rotation():
    source = trajectory()
    source.rotations[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        source.validate()


def test_svg_marks_non_empirical_result(tmp_path):
    report = audit_trajectory(LinearSolver(), trajectory(empirical=False))
    path = render_fragility_svg(report, tmp_path / "fragility.svg")
    contents = path.read_text()
    assert "empirical trajectory: false" in contents
    assert "physical robot failure" in contents
