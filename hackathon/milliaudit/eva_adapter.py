"""Adapter for the pinned EgoVerse Eva Mink solver."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def create_eva_solver():
    import egomimic
    from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver

    model_path = Path(os.path.dirname(egomimic.__file__)) / "resources/model_x5.xml"
    solver = EvaMinkKinematicsSolver(
        model_path=str(model_path),
        eef_link_name="tcp_match_trac",
        eef_frame_type="site",
        max_iterations=100,
        position_tolerance=1e-3,
        orientation_tolerance=1e-3,
    )
    limits = np.asarray([solver.model.joint(name).range for name in solver.JOINT_NAMES])
    return solver, limits[:, 0], limits[:, 1]
