"""CLI for a real Eva target-pose trajectory NPZ."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np

from .core import audit_trajectory, load_trajectory, render_fragility_svg, write_report


def _load_factory(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("solver factory must use module:callable syntax")
    return getattr(importlib.import_module(module_name), attribute)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--solver-factory", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--success-tolerance-mm", type=float, default=2.0)
    args = parser.parse_args()

    trajectory = load_trajectory(args.trajectory)
    factory = _load_factory(args.solver_factory)
    created = factory()
    if isinstance(created, tuple):
        solver, lower, upper = created
        lower, upper = np.asarray(lower), np.asarray(upper)
    else:
        solver, lower, upper = created, None, None
    report = audit_trajectory(
        solver,
        trajectory,
        joint_lower=lower,
        joint_upper=upper,
        success_tolerance_m=args.success_tolerance_mm / 1000.0,
    )
    report_path = write_report(report, args.output_dir / "fragility.json")
    plot_path = render_fragility_svg(report, args.output_dir / "fragility.svg")
    print(f"RESULT: {report_path}")
    print(f"PLOT: {plot_path}")
    if not trajectory.empirical:
        print("CLAIM BOUNDARY: input is not marked empirical; do not present as a real trajectory result")


if __name__ == "__main__":
    main()
