# MilliAudit (secondary, bounded experiment)

MilliAudit asks where `+/-1`, `+/-2`, or `+/-3 mm` Cartesian target errors
materially change Eva inverse-kinematics feasibility. It reports IK convergence,
final position residual, joint-space change from nominal, joint-limit margin
change, and temporal discontinuity. It does **not** infer contact or task failure.

## Current evidence status

No recorded Eva or retargeted target-pose trajectory is present in this
workspace. The selected public Mecka episodes expose human MP4s, not Eva target
poses or joint state. Consequently, the included Modal run uses the real pinned
EgoVerse Eva Mink solver and `model_x5.xml`, but only on a deterministic
FK-generated joint sweep. Its output is a **method/solver smoke test**, not an
empirical retargeting result and should not be shown as one.

## Real trajectory input contract

Provide an NPZ containing:

- `timestamps`: `[T]`, strictly increasing seconds
- `positions`: `[T, 3]`, Cartesian targets in meters
- `rotations`: `[T, 3, 3]`, target rotation matrices
- `seed_joints`: `[T, J]`, nominal/warm-start joints in radians
- `metadata_json`: scalar JSON string with `trajectory_id`, `provenance`, and
  `empirical: true`

Run against an environment containing the pinned EgoVerse package:

```bash
python -m hackathon.milliaudit.run \
  --trajectory /path/to/eva_targets.npz \
  --solver-factory hackathon.milliaudit.eva_adapter:create_eva_solver \
  --output-dir hackathon/milliaudit/results/real
```

Run the bounded CPU-only solver smoke on Modal:

```bash
uv run modal run hackathon/milliaudit/modal_app.py --sample-count 12
```

Download its artifacts:

```bash
uv run modal volume get egoverse-data /milliaudit/solver-smoke/fragility.json \
  hackathon/milliaudit/results/solver-smoke/fragility.json --force
uv run modal volume get egoverse-data /milliaudit/solver-smoke/fragility.svg \
  hackathon/milliaudit/results/solver-smoke/fragility.svg --force
```
