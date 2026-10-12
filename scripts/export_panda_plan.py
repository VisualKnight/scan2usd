"""Export the Panda rim-grasp plan (joint waypoints) for replay in another simulator.

    python scripts/export_panda_plan.py [--asset output/demo/asset/ceramic_mug.usda] [--out output/panda_plan.json]

The plan comes from scripts/demo_panda.py (MuJoCo Menagerie Panda IK). The Franka's kinematics
are the same in Isaac Sim, so scripts/isaac_franka_pick.py replays it there with PhysX contacts:
a sim-to-sim check of the same asset and the same motion in two physics engines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_panda import HOME, MUG_POS, PLACE_SHIFT, Arm, build_scene, plan_pick, rim_grasp  # noqa: E402

from scan2usd.usd_io import read_usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asset", type=Path, default=ROOT / "output" / "demo" / "asset" / "ceramic_mug.usda")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "panda_plan.json")
    args = p.parse_args()
    asset = read_usd(args.asset)
    model = build_scene(asset, "hulls", ROOT / "data" / "work" / "panda_plan")
    arm = Arm(model)
    mujoco.mj_resetDataKeyframe(model, arm.data, model.key("home").id)
    grasp, radial = rim_grasp(asset.render_mesh, MUG_POS)
    plan = plan_pick(model, arm, grasp, radial)
    finger_m = 0.04 / 255.0  # Menagerie gripper ctrl 0..255 -> 0..40 mm per finger
    out = {
        "asset": args.asset.resolve().relative_to(ROOT).as_posix(),
        "mug_pos": MUG_POS.tolist(), "place_shift": PLACE_SHIFT.tolist(), "home": HOME.tolist(),
        "finger_start_m": 0.022,
        "waypoints": [{"q": [float(v) for v in q], "finger_m": float(g) * finger_m, "seconds": float(d),
                       "ik_error_mm": float(e) * 1000} for q, g, d, e in plan],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}: {len(plan)} waypoints, max IK error {max(w['ik_error_mm'] for w in out['waypoints']):.2f} mm")


if __name__ == "__main__":
    main()
