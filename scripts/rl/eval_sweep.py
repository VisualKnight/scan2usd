"""Score learned-policy checkpoints and the scripted teacher on the same random scenes, at several scene jitters.

Shows how the network got there (behavior-cloning snapshots, DAgger rounds) and where it stops working
(scene jitter beyond the training range: the benchmark's +-2 cm / +-10 deg). One Isaac Lab process, one
batch of cells per job, the same scenes for every checkpoint at a given jitter.

    set OMNI_KIT_ACCEPT_EULA=YES
    D:\\local_projects\\il\\venv\\Scripts\\python.exe scripts\\rl\\eval_sweep.py --name curve ^
        --jobs bc100=output/rl/curve/bc_100.pt bc1000=output/rl/curve/bc_1000.pt final=scripts/rl/checkpoints/hand_place_policy.pt
    D:\\local_projects\\il\\venv\\Scripts\\python.exe scripts\\rl\\eval_sweep.py --name jitter --scales 1 2 3 4 ^
        --jobs teacher final=scripts/rl/checkpoints/hand_place_policy.pt

Writes output/rl/sweep_<name>.json (after every job).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--jobs", nargs="+", required=True, help="label=checkpoint, or 'teacher' for the scripted hand")
parser.add_argument("--scales", type=float, nargs="+", default=[1.0], help="scene jitter, in units of the training range")
parser.add_argument("--num_envs", type=int, default=400)
parser.add_argument("--seed", type=int, default=7000)
parser.add_argument("--power_w", type=float, default=400.0)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hand_teacher as T  # noqa: E402
import policy as PL  # noqa: E402
from hand_place_env import HandPlaceEnv, HandPlaceEnvCfg, SPEC  # noqa: E402


def rand(scale: float) -> T.Randomization:
    b = T.Randomization()
    return T.Randomization(b.holder_xy * scale, b.holder_yaw * scale, b.tray_xy * scale, b.controller_yaw * scale)


def main():
    cfg = HandPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 1.0
    cfg.episode_length_s = 1.0e5
    env = HandPlaceEnv(cfg)
    env.reset()
    dev, n = env.device, args.num_envs
    gov = PL.PowerGovernor(args.power_w)
    run = PL.Runner(env, SPEC, governor=gov, obs_set="scene")
    out_path = os.path.join(T.ROOT, "output", "rl", f"sweep_{args.name}.json")
    out = {"args": vars(args), "jobs": []}
    for scale in args.scales:
        rz = rand(scale)
        scenes = [SPEC.scene(np.random.default_rng([args.seed, i]), rz) for i in range(n)]
        for job in args.jobs:
            label, _, ck = job.partition("=")
            pol = PL.load(ck, dev) if ck else None
            use = torch.ones(n, dtype=torch.bool) if pol is not None else None
            t0 = time.time()
            res, _ = run.episode(scenes, policy=pol, use_policy=use, collect=False)
            row = {"label": label, "checkpoint": ck or "teacher", "scale": scale, "rand": vars(rz), **PL.summarize(res),
                   "add_mm": [round(float(v), 2) for v in res["add_mm"]], "placed_cells": res["placed"].tolist(),
                   "held_cells": res["held"].tolist(), "wall_s": time.time() - t0}
            out["jobs"].append(row)
            out["power"] = gov.summary()
            PL.write_json(out_path, out)
            print(f"[sweep] {label} x{scale:g}: {row['placed']}/{row['envs']} placed, held {row['held']}, "
                  f"{row['add_mm_median']:.1f} mm / {row['rot_deg_median']:.1f} deg ({row['wall_s']:.0f} s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
