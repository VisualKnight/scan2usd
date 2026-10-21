"""Closed-loop evaluation of the vision policy: the hand placing the controller from camera estimates.

    set OMNI_KIT_ACCEPT_EULA=YES
    tools\\il\\venv\\Scripts\\python.exe scripts\\rl\\vision_eval.py [--policy P] [--modes exact/exact,vision/vision,...]

Each mode is <scene>/<in-hand>: where the policy's scene poses (the controller's settled pose, its seated pose)
and its view of the controller in the hand come from: "exact" (the simulator), "vision" (the pose networks
on the two RGB-D views) or, for the hand, "mean" (the average grasp: no live estimate). Every mode runs the
same 400 fresh scenes as the state-based policy's evaluation (train_policy.py: seed 5000), under random looks
the networks never saw. Writes output/rl/vision/eval_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="", help="policy checkpoint (default: the state-based one)")
parser.add_argument("--scene_net", default="")
parser.add_argument("--hand_net", default="")
parser.add_argument("--modes", default="exact/exact,vision/exact,exact/vision,vision/vision,vision/mean")
parser.add_argument("--envs", type=int, default=400)
parser.add_argument("--seed", type=int, default=5000)
parser.add_argument("--scale", type=float, default=1.0, help="scene jitter, times the training range")
parser.add_argument("--looks", default="random", choices=["random", "nominal"])
parser.add_argument("--noise", action="store_true", help="a noisier sensor than training saw (vision_policy.corrupt)")
parser.add_argument("--name", default="")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hand_teacher as T  # noqa: E402
import policy as PL  # noqa: E402
import vision_policy as VP  # noqa: E402
from hand_place_env import SPEC  # noqa: E402
from vision_env import VisionHandPlaceEnv, VisionHandPlaceEnvCfg  # noqa: E402

VIS = os.path.join(T.ROOT, "output", "rl", "vision")
DEFAULT_POLICY = os.path.join(T.ROOT, "scripts", "rl", "checkpoints", "hand_place_policy.pt")


def main():
    cfg = VisionHandPlaceEnvCfg()
    cfg.scene.num_envs = args.envs
    cfg.episode_length_s = 1.0e5
    env = VisionHandPlaceEnv(cfg)
    env.reset()
    dev = env.device
    pol = PL.load(args.policy or DEFAULT_POLICY, dev)
    modes = [m.split("/") for m in args.modes.split(",")]
    scene_net = (VP.load_net(args.scene_net or VP.SCENE_NET, dev)
                 if any(s == "vision" for s, _ in modes) else None)
    hand_net = (VP.load_net(args.hand_net or VP.HAND_NET, dev)
                if any(h != "exact" for _, h in modes) else None)
    r = T.Randomization()
    s = args.scale
    rz = T.Randomization(r.holder_xy * s, r.holder_yaw * s, r.tray_xy * s, r.controller_yaw * s)
    scenes = [SPEC.scene(np.random.default_rng([args.seed, i]), rz) for i in range(args.envs)]
    name = args.name or f"{os.path.splitext(os.path.basename(args.policy or DEFAULT_POLICY))[0]}_x{s:g}_{args.looks}"
    out = {"policy": args.policy or DEFAULT_POLICY, "scale": s, "looks": args.looks, "envs": args.envs, "modes": {}}
    for mode in args.modes.split(","):
        scene_mode, hand_mode = mode.split("/")
        env.randomize_looks(None if args.looks == "nominal" else np.random.default_rng([args.seed, 99]))
        run = VP.VisionRunner(env, SPEC, scene_net, hand_net, scene=scene_mode, in_hand=hand_mode, noise=args.noise)
        t0 = time.time()
        res, _ = run.episode(scenes, policy=pol, use_policy=torch.ones(args.envs, dtype=torch.bool), collect=False)
        out["modes"][mode] = {**PL.summarize(res), "estimates": run.error_summary(), "wall_s": time.time() - t0,
                              "placed_cells": res["placed"].nonzero().squeeze(-1).tolist()}
        print(f"[eval] {mode}: {json.dumps({k: v for k, v in out['modes'][mode].items() if k != 'placed_cells'})}",
              flush=True)
        PL.write_json(os.path.join(VIS, f"eval_{name}.json"), out)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
