"""Training data for the vision policy's pose networks (vision_policy.py): camera frames with the true poses.

    set OMNI_KIT_ACCEPT_EULA=YES
    tools\\il\\venv\\Scripts\\python.exe scripts\\rl\\vision_data.py snap --batches 40
    tools\\il\\venv\\Scripts\\python.exe scripts\\rl\\vision_data.py hand --episodes 6

snap  scenes placed and settled as at the start of an episode (the hand parked above the controller, as the
      Runner parks it), one frame each: the holder's pose and the controller's settled pose. Scene jitter
      --scale times the policy's (the networks should not be the narrowest part).
hand  whole episodes, every --stride-th control step from just before the re-aim to the end: the controller's
      and the forearm's poses. Drivers by episode: the scripted teacher, the teacher with a slow random drift
      off its plan (DART: grasps and carries the policy might produce), and the state-based policy.

Looks are randomized per batch/episode (vision_env.randomize_looks; every 8th nominal). Writes
output/rl/vision/data/{snap,hand}_<k>.pt: uint8 colour, uint16 depth (mm), float poses (env-relative).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["snap", "hand"])
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--batches", type=int, default=40, help="snap: scene batches")
parser.add_argument("--episodes", type=int, default=6, help="hand: episodes")
parser.add_argument("--stride", type=int, default=5, help="hand: record every n-th control step")
parser.add_argument("--scale", type=float, default=1.5, help="scene jitter, times the policy's training range")
parser.add_argument("--seed", type=int, default=20000)
parser.add_argument("--policy", default="", help="hand: the state-based policy for the policy-driven episodes")
parser.add_argument("--start", type=int, default=0, help="first file index (to add to earlier data)")
parser.add_argument("--out", default="")
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

OUT = args.out or os.path.join(T.ROOT, "output", "rl", "vision", "data")
DEFAULT_POLICY = os.path.join(T.ROOT, "scripts", "rl", "checkpoints", "hand_place_policy.pt")


def make_env(n):
    cfg = VisionHandPlaceEnvCfg()
    cfg.scene.num_envs = n
    cfg.episode_length_s = 1.0e5
    env = VisionHandPlaceEnv(cfg)
    env.reset()
    return env


def rz(scale: float) -> T.Randomization:
    r = T.Randomization()
    return T.Randomization(r.holder_xy * scale, r.holder_yaw * scale, r.tray_xy * scale, r.controller_yaw * scale)


def scenes(seed: int, n: int, scale: float):
    return [SPEC.scene(np.random.default_rng([seed, i]), rz(scale)) for i in range(n)]


def looks(env, k: int, seed: int):
    env.randomize_looks(None if k % 8 == 0 else np.random.default_rng([seed, 77, k]))


def snap(env, run):
    for b in range(args.batches):
        t0 = time.time()
        looks(env, b, args.seed)
        sc = scenes(args.seed + b, env.num_envs, args.scale)
        parks = []
        for s in sc:  # as Runner.episode: 25 cm above the plan's first pose
            T0 = SPEC.plan(s["controller_start"], s)[0]["T"].copy()
            T0[2, 3] += 0.25
            parks.append(T0)
        env.on_substep = None
        env.place_scene(sc, hand_poses=parks)
        env.episode_length_buf[:] = 0
        run.step(int(0.6 * 30))
        rgb, dep = VP.to_storage(env.images())
        holder7 = env.rel(env.holder.data.root_link_pose_w.torch).cpu()
        ctrl7 = env.controller_pose().cpu()
        ok = torch.isfinite(ctrl7).all(-1) & (ctrl7[:, 2] > 0.05)  # the controller settled on its tray
        torch.save({"rgb": rgb[ok], "depth": dep[ok], "holder7": holder7[ok], "ctrl7": ctrl7[ok], "looks": env.looks},
                   os.path.join(OUT, f"snap_{b:03d}.pt"))
        print(f"[data] snap {b}: {int(ok.sum())}/{len(ok)} kept ({time.time() - t0:.1f} s)", flush=True)


def hand(env, run):
    pol = PL.load(args.policy or DEFAULT_POLICY, env.device)
    drivers = ["teacher", "dart", "policy"]
    for k in range(args.start, args.start + args.episodes):
        t0 = time.time()
        driver = drivers[k % 3]
        looks(env, k + 1, args.seed + 1)
        frames = {"rgb": [], "depth": [], "ctrl7": [], "forearm7": [], "t": []}
        count = [0]

        def cb():
            i = count[0]
            count[0] += 1
            tb = getattr(run, "t_b", None)
            if tb is None or i < 18:
                return
            t = (i - 18) / 30.0  # plan time: the Runner settles 18 control steps first
            if t < tb - VP.VisionRunner.WARMUP_S - 0.2 or (i - 18) % args.stride:
                return
            rgb, dep = VP.to_storage(env.images())
            frames["rgb"].append(rgb)
            frames["depth"].append(dep)
            frames["ctrl7"].append(env.controller_pose().cpu())
            frames["forearm7"].append(env.forearm_pose().cpu())
            frames["t"].append(t)

        sc = scenes(args.seed + 1000 + k, env.num_envs, 1.0)
        n = env.num_envs
        if driver == "policy":
            res, _ = run.episode(sc, policy=pol, use_policy=torch.ones(n, dtype=torch.bool), collect=False, frame_cb=cb)
        else:
            res, _ = run.episode(sc, collect=False, frame_cb=cb,
                                 dart=(0.003, 0.015, 0.0) if driver == "dart" else None)
        run.t_b = None
        rgb, dep = torch.stack(frames["rgb"], 1), torch.stack(frames["depth"], 1)  # (n, frames, V, ...)
        c7, f7 = torch.stack(frames["ctrl7"], 1), torch.stack(frames["forearm7"], 1)
        # keep frames where the controller is in (or by) the hand: not dropped on the floor, not blown up
        near = (c7[..., :3] - f7[..., :3]).norm(dim=-1) < 0.6
        ok = torch.isfinite(c7).all(-1) & torch.isfinite(f7).all(-1) & near & ~res["broken"][:, None]
        torch.save({"rgb": rgb[ok], "depth": dep[ok], "ctrl7": c7[ok], "forearm7": f7[ok],
                    "t": torch.tensor(frames["t"]).expand(n, -1)[ok], "env": torch.arange(n)[:, None].expand(n, ok.shape[1])[ok],
                    "driver": driver, "looks": env.looks,
                    "result": PL.summarize(res)}, os.path.join(OUT, f"hand_{k:03d}.pt"))
        print(f"[data] hand {k} ({driver}): {int(ok.sum())} frames over t {frames['t'][0]:.1f}-{frames['t'][-1]:.1f} s, "
              f"{PL.summarize(res)} ({time.time() - t0:.0f} s)", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    env = make_env(args.num_envs)
    run = PL.Runner(env, SPEC)
    (snap if args.mode == "snap" else hand)(env, run)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
