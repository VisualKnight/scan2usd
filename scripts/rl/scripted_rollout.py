"""Scripted rollouts of the hand-place task in Isaac Lab: many cells at once, scored, optionally filmed.

Setup (once; Isaac Lab v3.0.0-beta2.patch1 on Isaac Sim 6.0.1, Python 3.12, Windows):
    uv venv --python 3.12 D:\\il\\venv
    uv pip install --python D:\\il\\venv\\Scripts\\python.exe "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com
    uv pip install --python D:\\il\\venv\\Scripts\\python.exe torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
    git clone --branch v3.0.0-beta2.patch1 https://github.com/isaac-sim/IsaacLab  (then, with VIRTUAL_ENV set: isaaclab.bat -i)
    python scripts/demo_hand_place.py --export-task output/hand_task.json   (main venv: the task spec)

    set OMNI_KIT_ACCEPT_EULA=YES
    D:\\il\\venv\\Scripts\\python.exe scripts\\rl\\scripted_rollout.py --replay              # the 12 benchmark seeds
    D:\\il\\venv\\Scripts\\python.exe scripts\\rl\\scripted_rollout.py --num_envs 100 --seed 0 --video

--replay runs the exported benchmark plans (output/hand_place_plans.json: MuJoCo's scenes, settled
controller poses and waypoints), as scripts/isaac_hand_place.py does on CPU PhysX, here on GPU PhysX
with all seeds in parallel. Otherwise each env gets its own random scene (hand_teacher.Spec.scene), the
controller settles, and the plan is made from its settled pose. Both re-aim at the pre-place pose (the
controller's pose in the hand is measured and the insertion re-planned), as the benchmark's best variant.

Score per env as everywhere in the project: mean distance of the controller's surface points from the
pose measured in the scan of the real pair (ADD), its rotation, held (not on the floor), placed (< 2 cm).
Writes output/rl/rollout_<name>.json, and with --video output/rl/<name>.mp4 (+ a poster frame).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--replay", action="store_true", help="the exported benchmark seeds (output/hand_place_plans.json)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--rand", type=float, nargs=4, default=None, metavar=("HOLDER_XY", "HOLDER_YAW", "TRAY_XY", "CTRL_YAW"),
                    help="scene jitter (m, deg, m, deg); default: the benchmark's 0.02 10 0.02 10")
parser.add_argument("--physics_hz", type=int, default=240)
parser.add_argument("--video", action="store_true")
parser.add_argument("--name", default=None)
parser.add_argument("--spacing", type=float, default=1.0)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=args.video).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hand_teacher as T  # noqa: E402
from hand_place_env import HandPlaceEnv, HandPlaceEnvCfg, SPEC, quat_angle_deg  # noqa: E402

OUT = os.path.join(T.ROOT, "output", "rl")
name = args.name or ("replay" if args.replay else f"random_{args.num_envs}_s{args.seed}")


def controller_points(n=3000) -> np.ndarray:
    from pxr import Usd, UsdGeom

    from hand_place_env import CTRL_USD

    stage = Usd.Stage.Open(CTRL_USD)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/{stage.GetDefaultPrim().GetName()}/render"))
    pts = np.asarray(mesh.GetPointsAttr().Get(), float)
    return pts[np.random.default_rng(1).choice(len(pts), min(n, len(pts)), replace=False)]


def to4(p: torch.Tensor) -> np.ndarray:
    return np.array([T.from7(x) for x in p.cpu().numpy()])


def main():
    if args.replay:
        plans = json.load(open(os.path.join(T.ROOT, "output", "hand_place_plans.json")))["trials"]
        n = len(plans) if args.num_envs <= 0 else args.num_envs
        trials = [plans[i % len(plans)] for i in range(n)]
        scenes = []
        for tr in trials:
            sc = {k: np.array(tr[k]) for k in ("holder", "target", "c_goal", "approach", "tray")}
            sc["controller_start"] = np.array(tr["controller_start"])
            sc["controller_start"][2, 3] += 0.001  # as isaac_hand_place.py: MuJoCo's settled pose, 1 mm up
            scenes.append(sc)
    else:
        n = args.num_envs
        rz = T.Randomization(*args.rand) if args.rand else T.Randomization()
        scenes = [SPEC.scene(np.random.default_rng([args.seed, i]), rz) for i in range(n)]

    cfg = HandPlaceEnvCfg()
    cfg.scene.num_envs = n
    cfg.scene.env_spacing = args.spacing
    cfg.sim.dt = 1.0 / args.physics_hz
    cfg.decimation = max(1, round(args.physics_hz / 30))
    cfg.sim.render_interval = cfg.decimation
    env = HandPlaceEnv(cfg)
    dev, dt = env.device, env.physics_dt
    env.reset()
    zero = torch.zeros((n, env.cfg.action_space), device=dev)
    rec = Recorder(env, scenes, duration=19.4) if args.video else None

    def step(k=1):
        for _ in range(k):
            env.step(zero)
            if rec:
                rec.frame()

    # The hand waits 25 cm above where its plan will start (planned from the unsettled controller: within
    # millimetres of the real start), then glides down into the plan: no teleport into the scene.
    parks = []
    for sc in scenes:
        T0 = SPEC.plan(sc["controller_start"], sc)[0]["T"].copy()
        T0[2, 3] += 0.25
        parks.append(T0)
    env.place_scene(scenes, hand_poses=parks)
    step(int(0.6 * 30))  # the controller settles on its tray
    if args.replay:
        plans_w = [[{**w, "T": np.array(w["T"])} for w in tr["waypoints"]] for tr in trials]
    else:
        starts = to4(env.controller_pose())
        plans_w = [SPEC.plan(starts[i], scenes[i]) for i in range(n)]
    for w in plans_w:
        w[0] = {**w[0], "dur": 1.2}  # the glide from the park pose to the plan's start
    sampled = [T.sample(w, dt, env.keys, start=(parks[i], SPEC.open_key)) for i, w in enumerate(plans_w)]
    margin = int(1.0 / dt)
    L = max(len(p) for p, _, _ in sampled) + margin
    P = torch.zeros((n, L, 7), device=dev)
    Q = torch.zeros((n, L, len(env.joint_names)), device=dev)
    lengths = torch.zeros(n, dtype=torch.long, device=dev)
    preplace = torch.zeros(n, dtype=torch.long, device=dev)
    for i, (p, q, ev) in enumerate(sampled):
        P[i, :len(p)] = torch.from_numpy(p)
        P[i, len(p):] = torch.from_numpy(p[-1])
        Q[i, :len(q)] = torch.from_numpy(q)
        Q[i, len(q):] = torch.from_numpy(q[-1])
        lengths[i] = len(p)
        preplace[i] = ev["preplace"]
    P[:, :, :3] += env.scene.env_origins[:, None, :]
    cursor = torch.zeros(n, dtype=torch.long, device=dev)
    reaimed = torch.zeros(n, dtype=torch.bool, device=dev)
    rows = torch.arange(n, device=dev)

    def on_substep(e):
        due = (cursor == preplace) & ~reaimed
        if due.any():  # the re-aim: measure the controller in the hand, re-plan the insertion
            ids = torch.nonzero(due).squeeze(-1).tolist()
            C, F = to4(e.controller_pose()[ids]), to4(e.forearm_pose()[ids])
            for j, i in enumerate(ids):
                H_rel = np.linalg.inv(C[j]) @ F[j]
                wps = SPEC.replan_place(scenes[i], H_rel)
                start = (plans_w[i][[w["tag"] for w in plans_w[i]].index("preplace")]["T"], "grip")
                p, q, _ = T.sample(wps[1:], dt, e.keys, start=start)
                k0 = int(cursor[i]) + 1
                m = min(len(p), L - k0)
                pt = torch.from_numpy(p[:m]).to(dev)
                pt[:, :3] += e.scene.env_origins[i]
                P[i, k0:k0 + m] = pt
                Q[i, k0:k0 + m] = torch.from_numpy(q[:m]).to(dev)
                P[i, k0 + m:] = pt[-1]
                Q[i, k0 + m:] = Q[i, k0 + m - 1]
                lengths[i] = k0 + m
            reaimed[due] = True
        c = cursor.clamp(max=L - 1)
        e.wrist_target[:] = P[rows, c]
        e.joint_target[:] = Q[rows, c]
        cursor.add_(1)

    env.on_substep = on_substep
    t0 = time.time()
    while bool((cursor < lengths).any()):
        step()
    env.on_substep = None
    step(int(1.5 * 30))  # let it settle in the holder
    wall = time.time() - t0

    # score
    cp = torch.tensor(controller_points(), dtype=torch.float32, device=dev)
    C = env.controller_pose()
    tgt = torch.tensor(np.array([T.pose7(s["target"]) for s in scenes]), dtype=torch.float32, device=dev)

    def apply(p, pts):
        from isaaclab.utils.math import quat_apply

        return quat_apply(p[:, None, 3:7].expand(-1, len(pts), -1), pts[None].expand(len(p), -1, -1)) + p[:, None, :3]

    end, true = apply(C, cp), apply(tgt, cp)
    add = (end - true).norm(dim=-1).mean(-1) * 1000
    rot = quat_angle_deg(C[:, 3:7], tgt[:, 3:7])
    held = end[..., 2].min(-1).values > 0.02
    placed = held & (add < 20)
    res = [{"env": i, "add_mm": float(add[i]), "rot_deg": float(rot[i]), "held": bool(held[i]),
            "placed": bool(placed[i])} for i in range(n)]
    summary = {"envs": n, "held": int(held.sum()), "placed": int(placed.sum()),
               "add_mm_median": float(add.median()), "rot_deg_median": float(rot.median()),
               "physics_hz": args.physics_hz, "sim_s": float(cursor.max()) * dt + 2.1, "wall_s": wall}
    print("[rollout]", json.dumps(summary), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"rollout_{name}.json"), "w") as f:
        json.dump({"summary": summary, "results": res, "replay": args.replay, "seed": args.seed,
                   "rand": args.rand}, f, indent=1)
    if rec:
        rec.finish(os.path.join(OUT, f"{name}.mp4"), res)
    env.close()


class Recorder:
    """Frames from a moving USD camera via a Replicator render product: a wide shot of the whole grid
    that pushes in slowly on one cell of the front row, the others carrying on behind it."""

    def __init__(self, env, scenes, duration=17.0, size=(1920, 1080)):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        sys.path.insert(0, os.path.join(T.ROOT, "scripts"))
        import hand_motion

        self.env, self.frames, self.size = env, [], size
        stage = env.scene.stage
        cam = UsdGeom.Camera.Define(stage, "/World/FilmCam")
        cam.CreateHorizontalApertureAttr(36.0)
        cam.CreateVerticalApertureAttr(20.25)
        cam.CreateFocalLengthAttr(32.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
        self.xf = cam.AddTransformOp()
        self.rp = rep.create.render_product("/World/FilmCam", size)
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([self.rp])
        o = env.scene.env_origins.cpu().numpy()
        lo, hi = o.min(0), o.max(0)
        center = 0.5 * (lo + hi) + [0.12, 0.12, 0.0]
        W = float(max(hi[0] - lo[0], hi[1] - lo[1])) + 1.0
        # The hero cell: the front row's rightmost (nothing between it and the camera). The hand comes in from
        # its front-left and backs away there, the tray stands front-right: the close-up is from the front.
        front = o[np.isclose(o[:, 1], lo[1])]
        hero = front[np.argmax(front[:, 0])]
        cell = hero + [0.02, 0.18, 0.15]
        keys = [(0.0, center + [0, 0, 0.0], 1.05 * W, 122, -40),
                (0.3 * duration, 0.55 * center + 0.45 * cell, 0.6 * W, 126, -33),
                (0.62 * duration, cell, 2.0, 114, -30),
                (duration, cell + [-0.03, 0.04, -0.03], 1.3, 102, -26)]  # the tray off to the right of the holder
        self.track = hand_motion.camera_track(keys)
        self.eye = hand_motion.orbit_eye
        self.k = 0
        self.place(0.0)

    def place(self, t):
        from pxr import Gf

        eye, look = self.eye(self.track(t))
        self.xf.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1)).GetInverse())

    def frame(self):
        self.k += 1
        self.place(self.k / 30.0)
        # Headless with no visualizer nothing pumps Kit's loop, which renders the RTX render product; Isaac
        # Lab's own cameras do it this way (once per sim step, physics untouched).
        self.env.sim.render()
        try:
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

            ensure_isaac_rtx_render_update()
        except Exception:
            pass
        data = self.rgb.get_data()
        if data is None or not getattr(data, "size", 0):
            import omni.kit.app

            omni.kit.app.get_app().update()
            data = self.rgb.get_data()
        if data is not None and getattr(data, "size", 0):
            self.frames.append(np.asarray(data)[..., :3].copy())

    def finish(self, path, res):
        import subprocess
        import tempfile

        from PIL import Image, ImageDraw, ImageFont

        exe = os.path.join(T.ROOT, ".venv", "Lib", "site-packages", "imageio_ffmpeg", "binaries")
        exe = next(os.path.join(exe, f) for f in os.listdir(exe) if f.startswith("ffmpeg"))
        tmp = tempfile.mkdtemp(prefix="rl_video_")
        n = len(self.frames)
        if n == 0:
            raise RuntimeError("no frames were rendered")
        placed = sum(r["placed"] for r in res)

        def font(sz):
            for f in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
                if os.path.exists(f):
                    return ImageFont.truetype(f, sz)
            return ImageFont.load_default()

        f1, f2 = font(int(self.size[1] * 0.034)), font(int(self.size[1] * 0.024))
        W, H = self.size
        x0, y0 = int(W * 0.035), int(H * 0.045)
        for i, f in enumerate(self.frames):
            im = Image.fromarray(f).convert("RGBA")
            ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            t = i / 30.0
            a = max(0.0, min(1.0, (4.0 - t) / 0.7))  # the title, fading out
            if a > 0:
                l1 = f"{len(res)} scenes at once \u00b7 Isaac Sim 6 \u00b7 GPU PhysX"
                l2 = "the scripted teacher: each scene randomized, the plan made from where the controller settled"
                w = max(d.textlength(l1, font=f1), d.textlength(l2, font=f2))
                d.rounded_rectangle([x0 - 18, y0 - 12, x0 + w + 18, y0 + int(H * 0.088)], 10,
                                    fill=(255, 255, 255, int(205 * a)))
                d.text((x0, y0), l1, fill=(30, 41, 59, int(255 * a)), font=f1)
                d.text((x0, y0 + int(H * 0.048)), l2, fill=(71, 85, 105, int(255 * a)), font=f2)
            b = max(0.0, min(1.0, (t - (n / 30.0 - 2.6)) / 0.5))  # the result, fading in
            if b > 0:
                msg = f"{placed}/{len(res)} placed in the holder"
                w = d.textlength(msg, font=f1)
                d.rounded_rectangle([x0 - 18, y0 - 12, x0 + w + 18, y0 + int(H * 0.05)], 10,
                                    fill=(255, 255, 255, int(215 * b)))
                d.text((x0, y0), msg, fill=(21, 128, 61, int(255 * b)), font=f1)
            Image.alpha_composite(im, ov).convert("RGB").save(os.path.join(tmp, f"{i:05d}.jpg"), quality=92)
        subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", "30", "-i", os.path.join(tmp, "%05d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path], check=True)
        for k, t in (("wide", 0.5), ("mid", 0.45), ("close", 0.9)):
            Image.open(os.path.join(tmp, f"{int(t * (n - 1)) if k != 'wide' else 15:05d}.jpg")).save(
                path.replace(".mp4", f"_{k}.jpg"), quality=90)
        print(f"[rollout] wrote {path} ({n} frames)", flush=True)


if __name__ == "__main__":
    main()
    app.close()
