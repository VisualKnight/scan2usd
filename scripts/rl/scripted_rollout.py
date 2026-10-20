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

--replay runs the exported benchmark plans (--plans, default output/hand_pinch_plans.json: MuJoCo's scenes, settled
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
parser.add_argument("--replay", action="store_true", help="the exported benchmark seeds (--plans)")
parser.add_argument("--plans", default=os.path.join("output", "hand_pinch_plans.json"),
                    help="with --replay: demo_hand_pinch.py --export-plans (the pinch; hand_place_plans.json: the span)")
parser.add_argument("--track", choices=("grid", "plan"), default="grid",
                    help="video camera: grid (the quarter view over the cells, or --camera) or plan (env 0's "
                         "close-up: hand_motion.shot_keys as one move, the MuJoCo and Isaac Sim videos' camera)")
parser.add_argument("--title", default="", help="the video title's first line")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--rand", type=float, nargs=4, default=None, metavar=("HOLDER_XY", "HOLDER_YAW", "TRAY_XY", "CTRL_YAW"),
                    help="scene jitter (m, deg, m, deg); default: the benchmark's 0.02 10 0.02 10")
parser.add_argument("--physics_hz", type=int, default=240)
parser.add_argument("--video", action="store_true")
parser.add_argument("--gui", type=float, default=None, metavar="T",
                    help="open the Isaac Sim editor at T s to pick a camera (logged to output/rl/picked_camera.json); "
                         "static, or with --playback the whole episode replayable")
parser.add_argument("--camera", default="", help="a camera JSON from --gui (output/rl/picked_camera.json): stills from it")
parser.add_argument("--floor", default="gun30_r25", help="floor look with --camera (FLOORS); gun30_r25 as in the editor pick")
parser.add_argument("--stills", default="", help="comma-separated times (s): stills of every camera variant (VIEWS)")
parser.add_argument("--name", default=None)
parser.add_argument("--spacing", type=float, default=1.0)
parser.add_argument("--layout_yaw", type=float, default=0.0, help="turn every cell's layout (deg), for filming")
parser.add_argument("--policy", default="", help="a learned policy checkpoint (train_policy.py): it drives every cell")
parser.add_argument("--power_w", type=float, default=400.0, help="GPU power budget with --policy (policy.PowerGovernor)")
parser.add_argument("--subtitle", default="", help="the video title's second line (default: says who drives)")
parser.add_argument("--encode", action="store_true",
                    help="no simulation: re-encode output/rl/frames_<name>/ with rollout_<name>.json's results")
parser.add_argument("--dome", default="studio", help="sky (DOMES): studio, gradient, uniform...; comma-separated "
                                                     "with --camera --stills: one still each")
parser.add_argument("--playback", action="store_true",
                    help="with --gui: simulate the whole episode first, then a Playback window in the editor")
parser.add_argument("--hide_forearm", action="store_true",
                    help="render without the forearm's housing (its visual meshes only: physics unchanged), so it "
                         "hides less of the grid behind it")
parser.add_argument("--exposure", type=float, default=None,
                    help="the video grade's exposure (default GRADE's, set for the studio sky; ~0.93 for gradient)")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

if args.gui is None:
    app = AppLauncher(headless=True, enable_cameras=args.video or bool(args.stills)).app
else:  # the editor window: Isaac Lab's Kit visualizer
    app = AppLauncher(visualizer=["kit"], visualizer_explicit=True, enable_cameras=True).app

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
    if args.encode:
        res = json.load(open(os.path.join(OUT, f"rollout_{name}.json")))["results"]
        single = (f"Placed: {res[0]['add_mm']:.1f} mm / {res[0]['rot_deg']:.0f}\u00b0 from the real pose"
                  if len(res) == 1 and res[0]["placed"] else None)
        encode(os.path.join(OUT, f"frames_{name}"), os.path.join(OUT, f"{name}.mp4"), len(res),
               sum(r["placed"] for r in res), learned=bool(args.policy), subtitle=args.subtitle or None,
               single=single)
        return
    if args.replay:
        plans = json.load(open(os.path.join(T.ROOT, args.plans)))["trials"]
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
        scenes = [SPEC.scene(np.random.default_rng([args.seed, i]), rz, args.layout_yaw) for i in range(n)]

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
    rec = Recorder(env, scenes, duration=19.4, layout_yaw=args.layout_yaw,
                   frame_dir=os.path.join(OUT, f"frames_{name}")) if (args.video or args.stills) else None
    if args.gui is not None:
        rec = Recorder(env, scenes, layout_yaw=args.layout_yaw, render=False)
    if args.policy:  # the learned policy drives every cell (policy.Runner); the teacher only re-aims its own plan
        import policy as PL

        pol = PL.load(args.policy, dev)
        run = PL.Runner(env, SPEC, layout_yaw=args.layout_yaw, governor=PL.PowerGovernor(args.power_w), obs_set=pol.obs_set)
        t0 = time.time()
        r, _ = run.episode(scenes, policy=pol, use_policy=torch.ones(n, dtype=torch.bool), collect=False,
                           frame_cb=rec.frame if (rec and args.video) else None)
        res = [{"env": i, "add_mm": float(r["add_mm"][i]), "rot_deg": float(r["rot_deg"][i]), "held": bool(r["held"][i]),
                "placed": bool(r["placed"][i])} for i in range(n)]
        summary = {**PL.summarize(r), "policy": args.policy, "wall_s": time.time() - t0, "power": run.gov.summary()}
        print("[rollout]", json.dumps(summary), flush=True)
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, f"rollout_{name}.json"), "w") as f:
            json.dump({"summary": summary, "results": res, "seed": args.seed, "layout_yaw": args.layout_yaw}, f, indent=1)
        if rec and args.video:
            rec.finish(os.path.join(OUT, f"{name}.mp4"), res, learned=True)
        env.close()
        return
    still_steps = {int(round(float(t) * 30)): float(t) for t in args.stills.split(",") if t}
    count = [0]

    def step(k=1):
        for _ in range(k):
            env.step(zero)
            count[0] += 1
            if rec and args.video:
                rec.frame()
            if rec and count[0] in still_steps:
                rec.stills(os.path.join(OUT, f"stills_{name}"), still_steps[count[0]])
            if args.gui is not None and args.playback:
                rec.capture()
            elif args.gui is not None and count[0] == int(round(args.gui * 30)):  # static: the editor at T s
                rec.pick_camera(os.path.join(OUT, "picked_camera.json"))
                app.close()
                sys.exit(0)

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
    if rec and args.track == "plan":
        rec.plan_camera(plans_w[0], scenes[0])
    tracks = [T.Track(w, dt, env.keys, start=(parks[i], SPEC.open_key)) for i, w in enumerate(plans_w)]
    sampled = [(tr.P, tr.Q, tr.ev) for tr in tracks]
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
                k0, p, q = tracks[i].reaim(int(cursor[i]), SPEC.replan_place(scenes[i], H_rel))
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
    if args.gui is not None:  # --playback: the whole episode is cached; hand the editor over from T
        rec.pick_camera(os.path.join(OUT, "picked_camera.json"), start=int(round(args.gui * 30)) - 1)
        app.close()
        sys.exit(0)

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
    if rec and args.video:
        rec.finish(os.path.join(OUT, f"{name}.mp4"), res)
    env.close()


GRADE = {"exposure": 0.55, "lift": 0.75, "shoulder": 0.85}


def grade(img: np.ndarray, exposure=GRADE["exposure"], lift=GRADE["lift"], shoulder=GRADE["shoulder"]):
    """A lower-contrast grade in linear light: exposure down (the render is ~1.5 stops bright), the toe
    lifted (a gamma below 1 on luminance) so the near-black holder keeps its detail, and a soft shoulder
    so the white hands don't clip. Hue and saturation are kept (the luminance ratio scales RGB)."""
    x = img.astype(np.float32) / 255.0
    L = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4) * exposure
    y = 0.2126 * L[..., 0] + 0.7152 * L[..., 1] + 0.0722 * L[..., 2]
    y2 = np.power(np.clip(y, 1e-6, None), lift)
    y2 = np.where(y2 > shoulder, shoulder + (1 - shoulder) * np.tanh((y2 - shoulder) / (1 - shoulder)), y2)
    L = np.clip(L * (y2 / np.maximum(y, 1e-6))[..., None], 0, 1)
    out = np.where(L <= 0.0031308, 12.92 * L, 1.055 * np.power(L, 1 / 2.4) - 0.055)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def encode(frame_dir: str, path: str, n_envs: int, placed: int, size=(1920, 1080), fps=30, learned=False,
           subtitle: str | None = None, single: str | None = None, **g):
    """Raw frames (frame_dir/*.png) -> graded, captioned H.264 MP4 + stills. Separate from rendering so the
    grade can be tuned without re-simulating."""
    import glob
    import subprocess
    import tempfile

    from PIL import Image, ImageDraw, ImageFont

    exe = os.path.join(T.ROOT, ".venv", "Lib", "site-packages", "imageio_ffmpeg", "binaries")
    exe = next(os.path.join(exe, f) for f in os.listdir(exe) if f.startswith("ffmpeg"))
    files = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    n = len(files)
    if n == 0:
        raise RuntimeError("no frames were rendered")

    def font(sz):
        for f in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
            if os.path.exists(f):
                return ImageFont.truetype(f, sz)
        return ImageFont.load_default()

    W, H = size
    if args.exposure is not None:
        g.setdefault("exposure", args.exposure)
    f1, f2 = font(int(H * 0.034)), font(int(H * 0.024))
    x0, yb = int(W * 0.035), int(H * 0.955)  # bottom margin
    tmp = tempfile.mkdtemp(prefix="rl_video_")
    for i, fn in enumerate(files):
        raw = np.array(Image.open(fn).convert("RGB"))
        raw[0] = raw[1]  # RTX leaves a few stray dark pixels in the first row
        im = Image.fromarray(grade(raw, **g)).convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        t = i / fps
        a = max(0.0, min(1.0, (4.0 - t) / 0.7))  # the title, fading out
        if a > 0:
            l1 = args.title or f"{n_envs} scenes at once \u00b7 Isaac Sim 6 \u00b7 GPU PhysX"
            l2 = subtitle or ("a learned policy: a small neural network on the simulator state, trained by imitating the scripted hand"
                              if learned else
                              "the scripted teacher: each scene randomized, the plan made from where the controller settled")
            w = max(d.textlength(l1, font=f1), d.textlength(l2, font=f2))
            y0 = yb - int(H * 0.088)
            d.rounded_rectangle([x0 - 18, y0 - 12, x0 + w + 18, yb], 10, fill=(255, 255, 255, int(205 * a)))
            d.text((x0, y0), l1, fill=(30, 41, 59, int(255 * a)), font=f1)
            d.text((x0, y0 + int(H * 0.048)), l2, fill=(71, 85, 105, int(255 * a)), font=f2)
        b = max(0.0, min(1.0, (t - (n / fps - 2.6)) / 0.5))  # the result, fading in
        if b > 0:
            msg = single or f"{placed}/{n_envs} placed in the holder"
            w = d.textlength(msg, font=f1)
            xr, y0 = W - x0 - w, yb - int(H * 0.05)
            d.rounded_rectangle([xr - 18, y0 - 12, xr + w + 18, yb], 10, fill=(255, 255, 255, int(215 * b)))
            ok = placed >= 0.9 * n_envs
            d.text((xr, y0), msg, fill=(21, 128, 61, int(255 * b)) if ok else (185, 28, 28, int(255 * b)), font=f1)
        Image.alpha_composite(im, ov).convert("RGB").save(os.path.join(tmp, f"{i:05d}.jpg"), quality=93)
    subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", path], check=True)
    for k, t in (("start", 0.15), ("mid", 0.5), ("place", 0.68), ("end", 0.93)):
        Image.open(os.path.join(tmp, f"{int(t * (n - 1)):05d}.jpg")).save(path.replace(".mp4", f"_{k}.jpg"), quality=90)
    print(f"[rollout] wrote {path} ({n} frames)", flush=True)


# Camera variants for --stills, placed by eye: (eye height m, ground distance back from the near cell m,
# pitch deg, azimuth offset deg, focal length mm on a 36 mm wide 16:9 back). A quarter view from the grid's
# nearest corner, low; wider lenses get a steeper pitch so the frame's top edge stays below the horizon
# (top edge = pitch + half the vertical field of view).
VIEWS = {
    # round 3 (kept for reference)
    "f18_h0.80_g0.65_p-34": (0.80, 0.65, -34, 0, 18.0),
    # round 6: 18-24 mm, eye lowered to 0.70 m, looking down more; g keeps the near cell in the lower third
    # (the view's centre ray meets the floor ~0.4-0.45 m beyond it)
    "f24_h0.70_g0.52_p-36": (0.70, 0.52, -36, 0, 24.0),
    "f22_h0.70_g0.49_p-37": (0.70, 0.49, -37, 0, 22.0),
    "f20_h0.70_g0.46_p-38": (0.70, 0.46, -38, 0, 20.0),
    "f18_h0.70_g0.42_p-40": (0.70, 0.42, -40, 0, 18.0),
    "f18_h0.65_g0.36_p-42": (0.65, 0.36, -42, 0, 18.0),
}
# Floor looks (base color, metallic, roughness): gunmetal (base 0.30) from smooth to rough.
FLOORS = {
    "gun30_r25": ((0.30, 0.31, 0.33), 1.0, 0.25),
    "gun30_r50": ((0.30, 0.31, 0.33), 1.0, 0.50),
    "gun30_r60": ((0.30, 0.31, 0.33), 1.0, 0.60),
    "gun30_r70": ((0.30, 0.31, 0.33), 1.0, 0.70),
    "gun30_r75": ((0.30, 0.31, 0.33), 1.0, 0.75),
    "gun30_r80": ((0.30, 0.31, 0.33), 1.0, 0.80),
}
# Sky domes: (HDRI under NVIDIA's asset root, a local image, or None for a plain dome; intensity; colour). The
# dome lights the cells, is what the metal floor reflects, and is the background above the horizon.
DOMES = {
    "white": (None, 700.0, (1.0, 1.0, 1.0)),
    "studio": ("/Assets/Skies/Studio/photo_studio_01_4k.hdr", 900.0, (1.0, 1.0, 1.0)),
    "hall": ("/Assets/Skies/Indoor/ZetoCGcom_ExhibitionHall_Interior1.hdr", 900.0, (1.0, 1.0, 1.0)),
    # a smooth studio cyclorama: light grey, brightest at the horizon, a touch cooler and darker overhead
    "gradient": ("gradient_sky.png", 1000.0, (1.0, 1.0, 1.0)),
    "uniform": (None, 600.0, (0.86, 0.87, 0.88)),
}


def gradient_sky(path, w=1024, h=512):
    """A lat-long sky (zenith at the top row): sRGB, a smoothstep from zenith to horizon, flat below it."""
    from PIL import Image

    el = np.linspace(90.0, -90.0, h)[:, None]  # elevation of each row
    zen, hor, gnd = np.array([200, 206, 213.0]), np.array([238, 239, 240.0]), np.array([150, 151, 153.0])
    s = np.clip(el / 60.0, 0.0, 1.0)
    s = s * s * (3 - 2 * s)
    up = hor + (zen - hor) * s[..., None]
    g = np.clip(-el / 8.0, 0.0, 1.0)[..., None]  # a soft 8 deg blend under the horizon (hidden by the floor)
    img = np.where(el[..., None] >= 0, up, hor + (gnd - hor) * g)
    img = np.broadcast_to(img, (h, w, 3))
    Image.fromarray((img + 0.5).astype(np.uint8)).save(path)
V = "f20_h0.70_g0.46_p-38"
STILL_PLAN = ([(v, "gun30_r60", "studio", "PathTracing") for v in VIEWS if v != "f18_h0.80_g0.65_p-34"]
              + [(V, f, "studio", "PathTracing") for f in ("gun30_r25", "gun30_r50", "gun30_r70", "gun30_r80")])


class Recorder:
    """A quarter view from just outside the grid's nearest corner, low and close with a wide lens: the
    nearest cell large in front, the rest of the grid receding on both sides (foreshortening). The camera
    only drifts. Raw frames are kept (output/rl/frames_<name>/) for encode()."""

    def __init__(self, env, scenes, duration=17.0, size=(1920, 1080), layout_yaw=0.0, frame_dir=None, render=True):
        from pxr import Gf, UsdGeom

        if render:
            import omni.replicator.core as rep
        sys.path.insert(0, os.path.join(T.ROOT, "scripts"))
        import hand_motion

        self.env, self.size, self.k, self.n = env, size, 0, 0
        self.cache = []  # --gui: every cell's hand and controller state per frame, for playback in the editor
        self.frame_dir = frame_dir
        if frame_dir:
            os.makedirs(frame_dir, exist_ok=True)
            for f in os.listdir(frame_dir):
                os.remove(os.path.join(frame_dir, f))
        stage = env.scene.stage
        cam = UsdGeom.Camera.Define(stage, "/World/FilmCam")
        cam.CreateHorizontalApertureAttr(36.0)
        cam.CreateVerticalApertureAttr(20.25)
        cam.CreateFocalLengthAttr(24.0)
        self.cam = cam
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 200.0))
        self.xf = cam.AddTransformOp()
        if render:
            self.rp = rep.create.render_product("/World/FilmCam", size)
            self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb.attach([self.rp])
        o = env.scene.env_origins.cpu().numpy()
        # Looking diagonally across the grid from its (+x, -y) corner: azimuth 135 deg in the world. The
        # cells are turned by layout_yaw, so in a cell's own frame the view is from 135 - layout_yaw:
        # with ~33 deg, its front (the holder's face, the tray off to its right).
        az = 135.0
        d = np.array([np.cos(np.radians(az)), np.sin(np.radians(az))])
        hero = o[np.argmin(o[:, :2] @ d)]  # the corner cell nearest the camera
        G = T.yaw(layout_yaw)
        look = hero + G @ np.array([0.10, 0.16, 0.0]) + [0, 0, 0.17]  # between the tray and the holder
        self.look, self.az = look, az
        keys = [(0.0, look + [0, 0, 0.01], 1.30, az - 4, -26),  # ~25 deg down: the grid fills the frame
                (duration, look, 1.16, az + 4, -24)]
        self.track = hand_motion.camera_track(keys)
        self.eye = hand_motion.orbit_eye
        self.picked = None
        if args.hide_forearm:
            self.hide_forearm()
        if args.camera:  # the camera picked in the editor, static; its floor and sky
            self.picked = self.load_picked()
            self.set_floor(*FLOORS[args.floor])
            self.set_dome(*DOMES[args.dome.split(",")[0]])
        if args.track == "plan":  # env 0 close up, the 400-cell film's floor and sky, the shared lens
            self.pending = True
            self.set_floor(*FLOORS[args.floor])
            self.set_dome(*DOMES[args.dome.split(",")[0]])
            cam.GetFocalLengthAttr().Set(hand_motion.focal_mm(20.25))
        self.place(0.0)

    def plan_camera(self, wps, sc):
        """--track plan: hand_motion.shot_keys for env 0's plan as one continuous move (as demo_hand_pinch's
        video and isaac_hand_place.py), its clock starting now, with the plan."""
        import hand_motion

        o = self.env.scene.env_origins[0].cpu().numpy()
        c0, g = np.asarray(sc["controller_start"])[:3, 3] + o, np.asarray(sc["target"])[:3, 3] + o
        (_, a), (_, b) = hand_motion.shot_keys(c0, g, wps)
        self.plan_track = hand_motion.camera_track(a[:-1] + b)
        self.pending = False

    def hide_forearm(self):
        """Make the forearm housing's visual meshes (forearm_0/1 under rh_forearm) invisible in every cell. The
        collision shapes, masses and the weld are untouched."""
        from pxr import Usd, UsdGeom

        stage = self.env.scene.stage
        fa = next(p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Hand")) if p.GetName() == "rh_forearm")
        rel = [str(c.GetPath()).removeprefix("/World/envs/env_0/") for c in fa.GetChildren()
               if c.GetName().startswith("forearm_") and "collision" not in c.GetName()]
        k = 0
        for i in range(self.env.num_envs):
            for r in rel:
                p = stage.GetPrimAtPath(f"/World/envs/env_{i}/{r}")
                if p.IsValid():
                    UsdGeom.Imageable(p).MakeInvisible()
                    k += 1
        print(f"[rollout] forearm hidden: {rel} in {k // max(len(rel), 1)} cells", flush=True)

    def load_picked(self):
        """The picked camera (--camera), moved with the grid's near cell: the pose is kept relative to that
        cell, so the framing holds when --spacing changes. Sets the focal length too."""
        from pxr import Gf

        pc = json.load(open(args.camera))
        M = np.array(pc["camera_to_world"], float)
        if "near_cell_look" in pc:
            M[:3, 3] += self.look - np.array(pc["near_cell_look"])
        self.cam.GetFocalLengthAttr().Set(float(pc["focal_mm"]))
        return Gf.Matrix4d(*M.T.flatten().tolist())

    def set_view(self, x):
        """x: (lookat xyz, distance, azimuth, elevation) as hand_motion.camera_track samples."""
        from pxr import Gf

        eye, look = self.eye(x)
        self.xf.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1)).GetInverse())

    def set_eye(self, h, g, pitch, az):
        """Eye h above the floor, g back (horizontally) from the near cell's center, looking along azimuth az
        and pitch (deg, negative: down)."""
        from pxr import Gf

        a, e = np.radians(az), np.radians(pitch)
        fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        c = self.look.copy()
        eye = np.array([c[0] - g * np.cos(a), c[1] - g * np.sin(a), h])
        self.xf.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*(eye + fwd)), Gf.Vec3d(0, 0, 1)).GetInverse())

    def place(self, t):
        if getattr(self, "plan_track", None) is not None:
            self.set_view(self.plan_track(t))
            return
        if self.picked is not None:
            self.xf.Set(self.picked)
            return
        self.set_view(self.track(t))

    def _grab(self, pumps=1):
        self.env.sim.render()
        try:
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

            ensure_isaac_rtx_render_update()
        except Exception:
            pass
        import omni.kit.app

        for _ in range(pumps):
            omni.kit.app.get_app().update()
        data = self.rgb.get_data()
        return np.asarray(data)[..., :3] if data is not None and getattr(data, "size", 0) else None

    def stills(self, out_dir, t):
        """The scene as it is now, from the camera/floor/render-mode combinations of STILL_PLAN (graded JPEGs)."""
        import carb
        from PIL import Image

        os.makedirs(out_dir, exist_ok=True)
        st = carb.settings.get_settings()
        plan = [("picked", args.floor, d, "PathTracing") for d in args.dome.split(",")] if args.camera else STILL_PLAN
        for view, floor, dome, mode in plan:
            if view == "picked":  # the exact camera picked in the editor
                self.xf.Set(self.load_picked())
            else:
                h, g, pitch, az_off, focal = VIEWS[view]
                self.cam.GetFocalLengthAttr().Set(focal)
                self.set_eye(h, g, pitch, self.az + az_off)
            self.set_floor(*FLOORS[floor])
            self.set_dome(*DOMES[dome])
            st.set("/rtx/rendermode", mode)
            if mode == "PathTracing":
                st.set("/rtx/pathtracing/spp", 16)
                st.set("/rtx/pathtracing/totalSpp", 512)
                st.set("/rtx/pathtracing/maxBounces", 6)
                st.set("/rtx/pathtracing/optixDenoiser/enabled", True)
            img = self._grab(pumps=40 if mode == "PathTracing" else 8)
            if img is not None:
                # one exposure for every still of a time step (set on the first, from its highlights), so the
                # floors compare as they are; raw frames kept for regrading
                if getattr(self, "_ex_t", None) != t:
                    x = img.astype(np.float32) / 255.0
                    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
                    y = np.percentile(0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2], 99)
                    self._ex, self._ex_t = float(np.clip(0.75 / max(y, 1e-4), 0.1, 4.0)), t
                ex = self._ex
                tag = f"{view}__{floor}__{dome}__{'pt' if mode == 'PathTracing' else 'rt'}"
                Image.fromarray(grade(img, exposure=ex)).save(os.path.join(out_dir, f"t{t:04.1f}_{tag}.jpg"), quality=90)
                os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
                Image.fromarray(img).save(os.path.join(out_dir, "raw", f"t{t:04.1f}_{tag}.png"))
        st.set("/rtx/rendermode", "RaytracedLighting")
        if self.picked is None:
            self.cam.GetFocalLengthAttr().Set(24.0)
        else:
            self.picked = self.load_picked()
        print(f"[rollout] stills at {t:.1f} s -> {out_dir}", flush=True)

    def capture(self):
        e = self.env
        self.cache.append((e.hand.data.root_link_pose_w.torch.clone(), e.hand.data.joint_pos.torch.clone(),
                           e.controller.data.root_link_pose_w.torch.clone()))

    def show(self, k):
        """Put every cell back into cached frame k without stepping physics (the timeline is paused, so the
        hand's links and Fabric are updated by hand)."""
        from isaaclab_physx.physics.physx_manager import PhysxManager as PM

        hp, q, cp = self.cache[k]
        e = self.env
        e.hand.write_root_link_pose_to_sim_index(root_pose=hp)
        e.hand.write_joint_position_to_sim_index(position=q)
        e.controller.write_root_link_pose_to_sim_index(root_pose=cp)
        if PM._view is not None:
            PM._view.update_articulations_kinematic()
        if PM._update_fabric is not None:
            PM._update_fabric(0.0, 0.0)

    def playback_ui(self, start):
        """A small window: play / pause and a time slider over the cached frames (30 fps)."""
        import omni.ui as ui

        n = len(self.cache)
        st = {"k": min(max(start, 0), n - 1), "play": False, "shown": -1, "busy": False, "acc": 0.0}
        win = ui.Window("Playback", width=460, height=120)
        with win.frame:
            with ui.VStack(spacing=6, height=0):
                label = ui.Label("")
                slider = ui.IntSlider(min=0, max=n - 1)
                with ui.HStack(spacing=6):
                    ui.Button("Play / Pause", clicked_fn=lambda: st.update(play=not st["play"]))
                    ui.Button("|< Start", clicked_fn=lambda: st.update(k=0))
                    ui.Button("End >|", clicked_fn=lambda: st.update(k=n - 1, play=False))

        def moved(m):
            if not st["busy"]:
                st.update(k=m.get_value_as_int(), play=False)

        slider.model.add_value_changed_fn(moved)
        self._pb = (win, label, slider, st)

        def tick(dt):
            if st["play"]:
                st["acc"] += dt * 30.0
                adv = int(st["acc"])
                st["acc"] -= adv
                st["k"] = (st["k"] + adv) % n
            if st["k"] != st["shown"]:
                self.show(st["k"])
                st["shown"] = st["k"]
                st["busy"] = True
                slider.model.set_value(st["k"])
                st["busy"] = False
                label.text = f"t = {st['k'] / 30.0:5.2f} s  /  {(n - 1) / 30.0:.2f} s   {'playing' if st['play'] else 'paused'}"
        return tick

    def pick_camera(self, path, view="f18_h0.80_g0.65_p-34", floor="gun30_r25", start=0):
        """Freeze the simulation and hand the editor over: the viewport looks through /World/FilmCam (start:
        `view`), which you fly with the usual viewport navigation (right mouse + WASD, Alt + mouse to
        orbit), its focal length in the Property panel. Its pose is written to `path` whenever it changes,
        also as (eye height, distance back from the near cell, pitch, azimuth, focal). Close the window
        when done."""
        import json as _json

        import omni.kit.app
        import omni.timeline
        from pxr import Usd, UsdGeom

        omni.timeline.get_timeline_interface().pause()
        if self.picked is not None:  # --camera: start from the earlier pick (its pose, lens and --floor)
            self.xf.Set(self.picked)
            self.set_floor(*FLOORS[args.floor])
        else:
            h, g, pitch, az_off, focal = VIEWS[view]
            self.cam.GetFocalLengthAttr().Set(focal)
            self.set_eye(h, g, pitch, self.az + az_off)
            self.set_floor(*FLOORS[floor])
        self.set_dome(*DOMES[args.dome.split(",")[0]])
        try:
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().camera_path = "/World/FilmCam"
        except Exception as e:  # noqa: BLE001
            print("[pick] could not set the viewport camera:", e, flush=True)
        print("[pick] ready: fly the camera in the viewport, close the window when done", flush=True)
        kit = omni.kit.app.get_app()
        tick = self.playback_ui(start) if self.cache else None
        prim, last, k = self.cam.GetPrim(), None, 0
        t_prev = time.time()
        while kit.is_running():
            kit.update()
            now = time.time()
            if tick:
                tick(now - t_prev)
            t_prev = now
            k += 1
            if k % 20:
                continue
            M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float).T
            f = float(self.cam.GetFocalLengthAttr().Get())
            key = (np.round(M, 5).tobytes(), round(f, 3))
            if key == last:
                continue
            last = key
            eye, fwd = M[:3, 3], -M[:3, 2]  # USD cameras look down their -z
            pitch = float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1))))
            az = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
            g = float(np.linalg.norm(eye[:2] - self.look[:2]))
            vfov = float(np.degrees(2 * np.arctan(10.125 / f)))
            out = {"camera_to_world": M.tolist(), "focal_mm": f, "aperture_mm": [36.0, 20.25],
                   "eye_height_m": float(eye[2]), "ground_distance_from_near_cell_m": g, "pitch_deg": pitch,
                   "azimuth_deg": az, "azimuth_offset_deg": az - self.az, "top_edge_deg": pitch + vfov / 2,
                   "near_cell_look": self.look.tolist(), "num_envs": self.env.num_envs, "time_s": args.gui}
            with open(path, "w") as fh:
                _json.dump(out, fh, indent=1)
            print(f"[pick] focal {f:.1f} mm, eye height {eye[2]:.2f} m, back {g:.2f} m, pitch {pitch:+.1f}, "
                  f"azimuth {az:.1f} (top edge {pitch + vfov / 2:+.1f} deg)", flush=True)

    def set_dome(self, hdri, intensity, color=(1.0, 1.0, 1.0)):
        from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
        from pxr import Gf, UsdLux

        if hdri and not hdri.startswith("/Assets"):  # a local image, made here (gradient_sky)
            path = os.path.join(OUT, hdri)
            if not os.path.exists(path):
                gradient_sky(path)
            hdri = path.replace("\\", "/")
        elif hdri:
            hdri = f"{NVIDIA_NUCLEUS_DIR}{hdri}"
        d = UsdLux.DomeLight(self.env.scene.stage.GetPrimAtPath("/World/Dome"))
        d.GetIntensityAttr().Set(float(intensity))
        d.GetColorAttr().Set(Gf.Vec3f(*color))
        d.CreateTextureFileAttr().Set(hdri or "")

    def set_floor(self, color, metallic, roughness):
        from pxr import Gf, Usd, UsdShade

        stage = self.env.scene.stage
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/ground")):
            if prim.IsA(UsdShade.Shader):
                sh = UsdShade.Shader(prim)
                for name, val in (("diffuseColor", Gf.Vec3f(*color)), ("metallic", float(metallic)),
                                  ("roughness", float(roughness))):
                    inp = sh.GetInput(name)
                    if inp:
                        inp.Set(val)
                    else:
                        from pxr import Sdf

                        sh.CreateInput(name, Sdf.ValueTypeNames.Color3f if name == "diffuseColor"
                                       else Sdf.ValueTypeNames.Float).Set(val)

    def frame(self):
        from PIL import Image

        if getattr(self, "pending", False):  # --track plan: nothing to film until the plan starts
            return
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
            Image.fromarray(np.asarray(data)[..., :3]).save(os.path.join(self.frame_dir, f"{self.n:05d}.png"),
                                                             compress_level=1)
            self.n += 1

    def finish(self, path, res, learned=False):
        r0 = res[0]
        single = (f"Placed: {r0['add_mm']:.1f} mm / {r0['rot_deg']:.0f}\u00b0 from the real pose"
                  if len(res) == 1 and r0["placed"] else None)
        encode(self.frame_dir, path, len(res), sum(r["placed"] for r in res), self.size, learned=learned,
               subtitle=args.subtitle or None, single=single)


if __name__ == "__main__":
    main()
    app.close()
