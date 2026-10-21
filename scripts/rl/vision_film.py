"""Film the vision policy: one scene, close up, beside what its two RGB-D cameras see and what the pose networks
make of it (vision_policy.py).

    set OMNI_KIT_ACCEPT_EULA=YES
    tools\\il\\venv\\Scripts\\python.exe scripts\\rl\\vision_film.py --policy output\\rl\\vision\\policy\\policy.pt
    tools\\il\\venv\\Scripts\\python.exe scripts\\rl\\vision_film.py --preview      # film-camera stills only

Frame (1920 x 1080): the film camera; on the right the policy's two colour views and one depth view at their
real resolution (160 x 120, scaled up), with the network's estimates drawn in: the controller where it thinks
it is (green dots: the settled pose until the grasp, then the live in-hand estimate) and where it thinks the
controller belongs in the holder (amber). Below: the estimates' errors against the simulator, live.
Writes output/rl/vision/<name>.mp4, stills and <name>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="")
parser.add_argument("--scene_net", default="")
parser.add_argument("--hand_net", default="")
parser.add_argument("--seed", type=int, default=5000, help="scene seed (cell index --cell): the evaluation's scenes")
parser.add_argument("--cell", type=int, default=0)
parser.add_argument("--looks", type=int, default=-1, help="random-looks seed (-1: nominal)")
parser.add_argument("--name", default="vision_policy")
parser.add_argument("--preview", action="store_true", help="no policy: stills of the film camera during a teacher episode")
parser.add_argument("--eye", type=float, nargs=3, default=(0.60, 0.70, 0.82))
parser.add_argument("--look", type=float, nargs=3, default=(0.10, 0.13, 0.12))
parser.add_argument("--focal", type=float, default=22.0)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import hand_teacher as T  # noqa: E402
import policy as PL  # noqa: E402
import vision_env as VE  # noqa: E402
import vision_policy as VP  # noqa: E402
from hand_place_env import SPEC  # noqa: E402

VIS = os.path.join(T.ROOT, "output", "rl", "vision")
FW, FH = 1440, 810
PW, PH = 480, 360
GREEN, AMBER = (60, 220, 90), (255, 176, 32)


def font(sz):
    for f in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


class Projector:
    """Env-relative points -> pixels of one of the policy's cameras (pinhole, from vision_env.CAMS)."""

    def __init__(self, name: str, w: int, h: int):
        eye, target = VE.CAMS[name]
        self.eye = np.array(eye)
        self.R = Rotation.from_quat(VE.look_quat(eye, target)).as_matrix()  # columns: forward, left, up
        self.f = w * 24.0 / 20.955
        self.cx, self.cy = w / 2, h / 2

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        pc = (pts - self.eye) @ self.R
        z = np.clip(pc[:, 0], 1e-3, None)
        return np.c_[self.cx - self.f * pc[:, 1] / z, self.cy - self.f * pc[:, 2] / z]


def transform(p7: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ Rotation.from_quat(p7[3:7]).as_matrix().T + p7[:3]


def panel(rgb: np.ndarray, label: str, proj: Projector | None, overlays, f) -> Image.Image:
    im = Image.fromarray(rgb).resize((PW, PH), Image.NEAREST)
    d = ImageDraw.Draw(im)
    s = PW / rgb.shape[1]
    if proj is not None:
        for pts, col in overlays:
            for u, v in proj(pts) * s:
                d.ellipse([u - 1.6, v - 1.6, u + 1.6, v + 1.6], fill=col)
    d.rectangle([0, 0, d.textlength(label, font=f) + 16, 30], fill=(15, 23, 42))
    d.text((8, 3), label, fill=(226, 232, 240), font=f)
    return im


def depth_rgb(d: np.ndarray) -> np.ndarray:
    x = np.clip((1.3 - d) / 1.0, 0, 1) * (d > 0)
    return (np.stack([x * 0.55 + 0.1 * (d > 0), x * 0.8, x], -1) * 255).astype(np.uint8)


class Grade:
    """One gain for the whole film (from its first frame: the floor's 98th percentile to 0.55), in linear light."""

    def __init__(self):
        self.gain = None

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        x = rgb.astype(np.float32) / 255.0
        lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
        if self.gain is None:
            self.gain = float(np.clip(0.55 / max(np.percentile(lin, 98), 1e-3), 1.0, 4.0))
        lin = lin * self.gain
        lin = lin / (1 + 0.15 * lin)  # a soft shoulder
        y = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.clip(lin, 0, None), 1 / 2.4) - 0.055)
        return (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    cfg = VE.VisionHandPlaceEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1.0e5
    cfg.film = VE.camera_cfg("film", FW, FH, eye=tuple(args.eye), target=tuple(args.look), focal=args.focal,
                             data_types=("rgb",))
    env = VE.VisionHandPlaceEnv(cfg)
    env.reset()
    dev = env.device
    env.randomize_looks(None if args.looks < 0 else np.random.default_rng([args.looks, 5]))
    scene = [SPEC.scene(np.random.default_rng([args.seed, args.cell]))]
    pts = PL.controller_points(400)
    projs = {k: Projector(k, VE.IMG_W, VE.IMG_H) for k in VE.CAMS}
    os.makedirs(VIS, exist_ok=True)

    if args.preview:
        run = PL.Runner(env, SPEC)
        shots, k, grade = {}, [0], Grade()

        def cb():
            if k[0] in (20, 200, 330, 420, 520):
                shots[k[0]] = grade(env.cams["film"].data.output["rgb"][0, ..., :3].cpu().numpy())
            k[0] += 1

        run.episode(scene, collect=False, frame_cb=cb)
        Image.fromarray(np.concatenate([np.concatenate([shots[a], shots[b]], 1) for a, b in
                                        ((20, 200), (330, 520))], 0)).resize((FW, FH)).save(
            os.path.join(VIS, "film_preview.jpg"), quality=90)
        print("[film] preview written", flush=True)
        env.close()
        return

    pol = PL.load(args.policy or os.path.join(T.ROOT, "scripts", "rl", "checkpoints", "hand_place_policy.pt"), dev)
    nets = [VP.load_net(p or d, dev) for p, d in ((args.scene_net, VP.SCENE_NET), (args.hand_net, VP.HAND_NET))]
    run = VP.VisionRunner(env, SPEC, *nets)
    tmp = tempfile.mkdtemp(prefix="vision_film_")
    f_lab, f_big, f_mid, f_small = font(19), font(34), font(24), font(20)
    k, log, grade, grip = [0], [], Grade(), {}

    def cb():
        img = env.images()[0].cpu().numpy()  # (V, 4, H, W)
        film = grade(env.cams["film"].data.output["rgb"][0, ..., :3].cpu().numpy())
        true7 = env.controller_pose()[0].cpu().numpy()
        seen = getattr(run, "seen", None)
        live = getattr(run, "live", None)
        ov, errs = [], {}
        if seen is not None:
            ov.append((transform(seen["target"][0], pts), AMBER))
            held = True
            if live is not None:  # held while the controller keeps its distance to the forearm (display only)
                dist = float(np.linalg.norm(true7[:3] - env.forearm_pose()[0, :3].cpu().numpy()))
                grip.setdefault("d0", dist)
                held = abs(dist - grip["d0"]) < 0.02 and not grip.get("released")
                grip["released"] = not held
            if held:
                est = live[0] if live is not None else seen["start"][0]
                ov.append((transform(est, pts), GREEN))
            e = run.errors
            errs = {"start": float(e["start_mm"][0]), "goal": float(e["target_mm"][0])}
            if live is not None:
                errs["hand"] = float(np.linalg.norm(live[0][:3] - true7[:3]) * 1000) if held else None
        frame = Image.new("RGB", (1920, 1080), (241, 245, 249))
        frame.paste(Image.fromarray(film), (0, 0))
        rgb = [(img[v, :3].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8) for v in range(2)]
        names = list(VE.CAMS)
        frame.paste(panel(rgb[0], "camera A · colour", projs[names[0]], ov, f_lab), (FW, 0))
        frame.paste(panel(rgb[1], "camera B · colour", projs[names[1]], ov, f_lab), (FW, PH))
        frame.paste(panel(depth_rgb(img[0, 3]), "camera A · depth", projs[names[0]], ov, f_lab), (FW, 2 * PH))
        d = ImageDraw.Draw(frame)
        d.text((40, FH + 22), "Vision policy: the hand sees only two RGB-D cameras", fill=(15, 23, 42), font=f_big)
        d.text((40, FH + 72), "Pose networks find the controller and the holder in the images;", fill=(71, 85, 105),
               font=f_mid)
        d.text((40, FH + 104), "the learned hand acts on their estimates, never on the simulator's state.",
               fill=(71, 85, 105), font=f_mid)
        y = FH + 152
        d.ellipse([40, y + 7, 54, y + 21], fill=GREEN)
        d.text((62, y), "controller, as estimated", fill=(51, 65, 85), font=f_small)
        d.ellipse([320, y + 7, 334, y + 21], fill=AMBER)
        d.text((342, y), "where it belongs, as estimated", fill=(51, 65, 85), font=f_small)
        if errs:
            parts = [f"settled pose {errs['start']:.1f} mm", f"goal {errs['goal']:.1f} mm"]
            if "hand" in errs:
                parts.append(f"in the hand {errs['hand']:.1f} mm" if errs["hand"] is not None else "released")
            d.text((40, y + 36), "estimate error vs the simulator:  " + "  ·  ".join(parts), fill=(51, 65, 85), font=f_small)
        log.append({"frame": k[0], **errs})
        frame.save(os.path.join(tmp, f"{k[0]:05d}.jpg"), quality=92)
        k[0] += 1

    res, _ = run.episode(scene, policy=pol, use_policy=torch.ones(1, dtype=torch.bool, device=dev), collect=False,
                         frame_cb=cb)
    out = {**PL.summarize(res), "add_mm": float(res["add_mm"][0]), "rot_deg": float(res["rot_deg"][0]),
           "estimates": run.error_summary(), "seed": args.seed, "cell": args.cell, "looks": args.looks}
    # the result, over the last 2.5 s
    n = k[0]
    msg = (f"Placed: {out['add_mm']:.1f} mm / {out['rot_deg']:.0f}\u00b0 from the real pose" if out["placed"]
           else "Not placed")
    for i in range(max(0, n - 75), n):
        p = os.path.join(tmp, f"{i:05d}.jpg")
        im = Image.open(p)
        d = ImageDraw.Draw(im)
        w = d.textlength(msg, font=f_big)
        d.rounded_rectangle([FW - w - 70, FH - 80, FW - 30, FH - 22], 10, fill=(255, 255, 255))
        d.text((FW - w - 50, FH - 76), msg, fill=(21, 128, 61) if out["placed"] else (185, 28, 28), font=f_big)
        im.save(p, quality=92)
    exe = os.path.join(T.ROOT, ".venv", "Lib", "site-packages", "imageio_ffmpeg", "binaries")
    exe = next(os.path.join(exe, f) for f in os.listdir(exe) if f.startswith("ffmpeg"))
    mp4 = os.path.join(VIS, f"{args.name}.mp4")
    subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", "30", "-i", os.path.join(tmp, "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4], check=True)
    for tag, t in (("start", 0.1), ("grasp", 0.42), ("carry", 0.6), ("place", 0.8), ("end", 0.97)):
        Image.open(os.path.join(tmp, f"{int(t * (n - 1)):05d}.jpg")).save(mp4.replace(".mp4", f"_{tag}.jpg"), quality=90)
    out["frames"] = log
    PL.write_json(os.path.join(VIS, f"{args.name}.json"), out)
    print(f"[film] {mp4}: {json.dumps({k_: v for k_, v in out.items() if k_ != 'frames'})}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
