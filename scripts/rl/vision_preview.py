"""Look through the vision policy's cameras: contact sheets of both views (colour and depth) at a few moments of
scripted episodes, nominal and randomized looks; and, with --bench, the cost of rendering them for many cells.

    set OMNI_KIT_ACCEPT_EULA=YES
    tools\\il\\venv\\Scripts\\python.exe scripts\\rl\\vision_preview.py [--num_envs 4] [--bench 512]

Writes output/rl/vision/preview_<look>.jpg (and bench.json).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--bench", type=int, default=0, help="also time a teacher episode with this many cells")
parser.add_argument("--times", default="0.5,6,9,12,15", help="moments (s from the episode start) to show")
parser.add_argument("--dim", action="store_true",
                    help="also probe exposure: brightness at dimmer lights, with and without auto exposure, and at film ISOs")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import hand_teacher as T  # noqa: E402
import policy as PL  # noqa: E402
from hand_place_env import SPEC  # noqa: E402
from vision_env import VisionHandPlaceEnv, VisionHandPlaceEnvCfg  # noqa: E402

OUT = os.path.join(T.ROOT, "output", "rl", "vision")


def make_env(n):
    cfg = VisionHandPlaceEnvCfg()
    cfg.scene.num_envs = n
    cfg.episode_length_s = 1.0e5
    env = VisionHandPlaceEnv(cfg)
    env.reset()
    return env


def tile(img: torch.Tensor) -> Image.Image:
    """(4, H, W) colour + depth -> colour beside a depth map (0.3-1.3 m)."""
    rgb = (img[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    d = img[3].cpu().numpy()
    dn = np.clip((1.3 - d) / 1.0, 0, 1) * (d > 0)
    dep = (np.stack([dn] * 3, -1) * 255).astype(np.uint8)
    return Image.fromarray(np.concatenate([rgb, dep], 1))


def sheet(frames: dict, n_show: int, path: str):
    """frames: t -> (n, views, 4, H, W). Rows: cells x views; columns: moments."""
    ts = sorted(frames)
    first = frames[ts[0]]
    n, v, _, h, w = first.shape
    rows = min(n, n_show) * v
    W, H = 2 * w, h
    im = Image.new("RGB", (len(ts) * W, rows * H + 14), (20, 20, 20))
    dr = ImageDraw.Draw(im)
    for j, t in enumerate(ts):
        dr.text((j * W + 4, 1), f"t = {t:.1f} s", fill=(230, 230, 230))
        for i in range(min(n, n_show)):
            for k in range(v):
                im.paste(tile(frames[t][i, k]), (j * W, 14 + (i * v + k) * H))
    im.save(path, quality=92)
    print(f"[preview] {path}", flush=True)


def run(env, run_, seed, times, look_rng):
    env.randomize_looks(look_rng)
    want = {int(round(t * 30)): t for t in times}
    frames, count = {}, [0]

    def cb():
        if count[0] in want:
            frames[want[count[0]]] = env.images().clone()
        count[0] += 1

    scenes = [SPEC.scene(np.random.default_rng([seed, i])) for i in range(env.num_envs)]
    res, _ = run_.episode(scenes, collect=False, frame_cb=cb)
    return frames, res


def main():
    os.makedirs(OUT, exist_ok=True)
    times = [float(t) for t in args.times.split(",")]
    if args.bench:
        env = make_env(args.bench)
        run_ = PL.Runner(env, SPEC)
        t0 = time.time()
        _, res = run(env, run_, 7100, [], np.random.default_rng(3))
        wall = time.time() - t0
        out = {"envs": args.bench, "episode_s": wall, "result": PL.summarize(res),
               "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        PL.write_json(os.path.join(OUT, "bench.json"), out)
        print(f"[preview] bench {json.dumps(out)}", flush=True)
        env.close()
        return
    env = make_env(args.num_envs)
    run_ = PL.Runner(env, SPEC)
    for name, rng in (("nominal", None), ("random1", np.random.default_rng(1)), ("random2", np.random.default_rng(2))):
        frames, res = run(env, run_, 7000, times, rng)
        f0 = frames[min(frames)]
        print(f"[preview] {name}: {PL.summarize(res)} looks {env.looks} "
              f"rgb mean {float(f0[:, :, :3].mean()):.3f} p5 {float(f0[:, :, :3].quantile(0.05)):.3f}", flush=True)
        sheet(frames, 2, os.path.join(OUT, f"preview_{name}.jpg"))
    if args.dim:
        import carb

        st = carb.settings.get_settings()
        print(f"[preview] histogram {st.get('/rtx/post/histogram/enabled')} iso {st.get('/rtx/post/tonemap/filmIso')} "
              f"op {st.get('/rtx/post/tonemap/op')} cm2 {st.get('/rtx/post/tonemap/cm2Factor')}", flush=True)
        env.randomize_looks(None)
        stage = env.scene.stage
        for hist in (True, False):
            st.set("/rtx/post/histogram/enabled", hist)
            for s in (1.0, 0.3, 0.1):
                stage.GetPrimAtPath("/World/Dome").GetAttribute("inputs:intensity").Set(700.0 * s)
                stage.GetPrimAtPath("/World/Sun").GetAttribute("inputs:intensity").Set(750.0 * s)
                run_.step(6)
                f = env.images()
                print(f"[preview] histogram {hist} light x{s}: rgb mean {float(f[:, :, :3].mean()):.3f}", flush=True)
        stage.GetPrimAtPath("/World/Dome").GetAttribute("inputs:intensity").Set(700.0)
        stage.GetPrimAtPath("/World/Sun").GetAttribute("inputs:intensity").Set(750.0)
        for iso in (100.0, 50.0, 25.0):
            st.set("/rtx/post/tonemap/filmIso", iso)
            run_.step(6)
            f = env.images()
            print(f"[preview] iso {iso}: rgb mean {float(f[:, :, :3].mean()):.3f} p95 {float(f[:, :, :3].quantile(0.95)):.3f}",
                  flush=True)
            sheet({0.0: f}, 2, os.path.join(OUT, f"probe_iso{int(iso)}.jpg"))
    env.close()


if __name__ == "__main__":
    main()
    app.close()
