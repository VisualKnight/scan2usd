"""Demo reel: one ~90 s MP4 from the project's figures and animations.

    pip install "imageio[ffmpeg]"
    python scripts/make_demo_reel.py [--out docs/scan2usd_demo.mp4]

Title card -> photos to USD -> simulation demos (MuJoCo, Isaac Sim) -> perception -> self-aware assets ->
links. Every clip is an existing figure in docs/figures, so the reel stays in sync with them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from scan2usd.preview import _font

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
W, H, FPS = 1280, 720, 30
BAR = 92
INK, MUTED, ACCENT = (30, 41, 59), (100, 116, 139), (37, 99, 235)

def _holder_subtitle() -> str:
    import json

    try:
        st = json.loads((ROOT / "output" / "holder_fit" / "state.json").read_text())["sim"]
        isaac = json.loads((FIG / "isaac_holder_fit.json").read_text())["results"]["decomposed"]
    except (OSError, KeyError):
        return "Scanned controller dropped into the scanned holder, compared with the real pose"
    d, h = st["decomposed"], st["one hull"]
    return (f"MuJoCo: {d['held_in_holder']}/{d['of']} held, median {d['add_mm_median']:.1f} mm from the real pose "
            f"(Isaac Sim PhysX: {isaac['held_in_holder']}/{isaac['of']}); one convex hull: {h['held_in_holder']}/{h['of']}")


# (file, title, subtitle, seconds for stills / max seconds for animations)
SEGMENTS = [
    ("hero.jpg", "Phone photos in, physics-ready OpenUSD out",
     "67 real photos → USD asset → 21 convex colliders + mass → Isaac Sim / MuJoCo", 6),
    ("buddha_real.jpg", "Real photos → simulation asset",
     "COLMAP reconstruction, table removed automatically, watertight, convex decomposition", 4),
    ("texture_bake.jpg", "Scored against ground truth",
     "0.6–0.8 mm mean surface error on textured objects; size within ±1.4% from a ColorChecker", 3),
    ("render_beauty.jpg", "Path traced in Isaac Sim 6 RTX",
     "PBR look presets, baked 2K albedo + normal maps, HDRI lighting, physical depth of field", 5),
    ("panda_demo.webp", "Why collision geometry matters",
     "MuJoCo · Franka Panda rim-grasps the scanned mug: decomposition picks, one convex hull fails", 8),
    ("isaac_franka_pick.webp", "Same plan, second physics engine",
     "Isaac Sim 6 PhysX, plan from the grasp stored in the USD: placed within 5.5 mm; one hull fails again", 7),
    ("hand_grasp.webp", "Dexterous grasping",
     "Shadow Hand (24 joints): 1,440 sampled grasps over the asset library, lift + shake test", 5),
    
    ("sim2real_compare.jpg", "Sim-to-real: trained only on renders",
     "On 67 real photos: object found in 100%; distractors cut false positives by 84%", 5),
    ("perception_pick.webp", "See it, locate it, pick it",
     "Detector → fit the scanned lion to RGB-D → Panda picks at the estimate: 10/10, 0.6 mm median error", 7),
    ("holder_fit_lead.jpg", "Own phone photos: does the scan behave like the real thing?",
     "A holder and a controller, each scanned alone, then photographed together as the answer key", 5),
    ("holder_fit.webp", "Scanned controller settles into the scanned holder", _holder_subtitle(), 6),
    ("holder_fit_render.jpg", "Path traced beside the other scans",
     "Phone-scanned holder and controller in the measured resting pose, Isaac Sim 6 RTX", 4),
    ("unreal_flythrough.mp4", "The same USD files in Unreal Engine 5.8",
     "Lumen + MegaLights: 89 shadow-casting fairy lights, real time on an RTX 4090", 6),
    ("uncertainty.png", "Assets that know what they don't know",
     "Seen / guessed / sealed surface and mass intervals stored in USD: truth inside for 5 of 6 objects", 4),
    ("splat_buddha.jpg", "Gaussian splat for looks, convex hulls for physics",
     "Rendered from photos it never saw: beats the textured mesh on 18/18 (Buddha PSNR 19.7 → 27.0 dB)", 4),
    ("isaac_franka_pick_splat.webp", "Franka picks a splat mug",
     "Same plan and physics, the mug drawn by its Gaussian splat", 5),
]

def canvas(title: str, subtitle: str) -> Image.Image:
    img = Image.new("RGB", (W, H), (248, 250, 252))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, BAR], fill="white")
    d.rectangle([0, BAR - 3, W, BAR], fill=ACCENT)
    d.text((32, 14), title, fill=INK, font=_font(32))
    d.text((32, 56), subtitle, fill=MUTED, font=_font(19))
    return img


def place(frame: Image.Image, title: str, subtitle: str) -> np.ndarray:
    img = canvas(title, subtitle)
    box_w, box_h = W - 48, H - BAR - 24
    s = min(box_w / frame.width, box_h / frame.height)
    f = frame.convert("RGB").resize((round(frame.width * s), round(frame.height * s)), Image.LANCZOS)
    img.paste(f, ((W - f.width) // 2, BAR + 12 + (box_h - f.height) // 2))
    return np.asarray(img)


def animation_frames(path: Path, max_seconds: float):
    """Frames resampled to FPS; longer animations are sped up to fit max_seconds."""
    im = Image.open(path)
    frames, times, t = [], [], 0.0
    for k in range(im.n_frames):
        im.seek(k)
        frames.append(im.convert("RGB").copy())
        times.append(t)
        t += (im.info.get("duration", 33) or 33) / 1000.0
    speed = max(1.0, t / max_seconds)
    n = int(t / speed * FPS)
    idx = np.searchsorted(times, np.arange(n) * speed / FPS, side="right") - 1
    return [frames[max(0, i)] for i in idx]


def video_frames(path: Path, max_seconds: float):
    """MP4 frames, sped up (frames skipped) to fit max_seconds."""
    frames = [Image.fromarray(f) for f in imageio.mimread(path, memtest=False)]
    step = max(1, int(np.ceil(len(frames) / (max_seconds * FPS))))
    return frames[::step]


def end_card() -> np.ndarray:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 200), "Scan2USD", fill=INK, font=_font(64))
    d.text((80, 290), "Real-world capture → physics-ready OpenUSD for robotics simulation", fill=MUTED, font=_font(26))
    d.text((80, 380), "visualknight.github.io/scan2usd", fill=ACCENT, font=_font(30))
    d.text((80, 425), "github.com/VisualKnight/scan2usd", fill=ACCENT, font=_font(30))
    d.text((80, 520), "Yibo Liu · personal R&D project · 2026", fill=MUTED, font=_font(20))
    return np.asarray(img)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "scan2usd_demo.mp4")
    p.add_argument("--poster", type=Path, default=ROOT / "docs" / "figures" / "demo_poster.jpg")
    args = p.parse_args()
    writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p",
                                macro_block_size=16, ffmpeg_log_level="error")
    total = 0
    title = end_card()  # the same card opens and closes the reel
    for _ in range(int(2.5 * FPS)):
        writer.append_data(title)
    total += int(2.5 * FPS)
    for name, title, subtitle, seconds in SEGMENTS:
        path = FIG / name
        if path.suffix in (".webp", ".mp4"):
            frames = animation_frames(path, seconds) if path.suffix == ".webp" else video_frames(path, seconds)
            for f in frames:
                writer.append_data(place(f, title, subtitle))
            n = len(frames)
            for _ in range(FPS):  # hold the last frame (the result) for a second
                writer.append_data(place(frames[-1], title, subtitle))
            n += FPS
        else:
            still = place(Image.open(path), title, subtitle)
            if name == "hero.jpg":
                Image.fromarray(still).save(args.poster, quality=88)
            n = int(seconds * FPS)
            for _ in range(n):
                writer.append_data(still)
        total += n
        print(f"  {name}: {n / FPS:.1f} s")
    end = end_card()
    for _ in range(4 * FPS):
        writer.append_data(end)
    total += 4 * FPS
    writer.close()
    print(f"wrote {args.out} ({total / FPS:.0f} s, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
