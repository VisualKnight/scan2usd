"""Demo reel: one ~90 s MP4 from the project's figures and animations.

    pip install "imageio[ffmpeg]"
    python scripts/make_demo_reel.py [--out docs/scan2usd_demo.mp4]

Title card -> photos to USD -> simulation demos (MuJoCo, Isaac Sim) -> perception -> results ->
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

# (file, title, subtitle, seconds for stills / max seconds for animations)
SEGMENTS = [
    ("hero.jpg", "Phone photos in, physics-ready OpenUSD out",
     "67 real photos → USD asset → 21 convex colliders + mass → Isaac Sim / MuJoCo", 6),
    ("buddha_real.jpg", "Real photos → simulation asset",
     "COLMAP reconstruction, table removed automatically, watertight, convex decomposition", 4),
    ("texture_bake.jpg", "Scored against ground truth",
     "0.6–0.8 mm mean surface error on textured objects; size within ±1.4% from a ColorChecker", 4),
    ("panda_demo.webp", "Why collision geometry matters",
     "MuJoCo · Franka Panda rim-grasps the scanned mug: decomposition picks, one convex hull fails", 9),
    ("isaac_franka_pick.webp", "Same plan, second physics engine",
     "Isaac Sim 6 PhysX reproduces it: lifted 14.3 cm, placed within 14 mm; one hull fails again", 8),
    ("isaac_articulation.webp", "Articulated assets",
     "USD revolute joint from one scan; PhysX drive: 90° → 90.1°, stops at the 105° limit", 5),
    ("clutter_demo.webp", "Clutter",
     "14 scanned objects, 392 convex colliders, settle in a tote with 0.08 mm penetration", 6),
    ("isaac_library.jpg", "A USD asset library",
     "Payloads + collision / friction variant sets; every variant runs in PhysX", 4),
    ("isaac_synth_dataset.jpg", "Synthetic training data in Isaac Sim",
     "1,000 domain-randomized labeled frames in 3.5 minutes (Replicator)", 4),
    ("sim2real_compare.jpg", "Sim-to-real: trained only on renders",
     "On 67 real photos: object found in 100%; distractors cut false positives by 84%", 5),
    ("perception_pick.webp", "See it, locate it, pick it",
     "Detector → fit the scanned mesh to RGB-D → Panda picks at the estimate: 10/10, 4.6 mm median error", 9),
    ("robustness.png", "How accurate must a scan be?",
     "93 simulated picks: collision shape decides; size error within ±5% is tolerated", 6),
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
    for name, title, subtitle, seconds in SEGMENTS:
        path = FIG / name
        if path.suffix == ".webp":
            frames = animation_frames(path, seconds)
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
