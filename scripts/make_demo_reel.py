"""Demo reel: one ~2.5 min MP4 from the project's figures and animations.

    pip install "imageio[ffmpeg]"
    python scripts/make_demo_reel.py [--out docs/scan2usd_demo.mp4]

Title card -> photos to USD -> simulation demos (MuJoCo, Isaac Sim, Isaac Lab) -> perception -> self-aware assets ->
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


def _place_subtitle() -> str:
    import json

    try:
        s = json.loads((FIG / "holder_place.json").read_text())["summary"]
    except (OSError, KeyError):
        return "The Panda pinches the controller, turns it upright and sets it into the holder"
    d, h = s["hulls"], s["single"]
    return (f"Pick, turn upright, set down: {d['held']}/{d['trials']} held, median {d['add_mm_median']:.1f} mm "
            f"from the real pose; one convex hull: {h['held']}/{h['trials']}")


def _isaac_hand_subtitle() -> str:
    import json

    try:
        s = json.loads((FIG / "isaac_hand_place.json").read_text())["summary"]
    except (OSError, KeyError):
        return "Fingers on the top rim, thumb under the bottom edge; re-aimed from the in-hand pose"
    r = s["decomposed_reaimed"]
    return (f"Fingers on the top rim, thumb under the bottom edge, re-aimed from the in-hand pose; 12 jittered "
            f"trials: {r['placed']}/{r['trials']} placed, median {r['add_mm_median']:.1f} mm / {r['rot_deg_median']:.0f}\u00b0")


def _learned_subtitle() -> str:
    import json

    try:
        s = json.loads((FIG / "learned_hand_place.json").read_text())
    except (OSError, KeyError):
        return "Isaac Lab: a small neural network learned the placement from the scripted hand"
    ev = s["eval"]
    return (f"A small neural network learned it from the scripted hand (cloning + DAgger, "
            f"{s['train_minutes']:.0f} min): {ev['placed']}/{ev['envs']} fresh scenes placed, "
            f"{ev['add_mm_median']:.1f} mm / {ev['rot_deg_median']:.0f}\u00b0")


# (file, title, subtitle, seconds for stills / max seconds for animations)
SEGMENTS = [
    ("hero.jpg", "Phone photos in, physics-ready OpenUSD out",
     "67 real photos → USD asset → 21 convex colliders + mass → Isaac Sim / MuJoCo", 6),
    ("hand_hero.jpg", "From my phone photos to a robot hand placing the scans",
     "The real pair → Shadow Hand picks, turns and seats the scanned controller (Isaac Sim 6, PhysX, path traced)", 5),
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
    ("holder_place.webp", "A robot puts it there", _place_subtitle(), 8),
    ("isaac_hand_place.mp4", "Five fingers: Shadow Hand places it (Isaac Sim 6)", _isaac_hand_subtitle(), 13),
    ("learned_hand_place.mp4", "400 hands at once (Isaac Lab, GPU PhysX)", _learned_subtitle(), 15),
    ("holder_fit_render.jpg", "Path traced beside the other scans",
     "Phone-scanned holder and controller in the measured resting pose, Isaac Sim 6 RTX", 4),
    ("unreal_flythrough.mp4", "The same USD files in Unreal Engine 5.8",
     "Lumen + MegaLights: 80 shadow-casting fairy lights, real time on an RTX 4090", 6),
    ("unreal_hand_place.mp4", "The hand in Unreal: Lumen + MegaLights",
     "The MuJoCo pinch frame for frame, every link keyed in Sequencer, 108 shadow-casting fairy lights", 13),
    ("uncertainty.png", "Assets that know what they don't know",
     "Seen / guessed / sealed surface and mass intervals stored in USD: truth inside for 5 of 6 objects", 4),
    ("splat_buddha.jpg", "Gaussian splat for looks, convex hulls for physics",
     "Rendered from photos it never saw: beats the textured mesh on 18/18 (Buddha PSNR 19.7 → 27.0 dB)", 4),
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
    """MP4 frames at real time, or sped up just enough to fit max_seconds (frames resampled, not
    skipped in whole steps, so a 17 s clip in 11 s plays at 1.5x, not 2x)."""
    frames = [Image.fromarray(f) for f in imageio.mimread(path, memtest=False)]
    speed = max(1.0, len(frames) / (max_seconds * FPS))
    idx = np.minimum(np.round(np.arange(int(len(frames) / speed)) * speed).astype(int), len(frames) - 1)
    return [frames[i] for i in idx]


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
    raw = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=None, pixelformat="yuv420p",
                             macro_block_size=16, ffmpeg_log_level="error",
                             output_params=["-crf", "25", "-preset", "slow", "-movflags", "+faststart"])

    class Fader:
        """Dissolves each segment's first FADE frames from the previous segment's last frame."""
        FADE = 9

        def __init__(self):
            self.last, self.left = None, 0

        def cut(self):
            self.left = self.FADE if self.last is not None else 0

        def append_data(self, f):
            f = np.asarray(f)
            if self.left:
                a = 1.0 - self.left / (self.FADE + 1)
                out = (self.last.astype(np.float32) * (1 - a) + f.astype(np.float32) * a).astype(np.uint8)
                self.left -= 1
                raw.append_data(out)
            else:
                raw.append_data(f)
                self.last = f

        def close(self):
            raw.close()

    writer = Fader()
    total = 0
    title = end_card()  # the same card opens and closes the reel
    for _ in range(int(2.5 * FPS)):
        writer.append_data(title)
    total += int(2.5 * FPS)
    for name, title, subtitle, seconds in SEGMENTS:
        writer.cut()
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
    writer.cut()
    for _ in range(4 * FPS):
        writer.append_data(end)
    total += 4 * FPS
    writer.close()
    print(f"wrote {args.out} ({total / FPS:.0f} s, {args.out.stat().st_size / 1e6:.1f} MB)")
    import json  # the docs quote the length (scripts/sync_holder_numbers.py)

    (FIG / "demo_reel.json").write_text(json.dumps({"seconds": round(total / FPS)}))


if __name__ == "__main__":
    main()
