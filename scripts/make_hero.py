"""README hero image: real photo -> USD asset -> convex colliders -> Isaac Sim scene.

    python scripts/make_hero.py

Inputs: AliceVision Buddha photos (data/external/buddha), the built asset
(output/buddha/buddha_head.usda) and a path-traced Isaac Sim render
(output/pathtrace/hero_pt.png, scripts/isaac_pathtrace.py). Writes docs/figures/hero.jpg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from scan2usd.preview import _font, hull_colors, render
from scan2usd.usd_io import read_usd

ROOT = Path(__file__).resolve().parents[1]
INK = (30, 41, 59)
MUTED = (100, 116, 139)
ACCENT = (37, 99, 235)


def square(img: Image.Image, cx: float, cy: float = 0.5, side: float = 1.0) -> Image.Image:
    """Square crop of side * min(w, h), centered at (cx, cy) in relative coordinates, clamped."""
    s = round(min(img.size) * side)
    x = min(max(0, round(cx * img.width - s / 2)), img.width - s)
    y = min(max(0, round(cy * img.height - s / 2)), img.height - s)
    return img.crop((x, y, x + s, y + s))


def arrow(d: ImageDraw.ImageDraw, x0: int, x1: int, y: int) -> None:
    d.line([(x0, y), (x1 - 14, y)], fill=ACCENT, width=6)
    d.polygon([(x1, y), (x1 - 20, y - 14), (x1 - 20, y + 14)], fill=ACCENT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--photo", type=Path, default=ROOT / "data" / "external" / "buddha" / "buddha" / "00021._c.png")
    p.add_argument("--asset", type=Path, default=ROOT / "output" / "buddha" / "buddha_head.usda")
    p.add_argument("--sim-frame", type=Path, default=ROOT / "output" / "pathtrace" / "hero_pt.png")
    p.add_argument("--azimuth", type=float, default=232.0)
    p.add_argument("--tile", type=int, default=400)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "hero.jpg")
    args = p.parse_args()
    t = args.tile

    asset = read_usd(args.asset)
    view = {"azimuth": args.azimuth, "elevation": 28.0, "margin": 0.05}
    bounds = asset.render_mesh.bounds
    panels = [
        ("1 · Photos", "67 real photos, no scanner",
         square(ImageOps.exif_transpose(Image.open(args.photo)).convert("RGB"), 0.54).resize((t, t), Image.LANCZOS)),
        ("2 · OpenUSD asset", "meters, Z-up, color-calibrated",
         render([asset.render_mesh], size=t, bounds=bounds, **view)),
        ("3 · Physics", f"{len(asset.hulls)} convex colliders, mass, inertia",
         render(list(asset.hulls), colors=hull_colors(len(asset.hulls)), size=t, bounds=bounds, **view)),
        ("4 · Simulation", "Isaac Sim 6 · RTX path traced",
         square(Image.open(args.sim_frame).convert("RGB"), 0.5).resize((t, t), Image.LANCZOS)),
    ]

    gap, pad, head = 56, 24, 84
    w = 4 * t + 3 * gap + 2 * pad
    h = head + t + pad
    sheet = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(sheet)
    for i, (title, sub, img) in enumerate(panels):
        x = pad + i * (t + gap)
        d.text((x, 10), title, fill=INK, font=_font(30))
        d.text((x, 48), sub, fill=MUTED, font=_font(21))
        sheet.paste(img, (x, head))
        d.rectangle([x, head, x + t - 1, head + t - 1], outline=(226, 232, 240), width=2)
        if i < 3:
            arrow(d, x + t + 10, x + t + gap - 10, head + t // 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=88)
    print(f"wrote {args.out} ({w}x{h})")


if __name__ == "__main__":
    main()
