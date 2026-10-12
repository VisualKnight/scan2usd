"""Docs figures from the path-traced renders (output/pathtrace, scripts/isaac_pathtrace.py).

    python scripts/make_render_figures.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from scan2usd.preview import _font

ROOT = Path(__file__).resolve().parents[1]
PT = ROOT / "output" / "pathtrace"
FIG = ROOT / "docs" / "figures"


def label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    w = d.textlength(text, font=_font(20)) + 20
    d.rectangle([10, 10, 10 + w, 44], fill=(255, 255, 255))
    d.text((20, 14), text, fill=(30, 41, 59), font=_font(20))
    return img


def main() -> None:
    Image.open(PT / "beauty_pt.png").convert("RGB").save(FIG / "render_beauty.jpg", quality=90)
    rt = label(Image.open(PT / "beauty_rt.png").convert("RGB").resize((960, 540), Image.LANCZOS),
               "RTX real-time path tracing · 1 frame")
    pt = label(Image.open(PT / "beauty_pt.png").convert("RGB").resize((960, 540), Image.LANCZOS),
               "RTX path tracing · 1,024 spp · depth of field")
    s = Image.new("RGB", (1920, 540))
    s.paste(rt, (0, 0))
    s.paste(pt, (960, 0))
    s.save(FIG / "render_modes.jpg", quality=88)

    off = Image.open(PT / "normals_off.png").convert("RGB").resize((640, 640), Image.LANCZOS)
    on = Image.open(PT / "normals_on.png").convert("RGB").resize((640, 640), Image.LANCZOS)
    s = Image.new("RGB", (1280, 640))
    s.paste(label(off, "50k triangles, vertex colors"), (0, 0))
    s.paste(label(on, "50k triangles + baked 2K albedo & normal maps"), (640, 0))
    s.save(FIG / "render_normals.jpg", quality=88)
    print("wrote render_beauty.jpg, render_modes.jpg, render_normals.jpg")


if __name__ == "__main__":
    main()
