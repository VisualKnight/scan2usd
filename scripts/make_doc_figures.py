"""Generate the SVG figures used by docs/overview.html from a demo build.

    python scripts/make_doc_figures.py [output/demo]
"""

from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

from scan2usd.color import (
    PATCH_NAMES,
    Calibration,
    apply_ccm_linear,
    delta_e_2000,
    linear_rgb_to_lab,
    linear_to_srgb,
    reference_linear,
    sample_chart,
)
from scan2usd.usd_io import read_usd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def _hex(rgb01) -> str:
    r, g, b = (int(round(float(np.clip(v, 0, 1)) * 255)) for v in rgb01)
    return f"#{r:02x}{g:02x}{b:02x}"


def _poly(points: np.ndarray) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _silhouette_image(tris: np.ndarray, ox: float, oy: float, size: int, ss: int = 2) -> str:
    """Rasterize projected triangles into a translucent PNG embedded in the SVG."""
    import base64
    import io

    from PIL import ImageDraw

    mask = Image.new("L", (size * ss, size * ss), 0)
    draw = ImageDraw.Draw(mask)
    for t in tris:
        draw.polygon([((x - ox) * ss, (y - oy) * ss) for x, y in t], fill=255)
    rgba = Image.new("RGBA", mask.size, (100, 116, 139, 0))
    rgba.putalpha(mask.point(lambda v: v * 70 // 255))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG", optimize=True)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<image x="{ox}" y="{oy}" width="{size}" height="{size}" href="data:image/png;base64,{data}"/>'


def hull_figure(usd_path: Path) -> str:
    """Render mesh silhouette (grey) overlaid with each convex collision hull (colored)."""
    asset = read_usd(usd_path)
    views = [("Side view (X-Z)", 0, 2), ("Top view (X-Y)", 0, 1)]
    lo, hi = asset.render_mesh.bounds
    pad = 0.006
    span = float(np.max(hi - lo)) + 2 * pad
    size, gap, label_h = 300, 40, 28
    scale = size / span
    width = len(views) * size + (len(views) - 1) * gap
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {size + label_h}" '
           f'font-family="Segoe UI, sans-serif" font-size="13">']
    for vi, (title, a, b) in enumerate(views):
        ox = vi * (size + gap)

        def proj(p, a=a, b=b, ox=ox):
            x = ox + (p[:, a] - lo[a] + pad) * scale
            y = label_h + size - (p[:, b] - lo[b] + pad) * scale
            return np.c_[x, y]

        out.append(f'<text x="{ox + size / 2}" y="16" text-anchor="middle" fill="#cbd5e1">{title}</text>')
        out.append(f'<rect x="{ox}" y="{label_h}" width="{size}" height="{size}" rx="8" fill="#0b1220" stroke="#334155"/>')
        out.append(_silhouette_image(proj(asset.render_mesh.vertices)[asset.render_mesh.faces], ox, label_h, size))
        for k, h in enumerate(asset.hulls):
            p = proj(h.vertices)
            ring = p[ConvexHull(p).vertices]
            color = _hex(colorsys.hsv_to_rgb((k * 0.61803) % 1.0, 0.65, 0.95))
            out.append(f'<polygon points="{_poly(ring)}" fill="{color}" fill-opacity="0.28" '
                       f'stroke="{color}" stroke-width="1.2"/>')
    out.append("</svg>")
    return "\n".join(out)


def chart_figure(demo_dir: Path) -> str:
    """Each patch split in three: as captured | after CCM | reference, with CIEDE2000."""
    img = np.asarray(Image.open(demo_dir / "inputs" / "chart_photo.png").convert("RGB"))
    corners = np.array(json.loads((demo_dir / "inputs" / "chart_corners.json").read_text()))
    cal = Calibration.load(demo_dir / "asset" / "color_calibration.json")
    measured = sample_chart(img, corners)
    corrected = apply_ccm_linear(measured, cal.matrix)
    ref = reference_linear()
    de_before = delta_e_2000(linear_rgb_to_lab(measured), linear_rgb_to_lab(ref))
    de_after = delta_e_2000(linear_rgb_to_lab(corrected), linear_rgb_to_lab(ref))

    cols, cell, gap, h_cell = 6, 120, 10, 96
    sw = cell / 3
    width = cols * cell + (cols - 1) * gap
    height = 4 * (h_cell + gap)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'font-family="Segoe UI, sans-serif" font-size="11">']
    for i in range(24):
        r, c = divmod(i, cols)
        x, y = c * (cell + gap), r * (h_cell + gap)
        for j, rgb in enumerate((measured[i], corrected[i], ref[i])):
            out.append(f'<rect x="{x + j * sw:.1f}" y="{y}" width="{sw:.1f}" height="64" '
                       f'fill="{_hex(linear_to_srgb(rgb))}"/>')
        out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="64" fill="none" stroke="#1e293b"/>')
        out.append(f'<text x="{x + cell / 2}" y="{y + 78}" text-anchor="middle" fill="#cbd5e1">{PATCH_NAMES[i]}</text>')
        out.append(f'<text x="{x + cell / 2}" y="{y + 92}" text-anchor="middle" fill="#94a3b8">'
                   f'ΔE {de_before[i]:.1f} → <tspan fill="#4ade80">{de_after[i]:.2f}</tspan></text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    demo = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output" / "demo"
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "collision_hulls.svg").write_text(hull_figure(demo / "asset" / "ceramic_mug.usda"), encoding="utf-8")
    (FIG / "color_chart.svg").write_text(chart_figure(demo), encoding="utf-8")
    print(f"wrote figures to {FIG}")


if __name__ == "__main__":
    main()
