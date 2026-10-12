"""Generate the figures used by docs/overview.html from a demo build.

The architecture diagram (docs/figures/architecture.svg) is hand-authored.

    python scripts/make_doc_figures.py [output/demo]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def _hex(rgb01) -> str:
    r, g, b = (int(round(float(np.clip(v, 0, 1)) * 255)) for v in rgb01)
    return f"#{r:02x}{g:02x}{b:02x}"


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
        out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="64" fill="none" stroke="#cbd5e1"/>')
        out.append(f'<text x="{x + cell / 2}" y="{y + 78}" text-anchor="middle" fill="#334155">{PATCH_NAMES[i]}</text>')
        out.append(f'<text x="{x + cell / 2}" y="{y + 92}" text-anchor="middle" fill="#64748b">'
                   f'ΔE {de_before[i]:.1f} → <tspan fill="#15803d" font-weight="600">{de_after[i]:.2f}</tspan></text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    demo = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output" / "demo"
    FIG.mkdir(parents=True, exist_ok=True)
    preview = demo / "preview.png"
    if not preview.exists():
        raise SystemExit(f"{preview} not found; run `scan2usd demo` first")
    shutil.copyfile(preview, FIG / "demo_preview.png")
    (FIG / "color_chart.svg").write_text(chart_figure(demo), encoding="utf-8")
    print(f"wrote figures to {FIG}")


if __name__ == "__main__":
    main()
