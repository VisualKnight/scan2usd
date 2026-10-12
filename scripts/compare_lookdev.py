"""Look-dev check: compare the path-traced render with the real photo from the same camera.

    python scripts/compare_lookdev.py

Inputs: output/lookdev/camera.json (scripts/lookdev_camera.py) and output/pathtrace/lookdev{,_mask}.png
(scripts/isaac_pathtrace.py). Measures
  * registration: distance (px) from the render's silhouette to the nearest strong edge in the photo
    (geometry + scale + solved camera, all at once);
  * appearance: median CIEDE2000 inside the object after one global exposure match (the real
    lighting is unknown, so absolute brightness isn't comparable).
Writes docs/figures/lookdev.{jpg,json}.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from scan2usd.color import delta_e_2000, linear_rgb_to_lab, linear_to_srgb, srgb_to_linear
from scan2usd.preview import _font

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cam = json.loads((ROOT / "output" / "lookdev" / "camera.json").read_text())
    render = np.asarray(Image.open(ROOT / "output" / "pathtrace" / "lookdev.png").convert("RGB"))
    mask = np.asarray(Image.open(ROOT / "output" / "pathtrace" / "lookdev_mask.png")) > 127
    h, w = render.shape[:2]
    photo = np.asarray(Image.open(cam["photo"]).convert("RGB").resize((w, h), Image.LANCZOS))

    # Registration: render silhouette vs. photo edges.
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY), (5, 5), 0), 40, 110) > 0
    dist_to_edge = ndimage.distance_transform_edt(~edges)
    contour = mask & ~ndimage.binary_erosion(mask, iterations=1)
    d = dist_to_edge[contour]
    inner = ndimage.binary_erosion(mask, iterations=6)  # skip the anti-aliased rim

    # Appearance: color difference inside the object after matching exposure (one gain) and after
    # matching white balance (per-channel gains, as a camera would); the real lighting is unknown.
    p_lin = srgb_to_linear(photo[inner] / 255.0)
    r_lin = srgb_to_linear(render[inner] / 255.0)
    gain = float(np.median(p_lin.sum(1)) / max(np.median(r_lin.sum(1)), 1e-6))
    wb = np.median(p_lin, axis=0) / np.maximum(np.median(r_lin, axis=0), 1e-6)
    de = delta_e_2000(linear_rgb_to_lab(p_lin), linear_rgb_to_lab(np.clip(r_lin * gain, 0, 1)))
    de_wb = delta_e_2000(linear_rgb_to_lab(p_lin), linear_rgb_to_lab(np.clip(r_lin * wb, 0, 1)))
    res = {"silhouette_to_photo_edge_px": {"median": float(np.median(d)), "p90": float(np.percentile(d, 90))},
           "image_size": [w, h], "silhouette_fraction_within_3px": float(np.mean(d <= 3)),
           "exposure_gain": gain, "delta_e2000_median": float(np.median(de)), "delta_e2000_p90": float(np.percentile(de, 90)),
           "white_balance_gains": wb.tolist(), "delta_e2000_wb_median": float(np.median(de_wb)),
           "delta_e2000_wb_p90": float(np.percentile(de_wb, 90)), "photo": Path(cam["photo"]).name}

    # Figure: photo | render (white-balanced, over the photo) | wipe | silhouette on photo.
    matched = (linear_to_srgb(np.clip(srgb_to_linear(render / 255.0) * wb, 0, 1)) * 255).astype(np.uint8)
    comp = photo.copy()
    comp[mask] = matched[mask]
    wipe = photo.copy()
    wipe[:, w // 2:][mask[:, w // 2:]] = matched[:, w // 2:][mask[:, w // 2:]]
    sil = photo.copy()
    sil[ndimage.binary_dilation(contour, iterations=1)] = (255, 60, 30)
    tiles = [(photo, "real photo (undistorted)"), (comp, "path-traced asset, same camera, white-balanced"),
             (wipe, "left: photo · right: render"), (sil, "render silhouette on the photo")]
    tw = 480
    th = round(h * tw / w)
    sheet = Image.new("RGB", (tw * 4, th + 64), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (img, title) in enumerate(tiles):
        sheet.paste(Image.fromarray(img).resize((tw, th), Image.LANCZOS), (i * tw, 34))
        dr.text((i * tw + 8, 8), title, fill=(30, 41, 59), font=_font(17))
    dr.line([(2 * tw + tw // 2, 34), (2 * tw + tw // 2, 34 + th)], fill=(255, 255, 255), width=2)
    dr.text((8, th + 40), f"silhouette within 3 px of a photo edge: {res['silhouette_fraction_within_3px'] * 100:.0f}% "
            f"(median {res['silhouette_to_photo_edge_px']['median']:.1f} px at {w}x{h}) · color inside the object: "
            f"median ΔE2000 {res['delta_e2000_median']:.1f} after exposure match, {res['delta_e2000_wb_median']:.1f} after "
            f"white balance", fill=(71, 85, 105), font=_font(15))
    out = ROOT / "docs" / "figures"
    sheet.save(out / "lookdev.jpg", quality=88)
    (out / "lookdev.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
