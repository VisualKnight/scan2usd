"""Held-out photos vs. the asset rendered from the same solved camera: textured mesh vs. Gaussian splat.

    python scripts/compare_splat.py [--object buddha]

Inputs: output/splat/cams*/camera_*.json (scripts/lookdev_camera.py) and the Isaac renders
output/splat/isaac/<prefix><photo>_{mesh,splat}.png (+ the mesh's object mask) from
scripts/isaac_splat.py. The splat never saw these photos; the mesh's texture was baked from all
photos, these included, which can only favor the mesh.

Same protocol as scripts/compare_lookdev.py, applied to both renders: inside the object (mask eroded
to skip the anti-aliased rim), match white balance with per-channel gains (the real lighting is
unknown), then CIEDE2000, PSNR and SSIM against the photo.
Writes docs/figures/splat_<object>.{jpg,json}; with --video, also docs/figures/splat_drop.mp4.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.metrics import structural_similarity

from scan2usd.color import delta_e_2000, linear_rgb_to_lab, linear_to_srgb, srgb_to_linear
from scan2usd.preview import _font

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = {
    "buddha": {"cams": "output/splat/cams/camera_*.json", "prefix": "heldout_", "title": "Buddha head (real photos)"},
    "mug": {"cams": "output/splat/cams_mug/camera_*.json", "prefix": "mug_heldout_",
            "title": "speckled mug (synthetic photos)"},
}


def score(photo, render, inner):
    p_lin = srgb_to_linear(photo[inner] / 255.0)
    r_lin = srgb_to_linear(render[inner] / 255.0)
    wb = np.median(p_lin, axis=0) / np.maximum(np.median(r_lin, axis=0), 1e-6)
    matched = (linear_to_srgb(np.clip(srgb_to_linear(render / 255.0) * wb, 0, 1)) * 255).astype(np.uint8)
    de = delta_e_2000(linear_rgb_to_lab(p_lin), linear_rgb_to_lab(np.clip(r_lin * wb, 0, 1)))
    err = (photo[inner].astype(float) - matched[inner].astype(float)) / 255.0
    psnr = float(-10 * np.log10(np.mean(err ** 2)))
    g = lambda im: im.astype(float).mean(2) / 255.0  # noqa: E731
    _, ssim_map = structural_similarity(g(photo), g(matched), data_range=1.0, full=True)
    return {"delta_e2000_wb_median": float(np.median(de)), "psnr_db": psnr,
            "ssim": float(ssim_map[inner].mean()), "wb_gains": wb.tolist()}, matched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="buddha", choices=sorted(OBJECTS))
    ap.add_argument("--renders", type=Path, default=ROOT / "output" / "splat" / "isaac")
    ap.add_argument("--rows", type=int, default=3, help="views shown in the figure")
    ap.add_argument("--video", action="store_true", help="also assemble drop_frames/ into splat_drop.mp4")
    args = ap.parse_args()
    cfg = OBJECTS[args.object]
    out = ROOT / "docs" / "figures"

    views, tiles = [], []
    for c in sorted(glob.glob(str(ROOT / cfg["cams"]))):
        cam = json.loads(Path(c).read_text())
        stem = Path(cam["photo"]).stem.replace("._c", "")
        base = args.renders / f"{cfg['prefix']}{stem}"
        if not Path(f"{base}_mesh.png").exists():
            continue
        mesh = np.asarray(Image.open(f"{base}_mesh.png").convert("RGB"))
        splat = np.asarray(Image.open(f"{base}_splat.png").convert("RGB"))
        mask = np.asarray(Image.open(f"{base}_mesh_mask.png")) > 127
        h, w = mesh.shape[:2]
        photo = np.asarray(Image.open(cam["photo"]).convert("RGB").resize((w, h), Image.LANCZOS))
        inner = ndimage.binary_erosion(mask, iterations=6)
        if inner.sum() < 500:
            continue
        sm, mm = score(photo, mesh, inner)
        ss, ms = score(photo, splat, inner)
        views.append({"photo": Path(cam["photo"]).name, "object_pixels": int(inner.sum()), "mesh": sm, "splat": ss})
        tiles.append((inner.sum(), photo, mm, ms, mask))

    keys = ("delta_e2000_wb_median", "psnr_db", "ssim")
    summary = {look: {k: float(np.median([v[look][k] for v in views])) for k in keys} for look in ("mesh", "splat")}
    summary["splat_better"] = {
        "delta_e2000": sum(v["splat"]["delta_e2000_wb_median"] < v["mesh"]["delta_e2000_wb_median"] for v in views),
        "psnr": sum(v["splat"]["psnr_db"] > v["mesh"]["psnr_db"] for v in views),
        "ssim": sum(v["splat"]["ssim"] > v["mesh"]["ssim"] for v in views), "of": len(views)}
    res = {"object": args.object, "heldout_views": len(views), "summary": summary, "views": views}
    ablation = ROOT / "output" / "splat" / f"splat_{args.object}_unmasked.json"
    if ablation.exists():  # same protocol, splat trained on the raw photos (background cropped afterwards)
        res["unmasked_splat"] = json.loads(ablation.read_text())["summary"]["splat"]

    # Figure: for the largest views, a crop around the object: photo | mesh | splat (both white-balanced).
    tiles.sort(key=lambda t: -t[0])
    tw = 300
    rows = []
    for _, photo, mm, ms, mask in tiles[: args.rows]:
        ys, xs = np.nonzero(mask)
        pad = int(0.08 * max(np.ptp(ys), np.ptp(xs)))
        y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
        side = max(y1 - y0, x1 - x0)
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
        crop = lambda im: Image.fromarray(im[y0:y0 + side, x0:x0 + side]).resize((tw, tw), Image.LANCZOS)  # noqa: E731
        rows.append([crop(photo), crop(mm), crop(ms)])
    titles = [f"held-out photo ({'real' if args.object == 'buddha' else 'synthetic'})", "textured mesh (RTX)",
              "Gaussian splat (RTX)"]
    sheet = Image.new("RGB", (tw * 3, 34 + tw * len(rows) + 40), "white")
    dr = ImageDraw.Draw(sheet)
    for k, t in enumerate(titles):
        dr.text((k * tw + 8, 8), t, fill=(30, 41, 59), font=_font(17))
    for r, row in enumerate(rows):
        for k, im in enumerate(row):
            sheet.paste(im, (k * tw, 34 + r * tw))
    s_m, s_s = summary["mesh"], summary["splat"]
    dr.text((8, 34 + tw * len(rows) + 10),
            f"{len(views)} held-out views · median ΔE2000 {s_m['delta_e2000_wb_median']:.1f} → "
            f"{s_s['delta_e2000_wb_median']:.1f} · PSNR {s_m['psnr_db']:.1f} → {s_s['psnr_db']:.1f} dB · "
            f"SSIM {s_m['ssim']:.2f} → {s_s['ssim']:.2f}", fill=(71, 85, 105), font=_font(15))
    sheet.save(out / f"splat_{args.object}.jpg", quality=88)
    (out / f"splat_{args.object}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"object": args.object, "summary": summary}, indent=2))

    if args.video:
        import imageio.v2 as imageio

        frames = sorted((args.renders / "drop_frames").glob("*.png"))
        if frames:
            imageio.mimwrite(out / "splat_drop.mp4", [np.asarray(Image.open(f).convert("RGB")) for f in frames],
                             fps=30, quality=8, macro_block_size=None)
            print(f"wrote {out / 'splat_drop.mp4'} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
