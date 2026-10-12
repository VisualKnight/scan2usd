"""Experiment: can visibility (free-space) carving unseal cavities that photogrammetry missed?

    python scripts/study_carving.py

Poisson reconstruction closes every gap, so a mug whose plain white interior wasn't matched comes
out sealed partway down (+191% volume on the benchmark). Each fused COLMAP point records which
cameras saw it (fused.ply.vis), and the line of sight from a camera to a point it saw is empty.
Voxels of the sealed solid crossed by >= 2 such rays (and not next to any observed point) are
carved. Result on the benchmark: small gains only, because the bottom of each cavity was never
observed: no camera ray reaches it, so nothing proves it empty. The fix is in the capture.

Writes docs/figures/carving_study.{json,png} (cross-section through the printed mug).
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage

from scan2usd.chart import read_colmap_model
from scan2usd.preview import _font
from scan2usd.reconstruct import _median_spacing, isolate_object, read_ply_points

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = {  # benchmark volume error before carving (docs/benchmark.md)
    "ace_coffee_mug_kristen_16_oz_cup": ("Printed mug", 191.5),
    "cole_hardware_bowl_scirocco_yellowblue": ("Ceramic bowl", 82.0),
    "womens_bluefish_2eye_boat_shoe_in_tan": ("Boat shoe", 32.2),
}


def read_visibility(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP fused.ply.vis as CSR: (offsets[n+1], image indices)."""
    data = path.read_bytes()
    n = struct.unpack_from("<Q", data, 0)[0]
    raw = np.frombuffer(data, dtype="<u4", offset=8)
    offsets, idx, k = np.zeros(n + 1, dtype=np.int64), [], 0
    for i in range(n):
        c = int(raw[k])
        idx.append(raw[k + 1: k + 1 + c])
        k += 1 + c
        offsets[i + 1] = offsets[i] + c
    return offsets, np.concatenate(idx).astype(np.int64)


def carve(mesh, points, offsets, vis, centers, min_votes=2, rays_per_point=8, keep_radius=2.0, divisions=160, seed=0):
    """Voxel masks (inside, carved), grid origin and pitch, ray count, max observed depth (voxels)."""
    pitch = float((mesh.bounds[1] - mesh.bounds[0]).max()) / divisions
    vg = mesh.voxelized(pitch).fill()
    inside = np.pad(vg.matrix, 2)
    lo = np.asarray(vg.transform[:3, 3]) - 2.5 * pitch
    shape = np.array(inside.shape)

    rng = np.random.default_rng(seed)
    starts, ends = [], []
    for i in range(len(points)):
        cams = vis[offsets[i]:offsets[i + 1]]
        if len(cams) > rays_per_point:
            cams = rng.choice(cams, rays_per_point, replace=False)
        starts.append(centers[cams])
        ends.append(np.repeat(points[i:i + 1], len(cams), axis=0))
    starts, ends = np.concatenate(starts), np.concatenate(ends)

    d = ends - starts
    length = np.linalg.norm(d, axis=1)
    u = d / length[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t0, t1 = (lo - starts) / u, (lo + shape * pitch - starts) / u
    t_in = np.nanmax(np.minimum(t0, t1), axis=1).clip(min=0)
    t_out = np.minimum(np.nanmin(np.maximum(t0, t1), axis=1), length - keep_radius * pitch)
    votes = np.zeros(int(np.prod(shape)), dtype=np.int32)
    step = 0.5 * pitch
    for c in range(0, len(starts), 20000):
        sl = slice(c, c + 20000)
        n = np.ceil((t_out[sl] - t_in[sl]).clip(min=0) / step).astype(int)
        ray = np.repeat(np.arange(len(n)), n)
        if not len(ray):
            continue
        k = np.arange(len(ray)) - np.repeat(np.cumsum(n) - n, n)
        pts = starts[sl][ray] + u[sl][ray] * (t_in[sl][ray] + k * step)[:, None]
        ijk = np.floor((pts - lo) / pitch).astype(int)
        ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
        key = np.unique(ray[ok].astype(np.int64) * votes.size + np.ravel_multi_index(ijk[ok].T, shape))
        np.add.at(votes, key % votes.size, 1)  # one vote per (ray, voxel)
    votes = votes.reshape(shape)

    seen = np.zeros(shape, dtype=bool)
    ijk = np.floor((points - lo) / pitch).astype(int)
    ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
    seen[tuple(ijk[ok].T)] = True
    near = ndimage.binary_dilation(seen, iterations=int(np.ceil(keep_radius)))
    carved = inside & (votes >= min_votes) & ~near
    depth = ndimage.distance_transform_edt(inside)
    return {"inside": inside, "carved": carved, "seen": seen, "lo": lo, "pitch": pitch, "rays": len(starts),
            "observed_depth_vox": float(depth[seen & inside].max()) if (seen & inside).any() else 0.0,
            "sealed_depth_vox": float(depth.max())}


def load(obj: str):
    w = ROOT / "data" / "synthetic" / obj / "work" / "colmap"
    ply = read_ply_points(w / "dense" / "0" / "fused.ply")
    xyz = np.c_[ply["x"], ply["y"], ply["z"]].astype(np.float64)
    off, vis = read_visibility(w / "dense" / "0" / "fused.ply.vis")
    sp = _median_spacing(xyz)
    aligned, keep, rot, origin, _ = isolate_object(xyz, sp, 2.0 * sp)  # same frame as object_mesh.ply
    _, images = read_colmap_model(w / "dense" / "0" / "sparse")
    centers = (np.array([-(r.T @ np.asarray(t)) for _, r, t in images.values()]) - origin) @ rot.T
    sub_off = np.r_[0, np.cumsum(off[keep + 1] - off[keep])]
    sub_vis = np.concatenate([vis[off[i]:off[i + 1]] for i in keep])
    return trimesh.load(w / "object_mesh.ply", force="mesh"), aligned[keep], sub_off, sub_vis, centers


def section(res, points, title, scale=4):
    """Vertical slice through the solid's center: sealed solid, carved voxels, observed points."""
    inside, carved, lo, pitch = res["inside"], res["carved"], res["lo"], res["pitch"]
    j = int(np.argmax(inside.sum(axis=(0, 2))))  # the y-slice with most material
    sl_in, sl_cv = inside[:, j, :].T[::-1], carved[:, j, :].T[::-1]
    img = np.full(sl_in.shape + (3,), 255, np.uint8)
    img[sl_in] = (203, 213, 225)
    img[sl_cv] = (251, 146, 60)
    y0 = lo[1] + (j + 0.5) * pitch
    band = points[np.abs(points[:, 1] - y0) < 1.5 * pitch]
    ix = np.floor((band[:, 0] - lo[0]) / pitch).astype(int)
    iz = sl_in.shape[0] - 1 - np.floor((band[:, 2] - lo[2]) / pitch).astype(int)
    ok = (ix >= 0) & (ix < sl_in.shape[1]) & (iz >= 0) & (iz < sl_in.shape[0])
    img[iz[ok], ix[ok]] = (37, 99, 235)
    pic = Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    out = Image.new("RGB", (pic.width + 20, pic.height + 46), "white")
    out.paste(pic, (10, 40))
    ImageDraw.Draw(out).text((10, 8), title, fill=(30, 41, 59), font=_font(18))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = ap.parse_args()
    results, panels = {}, []
    for obj, (label, err) in OBJECTS.items():
        t = time.time()
        mesh, pts, off, vis, cent = load(obj)
        r = carve(mesh, pts, off, vis, cent)
        frac = r["carved"].sum() / r["inside"].sum()
        true_fraction = 1 / (1 + err / 100)
        results[obj] = {"label": label, "rays": r["rays"], "removed_pct": round(100 * float(frac), 1),
                        "volume_error_before_pct": err,
                        "volume_error_after_pct": round(((1 - frac) / true_fraction - 1) * 100, 1),
                        "deepest_observed_point_vox": round(r["observed_depth_vox"], 1),
                        "sealed_depth_vox": round(r["sealed_depth_vox"], 1), "seconds": round(time.time() - t)}
        print(label, results[obj], flush=True)
        panels.append(section(r, pts, f"{label}: {err:+.0f}% → {results[obj]['volume_error_after_pct']:+.0f}% volume"))
    h = max(p.height for p in panels)
    sheet = Image.new("RGB", (sum(p.width for p in panels), h + 40), "white")
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width
    ImageDraw.Draw(sheet).text(
        (10, h + 8), "Cross-sections · gray: Poisson solid · orange: carved (cameras saw through it) · "
        "blue: observed points. The sealed bottoms were never seen, so they stay.", fill=(71, 85, 105), font=_font(15))
    sheet.save(args.out / "carving_study.png")
    (args.out / "carving_study.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
