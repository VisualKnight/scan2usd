"""Look-dev camera: put a USD camera exactly where a real photo was taken, in the asset's frame.

    python scripts/lookdev_camera.py [--photo 00021._c.png] [--asset output/pbr/buddha_head/buddha_head.usda]

Chains the transforms the pipeline applied: COLMAP world -> table-aligned frame (isolate_object:
rotation + origin) -> asset frame (uniform scale s, then recentered on the bounding box bottom).
The photo's COLMAP pose (world -> camera, +Y down, +Z forward) becomes a USD camera (+Y up,
looking down -Z). Intrinsics come from COLMAP's undistorted (pinhole) model, so the undistorted
photo is the reference image. Writes output/lookdev/camera.json for scripts/isaac_pathtrace.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from scan2usd.chart import read_colmap_model
from scan2usd.reconstruct import _median_spacing, isolate_object, read_ply_points
from scan2usd.usd_io import read_usd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, default=ROOT / "data" / "work" / "buddha")
    p.add_argument("--photo", default="00021._c.png")
    p.add_argument("--asset", type=Path, default=ROOT / "output" / "pbr" / "buddha_head" / "buddha_head.usda")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "lookdev" / "camera.json")
    args = p.parse_args()
    dense = args.workspace / "dense" / "0"

    ply = read_ply_points(dense / "fused.ply")
    xyz = np.c_[ply["x"], ply["y"], ply["z"]].astype(np.float64)
    sp = _median_spacing(xyz)
    _, _, rot, origin, _ = isolate_object(xyz, sp, 2.0 * sp)  # same call as reconstruct()

    # Asset frame: P = s * A + t, with A the aligned reconstruction (object_mesh.ply).
    report = json.loads((args.asset.parent / "build_report.json").read_text())
    s = float(report["scale_applied"])
    recon = trimesh.load(args.workspace / "object_mesh.ply", force="mesh")
    asset = read_usd(args.asset)
    t = asset.render_mesh.bounds[0] - s * recon.bounds[0]
    fit = np.abs((s * recon.bounds[1] + t) - asset.render_mesh.bounds[1]).max()

    cameras, images = read_colmap_model(dense / "sparse")
    cam_id, r_w2c, t_w2c = images[args.photo]
    model, w, h, params = cameras[cam_id]
    if model not in ("PINHOLE", "SIMPLE_PINHOLE"):
        raise SystemExit(f"expected an undistorted pinhole model, got {model}")
    fx, fy, cx, cy = (params[0], params[0], params[1], params[2]) if model == "SIMPLE_PINHOLE" else params[:4]

    # x_cam = R (rot^T (P - t) / s + origin) + t_c   ->  metric: R rot^T (P - t) + s (R origin + t_c)
    r = np.asarray(r_w2c) @ rot.T
    tc = s * (np.asarray(r_w2c) @ origin + np.asarray(t_w2c)) - r @ t
    flip = np.diag([1.0, -1.0, -1.0])  # COLMAP camera axes -> USD camera axes
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = flip @ r, flip @ tc
    c2w = np.linalg.inv(w2c)

    aperture = 36.0  # mm; any value works, focal length scales with it
    cam = {
        "photo": str((dense / "images" / args.photo).resolve()), "width": int(w), "height": int(h),
        "focal_length_mm": float(fx * aperture / w), "horizontal_aperture_mm": aperture,
        "vertical_aperture_mm": float(aperture * h / w * fx / fy),
        "horizontal_aperture_offset_mm": float((cx - w / 2) * aperture / w),
        "vertical_aperture_offset_mm": float(-(cy - h / 2) * aperture / w),
        # USD matrices are row-major with row vectors: the transpose of c2w.
        "camera_to_world_usd": c2w.T.tolist(), "camera_center_m": c2w[:3, 3].tolist(),
        "asset": str(args.asset.resolve()), "scale": s, "bounds_fit_error_m": float(fit),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cam, indent=2))
    dist = np.linalg.norm(c2w[:3, 3] - asset.render_mesh.bounds.mean(0))
    print(f"camera {np.round(c2w[:3, 3], 3)} ({dist:.2f} m from the object), f = {cam['focal_length_mm']:.1f} mm "
          f"on {aperture:.0f} mm, {w}x{h}; bounds fit {fit * 1000:.2f} mm")


if __name__ == "__main__":
    main()
