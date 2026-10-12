"""Synthetic demo capture: a ceramic mug "scanned" under a warm, color-cast light.

The demo exercises every pipeline stage without real photos, and because the true
glaze color is known it measures how well color calibration recovers it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from .color import delta_e_2000, linear_rgb_to_lab, render_chart, srgb_to_linear

# Ground-truth glaze albedo (8-bit sRGB) and a warm tungsten-like camera response with
# channel cross-talk and slight under-exposure.
TRUE_GLAZE_SRGB8 = np.array([46, 92, 140])
DEBRIS_SRGB8 = [90, 80, 70]
CAMERA_CAST = 0.85 * np.array([
    [1.10, 0.06, -0.02],
    [0.03, 0.95, 0.02],
    [-0.03, 0.08, 0.70],
])


def make_mug(sections: int = 96) -> trimesh.Trimesh:
    """Watertight mug, Z-up, meters, base on z=0: 8.2 cm wide body, 9.5 cm tall."""
    h, r_out, r_in, floor = 0.095, 0.041, 0.036, 0.007
    outer = trimesh.creation.cylinder(radius=r_out, height=h, sections=sections)
    outer.apply_translation([0, 0, h / 2])
    inner = trimesh.creation.cylinder(radius=r_in, height=h, sections=sections)
    inner.apply_translation([0, 0, floor + h / 2])
    handle = trimesh.creation.torus(major_radius=0.026, minor_radius=0.0065, major_sections=48, minor_sections=16)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    handle.apply_translation([r_out + 0.005, 0, 0.05])
    body = trimesh.boolean.union([outer, handle], engine="manifold")
    return trimesh.boolean.difference([body, inner], engine="manifold")


def _captured_srgb8(true_srgb8: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(np.asarray(true_srgb8, dtype=np.float64) / 255.0)
    cap = np.clip(lin @ CAMERA_CAST.T, 0, 1)
    return np.round(np.where(cap <= 0.0031308, cap * 12.92, 1.055 * cap ** (1 / 2.4) - 0.055) * 255).astype(int)


def make_scan(seed: int = 0) -> trimesh.Trimesh:
    """Degrade the clean mug like a real photogrammetry export would be."""
    rng = np.random.default_rng(seed)
    mug = make_mug()
    mug = mug.subdivide_to_size(max_edge=0.0025, max_iter=10)
    mug.vertices += rng.normal(0, 0.0002, mug.vertices.shape)  # ~0.2 mm scan noise

    glaze = _captured_srgb8(TRUE_GLAZE_SRGB8)
    shade = rng.normal(0, 2.0, (len(mug.vertices), 1))  # texture noise
    colors = np.clip(glaze[None, :] + shade, 0, 255).astype(np.uint8)
    mug.visual = trimesh.visual.ColorVisuals(mug, vertex_colors=np.c_[colors, np.full(len(colors), 255, np.uint8)])

    debris = []
    for pos in ([0.09, 0.02, 0.0], [-0.07, -0.08, 0.01]):
        s = trimesh.creation.icosphere(subdivisions=2, radius=0.004)
        s.apply_translation(pos)
        s.visual = trimesh.visual.ColorVisuals(s, vertex_colors=np.tile([*DEBRIS_SRGB8, 255], (len(s.vertices), 1)))
        debris.append(s)
    scan = trimesh.util.concatenate([mug] + debris)

    # Typical photogrammetry export: Y-up, centimeters, off-center.
    scan.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    scan.apply_scale(100.0)
    scan.apply_translation([12.0, 3.0, -7.0])
    return scan


def glaze_delta_e(colors_srgb8: np.ndarray) -> float:
    """Mean CIEDE2000 between per-vertex albedo and the ground-truth glaze."""
    lin = srgb_to_linear(np.asarray(colors_srgb8)[:, :3] / 255.0)
    ref = srgb_to_linear(TRUE_GLAZE_SRGB8 / 255.0)
    return float(delta_e_2000(linear_rgb_to_lab(lin), linear_rgb_to_lab(ref[None, :])).mean())


def write_demo_inputs(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scan = make_scan()
    scan_path = out / "mug_scan.ply"
    scan.export(str(scan_path))
    img, corners = render_chart(CAMERA_CAST)
    chart_path = out / "chart_photo.png"
    Image.fromarray(img).save(chart_path)
    corners_path = out / "chart_corners.json"
    corners_path.write_text(json.dumps(corners.tolist()))
    return {
        "scan": scan_path,
        "chart_image": chart_path,
        "chart_corners": corners.tolist(),
        "corners_path": corners_path,
        "captured_colors": np.asarray(scan.visual.vertex_colors),
    }


def run_demo(out_dir: str | Path, target_faces: int = 20_000, run_sim: bool = True):
    """Write the synthetic capture and build it. Returns (BuildResult, glaze dE2000 dict)."""
    from .pipeline import BuildOptions, build_asset

    out = Path(out_dir)
    inputs = write_demo_inputs(out / "inputs")
    opts = BuildOptions(
        name="ceramic_mug", material="ceramic", up_axis="y", units="cm",
        target_faces=target_faces, chart_image=inputs["chart_image"],
        chart_corners=inputs["chart_corners"], usda=True, run_sim=run_sim,
    )
    res = build_asset(inputs["scan"], out / "asset", opts)
    colors = inputs["captured_colors"]
    mug_only = colors[~np.all(colors[:, :3] == DEBRIS_SRGB8, axis=1)]
    glaze = {
        "captured": glaze_delta_e(mug_only),
        "calibrated": glaze_delta_e(np.asarray(res.mesh.visual.vertex_colors)),
    }
    res.summary["demo_glaze_delta_e2000"] = glaze
    res.report_path.write_text(json.dumps(res.summary, indent=2), encoding="utf-8")
    return res, glaze
