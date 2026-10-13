"""Silhouette carving: remove volume that the photos see straight through.

Poisson meshing closes narrow openings (the gap between a figure's legs) because it has no notion
of free space. Photos do: wherever a camera sees bright background through a spot, nothing solid can
be there (the visual-hull argument). `background_votes` counts, per 3D point, how many photos see
background through it; `subtract_volume` cuts a voxel region out of a watertight mesh.

Thresholding on brightness only works where the object is darker than what is behind it, so callers
restrict votes to a region where that holds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def background_votes(points: np.ndarray, cameras: dict, images: dict, image_dir: str | Path,
                     bright: float = 110.0, erode: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Per point: (views that see bright background through it, views it projects into).

    points: (n, 3) in the COLMAP world frame. cameras/images: from `chart.read_colmap_model` of an
    undistorted (PINHOLE / SIMPLE_PINHOLE) model, e.g. dense/0/sparse. A pixel counts as background
    if its gray level exceeds `bright` after the dark (object) region is grown by `erode` pixels, so
    object edges never vote.
    """
    from PIL import Image
    from scipy import ndimage

    votes = np.zeros(len(points), np.int32)
    seen = np.zeros(len(points), np.int32)
    for name, (cam_id, rot, t) in images.items():
        model, w, h, p = cameras[cam_id]
        if model == "PINHOLE":
            fx, fy, cx, cy = p[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        else:
            raise ValueError(f"{name}: {model} camera; use the undistorted model (dense/0/sparse)")
        gray = np.asarray(Image.open(Path(image_dir) / name).convert("L"), dtype=np.float32)
        background = ndimage.binary_erosion(gray > bright, iterations=erode) if erode else gray > bright
        pc = points @ rot.T + t
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * pc[:, 0] / pc[:, 2] + cx
            v = fy * pc[:, 1] / pc[:, 2] + cy
        ok = (pc[:, 2] > 0) & (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
        idx = np.nonzero(ok)[0]
        seen[idx] += 1
        votes[idx[background[v[idx].astype(int), u[idx].astype(int)]]] += 1
    return votes, seen


def subtract_volume(mesh: trimesh.Trimesh, mask: np.ndarray, origin, pitch: float,
                    sigma: float = 0.8) -> trimesh.Trimesh:
    """`mesh` minus the solid occupied by `mask` (a voxel grid: index (i, j, k) is at origin + pitch * (i, j, k)).

    The mask is smoothed (Gaussian, `sigma` voxels) before marching cubes so the cut faces aren't
    stair-stepped; the difference is computed exactly by manifold3d. Fragments are dropped (largest
    body kept) and vertex colors carried over from the nearest original vertex.
    """
    import manifold3d as mf
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage import measure

    if not mask.any():
        return mesh.copy()
    field = ndimage.gaussian_filter(np.pad(mask, 2).astype(np.float32), sigma) if sigma else np.pad(mask, 2).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(field, 0.5)
    cut = trimesh.Trimesh((verts - 2) * pitch + np.asarray(origin, float), faces, process=True)
    if cut.volume < 0:
        cut.invert()

    def manifold(t):
        return mf.Manifold(mf.Mesh(vert_properties=np.asarray(t.vertices, np.float32),
                                   tri_verts=np.asarray(t.faces, np.uint32)))

    res = (manifold(mesh) - manifold(cut)).to_mesh()
    out = trimesh.Trimesh(np.asarray(res.vert_properties)[:, :3], np.asarray(res.tri_verts), process=True)
    bodies = out.split(only_watertight=False)
    if len(bodies) > 1:
        out = max(bodies, key=lambda b: len(b.faces))
    if mesh.visual.kind == "vertex":
        _, j = cKDTree(mesh.vertices).query(out.vertices)
        out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=np.asarray(mesh.visual.vertex_colors)[j])
    return out
