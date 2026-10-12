"""Texture baking: transfer color from a dense source onto a low-poly mesh with a new UV atlas.

Photogrammetry produces millions of colored points or vertices; simulation and rendering want
a light mesh with a texture. This module unwraps the decimated mesh (xatlas), rasterizes the
atlas to find each texel's 3D position, looks up the nearest source color, and pads the UV
islands so mip-mapping doesn't bleed the background into seams.
"""

from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree


def unwrap(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Copy of `mesh` with an xatlas UV atlas (vertices split along seams, smooth normals kept)."""
    import xatlas

    vmap, faces, uvs = xatlas.parametrize(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.uint32))
    # Normals from the welded mesh: recomputing them after the seam split would shade the seams.
    out = trimesh.Trimesh(mesh.vertices[vmap], faces, vertex_normals=np.asarray(mesh.vertex_normals)[vmap],
                          process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=uvs)
    return out


def _source_colors(source: trimesh.Trimesh, samples: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dense (points, linear-free sRGB8 colors) describing the source's appearance."""
    if source.visual.kind == "vertex":
        return np.asarray(source.vertices), np.asarray(source.visual.vertex_colors[:, :3], np.float64)
    if source.visual.kind == "texture":
        from .mesh_ops import get_base_color_image

        image = get_base_color_image(source)
        uv = getattr(source.visual, "uv", None)
        if image is not None and uv is not None:
            pts, fid = trimesh.sample.sample_surface(source, samples, seed=seed)
            tri = source.vertices[source.faces[fid]]
            bary = trimesh.triangles.points_to_barycentric(tri, pts)
            tuv = (uv[source.faces[fid]] * bary[:, :, None]).sum(1)
            img = np.asarray(image.convert("RGB"), np.float64)
            h, w = img.shape[:2]
            x = np.clip((tuv[:, 0] % 1.0) * (w - 1), 0, w - 1).round().astype(int)
            y = np.clip((1.0 - tuv[:, 1] % 1.0) * (h - 1), 0, h - 1).round().astype(int)
            return pts, img[y, x]
    pts = np.asarray(source.vertices)
    return pts, np.full((len(pts), 3), 200.0)


def _rasterize(uvs: np.ndarray, faces: np.ndarray, size: int):
    """Per texel: covering face index (or -1) and barycentric weights."""
    face_id = np.full((size, size), -1, np.int64)
    bary = np.zeros((size, size, 3))
    p = uvs * size - 0.5  # texel centers
    p[:, 1] = size - 1 - p[:, 1]  # image rows go down
    for f, (a, b, c) in enumerate(faces):
        x, y = p[[a, b, c], 0], p[[a, b, c], 1]
        x0, x1 = int(max(np.floor(x.min()), 0)), int(min(np.ceil(x.max()), size - 1))
        y0, y1 = int(max(np.floor(y.min()), 0)), int(min(np.ceil(y.max()), size - 1))
        if x0 > x1 or y0 > y1:
            continue
        area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        if abs(area) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w0 = ((x[1] - gx) * (y[2] - gy) - (x[2] - gx) * (y[1] - gy)) / area
        w1 = ((x[2] - gx) * (y[0] - gy) - (x[0] - gx) * (y[2] - gy)) / area
        w2 = 1.0 - w0 - w1
        eps = -1e-6
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            continue
        face_id[y0: y1 + 1, x0: x1 + 1][inside] = f
        bary[y0: y1 + 1, x0: x1 + 1][inside] = np.stack([w0, w1, w2], -1)[inside]
    return face_id, bary


def _pad(img: np.ndarray, covered: np.ndarray, padding: int) -> np.ndarray:
    """Edge padding: grow each island outward so filtering at seams samples island values."""
    filled = covered.copy()
    for _ in range(padding):
        grown = ndimage.binary_dilation(filled) & ~filled
        if not grown.any():
            break
        idx = ndimage.distance_transform_edt(~filled, return_distances=False, return_indices=True)
        img[grown] = img[idx[0][grown], idx[1][grown]]
        filled |= grown
    return img


def bake_color(source: trimesh.Trimesh, target: trimesh.Trimesh, size: int = 1024, padding: int = 4,
               samples: int = 2_000_000) -> tuple[trimesh.Trimesh, Image.Image, dict]:
    """Bake the source's color onto `target` (unwrapped here). Returns (mesh with UVs, texture, stats)."""
    low = unwrap(target)
    uvs = np.asarray(low.visual.uv)
    face_id, bary = _rasterize(uvs, np.asarray(low.faces), size)
    covered = face_id >= 0
    tri = low.vertices[low.faces[face_id[covered]]]
    pos = (tri * bary[covered][:, :, None]).sum(1)

    pts, cols = _source_colors(source, samples)
    dist, nearest = cKDTree(pts).query(pos, k=4)
    w = 1.0 / np.maximum(dist, 1e-9)
    texel = (cols[nearest] * w[:, :, None]).sum(1) / w.sum(1, keepdims=True)

    img = np.zeros((size, size, 3))
    img[covered] = texel
    img = _pad(img, covered, padding)
    texture = Image.fromarray(np.clip(img, 0, 255).round().astype(np.uint8))
    low.visual = trimesh.visual.TextureVisuals(uv=uvs, image=texture)
    stats = {"texture_size": size, "atlas_coverage": float(covered.mean()),
             "bake_distance_p95_mm": float(np.percentile(dist[:, 0], 95) * 1000), "uv_vertices": int(len(uvs))}
    return low, texture, stats


def vertex_tangents(mesh: trimesh.Trimesh, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex tangent frames (T, handedness) from UVs, accumulated per face (MikkTSpace-style basis).

    T follows +u on the surface; the bitangent is handedness * cross(N, T).
    """
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    t0, t1, t2 = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-14, 1.0 / np.where(det == 0, 1, det), 0.0)
    tan = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    bit = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
    T = np.zeros_like(v)
    B = np.zeros_like(v)
    for k in range(3):
        np.add.at(T, f[:, k], tan)
        np.add.at(B, f[:, k], bit)
    n = np.asarray(mesh.vertex_normals)
    T = T - n * (T * n).sum(1, keepdims=True)  # Gram-Schmidt against the normal
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    hand = np.where((np.cross(n, T) * B).sum(1) < 0, -1.0, 1.0)
    return T, hand


def bake_normal_map(source: trimesh.Trimesh, low: trimesh.Trimesh, size: int = 1024, padding: int = 4,
                    k: int = 12) -> tuple[Image.Image, dict]:
    """Tangent-space normal map: the dense `source`'s surface normals on the unwrapped `low` mesh.

    The game-art "high-to-low" bake: per texel, the low-poly surface point and its smooth normal and
    UV tangent frame; the high-poly normal there (k nearest dense vertices, distance-weighted),
    expressed in that frame and encoded as RGB = 0.5 * n + 0.5 (OpenGL convention, +Y up in UV space).
    """
    uvs = np.asarray(low.visual.uv)
    face_id, bary = _rasterize(uvs, np.asarray(low.faces), size)
    covered = face_id >= 0
    faces = np.asarray(low.faces)[face_id[covered]]
    w3 = bary[covered][:, :, None]
    pos = (low.vertices[faces] * w3).sum(1)
    n_low = (np.asarray(low.vertex_normals)[faces] * w3).sum(1)
    T, hand = vertex_tangents(low, uvs)
    t_low = (T[faces] * w3).sum(1)
    h = np.sign((hand[faces] * bary[covered]).sum(1))
    n_low /= np.linalg.norm(n_low, axis=1, keepdims=True)
    t_low -= n_low * (t_low * n_low).sum(1, keepdims=True)
    t_low /= np.maximum(np.linalg.norm(t_low, axis=1, keepdims=True), 1e-12)
    b_low = np.cross(n_low, t_low) * h[:, None]

    dist, nearest = cKDTree(np.asarray(source.vertices)).query(pos, k=k)
    cand = np.asarray(source.vertex_normals)[nearest]  # (texels, k, 3)
    # Like a baking cage: ignore source points facing away from the low-poly surface (the other side
    # of a thin wall or a cup's bottom); texels with no valid source keep the flat normal.
    agree = (cand * n_low[:, None, :]).sum(-1) > 0.3
    w = agree / np.maximum(dist, 1e-9)
    n_high = (cand * w[:, :, None]).sum(1)
    norm = np.linalg.norm(n_high, axis=1, keepdims=True)
    n_high = np.where(norm > 1e-12, n_high / np.maximum(norm, 1e-12), n_low)
    ts = np.stack([(n_high * t_low).sum(1), (n_high * b_low).sum(1), (n_high * n_low).sum(1)], -1)
    ts[:, 2] = np.clip(ts[:, 2], 0.05, None)  # never point into the surface
    ts /= np.linalg.norm(ts, axis=1, keepdims=True)

    img = np.zeros((size, size, 3))
    img[..., 2] = 1.0  # flat normal outside islands
    img[covered] = ts
    img = _pad(img, covered, padding)
    rgb = np.clip((img * 0.5 + 0.5) * 255, 0, 255).round().astype(np.uint8)
    deviation = np.degrees(np.arccos(np.clip(ts[:, 2], -1, 1)))
    stats = {"normal_map_size": size, "mean_deviation_deg": float(deviation.mean()),
             "p95_deviation_deg": float(np.percentile(deviation, 95))}
    return Image.fromarray(rgb), stats
