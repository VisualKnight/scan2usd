"""Dependency-free preview renders (numpy z-buffer rasterizer, orthographic camera).

Lets you inspect meshes and USD assets without a 3D application, and produces
deterministic thumbnails for docs and reports.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from .color import linear_to_srgb, srgb_to_linear

DEFAULT_RGB = (200, 200, 205)


def _camera_basis(azimuth: float, elevation: float):
    az, el = np.radians(azimuth), np.radians(elevation)
    to_cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(-to_cam, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, -to_cam)
    return right, up, to_cam


def _vertex_rgb(mesh: trimesh.Trimesh, color) -> np.ndarray:
    if color is not None:
        return np.tile(np.asarray(color, dtype=np.float64)[:3], (len(mesh.vertices), 1))
    if mesh.visual.kind == "vertex":
        return np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float64)
    if mesh.visual.kind == "texture":
        try:
            return np.asarray(mesh.visual.to_color().vertex_colors[:, :3], dtype=np.float64)
        except Exception:
            pass
    return np.tile(np.asarray(DEFAULT_RGB, dtype=np.float64), (len(mesh.vertices), 1))


def render(
    meshes: list[trimesh.Trimesh],
    colors: list | None = None,
    size: int = 420,
    azimuth: float = -60.0,
    elevation: float = 22.0,
    background=(255, 255, 255),
    supersample: int = 2,
    margin: float = 0.08,
    bounds: np.ndarray | None = None,
) -> Image.Image:
    """Render Z-up meshes with per-vertex colors and simple key + fill lighting."""
    colors = colors or [None] * len(meshes)
    right, up, to_cam = _camera_basis(azimuth, elevation)
    res = size * supersample

    all_v = np.vstack([m.vertices for m in meshes]) if bounds is None else trimesh.bounds.corners(bounds)
    sx, sy = all_v @ right, all_v @ up
    cx, cy = (sx.max() + sx.min()) / 2, (sy.max() + sy.min()) / 2
    extent = max(sx.max() - sx.min(), sy.max() - sy.min()) * (1 + 2 * margin)
    scale = res / extent

    key = 0.75 * to_cam + 0.45 * up + 0.35 * right
    key /= np.linalg.norm(key)
    color_buf = np.empty((res, res, 3))
    color_buf[:] = srgb_to_linear(np.asarray(background, dtype=np.float64) / 255.0)
    depth = np.full((res, res), -np.inf)

    for mesh, override in zip(meshes, colors):
        v = np.asarray(mesh.vertices, dtype=np.float64)
        px = (v @ right - cx) * scale + res / 2
        py = res / 2 - (v @ up - cy) * scale
        pz = v @ to_cam
        n = np.asarray(mesh.vertex_normals, dtype=np.float64)
        facing = n @ to_cam
        n = np.where(facing[:, None] < 0, -n, n)  # light both sides of open surfaces
        shade = 0.30 + 0.62 * np.clip(n @ key, 0, 1) + 0.08 * np.clip(n[:, 2], 0, 1)
        lin = srgb_to_linear(_vertex_rgb(mesh, override) / 255.0) * shade[:, None]

        for f in np.asarray(mesh.faces):
            x, y, z = px[f], py[f], pz[f]
            x0, x1 = int(max(np.floor(x.min()), 0)), int(min(np.ceil(x.max()), res - 1))
            y0, y1 = int(max(np.floor(y.min()), 0)), int(min(np.ceil(y.max()), res - 1))
            if x0 > x1 or y0 > y1:
                continue
            area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
            if abs(area) < 1e-12:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w0 = ((x[1] - gx) * (y[2] - gy) - (x[2] - gx) * (y[1] - gy)) / area
            w1 = ((x[2] - gx) * (y[0] - gy) - (x[0] - gx) * (y[2] - gy)) / area
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
            region = depth[y0: y1 + 1, x0: x1 + 1]
            hit = inside & (zz > region)
            if not hit.any():
                continue
            region[hit] = zz[hit]
            c = lin[f]
            col = w0[..., None] * c[0] + w1[..., None] * c[1] + w2[..., None] * c[2]
            color_buf[y0: y1 + 1, x0: x1 + 1][hit] = col[hit]

    img = Image.fromarray(np.round(linear_to_srgb(np.clip(color_buf, 0, 1)) * 255).astype(np.uint8))
    return img.resize((size, size), Image.LANCZOS) if supersample > 1 else img


def hull_colors(n: int) -> list[tuple[int, int, int]]:
    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb((k * 0.61803) % 1.0, 0.55, 0.95)) for k in range(n)
    ]


def _font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(panels: list[tuple[str, Image.Image]], caption_h: int = 34) -> Image.Image:
    w = sum(p.width for _, p in panels)
    h = max(p.height for _, p in panels) + caption_h
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = _font(17)
    x = 0
    for title, img in panels:
        sheet.paste(img, (x, caption_h))
        tw = draw.textlength(title, font=font)
        draw.text((x + (img.width - tw) / 2, 8), title, fill=(40, 48, 64), font=font)
        x += img.width
    return sheet


def preview_file(path: str | Path, size: int = 420) -> Image.Image:
    """Preview a mesh file, or a USD asset (render mesh + collision hulls)."""
    path = Path(path)
    if path.suffix.lower() in (".usd", ".usda", ".usdc"):
        from .usd_io import read_usd

        asset = read_usd(path)
        b = asset.render_mesh.bounds
        panels = [("render mesh", render([asset.render_mesh], size=size, bounds=b))]
        if asset.hulls:
            panels.append((f"collision: {len(asset.hulls)} convex hulls",
                           render(asset.hulls, hull_colors(len(asset.hulls)), size=size, bounds=b)))
        return contact_sheet(panels)
    mesh = trimesh.load(str(path), force="mesh")
    return contact_sheet([(path.name, render([mesh], size=size))])
