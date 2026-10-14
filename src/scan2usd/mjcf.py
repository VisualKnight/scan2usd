"""Turn Scan2USD assets into MuJoCo (MJCF) bodies, for robotics demos and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .usd_io import AssetData, read_usd


@dataclass
class MjAsset:
    """MJCF snippets for one asset: <asset> entries, a <body>, and mesh bytes for the VFS."""

    name: str
    assets_xml: str
    body_xml: str
    files: dict[str, bytes]


def _mean_rgb(mesh: trimesh.Trimesh) -> tuple[float, float, float]:
    if mesh.visual.kind == "vertex":
        c = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float64).mean(0) / 255.0
        return tuple(float(v) for v in c)
    image = getattr(getattr(mesh.visual, "material", None), "image", None)
    if image is not None:  # MuJoCo geoms take one color: use the texture's average
        c = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float64).reshape(-1, 3).mean(0) / 255.0
        return tuple(float(v) for v in c)
    return (0.75, 0.75, 0.78)


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def asset_body(asset: AssetData | str, name: str, pos=(0, 0, 0), quat=(1, 0, 0, 0), collision: str = "hulls",
               friction: float | None = None, free: bool = True, rgba=None, visual_faces: int = 6000) -> MjAsset:
    """Build a free-floating MuJoCo body from a USD asset.

    collision: "hulls" uses the asset's convex decomposition; "single" uses one convex hull
    of the whole object (what a naive pipeline would produce).
    """
    if not isinstance(asset, AssetData):
        asset = read_usd(asset)
    files: dict[str, bytes] = {}
    if collision == "hulls":
        hulls = asset.hulls
    elif collision == "single":
        hulls = [asset.render_mesh.convex_hull]
    else:
        raise ValueError("collision must be 'hulls' or 'single'")

    visual = asset.render_mesh
    if len(visual.faces) > visual_faces:
        # Weld first: textured scans duplicate vertices along UV seams, and decimating those
        # separately tears every seam open (cracks you can see through). MuJoCo draws a flat color,
        # so the UVs aren't needed.
        import trimesh

        visual = trimesh.Trimesh(visual.vertices, visual.faces, process=False)
        visual.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=6)
        visual = visual.simplify_quadric_decimation(face_count=visual_faces)
    files[f"{name}_visual.stl"] = visual.export(file_type="stl")
    assets = [f'<mesh name="{name}_visual" file="{name}_visual.stl"/>']
    for k, h in enumerate(hulls):
        files[f"{name}_hull_{k}.stl"] = h.export(file_type="stl")
        assets.append(f'<mesh name="{name}_hull_{k}" file="{name}_hull_{k}.stl"/>')

    mu = friction if friction is not None else float(asset.physics_material.get("dynamic_friction", 0.5) or 0.5)
    if rgba is None and asset.texture_paths and asset.texture_paths[0].exists():
        from PIL import Image

        img = Image.open(asset.texture_paths[0]).convert("RGB").resize((64, 64))
        rgba = (*(np.asarray(img, dtype=np.float64).reshape(-1, 3).mean(0) / 255.0), 1.0)
    rgba = rgba or (*_mean_rgb(asset.render_mesh), 1.0)
    inertia = asset.inertia_tensor()
    full = f"{inertia[0, 0]:.6g} {inertia[1, 1]:.6g} {inertia[2, 2]:.6g} {inertia[0, 1]:.6g} {inertia[0, 2]:.6g} {inertia[1, 2]:.6g}"
    geoms = [f'<geom name="{name}_visual" type="mesh" mesh="{name}_visual" contype="0" conaffinity="0" group="1" '
             f'rgba="{_fmt(rgba)}"/>']
    geoms += [f'<geom name="{name}_col_{k}" type="mesh" mesh="{name}_hull_{k}" group="3" rgba="1 0.45 0.1 0.22" '
              f'friction="{mu:.4g} 0.005 0.0001" solref="0.005 1"/>' for k in range(len(hulls))]
    body = (f'<body name="{name}" pos="{_fmt(pos)}" quat="{_fmt(quat)}">'
            + ("<freejoint/>" if free else "")
            + f'<inertial pos="{_fmt(asset.center_of_mass)}" mass="{asset.mass:.6g}" fullinertia="{full}"/>'
            + "".join(geoms) + "</body>")
    return MjAsset(name, "".join(assets), body, files)
