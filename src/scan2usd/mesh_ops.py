"""Mesh loading and cleanup for photogrammetry / splat-derived meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254}

# Rotations that bring an input up-axis to +Z (the asset convention, as in Isaac Sim).
_UP_TO_Z = {
    "z": np.eye(4),
    "y": trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0]),
    "-y": trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1, 0, 0]),
    "-z": trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]),
}


@dataclass
class CleanReport:
    faces_in: int = 0
    faces_out: int = 0
    components_in: int = 0
    components_kept: int = 0
    degenerate_removed: int = 0
    duplicate_removed: int = 0
    holes_filled: bool = False
    watertight: bool = False
    notes: list[str] = field(default_factory=list)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    path = Path(path)
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"no triangle mesh found in {path}")
    if path.suffix.lower() == ".obj" and mesh.visual.kind == "texture" and get_base_color_image(mesh) is None:
        image = _find_obj_texture(path)
        if image is not None:
            mesh.visual.material.image = image
    return mesh


def _find_obj_texture(obj_path: Path):
    """Locate a .mtl's map_Kd image that isn't next to the OBJ (common in asset archives)."""
    from PIL import Image

    names = []
    for mtl in obj_path.parent.glob("*.mtl"):
        for line in mtl.read_text(errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].lower() == "map_kd":
                names.append(Path(parts[-1]).name)
    for name in names:
        for root in (obj_path.parent, obj_path.parent.parent):
            hits = sorted(root.rglob(name))
            if hits:
                return Image.open(hits[0])
    return None


def has_texture(mesh: trimesh.Trimesh) -> bool:
    vis = mesh.visual
    if vis.kind != "texture" or getattr(vis, "uv", None) is None:
        return False
    return get_base_color_image(mesh) is not None


def get_base_color_image(mesh: trimesh.Trimesh):
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return None
    return getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)


def has_vertex_colors(mesh: trimesh.Trimesh) -> bool:
    return mesh.visual.kind == "vertex"


def clean_mesh(mesh: trimesh.Trimesh, min_component_ratio: float = 0.02) -> tuple[trimesh.Trimesh, CleanReport]:
    """Remove scan debris and fix common topology problems.

    Components whose surface area is below `min_component_ratio` of the largest are
    treated as floaters (background fragments, splat noise) and dropped.
    """
    rep = CleanReport(faces_in=len(mesh.faces))
    mesh = mesh.copy()

    n = len(mesh.faces)
    mesh.update_faces(mesh.nondegenerate_faces())
    rep.degenerate_removed = n - len(mesh.faces)
    n = len(mesh.faces)
    mesh.update_faces(mesh.unique_faces())
    rep.duplicate_removed = n - len(mesh.faces)
    mesh.remove_unreferenced_vertices()

    parts = mesh.split(only_watertight=False)
    rep.components_in = len(parts)
    if len(parts) > 1:
        areas = np.array([p.area for p in parts])
        keep = [p for p, a in zip(parts, areas) if a >= min_component_ratio * areas.max()]
        rep.components_kept = len(keep)
        if len(keep) < len(parts):
            rep.notes.append(f"dropped {len(parts) - len(keep)} floating fragment(s)")
        mesh = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    else:
        rep.components_kept = 1

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        rep.holes_filled = True
    trimesh.repair.fix_normals(mesh)
    rep.watertight = bool(mesh.is_watertight)
    if not rep.watertight:
        rep.notes.append("mesh is not watertight; mass will be estimated from collision hulls")
    rep.faces_out = len(mesh.faces)
    return mesh, rep


def orient_and_scale(
    mesh: trimesh.Trimesh,
    up_axis: str = "z",
    units: str = "m",
    scale_to: float | None = None,
    scale_axis: str = "max",
) -> tuple[trimesh.Trimesh, float]:
    """Convert to meters / Z-up, optionally rescale so one dimension matches a measurement,
    then place the origin at the bottom-center of the bounding box (resting on z=0).

    Returns the mesh and the total uniform scale applied.
    """
    if up_axis not in _UP_TO_Z:
        raise ValueError(f"up_axis must be one of {sorted(_UP_TO_Z)}")
    if units not in UNIT_SCALE:
        raise ValueError(f"units must be one of {sorted(UNIT_SCALE)}")
    mesh = mesh.copy()
    mesh.apply_transform(_UP_TO_Z[up_axis])
    scale = UNIT_SCALE[units]
    if scale_to is not None:
        ext = mesh.extents
        axes = {"x": 0, "y": 1, "z": 2}
        current = ext.max() if scale_axis == "max" else ext[axes[scale_axis]]
        if current <= 0:
            raise ValueError("mesh has zero extent along the scale axis")
        scale = scale_to / current
    mesh.apply_scale(scale)
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
    return mesh, float(scale)


def _meshlab_decimate(mesh: trimesh.Trimesh, target_faces: int, colors: np.ndarray | None = None):
    """Topology-preserving quadric edge collapse (keeps watertight meshes watertight)."""
    import pymeshlab

    kwargs = {}
    if colors is not None:
        kwargs["v_color_matrix"] = np.asarray(colors, dtype=np.float64) / 255.0
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32), **kwargs))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        preservetopology=True,
        preserveboundary=True,
        preservenormal=True,
        optimalplacement=True,
        planarquadric=True,
        qualitythr=0.3,
    )
    out = ms.current_mesh()
    low = trimesh.Trimesh(out.vertex_matrix(), out.face_matrix(), process=False)
    low_colors = None
    if colors is not None:
        low_colors = np.round(np.clip(out.vertex_color_matrix(), 0, 1) * 255).astype(np.uint8)
    return low, low_colors


def simplify_geometry(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Geometry-only decimation, e.g. a lightweight proxy for collision generation."""
    if len(mesh.faces) <= target_faces:
        return mesh
    low, _ = _meshlab_decimate(mesh, target_faces)
    return low


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> tuple[trimesh.Trimesh, list[str]]:
    """Quadric decimation to `target_faces`, keeping per-vertex colors when present.

    Textured meshes are left untouched: decimation would invalidate their UV layout,
    and photogrammetry tools can export a baked low-poly mesh directly.
    """
    notes: list[str] = []
    if len(mesh.faces) <= target_faces:
        return mesh, notes
    if has_texture(mesh):
        notes.append(
            f"skipped decimation of textured mesh ({len(mesh.faces)} faces); "
            "export a decimated, re-baked mesh from the reconstruction tool instead"
        )
        return mesh, notes
    colors = np.asarray(mesh.visual.vertex_colors) if has_vertex_colors(mesh) else None
    low, low_colors = _meshlab_decimate(mesh, target_faces, colors)
    if low_colors is not None:
        low.visual = trimesh.visual.ColorVisuals(low, vertex_colors=low_colors)
    trimesh.repair.fix_normals(low)
    if mesh.is_watertight and not low.is_watertight:
        notes.append("decimation opened the surface; consider a higher --target-faces")
    notes.append(f"decimated {len(mesh.faces)} -> {len(low.faces)} faces")
    return low, notes


def fit_disc_base(mesh: trimesh.Trimesh, points: np.ndarray, normals: np.ndarray, max_height_frac: float = 0.2,
                  segments: int = 128) -> tuple[trimesh.Trimesh, dict]:
    """Replace a scanned round base (figure stands, trophies, lamps) with a fitted cylinder.

    The sides of a dark or glossy base are rarely matched by photogrammetry, so Poisson fills them
    with pits and bumps; its flat top, seen from above, is usually measured densely. The top plane is
    the dominant height of upward-facing points in the lowest `max_height_frac` of the object; the
    circle is a least-squares fit to that top's outer rim. The mesh below the top is replaced by the
    cylinder (boolean union). `points`/`normals`: the measured cloud, same frame as `mesh` (Z-up,
    table at z = 0). Returns the new mesh and the fit.
    """
    h = float(points[:, 2].max())
    up = (normals[:, 2] > 0.9) & (points[:, 2] < max_height_frac * h)
    if up.sum() < 200:
        raise ValueError("no flat base top found")
    hist, edges = np.histogram(points[up, 2], bins=100, range=(0, max_height_frac * h))
    k = int(hist.argmax())
    z_top = float(points[up, 2][(points[up, 2] >= edges[k]) & (points[up, 2] < edges[k + 1])].mean())
    top = points[up & (np.abs(points[:, 2] - z_top) < 2 * (edges[1] - edges[0]))][:, :2]
    from scipy.spatial import ConvexHull

    rim = top[ConvexHull(top).vertices]
    # Algebraic circle fit: x^2 + y^2 + a x + b y + c = 0.
    a_, b_, c_ = np.linalg.lstsq(np.c_[rim, np.ones(len(rim))], -(rim ** 2).sum(1), rcond=None)[0]
    center = np.array([-a_ / 2, -b_ / 2])
    radius = float(np.sqrt(center @ center - c_))
    rim_err = np.abs(np.linalg.norm(rim - center, axis=1) - radius)
    disc = trimesh.creation.cylinder(radius=radius, height=z_top, sections=segments)
    disc.apply_translation([center[0], center[1], z_top / 2])
    upper = mesh.slice_plane([0, 0, z_top * 0.98], [0, 0, 1], cap=True)
    merged = trimesh.boolean.union([upper, disc], engine="manifold")
    if mesh.visual.kind == "vertex":
        from scipy.spatial import cKDTree

        _, j = cKDTree(mesh.vertices).query(merged.vertices)
        col = np.asarray(mesh.visual.vertex_colors)[j].copy()
        # One color for the whole disc: the scan's own base top, away from the feet and the rim (the rim
        # mixes in table and paper colors; per-vertex transfer onto long side faces gives streaks).
        mv = mesh.vertices
        r_m = np.linalg.norm(mv[:, :2] - center, axis=1)
        on_top = (np.abs(mv[:, 2] - z_top) < 0.02 * radius) & (r_m > 0.6 * radius) & (r_m < 0.9 * radius)
        src = np.asarray(mesh.visual.vertex_colors)[on_top if on_top.sum() > 50 else mv[:, 2] < z_top]
        base_col = np.median(src[:, :4], 0).astype(np.uint8)
        r_n = np.linalg.norm(merged.vertices[:, :2] - center, axis=1)
        on_disc = (merged.vertices[:, 2] < z_top * 0.995) | (r_n > 0.9 * radius)
        col[on_disc] = base_col
        merged.visual = trimesh.visual.ColorVisuals(merged, vertex_colors=col)
    fit = {"z_top": z_top, "center": center.tolist(), "radius": radius, "rim_points": int(len(rim)),
           "rim_rms": float(np.sqrt(np.mean(rim_err ** 2)))}
    return merged, fit
