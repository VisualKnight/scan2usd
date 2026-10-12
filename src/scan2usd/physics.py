"""Collision geometry and rigid-body mass properties."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

# Bulk densities (kg/m^3) and contact parameters for common household materials.
# Friction values are typical dry contact against wood/plastic surfaces.
MATERIALS: dict[str, dict[str, float]] = {
    "ceramic":   {"density": 2400.0, "static_friction": 0.60, "dynamic_friction": 0.50, "restitution": 0.10},
    "glass":     {"density": 2500.0, "static_friction": 0.50, "dynamic_friction": 0.40, "restitution": 0.10},
    "wood":      {"density": 700.0,  "static_friction": 0.50, "dynamic_friction": 0.40, "restitution": 0.20},
    "plastic":   {"density": 1050.0, "static_friction": 0.40, "dynamic_friction": 0.30, "restitution": 0.30},
    "rubber":    {"density": 1150.0, "static_friction": 1.00, "dynamic_friction": 0.80, "restitution": 0.60},
    "aluminum":  {"density": 2700.0, "static_friction": 0.45, "dynamic_friction": 0.35, "restitution": 0.15},
    "steel":     {"density": 7850.0, "static_friction": 0.50, "dynamic_friction": 0.40, "restitution": 0.15},
    "cardboard": {"density": 690.0,  "static_friction": 0.55, "dynamic_friction": 0.45, "restitution": 0.05},
    "foam":      {"density": 30.0,   "static_friction": 0.70, "dynamic_friction": 0.60, "restitution": 0.40},
    "fruit":     {"density": 900.0,  "static_friction": 0.60, "dynamic_friction": 0.50, "restitution": 0.10},
    "concrete":  {"density": 2300.0, "static_friction": 0.70, "dynamic_friction": 0.60, "restitution": 0.10},
}

# PhysX limits a convex hull to 255 vertices; stay well under it.
MAX_HULL_VERTICES = 64


@dataclass
class MassProperties:
    mass: float
    volume: float
    density: float
    center_of_mass: list[float]
    inertia_tensor: list[list[float]]
    principal_moments: list[float]
    principal_axes_quat_wxyz: list[float]
    volume_source: str

    def to_dict(self) -> dict:
        return asdict(self)


def single_hull(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    return [_limit_hull(mesh.convex_hull)]


def convex_decompose(
    mesh: trimesh.Trimesh,
    threshold: float = 0.05,
    max_hulls: int = 32,
    seed: int = 0,
    proxy_faces: int = 4000,
) -> list[trimesh.Trimesh]:
    """Approximate convex decomposition with CoACD (Wei et al., SIGGRAPH 2022).

    `threshold` is CoACD's concavity tolerance: lower gives tighter, more numerous hulls.
    CoACD runs on a `proxy_faces` simplification; collision does not need render detail
    and the search time grows quickly with face count.
    """
    import coacd

    from .mesh_ops import simplify_geometry

    coacd.set_log_level("error")
    proxy = simplify_geometry(mesh, proxy_faces)
    cmesh = coacd.Mesh(np.asarray(proxy.vertices, dtype=np.float64), np.asarray(proxy.faces, dtype=np.int64))
    parts = coacd.run_coacd(
        cmesh,
        threshold=threshold,
        max_convex_hull=max_hulls,
        max_ch_vertex=MAX_HULL_VERTICES,
        decimate=True,
        seed=seed,
    )
    hulls = []
    for verts, faces in parts:
        hull = trimesh.Trimesh(verts, faces, process=True).convex_hull
        if hull.volume > 1e-12:
            hulls.append(_limit_hull(hull))
    if not hulls:
        return single_hull(mesh)
    return hulls


def _limit_hull(hull: trimesh.Trimesh) -> trimesh.Trimesh:
    if len(hull.vertices) <= MAX_HULL_VERTICES:
        return hull
    target = max(4, 2 * MAX_HULL_VERTICES - 4)
    low = hull.simplify_quadric_decimation(face_count=target).convex_hull
    return low if len(low.vertices) <= MAX_HULL_VERTICES else hull


def union_volume_mesh(hulls: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    if len(hulls) == 1:
        return hulls[0]
    return trimesh.boolean.union(hulls, engine="manifold")


def voxel_solid(mesh: trimesh.Trimesh, resolution: int = 160):
    """Solid voxel centers for a mesh with small holes (typical of raw scans).

    Surface voxels are dilated once to seal gaps, enclosed space is filled, then the
    dilation is undone. Open cavities (a bowl, the inside of a mug) stay empty because
    they connect to the outside. Returns (centers, pitch, weights), or None if nothing
    was enclosed.
    """
    from scipy import ndimage

    pitch = float(max(mesh.extents)) / resolution
    vox = mesh.voxelized(pitch)
    pad = 10
    shell = np.pad(vox.matrix, pad)
    # Escalate the closing until the interior is sealed; capped so real openings stay open.
    for k in (1, 2, 4, 8):
        closed = ndimage.binary_dilation(shell, iterations=k)
        filled = ndimage.binary_fill_holes(closed)
        if filled.sum() > closed.sum() * 1.02:
            break
    else:
        return None
    solid = ndimage.binary_erosion(filled, iterations=k) | shell
    # Surface voxels straddle the true surface: count them as half inside.
    weights = np.where(shell, 0.5, 1.0)[solid]
    idx = np.argwhere(solid) - pad
    return vox.indices_to_points(idx), pitch, weights


def _mass_properties_from_points(points: np.ndarray, weights: np.ndarray, rho: float, pitch: float):
    m = rho * pitch ** 3 * weights
    com = (points * m[:, None]).sum(0) / m.sum()
    r = points - com
    inertia = (m * (r ** 2).sum(1)).sum() * np.eye(3) - (r * m[:, None]).T @ r
    inertia += np.eye(3) * m.sum() * pitch ** 2 / 6.0  # each voxel's own inertia
    return com, inertia


def principal_frame(inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose an inertia tensor into principal moments and a proper rotation."""
    moments, axes = np.linalg.eigh(np.asarray(inertia, dtype=np.float64))
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
    return moments, axes


def estimate_mass_properties(
    mesh: trimesh.Trimesh,
    hulls: list[trimesh.Trimesh] | None = None,
    density: float | None = None,
    mass: float | None = None,
) -> MassProperties:
    """Mass, center of mass and inertia of a solid body.

    Uses the watertight render mesh when possible; for scans with holes, a voxel-filled
    solid; and as a last resort the union of collision hulls (which over-fills cavities).
    Give either a material `density` (kg/m^3) or a measured `mass` (kg).
    """
    if (density is None) == (mass is None):
        raise ValueError("specify exactly one of density or mass")
    # Textured meshes duplicate vertices along UV seams; judge closedness by position only.
    welded = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
    welded.merge_vertices()
    if welded.is_watertight and welded.is_winding_consistent and welded.volume > 0:
        solid, source = welded, "render_mesh"
    else:
        vs = voxel_solid(welded)
        if vs is not None:
            points, pitch, weights = vs
            volume = float(weights.sum()) * pitch ** 3
            rho = float(mass / volume) if mass is not None else float(density)
            com, inertia = _mass_properties_from_points(points, weights, rho, pitch)
            return _pack(rho * volume, volume, rho, com, inertia, "voxel_fill")
        solid, source = union_volume_mesh(hulls or single_hull(mesh)), "collision_hull_union"
    volume = float(solid.volume)
    if volume <= 0:
        raise ValueError("solid has non-positive volume; cannot estimate mass")
    rho = float(mass / volume) if mass is not None else float(density)

    solid = solid.copy()
    solid.density = rho
    return _pack(rho * volume, volume, rho, solid.center_mass, np.asarray(solid.moment_inertia, dtype=np.float64),
                 source)


def _pack(mass: float, volume: float, rho: float, com, inertia: np.ndarray, source: str) -> MassProperties:
    moments, axes = principal_frame(inertia)
    x, y, z, w = Rotation.from_matrix(axes).as_quat()
    return MassProperties(
        mass=float(mass),
        volume=float(volume),
        density=float(rho),
        center_of_mass=[float(v) for v in com],
        inertia_tensor=np.asarray(inertia).tolist(),
        principal_moments=[float(v) for v in moments],
        principal_axes_quat_wxyz=[float(w), float(x), float(y), float(z)],
        volume_source=source,
    )
