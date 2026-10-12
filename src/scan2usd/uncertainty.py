"""Uncertainty-aware assets: what the scan actually saw, and what follows for physics.

Photogrammetry produces one surface, but only part of it was observed: Poisson reconstruction
fills everything else (a mug's unseen interior becomes a solid "lid"). This module turns the
capture itself into explicit uncertainty that ships with the asset:

* **Observation confidence** per render-mesh vertex (``primvars:scan2usd:observed``), from the
  distance to the dense points the cameras actually reconstructed: observed / inferred / sealed,
  plus the flat base the object stood on.
* **Geometry hypothesis** ``carved``: sealed surface patches could be openings. The volume behind
  them that isn't pinned by observed points (kept as a wall of thickness ``wall``) may be empty;
  carving it gives the other plausible shape (an open cup instead of a sealed one). Stored as a
  sibling asset and listed in ``scan2usd:uncertainty:geometryHypotheses``.
* **Property intervals** (``Scan2UsdUncertaintyAPI``): scale from the measurement source, volume
  between the two geometry hypotheses, mass from volume x density range (or a measured mass), center
  of mass box, friction / restitution ranges per material.
* **Sampling** physical variants from all of the above for domain randomization and robust planning.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from .schemas import Uncertainty

OBSERVED, INFERRED, SEALED, BASE = 0, 1, 2, 3
LABEL_NAMES = {OBSERVED: "observed", INFERRED: "inferred", SEALED: "sealed", BASE: "base"}

# Relative half-widths of material property uncertainty (density, friction, restitution).
MATERIAL_SPREAD = {"density": 0.12, "friction": 0.3, "restitution": 0.5}
SCALE_SPREAD = {"chart": 0.015, "assumed": 0.25}  # ruler: precision / measured length


@dataclass
class ObservationField:
    confidence: np.ndarray  # per vertex, 0..1
    labels: np.ndarray  # per vertex, OBSERVED / INFERRED / SEALED / BASE
    near: float  # distance (m) up to which a vertex counts as observed
    far: float  # distance (m) beyond which it is sealed
    observed_fraction: float  # area fraction (excluding the base) with confidence > 0.5
    label_area: dict = field(default_factory=dict)


def observation_field(mesh: trimesh.Trimesh, points: np.ndarray, near: float | None = None,
                      far: float | None = None, base_height: float = 0.0008) -> ObservationField:
    """Confidence of each vertex of `mesh` (asset frame) given the reconstructed `points` (same frame)."""
    tree = cKDTree(points)
    if near is None:
        sample = points[np.random.default_rng(0).choice(len(points), min(20000, len(points)), replace=False)]
        spacing = float(np.median(tree.query(sample, k=2)[0][:, 1]))
        near = max(3.0 * spacing, 0.0015)
    far = far or 3.0 * near
    d, _ = tree.query(mesh.vertices)
    conf = np.clip((far - d) / (far - near), 0.0, 1.0)
    labels = np.where(d <= near, OBSERVED, np.where(d <= far, INFERRED, SEALED))
    n = np.asarray(mesh.vertex_normals)
    labels[(mesh.vertices[:, 2] < mesh.bounds[0][2] + base_height) & (n[:, 2] < -0.7)] = BASE
    face_label = np.max(labels[mesh.faces], axis=1)  # a face is as uncertain as its worst vertex
    face_conf = conf[mesh.faces].mean(1)
    areas = mesh.area_faces
    not_base = face_label != BASE
    observed_fraction = float((areas * (face_conf > 0.5))[not_base].sum() / max(areas[not_base].sum(), 1e-12))
    label_area = {LABEL_NAMES[k]: float(areas[face_label == k].sum() / areas.sum()) for k in LABEL_NAMES}
    return ObservationField(conf, labels, near, far, observed_fraction, label_area)


def cavity_hypothesis(mesh: trimesh.Trimesh, points: np.ndarray, field: ObservationField, wall: float = 0.003,
                      min_patch_area: float = 1e-4, divisions: int = 128) -> tuple[trimesh.Trimesh | None, float]:
    """The 'carved' shape: volume behind large sealed patches, not within `wall` of any observed point.

    Returns (carved mesh or None if nothing to carve, fraction of the solid volume carved).
    """
    from skimage.measure import marching_cubes

    sealed_faces = np.all(field.labels[mesh.faces] == SEALED, axis=1)
    if not sealed_faces.any():
        return None, 0.0
    # Sealed patches big enough to be an opening (tiny crevices are not cavities).
    sub = mesh.submesh([np.nonzero(sealed_faces)[0]], append=True)
    patches = [p for p in sub.split(only_watertight=False) if p.area >= min_patch_area]
    if not patches:
        return None, 0.0
    seeds_pts = np.vstack([p.vertices for p in patches])

    pitch = float(mesh.extents.max()) / divisions
    vg = mesh.voxelized(pitch).fill()
    solid = np.pad(vg.matrix, 2)
    lo = np.asarray(vg.transform[:3, 3]) - 2.5 * pitch  # corner of voxel (0, 0, 0)

    def to_ijk(p):
        ijk = np.floor((p - lo) / pitch).astype(int)
        return ijk[np.all((ijk >= 0) & (ijk < solid.shape), axis=1)]

    observed = np.zeros(solid.shape, bool)
    observed[tuple(to_ijk(points).T)] = True
    dist_obs = ndimage.distance_transform_edt(~observed) * pitch
    free = solid & (dist_obs > wall)
    seeds = np.zeros(solid.shape, bool)
    seeds[tuple(to_ijk(seeds_pts).T)] = True
    seeds = ndimage.binary_dilation(seeds, iterations=2) & free
    if not seeds.any():
        return None, 0.0
    region = ndimage.binary_propagation(seeds, mask=free)
    carved = solid & ~region
    lab, num = ndimage.label(carved)
    if num > 1:
        carved = lab == 1 + int(np.argmax(ndimage.sum(carved, lab, range(1, num + 1))))
    fraction = float(region.sum() / max(solid.sum(), 1))
    if fraction < 0.01:
        return None, fraction
    verts, faces, _, _ = marching_cubes(np.pad(carved, 1).astype(np.float32), level=0.5)
    out = trimesh.Trimesh((verts - 1 + 0.5) * pitch + lo, faces[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(out, iterations=10)
    if out.volume < 0:
        out.invert()
    return out, fraction


def property_intervals(asset, material: str, physics_material: dict, scale_source: str = "ruler",
                       scale_precision: float = 0.001, measured_mass: float | None = None,
                       carved_volume: float | None = None, carved_com: np.ndarray | None = None,
                       observed_fraction: float = 1.0, sealed_fraction: float = 0.0,
                       density_spread: float | None = None) -> Uncertainty:
    """Intervals for an asset (AssetData), given the measurement sources and the geometry hypotheses."""
    ext = asset.render_mesh.extents
    if scale_source == "ruler":
        e = scale_precision / float(ext.max())
    else:
        e = SCALE_SPREAD.get(scale_source, 0.05)
    s_lo, s_hi = 1.0 - e, 1.0 + e
    v_filled = float(asset.render_mesh.volume) if asset.render_mesh.is_watertight else None
    if not v_filled or v_filled <= 0:
        v_filled = float(trimesh.convex.convex_hull(asset.render_mesh.vertices).volume)
    v_lo = (carved_volume if carved_volume else v_filled) * s_lo ** 3
    v_hi = v_filled * s_hi ** 3
    rho = asset.mass / v_filled
    dd = MATERIAL_SPREAD["density"] if density_spread is None else density_spread
    if measured_mass is not None:
        mass = (measured_mass * 0.99, measured_mass * 1.01)
        mass_source = "measured"
    else:
        mass = (rho * (1 - dd) * v_lo, rho * (1 + dd) * v_hi)
        mass_source = "density+sealed-volume" if carved_volume else "density"
    coms = [np.asarray(asset.center_of_mass)]
    if carved_com is not None:
        coms.append(np.asarray(carved_com))
    coms = np.array([c * s for c in coms for s in (s_lo, s_hi)])
    margin = 0.02 * ext  # density inhomogeneity
    com_min, com_max = coms.min(0) - margin, coms.max(0) + margin

    def spread(v, rel, lo=0.0, hi=None):
        a, b = v * (1 - rel), v * (1 + rel)
        return (max(a, lo), min(b, hi) if hi is not None else b)

    sf = physics_material.get("static_friction", 0.5) or 0.5
    df = physics_material.get("dynamic_friction", 0.4) or 0.4
    rs = physics_material.get("restitution", 0.1) or 0.1
    return Uncertainty(
        scale=(s_lo, s_hi), scale_source=scale_source, mass=tuple(float(x) for x in mass), mass_source=mass_source,
        com_min=tuple(float(x) for x in com_min), com_max=tuple(float(x) for x in com_max),
        static_friction=spread(sf, MATERIAL_SPREAD["friction"]), dynamic_friction=spread(df, MATERIAL_SPREAD["friction"]),
        restitution=spread(rs, MATERIAL_SPREAD["restitution"], 0.0, 1.0),
        observed_fraction=float(observed_fraction), sealed_volume_fraction=float(sealed_fraction),
    )


def scale_asset(asset, s: float):
    """Uniformly scaled copy of an AssetData (mass scales with volume, inertia with s^5)."""
    a = copy.deepcopy(asset)
    if s != 1.0:
        a.render_mesh.apply_scale(s)
        for h in a.hulls:
            h.apply_scale(s)
        a.center_of_mass = np.asarray(a.center_of_mass) * s
        a.mass = a.mass * s ** 3
        a.diagonal_inertia = np.asarray(a.diagonal_inertia) * s ** 5
    return a


def sample_physical(asset, u: Uncertainty, rng: np.random.Generator, hypotheses: dict | None = None,
                    p_carved: float = 0.5):
    """One physical variant consistent with the asset's uncertainty (for domain randomization).

    hypotheses: optional {"carved": AssetData}; chosen with probability p_carved.
    """
    base = asset
    geometry = "filled"
    if hypotheses and "carved" in hypotheses and rng.random() < p_carved:
        base, geometry = hypotheses["carved"], "carved"
    s = rng.uniform(*u.scale)
    a = scale_asset(base, s)
    # Mass: this geometry's mass with density spread, clipped to the stored interval.
    rho_scale = rng.uniform(1 - MATERIAL_SPREAD["density"], 1 + MATERIAL_SPREAD["density"])
    m = float(np.clip(a.mass * rho_scale, *u.mass)) if u.mass[1] > 0 else a.mass * rho_scale
    if a.mass > 0:
        a.diagonal_inertia = np.asarray(a.diagonal_inertia) * (m / a.mass)
    a.mass = m
    # Center of mass: shift within the margin around this geometry's own COM (stays inside the box).
    margin = 0.02 * a.render_mesh.extents
    a.center_of_mass = np.clip(np.asarray(a.center_of_mass) + rng.uniform(-margin, margin), u.com_min, u.com_max)
    a.physics_material = dict(a.physics_material)
    a.physics_material["static_friction"] = rng.uniform(*u.static_friction)
    a.physics_material["dynamic_friction"] = rng.uniform(*u.dynamic_friction)
    a.physics_material["restitution"] = rng.uniform(*u.restitution)
    return a, {"geometry": geometry, "scale": s, "mass": m,
               "dynamic_friction": a.physics_material["dynamic_friction"]}


# --- applying to an asset on disk ---------------------------------------------------------------
@dataclass
class AnalysisResult:
    uncertainty: Uncertainty
    field: ObservationField
    carved_path: Path | None
    sealed_fraction: float
    points_in_asset_frame: int
    transform_fit_m: float


def workspace_points(asset, workspace: Path, scale: float) -> tuple[np.ndarray, float]:
    """The reconstructed (observed) object points, mapped into the asset frame.

    reconstruct() writes object_points.ply in the same frame as object_mesh.ply; the build then
    scales by `scale` and recenters (bounding-box bottom-center at the origin).
    """
    from .reconstruct import read_ply_points

    ply = read_ply_points(workspace / "object_points.ply")
    pts = np.c_[ply["x"], ply["y"], ply["z"]].astype(np.float64)
    recon = trimesh.load(workspace / "object_mesh.ply", force="mesh")
    t = asset.render_mesh.bounds[0] - scale * recon.bounds[0]
    fit = float(np.abs((scale * recon.bounds[1] + t) - asset.render_mesh.bounds[1]).max())
    return pts * scale + t, fit


def analyze_asset(usd_path: str | Path, workspace: str | Path, material: str | None = None,
                  scale_source: str = "ruler", scale_precision: float = 0.001, measured_mass: float | None = None,
                  wall: float = 0.003, write: bool = True) -> AnalysisResult:
    """Compute and (optionally) store observation confidence, the carved hypothesis and the intervals."""
    import json

    from pxr import Sdf, Usd, UsdGeom, Vt

    from .bake import _source_colors
    from .physics import MATERIALS, convex_decompose, estimate_mass_properties
    from .schemas import author_uncertainty
    from .usd_io import LOOKS, PhysicsMaterial, SurfaceMaterial, read_usd, write_usd

    usd_path, workspace = Path(usd_path), Path(workspace)
    asset = read_usd(usd_path)
    report = json.loads((usd_path.parent / "build_report.json").read_text())
    material = material or (json.loads(Usd.Stage.Open(str(usd_path)).GetDefaultPrim().GetCustomDataByKey(
        "scan2usd:build") or "{}").get("material")) or "plastic"
    points, fit = workspace_points(asset, workspace, float(report["scale_applied"]))
    fld = observation_field(asset.render_mesh, points)

    carved_mesh, sealed_fraction = cavity_hypothesis(asset.render_mesh, points, fld, wall=wall)
    carved_path = carved_volume = carved_com = None
    if carved_mesh is not None:
        rho = asset.mass / max(float(asset.render_mesh.volume), 1e-12)
        hulls = convex_decompose(carved_mesh, 0.05, 32)
        mp = estimate_mass_properties(carved_mesh, hulls, density=rho)
        carved_volume, carved_com = mp.volume, np.asarray(mp.center_of_mass)
        pts, cols = _source_colors(asset.render_mesh, 400_000)
        _, nearest = cKDTree(pts).query(carved_mesh.vertices)
        rgb = np.clip(cols[nearest], 0, 255).astype(np.uint8)
        carved_mesh.visual = trimesh.visual.ColorVisuals(carved_mesh, vertex_colors=np.c_[rgb, np.full(len(rgb), 255, np.uint8)])
        look = LOOKS.get(material, {})
        carved_path = usd_path.with_name(f"{usd_path.stem}_carved.usda")
        pm = asset.physics_material or {}
        write_usd(carved_path, asset.name, carved_mesh, hulls, mp, SurfaceMaterial(use_vertex_colors=True, **look),
                  PhysicsMaterial(pm.get("static_friction", 0.5), pm.get("dynamic_friction", 0.4), pm.get("restitution", 0.1)),
                  {"hypothesis": "carved", "of": usd_path.name, "sealed_volume_fraction": sealed_fraction})

    mat = MATERIALS.get(material, {})
    u = property_intervals(asset, material, asset.physics_material or mat, scale_source, scale_precision, measured_mass,
                           carved_volume, carved_com, fld.observed_fraction, sealed_fraction)
    if write:
        stage = Usd.Stage.Open(str(usd_path))
        root = stage.GetDefaultPrim()
        author_uncertainty(root, u)
        hyp = [Sdf.AssetPath(f"./{carved_path.name}")] if carved_path else []
        root.CreateAttribute("scan2usd:uncertainty:geometryHypotheses", Sdf.ValueTypeNames.AssetArray).Set(hyp)
        render = next(p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh) and p.GetName() == "render")
        pv = UsdGeom.PrimvarsAPI(render)
        conf = pv.CreatePrimvar("scan2usd:observed", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        n_usd = len(UsdGeom.Mesh(render).GetPointsAttr().Get())
        if n_usd == len(fld.confidence):
            conf.Set(Vt.FloatArray.FromNumpy(fld.confidence.astype(np.float32)))
            pv.CreatePrimvar("scan2usd:observedLabel", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex).Set(
                Vt.IntArray.FromNumpy(fld.labels.astype(np.int32)))
        stage.GetRootLayer().Save()
    return AnalysisResult(u, fld, carved_path, sealed_fraction, len(points), fit)


def load_hypotheses(usd_path: str | Path) -> dict:
    """{"carved": AssetData} for the geometry hypotheses listed in an analyzed asset."""
    from pxr import Usd

    from .usd_io import read_usd

    stage = Usd.Stage.Open(str(usd_path))
    attr = stage.GetDefaultPrim().GetAttribute("scan2usd:uncertainty:geometryHypotheses")
    out = {}
    for ap in (attr.Get() or []) if attr else []:
        p = (Path(usd_path).parent / ap.path).resolve()
        if p.exists():
            out[p.stem.rsplit("_", 1)[-1]] = read_usd(p)
    return out
