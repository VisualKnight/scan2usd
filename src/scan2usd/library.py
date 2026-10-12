"""Asset library: many Scan2USD assets composed into one USD stage.

Each asset is a *payload* (loaded on demand) under `/Library`, laid out on a grid, with two
variant sets authored in the library layer, so the asset files stay untouched:

* ``collision``: ``decomposed`` (the asset's convex decomposition, default), ``convex_hull``
  (one hull per rigid body) and ``bounding_box`` (one box per rigid body). Simpler colliders
  are cheaper to simulate and are what a naive pipeline would give you; switching is one
  variant selection, e.g. for an ablation.
* ``friction``: ``measured`` (the asset's physics material, default), ``low`` and ``high``,
  for domain randomization of contact properties.

Variant opinions are stronger than payload opinions (LIVRPS), so they can deactivate the
asset's own collision scope and add replacements. Articulated assets get one simple collider
per rigid body.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

COLLISION_VARIANTS = ("decomposed", "convex_hull", "bounding_box")
FRICTION_VARIANTS = {"measured": None, "low": (0.25, 0.2), "high": (1.0, 0.9)}  # (static, dynamic)


@dataclass
class LibraryEntry:
    name: str
    prim_path: str
    asset_path: str
    bodies: list[str]
    mass_kg: float
    size_m: list[float]
    colliders: dict[str, int] = field(default_factory=dict)


def _prim_name(name: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return out if out and not out[0].isdigit() else f"a_{out}"


def _points(prim: Usd.Prim, relative_to: Usd.Prim) -> np.ndarray:
    pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
    xf, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, relative_to)
    return (np.c_[pts, np.ones(len(pts))] @ np.asarray(xf))[:, :3]


def _rigid_bodies(root: Usd.Prim) -> list[Usd.Prim]:
    return [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def _render_points(body: Usd.Prim) -> np.ndarray:
    """Render-mesh points of one rigid body (not descending into nested bodies), in body space."""
    pts = []
    it = iter(Usd.PrimRange(body))
    for p in it:
        if p != body and p.HasAPI(UsdPhysics.RigidBodyAPI):
            it.PruneChildren()
            continue
        if p.IsA(UsdGeom.Mesh) and not p.HasAPI(UsdPhysics.CollisionAPI) and \
                UsdGeom.Imageable(p).ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            pts.append(_points(p, body))
    return np.vstack(pts)


def _physics_material(body: Usd.Prim) -> Sdf.Path | None:
    for p in Usd.PrimRange(body):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            bound, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
            if bound:
                return bound.GetPath()
    return None


def _author_collider(stage: Usd.Stage, path: Sdf.Path, mesh: trimesh.Trimesh, phys_mat: Sdf.Path | None) -> None:
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).ravel()))
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    geom.CreatePurposeAttr(UsdGeom.Tokens.guide)
    prim = geom.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    if phys_mat is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(phys_mat)),
                                                     UsdShade.Tokens.weakerThanDescendants, "physics")


def _simple_collider(points: np.ndarray, kind: str) -> trimesh.Trimesh:
    if kind == "convex_hull":
        return trimesh.convex.convex_hull(points)
    lo, hi = points.min(0), points.max(0)
    return trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))


def build_library(assets: list[str | Path], out: str | Path, spacing: float = 0.1, columns: int = 4) -> list[LibraryEntry]:
    """Write `out` (a .usda root layer) composing `assets`; returns one entry per asset."""
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    lib = UsdGeom.Xform.Define(stage, "/Library").GetPrim()
    stage.SetDefaultPrim(lib)
    Usd.ModelAPI(lib).SetKind(Kind.Tokens.assembly)
    stage.GetRootLayer().documentation = "Scan2USD asset library: payloads + collision/friction variant sets"

    entries: list[LibraryEntry] = []
    x = y = row_depth = 0.0
    used: set[str] = set()
    for k, asset in enumerate(assets):
        asset = Path(asset).resolve()
        name = _prim_name(asset.stem)
        while name in used:
            name += "_"
        used.add(name)
        prim = UsdGeom.Xform.Define(stage, f"/Library/{name}").GetPrim()
        try:  # relative payload paths keep the library portable
            ref = Path(os.path.relpath(asset, out.parent)).as_posix()
            ref = ref if ref.startswith(".") else f"./{ref}"
        except ValueError:  # different drive on Windows
            ref = asset.as_posix()
        prim.GetPayloads().AddPayload(ref)
        Usd.ModelAPI(prim).SetKind(Kind.Tokens.component)

        bodies = _rigid_bodies(prim)
        if not bodies:
            raise ValueError(f"{asset} has no rigid body")
        body_pts = {b.GetPath(): _render_points(b) for b in bodies}
        all_pts = np.vstack([_points_world(b, prim, p) for b, p in zip(bodies, body_pts.values())])
        lo, hi = all_pts.min(0), all_pts.max(0)
        size = hi - lo

        # Grid layout: rows along +x, wrapping every `columns` assets.
        if k and k % columns == 0:
            x, y, row_depth = 0.0, y + row_depth + spacing, 0.0
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(x - lo[0], y - lo[1], 0.0))
        x += size[0] + spacing
        row_depth = max(row_depth, size[1])

        mass = sum(float(UsdPhysics.MassAPI(b).GetMassAttr().Get() or 0.0) for b in bodies)
        prim.SetAssetInfo({"name": name, "identifier": Sdf.AssetPath(ref)})
        colliders = {"decomposed": sum(1 for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI))}

        vsets = prim.GetVariantSets()
        col = vsets.AddVariantSet("collision")
        for v in COLLISION_VARIANTS:
            col.AddVariant(v)
        for v in COLLISION_VARIANTS[1:]:
            col.SetVariantSelection(v)
            with col.GetVariantEditContext():
                for b in bodies:
                    mat = _physics_material(b)
                    for child in b.GetChildren():
                        if child.GetName() == "collision":
                            stage.OverridePrim(child.GetPath()).SetActive(False)
                    _author_collider(stage, b.GetPath().AppendChild(f"collision_{v}"),
                                     _simple_collider(body_pts[b.GetPath()], v), mat)
            colliders[v] = len(bodies)
        col.SetVariantSelection("decomposed")

        fr = vsets.AddVariantSet("friction")
        mats = {_physics_material(b) for b in bodies} - {None}
        for v, values in FRICTION_VARIANTS.items():
            fr.AddVariant(v)
            if values is None:
                continue
            fr.SetVariantSelection(v)
            with fr.GetVariantEditContext():
                for m in mats:
                    over = stage.OverridePrim(m)
                    over.CreateAttribute("physics:staticFriction", Sdf.ValueTypeNames.Float).Set(values[0])
                    over.CreateAttribute("physics:dynamicFriction", Sdf.ValueTypeNames.Float).Set(values[1])
        fr.SetVariantSelection("measured")

        entry = LibraryEntry(name=name, prim_path=str(prim.GetPath()), asset_path=ref,
                             bodies=[str(b.GetPath()) for b in bodies], mass_kg=mass,
                             size_m=[float(v) for v in size], colliders=colliders)
        prim.SetCustomDataByKey("scan2usd:library", json.dumps(
            {"mass_kg": round(mass, 4), "size_m": [round(float(v), 4) for v in size], "colliders": colliders}))
        entries.append(entry)

    stage.GetRootLayer().Save()
    return entries


def _points_world(body: Usd.Prim, asset_root: Usd.Prim, pts: np.ndarray) -> np.ndarray:
    xf, _ = UsdGeom.XformCache().ComputeRelativeTransform(body, asset_root)
    return (np.c_[pts, np.ones(len(pts))] @ np.asarray(xf))[:, :3]


def asset_files(path: str | Path) -> dict[str, Path]:
    """Library prim name -> the asset file it payloads (resolved against the library's folder)."""
    path = Path(path).resolve()
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadNone)
    out = {}
    for prim in stage.GetDefaultPrim().GetFilteredChildren(Usd.PrimIsActive & Usd.PrimIsDefined):
        op = prim.GetMetadata("payload")
        items = list(op.GetAddedOrExplicitItems()) if op else []
        if items:
            out[prim.GetName()] = (path.parent / items[0].assetPath).resolve()
    return out


def library_entries(path: str | Path) -> list[dict]:
    """Summaries of a library file's assets (opens with payloads unloaded: cheap)."""
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadNone)
    out = []
    for prim in stage.GetDefaultPrim().GetFilteredChildren(Usd.PrimIsActive & Usd.PrimIsDefined):
        info = json.loads(prim.GetCustomDataByKey("scan2usd:library") or "{}")
        vsets = prim.GetVariantSets()
        out.append({"name": prim.GetName(), "path": str(prim.GetPath()), **info,
                    "variants": {n: vsets.GetVariantSet(n).GetVariantNames() for n in vsets.GetNames()}})
    return out
