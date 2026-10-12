"""Scan2USD's own USD schemas: grasps and uncertainty that travel inside the asset.

Two codeless applied API schemas live in ``scan2usd/schema`` (``generatedSchema.usda`` +
``plugInfo.json``): no compiled code, so any USD runtime can load them via
``PXR_PLUGINPATH_NAME=<that folder>`` (Isaac Sim, usdview, Houdini, ...). Without the plugin,
files still open: the data is plain namespaced attributes and the API names are just tokens.

* ``Scan2UsdGraspAPI``: on prims under ``<asset>/grasps``. The prim's local transform is the
  grasp frame in the asset frame. Convention per gripper:
  ``panda``: tool center point, +Z approaches the object, +Y is the closing direction;
  ``shadow_hand``: the Menagerie hand's ``grasp_site`` (palm) frame.
* ``Scan2UsdUncertaintyAPI``: on the asset root; intervals for scale, mass, center of mass,
  friction and restitution, plus how much of the surface / volume was actually observed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from pxr import Gf, Plug, Sdf, Usd, UsdGeom

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
GRASP_API = "Scan2UsdGraspAPI"
UNCERTAINTY_API = "Scan2UsdUncertaintyAPI"
_registered: bool | None = None


def register() -> bool:
    """Register the codeless schemas with this process's USD (idempotent). True if available."""
    global _registered
    if _registered is None:
        Plug.Registry().RegisterPlugins(str(SCHEMA_DIR))
        _registered = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(GRASP_API) is not None
    return _registered


def _apply(prim: Usd.Prim, api: str) -> None:
    if register():
        prim.ApplyAPI(api)
    else:  # still record the schema name; readers fall back to raw attributes
        prim.AddAppliedSchema(api)


# --- grasps -----------------------------------------------------------------------------------
@dataclass
class Grasp:
    pose: np.ndarray  # 4x4, grasp frame in the asset frame (column vectors)
    gripper: str = "panda"
    type: str = "parallel"
    width: float = 0.0
    success_rate: float = 0.0
    trials: int = 0
    expected_success: float | None = None
    evaluated_with: str = ""
    by_variant: dict = field(default_factory=dict)  # e.g. {"decomposed": 1.0, "convex_hull": 0.0}

    @property
    def position(self) -> np.ndarray:
        return self.pose[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        return self.pose[:3, :3]


def write_grasps(path: str | Path, grasps: list[Grasp], replace: bool = True, gripper: str | None = None) -> int:
    """Store grasps in the asset file at `path` under ``<defaultPrim>/grasps``.

    replace: drop existing grasps first (only those of `gripper`, if given). Returns the count written.
    """
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    scope_path = root.GetPath().AppendChild("grasps")
    scope = UsdGeom.Scope.Define(stage, scope_path)
    scope.GetPrim().SetMetadata("comment", "Scan2UsdGraspAPI grasps, best first per gripper")
    if replace:
        for child in list(scope.GetPrim().GetChildren()):
            if gripper is None or child.GetAttribute("scan2usd:grasp:gripper").Get() == gripper:
                stage.RemovePrim(child.GetPath())
    taken = {c.GetName() for c in scope.GetPrim().GetChildren()}
    k = 0
    for g in grasps:
        while f"g_{k:03d}" in taken:
            k += 1
        name = f"g_{k:03d}"
        taken.add(name)
        xf = UsdGeom.Xform.Define(stage, scope_path.AppendChild(name))
        xf.AddTransformOp().Set(Gf.Matrix4d(np.asarray(g.pose, dtype=float).T.tolist()))
        prim = xf.GetPrim()
        UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
        _apply(prim, GRASP_API)
        prim.CreateAttribute("scan2usd:grasp:gripper", Sdf.ValueTypeNames.Token, variability=Sdf.VariabilityUniform).Set(g.gripper)
        prim.CreateAttribute("scan2usd:grasp:type", Sdf.ValueTypeNames.Token, variability=Sdf.VariabilityUniform).Set(g.type)
        prim.CreateAttribute("scan2usd:grasp:width", Sdf.ValueTypeNames.Float).Set(float(g.width))
        prim.CreateAttribute("scan2usd:grasp:successRate", Sdf.ValueTypeNames.Float).Set(float(g.success_rate))
        prim.CreateAttribute("scan2usd:grasp:trials", Sdf.ValueTypeNames.Int).Set(int(g.trials))
        prim.CreateAttribute("scan2usd:grasp:evaluatedWith", Sdf.ValueTypeNames.Token).Set(g.evaluated_with)
        if g.expected_success is not None:
            prim.CreateAttribute("scan2usd:grasp:expectedSuccess", Sdf.ValueTypeNames.Float).Set(float(g.expected_success))
        if g.by_variant:
            prim.SetCustomDataByKey("scan2usd:successByVariant", json.dumps(g.by_variant))
    stage.GetRootLayer().Save()
    return len(grasps)


def read_grasps(stage_or_path, root_path: str | None = None, gripper: str | None = None) -> list[Grasp]:
    """Grasps under ``<root>/grasps`` (any prim carrying the grasp attributes), best first."""
    stage = stage_or_path if isinstance(stage_or_path, Usd.Stage) else Usd.Stage.Open(str(stage_or_path))
    root = stage.GetPrimAtPath(root_path) if root_path else stage.GetDefaultPrim()
    scope = root.GetChild("grasps")
    out = []
    if not scope:
        return out
    cache = UsdGeom.XformCache()
    for prim in scope.GetChildren():
        attr = prim.GetAttribute("scan2usd:grasp:gripper")
        if not attr or not attr.HasAuthoredValue():
            continue
        if gripper and attr.Get() != gripper:
            continue
        m, _ = cache.ComputeRelativeTransform(prim, root)

        def get(name, default):
            a = prim.GetAttribute(f"scan2usd:grasp:{name}")
            return a.Get() if a and a.HasAuthoredValue() else default

        exp = get("expectedSuccess", None)
        by = prim.GetCustomDataByKey("scan2usd:successByVariant")
        out.append(Grasp(pose=np.asarray(m, dtype=float).T, gripper=str(attr.Get()), type=str(get("type", "parallel")),
                         width=float(get("width", 0.0)), success_rate=float(get("successRate", 0.0)),
                         trials=int(get("trials", 0)), expected_success=None if exp is None else float(exp),
                         evaluated_with=str(get("evaluatedWith", "")), by_variant=json.loads(by) if by else {}))
    out.sort(key=lambda g: -(g.expected_success if g.expected_success is not None else g.success_rate))
    return out


# --- uncertainty --------------------------------------------------------------------------------
@dataclass
class Uncertainty:
    scale: tuple[float, float] = (1.0, 1.0)
    scale_source: str = "assumed"
    mass: tuple[float, float] = (0.0, 0.0)
    mass_source: str = ""
    com_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    com_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    static_friction: tuple[float, float] = (0.0, 0.0)
    dynamic_friction: tuple[float, float] = (0.0, 0.0)
    restitution: tuple[float, float] = (0.0, 0.0)
    observed_fraction: float = 1.0
    sealed_volume_fraction: float = 0.0
    confidence_level: float = 0.9

    def to_dict(self) -> dict:
        return asdict(self)


_UNC_ATTRS = {  # field -> (attribute, type)
    "scale": ("scale", Sdf.ValueTypeNames.Float2), "scale_source": ("scaleSource", Sdf.ValueTypeNames.Token),
    "mass": ("mass", Sdf.ValueTypeNames.Float2), "mass_source": ("massSource", Sdf.ValueTypeNames.Token),
    "com_min": ("centerOfMassMin", Sdf.ValueTypeNames.Point3f), "com_max": ("centerOfMassMax", Sdf.ValueTypeNames.Point3f),
    "static_friction": ("staticFriction", Sdf.ValueTypeNames.Float2),
    "dynamic_friction": ("dynamicFriction", Sdf.ValueTypeNames.Float2),
    "restitution": ("restitution", Sdf.ValueTypeNames.Float2),
    "observed_fraction": ("observedFraction", Sdf.ValueTypeNames.Float),
    "sealed_volume_fraction": ("sealedVolumeFraction", Sdf.ValueTypeNames.Float),
    "confidence_level": ("confidenceLevel", Sdf.ValueTypeNames.Float),
}


def author_uncertainty(prim: Usd.Prim, u: Uncertainty) -> None:
    _apply(prim, UNCERTAINTY_API)
    for f, (name, vtype) in _UNC_ATTRS.items():
        v = getattr(u, f)
        if isinstance(v, tuple):
            v = tuple(float(x) for x in v)
        elif isinstance(v, float):
            v = float(v)
        prim.CreateAttribute(f"scan2usd:uncertainty:{name}", vtype).Set(v)


def write_uncertainty(path: str | Path, u: Uncertainty) -> None:
    stage = Usd.Stage.Open(str(path))
    author_uncertainty(stage.GetDefaultPrim(), u)
    stage.GetRootLayer().Save()


def read_uncertainty(stage_or_path, root_path: str | None = None) -> Uncertainty | None:
    stage = stage_or_path if isinstance(stage_or_path, Usd.Stage) else Usd.Stage.Open(str(stage_or_path))
    root = stage.GetPrimAtPath(root_path) if root_path else stage.GetDefaultPrim()
    if not root.GetAttribute("scan2usd:uncertainty:mass").HasAuthoredValue():
        return None
    kw = {}
    for f, (name, _) in _UNC_ATTRS.items():
        a = root.GetAttribute(f"scan2usd:uncertainty:{name}")
        if a and a.HasAuthoredValue():
            v = a.Get()
            kw[f] = tuple(float(x) for x in v) if hasattr(v, "__len__") and not isinstance(v, str) else (
                str(v) if isinstance(v, str) else float(v))
    return Uncertainty(**kw)
