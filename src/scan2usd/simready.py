"""SimReady packaging: a Scan2USD asset -> an NVIDIA SimReady Foundation package, and its validation.

SimReady Foundation (github.com/NVIDIA/simready-foundation) specifies "simulation-ready" OpenUSD content as
requirements, grouped into features, bundled into profiles (``Robotics-Prop`` for props a robot handles).
This module writes a validated package from an asset the pipeline has already built; the source asset is
not modified.

Layout (NP.005): ``<root>/<name>/simready_usd/sm_<name>_01.usda`` (the only USD file in that folder),
textures in ``simready_usd/textures/``, a thumbnail at ``simready_usd/.thumbs/256x256/<file>.png`` (SR.002).
Every directory in the absolute path must use letters, digits, ``.``, ``-`` or ``_`` (NP.003): no spaces.

What the package adds to the (flattened) asset:

* semantic labels (``UsdSemantics.LabelsAPI``): the object's class and its Wikidata Q-code on the asset
  root (SL.001, SL.QCODE.001), what it is made of on its rendered materials (SL.MAT.001)
* provenance metadata, ``SimReady_Metadata`` in the root layer's ``customLayerData`` (SR.004)
* a display colour on the render meshes, the albedo's mean (DISP.001)
* one grasp line per stored grasp (``Scan2UsdGraspAPI``): from the tested grasp frame through the
  object's center (GSP.001)
* normals on the collision hulls (VG.027) and ``float3`` texture outputs (VM.PS.001), for assets built
  before ``usd_io`` authored them

``validate`` runs ``simready-validate`` (``pip install simready-validate simready-foundation-tier-core``,
Python 3.12, its own venv: it pins its own ``usd-core``) and reports each feature of the profile.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

PROFILE, PROFILE_VERSION = "Robotics-Prop", "4.0.0"
INTERMEDIATE = "simready_usd"
VALID_DIR = re.compile(r"^[A-Za-z0-9._-]+$")  # NP.003's directory-name rule


@dataclass
class SimReadyInfo:
    """What the package states about the asset (SimReady_Metadata and the semantic labels)."""

    category: str  # the object's class, e.g. "mug" (SL.001, metadata `category`)
    qcode: str  # its Wikidata item, e.g. "Q386215" (mug)
    material: str  # what it is made of, e.g. "ceramic" (SL.MAT.001)
    license: str  # e.g. "CC-BY-4.0", "Apache-2.0", "proprietary"
    author: str = "Yibo Liu (Scan2USD)"
    source_file: str = ""  # the capture or mesh it was built from (default: the build metadata's source)
    extra: dict = field(default_factory=dict)  # more SimReady_Metadata fields (custom ones are allowed)

    def __post_init__(self):
        if not re.fullmatch(r"Q\d+", self.qcode):
            raise ValueError(f"not a Wikidata Q-code: {self.qcode!r} (SL.QCODE.001: 'Q' followed by digits)")


def package_name(asset_usd: str | Path) -> str:
    """The package folder name: the asset's default prim, snake_case."""
    stage = Usd.Stage.Open(str(asset_usd))
    root = stage.GetDefaultPrim()
    name = root.GetName() if root else Path(asset_usd).stem
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
    return re.sub(r"[^a-z0-9_]+", "_", name).strip("_")


def bad_path_parts(path: str | Path) -> list[str]:
    """Directories in `path` that break NP.003 (it checks every folder of the absolute path)."""
    p = Path(path).resolve()
    return [d for d in p.parts if d and d != p.anchor and not VALID_DIR.match(d)]


def package(asset_usd: str | Path, out_root: str | Path, info: SimReadyInfo, name: str | None = None,
            thumbnail: bool = True) -> Path:
    """Write the SimReady package of `asset_usd` under `out_root`; returns the main USD file."""
    asset_usd = Path(asset_usd).resolve()
    name = name or package_name(asset_usd)
    pkg = Path(out_root) / name / INTERMEDIATE
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    main = pkg / f"sm_{name}_01.usda"

    src = Usd.Stage.Open(str(asset_usd))
    src.Flatten().Export(str(main))
    stage = Usd.Stage.Open(str(main))
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError(f"{asset_usd} has no defaultPrim")
    _copy_dependencies(stage, asset_usd.parent, pkg)
    _fix_texture_outputs(stage)
    _hull_normals(stage, root)
    render = _render_meshes(stage, root)
    colors = {m.GetPath(): _albedo(m) for m in render}
    for m in render:
        pv = UsdGeom.PrimvarsAPI(m)
        if pv.HasPrimvar("displayColor") and pv.GetPrimvar("displayColor").HasAuthoredValue():
            continue  # vertex-colored scans keep theirs (their material reads it)
        mean = colors[m.GetPath()][1]
        pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                         UsdGeom.Tokens.constant).Set(Vt.Vec3fArray([Gf.Vec3f(*mean)]))
    _labels(stage, root, render, info)
    _grasp_lines(stage, root, render)
    _metadata(stage, root, render, info, asset_usd, name)
    stage.GetRootLayer().Save()
    if thumbnail:
        write_thumbnail(stage, root, render, colors, pkg / ".thumbs" / "256x256" / f"{main.name}.png")
    return main


# --- content fixes ---------------------------------------------------------------------------------------


def _copy_dependencies(stage: Usd.Stage, src_dir: Path, pkg: Path) -> None:
    """Copy every file an asset-valued attribute points at into ``textures/`` and re-anchor it there."""
    tex = pkg / "textures"
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            val = attr.Get()
            if not val or not val.path:
                continue
            p = Path(val.path)
            f = p if p.is_absolute() else (src_dir / p)
            if not f.exists() and val.resolvedPath:
                f = Path(val.resolvedPath)
            if not f.exists():
                raise FileNotFoundError(f"{prim.GetPath()}.{attr.GetName()}: {val.path} not found")
            tex.mkdir(exist_ok=True)
            shutil.copy2(f, tex / f.name)
            attr.Set(Sdf.AssetPath(f"./textures/{f.name}"))


_SPEC_OUTPUTS = {  # UsdPreviewSurface spec: output types of the nodes the pipeline writes
    "UsdUVTexture": {"rgb": Sdf.ValueTypeNames.Float3},
    "UsdPrimvarReader_float3": {"result": Sdf.ValueTypeNames.Float3},
    "UsdPrimvarReader_float2": {"result": Sdf.ValueTypeNames.Float2},
}


def _fix_texture_outputs(stage: Usd.Stage) -> None:
    """Retype node outputs to the UsdPreviewSurface spec (VM.PS.001): a texture's rgb and a float3 primvar
    reader's result are float3 even when they feed a color3f input; connections are by path, so retyping
    the output keeps them."""
    for prim in stage.Traverse():
        sh = UsdShade.Shader(prim)
        if not sh:
            continue
        for name, typ in _SPEC_OUTPUTS.get(sh.GetIdAttr().Get(), {}).items():
            out = sh.GetOutput(name)
            if out and out.GetTypeName() != typ:
                prim.RemoveProperty(f"outputs:{name}")
                sh.CreateOutput(name, typ)


def face_normals(points: np.ndarray, counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Unit per-face normals of a polygon mesh (Newell's method); degenerate faces point away from the
    mesh's centroid."""
    out = np.zeros((len(counts), 3))
    center = points.mean(0)
    k = 0
    for f, n in enumerate(counts):
        poly = points[indices[k:k + n]]
        nxt = np.roll(poly, -1, axis=0)
        v = np.cross(poly, nxt).sum(0)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            v = poly.mean(0) - center
            norm = np.linalg.norm(v) or 1.0
        out[f] = v / norm
        k += n
    return out


def _hull_normals(stage: Usd.Stage, root: Usd.Prim) -> None:
    """Flat normals on collider meshes that have none (VG.027 asks it of every non-subdivided mesh)."""
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.GetNormalsAttr().HasAuthoredValue() or UsdGeom.PrimvarsAPI(prim).HasPrimvar("normals"):
            continue
        n = face_normals(np.asarray(mesh.GetPointsAttr().Get(), float),
                         np.asarray(mesh.GetFaceVertexCountsAttr().Get(), int),
                         np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)


def _render_meshes(stage: Usd.Stage, root: Usd.Prim) -> list[UsdGeom.Mesh]:
    out = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI):
            if UsdGeom.Imageable(prim).ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                out.append(UsdGeom.Mesh(prim))
    return out


def _bound_material(mesh: UsdGeom.Mesh) -> UsdShade.Material | None:
    mat, _ = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()
    return mat if mat else None


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _albedo(mesh: UsdGeom.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Per-point linear colors of a render mesh (its albedo texture sampled at the points' UVs, a
    vertex-color primvar its material reads, or the material's diffuse color) and their mean."""
    pts = np.asarray(mesh.GetPointsAttr().Get(), float)
    rgb = np.full((len(pts), 3), 0.3)
    mat = _bound_material(mesh)
    pv_api = UsdGeom.PrimvarsAPI(mesh)
    if mat is not None:
        surf = next((s for s in (UsdShade.Shader(p) for p in Usd.PrimRange(mat.GetPrim()))
                     if s and s.GetIdAttr().Get() == "UsdPreviewSurface"), None)
        diffuse = surf.GetInput("diffuseColor") if surf else None
        if diffuse:
            src = diffuse.GetConnectedSources()[0] if diffuse.HasConnectedSource() else []
            node = UsdShade.Shader(src[0].source.GetPrim()) if src else None
            node_id = node.GetIdAttr().Get() if node else None
            if node_id == "UsdUVTexture":
                f = node.GetInput("file").Get() if node.GetInput("file") else None
                st = pv_api.GetPrimvar("st")
                if f and f.resolvedPath and st and st.GetInterpolation() == UsdGeom.Tokens.vertex:
                    from PIL import Image

                    img = np.asarray(Image.open(f.resolvedPath).convert("RGB"), float) / 255.0
                    uv = np.asarray(st.Get(), float) % 1.0
                    h, w = img.shape[:2]
                    x = np.clip((uv[:, 0] * (w - 1)).round().astype(int), 0, w - 1)
                    y = np.clip(((1 - uv[:, 1]) * (h - 1)).round().astype(int), 0, h - 1)
                    rgb = _srgb_to_linear(img[y, x])  # an sRGB albedo texture
            elif node_id == "UsdPrimvarReader_float3":
                name = node.GetInput("varname").Get() if node.GetInput("varname") else None
                pv = pv_api.GetPrimvar(str(name)) if name else None
                if pv and pv.HasAuthoredValue():
                    vals = np.asarray(pv.Get(), float)
                    rgb = vals if len(vals) == len(pts) else np.tile(vals.mean(0), (len(pts), 1))
            elif diffuse.Get() is not None:
                rgb[:] = np.asarray(diffuse.Get(), float)
    return rgb, rgb.mean(0)


def _labels(stage: Usd.Stage, root: Usd.Prim, render: list[UsdGeom.Mesh], info: SimReadyInfo) -> None:
    from pxr import UsdSemantics

    UsdSemantics.LabelsAPI.Apply(root, "class").CreateLabelsAttr().Set([info.category])
    UsdSemantics.LabelsAPI.Apply(root, "wikidata_qcode").CreateLabelsAttr().Set([info.qcode])
    # SL.MAT.001's text scopes it to rendered materials, but its checker (2026.8.0) asks it of every
    # Material prim, the physics one included; both describe the same stuff, so both get the label.
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdShade.Material):
            UsdSemantics.LabelsAPI.Apply(prim, "material").CreateLabelsAttr().Set([info.material])


def _render_bounds(render: list[UsdGeom.Mesh], root: Usd.Prim) -> np.ndarray:
    cache = UsdGeom.XformCache()
    to_root = np.array(cache.GetLocalToWorldTransform(root)).T
    inv = np.linalg.inv(to_root)
    pts = []
    for m in render:
        mt = inv @ np.array(cache.GetLocalToWorldTransform(m.GetPrim())).T
        p = np.asarray(m.GetPointsAttr().Get(), float)
        pts.append(p @ mt[:3, :3].T + mt[:3, 3])
    p = np.concatenate(pts)
    return np.array([p.min(0), p.max(0)])


def _grasp_lines(stage: Usd.Stage, root: Usd.Prim, render: list[UsdGeom.Mesh]) -> int:
    """A guide-purpose line per stored grasp: from the grasp frame through the object's center and 25% of
    that distance beyond it, so it runs through the object."""
    grasps = [p for p in Usd.PrimRange(root) if p.GetAttribute("scan2usd:grasp:gripper")]
    if not grasps or not render:
        return 0
    b = _render_bounds(render, root)
    center = b.mean(0)
    cache = UsdGeom.XformCache()
    root_inv = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(root)).T)
    for k, g in enumerate(grasps):
        at = (root_inv @ np.array(cache.GetLocalToWorldTransform(g)).T)[:3, 3]
        end = center + 0.25 * (center - at)
        parent = g.GetParent()
        p_inv = np.linalg.inv(root_inv @ np.array(cache.GetLocalToWorldTransform(parent)).T)
        a, e = (p_inv @ np.append(at, 1))[:3], (p_inv @ np.append(end, 1))[:3]
        curve = UsdGeom.BasisCurves.Define(stage, parent.GetPath().AppendChild(f"grasp_identifier_{k:03d}"))
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*a), Gf.Vec3f(*e)]))
        curve.CreateWidthsAttr(Vt.FloatArray([0.002]))
        curve.CreatePurposeAttr(UsdGeom.Tokens.guide)
        curve.CreateExtentAttr(UsdGeom.Boundable.ComputeExtentFromPlugins(curve, Usd.TimeCode.Default()))  # VG.002
        curve.GetPrim().SetCustomDataByKey("scan2usd:grasp", g.GetName())
    return len(grasps)


def _metadata(stage: Usd.Stage, root: Usd.Prim, render: list[UsdGeom.Mesh], info: SimReadyInfo,
              asset_usd: Path, name: str) -> None:
    from . import __version__

    bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    mass = sum(float(UsdPhysics.MassAPI(p).GetMassAttr().Get() or 0.0) for p in bodies
               if p.HasAPI(UsdPhysics.MassAPI))
    ext = _render_bounds(render, root)
    build = root.GetCustomDataByKey("scan2usd:build")
    source = info.source_file or (json.loads(build).get("source", "") if build else "") or asset_usd.name
    md = {
        "author": info.author, "asset_name": name, "asset_type": "prop", "asset_license": info.license,
        "category": info.category, "source_file": source,
        "usd_date_generated": _dt.date.today().isoformat(), "qcode": info.qcode,
        "rigid_body_count": len(bodies), "asset_extents": Gf.Vec3f(*map(float, ext[1] - ext[0])),
        "mass": float(mass), "generator": f"scan2usd {__version__}", "material": info.material,
        **info.extra,
    }
    layer = stage.GetRootLayer()
    data = dict(layer.customLayerData)
    # SR.001 wants four identity keys directly in customLayerData (plus the SimReady_Metadata key); SR.004
    # reads the union of those and the nested dictionary, so each field is authored exactly once.
    top = ("asset_name", "asset_type", "source_file", "usd_date_generated")
    data.update({k: md.pop(k) for k in top})
    data["SimReady_Metadata"] = md
    layer.customLayerData = data


def write_thumbnail(stage: Usd.Stage, root: Usd.Prim, render: list[UsdGeom.Mesh], colors: dict,
                    path: Path, size: int = 256) -> Path:
    """The render meshes in their albedo colors, on white, three-quarter view (SR.002)."""
    import trimesh

    from .preview import render as draw

    cache = UsdGeom.XformCache()
    root_inv = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(root)).T)
    meshes = []
    for m in render:
        mt = root_inv @ np.array(cache.GetLocalToWorldTransform(m.GetPrim())).T
        pts = np.asarray(m.GetPointsAttr().Get(), float) @ mt[:3, :3].T + mt[:3, 3]
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), int)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), int)
        tris, k = [], 0
        for n in counts:
            tris += [[idx[k], idx[k + i], idx[k + i + 1]] for i in range(1, n - 1)]
            k += n
        rgb = (_linear_to_srgb(colors[m.GetPath()][0]) * 255).round().astype(np.uint8)
        meshes.append(trimesh.Trimesh(pts, np.asarray(tris), vertex_colors=rgb, process=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    draw(meshes, size=size, margin=0.06).save(path)
    return path


# --- validation --------------------------------------------------------------------------------------------


def find_validator(exe: str | Path | None = None) -> Path:
    """The simready-validate executable: `exe`, $SIMREADY_VALIDATE, or on PATH."""
    cand = exe or os.environ.get("SIMREADY_VALIDATE") or shutil.which("simready-validate")
    if not cand or not Path(cand).exists():
        raise FileNotFoundError("simready-validate not found: pip install simready-validate "
                                "simready-foundation-tier-core (Python 3.12, own venv), then pass --validator "
                                "or set SIMREADY_VALIDATE")
    return Path(cand)


def profile_features(exe: Path, profile: str = PROFILE, version: str = PROFILE_VERSION) -> dict[str, bool]:
    """{feature: optional} for a profile version, from the tier package installed next to the validator
    (read by that environment's Python 3.12, which has tomllib)."""
    exe = Path(exe).resolve()
    py = next((p for p in (exe.parent / "python.exe", exe.parent / "python") if p.exists()), None)
    if py is None:
        raise FileNotFoundError(f"no python next to {exe}")
    code = (
        "import json, sys, tomllib, pathlib\n"
        "import simready.foundation.tier_core as t\n"
        "stem, prof, ver = sys.argv[1:4]\n"
        "for base in t.__path__:\n"
        "    f = pathlib.Path(base) / 'profiles' / (stem + '.toml')\n"
        "    if f.exists():\n"
        "        feats = tomllib.loads(f.read_text(encoding='utf-8'))[prof][ver]['features']\n"
        "        out = {}\n"
        "        for e in feats:\n"
        "            out[next(k for k in e if k != 'optional')] = bool(e.get('optional', False))\n"
        "        print(json.dumps(out)); break\n"
    )
    r = subprocess.run([str(py), "-c", code, profile.lower().replace("-", "_"), profile, version],
                       capture_output=True, text=True, timeout=120)
    if r.returncode or not r.stdout.strip():
        raise RuntimeError(f"cannot read the {profile} {version} profile: {r.stderr.strip()[-300:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def validate(usd: str | Path, exe: str | Path | None = None, profile: str = PROFILE,
             version: str = PROFILE_VERSION) -> dict:
    """Run simready-validate on one asset: {"passed": bool, "features": {fid: {passed, optional,
    failing: [requirement codes], messages: {code: [..]}}}}. A profile passes when every required
    feature does."""
    exe = find_validator(exe)
    optional = profile_features(exe, profile, version)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        subprocess.run([str(exe), "--profile", profile, "--version", version, "--output", str(out), str(usd)],
                       capture_output=True, text=True, timeout=600)
        if not out.exists():
            raise RuntimeError(f"simready-validate wrote no report for {usd}")
        report = next(iter(json.loads(out.read_text(encoding="utf-8")).values()))
    import ast

    feats = {}
    for fid, v in report["features_summary"].items():
        failing = (v.get("failing requirements") or v.get("optional requirements") or "[]") if not v["passed"] else "[]"
        msgs = json.loads(v.get("requirement_messages") or "{}")
        # blocking: a required requirement of a required feature is unmet (what fails the profile); a
        # feature can be "not passed" without blocking when only its optional requirements are unmet
        feats[fid] = {"passed": bool(v["passed"]), "blocking": bool(v.get("blocking_failure")),
                      "optional": optional.get(fid, True),
                      "failing": [c.replace("com.nvidia.simready.", "").replace("com.nvidia.usd.", "")
                                  for c in ast.literal_eval(failing)],
                      "messages": {k.split(".", 3)[-1]: m[:3] for k, m in msgs.items()}}
    required_ok = not any(f["blocking"] and not f["optional"] for f in feats.values())
    return {"passed": required_ok, "profile": profile, "version": version,
            "validator": report.get("simready_validate_version"), "tiers": report.get("tier_versions"),
            "features": feats}
