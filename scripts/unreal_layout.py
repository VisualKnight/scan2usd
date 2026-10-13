"""Tabletop scene for the Unreal Engine render, written as plain USD (python scripts/unreal_layout.py).

Everything that makes up the scene is USD, so the same stage could be opened in Isaac Sim or usdview:
  - the Scan2USD asset files, referenced and placed on the table (meters, Z-up), with the phone-scanned
    controller resting in the phone-scanned holder at its measured pose (study_holder_fit.py);
  - a wooden tabletop (a mesh with UVs and a UsdPreviewSurface using Poly Haven's CC0 wood_table_001);
  - two strings of copper-wire fairy lights: a tube mesh for the wire, small emissive bulbs, and one
    shadow-casting UsdLux SphereLight per bulb. With ~100 shadowed lights this is the kind of scene
    Unreal's MegaLights is built for;
  - a soft rect key light.
Unreal's USD importer turns the stage into actors and does the axis and unit conversion. Physics is
switched off: this layout is for rendering.

Writes output/unreal/layout.usda (+ output/unreal/layout.json: the camera and bulb counts).
Textures: output/unreal/textures/ (Poly Haven, CC0; fetched by scripts/unreal_render.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "unreal"
FIT = ROOT / "output" / "holder_fit"

# name -> (x, y, yaw deg), meters on the tabletop. The holder pair is the subject; scans from the
# benchmark sit around it.
LAYOUT = {
    "cod_holder": (0.0, 0.0, 0.0),
    "speckled_mug": (0.175, 0.045, -35.0),
    "schleich_lion_action_figure": (-0.135, -0.075, 20.0),
    "buddha_head": (-0.25, 0.25, 120.0),
    "ace_coffee_mug_kristen_16_oz_cup": (0.285, 0.31, 70.0),
}
TABLE = (1.6, 0.9, 0.035)  # width, depth, thickness (m); top at z = 0
TABLE_CENTER = (0.0, 0.12)
# Fairy-light strands: control points (m). One lies on the table, weaving in front of and behind the
# objects; one hangs in two swags behind the table and becomes bokeh.
TABLE_STRAND = [(-0.55, -0.22), (-0.36, -0.17), (-0.20, -0.17), (-0.04, -0.13), (0.10, -0.15), (0.25, -0.09),
                (0.30, 0.06), (0.24, 0.16), (0.10, 0.12), (-0.04, 0.14), (-0.16, 0.10), (-0.24, 0.12),
                (-0.40, 0.10), (-0.58, 0.18)]
GARLAND_Y, GARLAND_X, GARLAND_TOP, GARLAND_SAG = 0.62, (-0.75, 0.75), 0.44, 0.20
BULB_SPACING = 0.042
WIRE_R, BULB_R = 0.0007, 0.0032
WARM = (1.0, 0.62, 0.30)  # ~2200 K LED
CAMERA = {"eye": (0.04, -0.60, 0.16), "target": (0.0, 0.05, 0.10), "focus": (0.0, 0.0, 0.11), "focal_mm": 45.0,
          "fstop": 2.8}
# Look-dev in the scene layer: a stronger USD opinion than the referenced asset, the asset file is
# untouched. The holder's painted figure is satin, not the generic plastic preset (roughness 0.42), and
# with a scan's surface noise a glossier value reads as chrome.
LOOK_OVERRIDES = {"cod_holder": {"roughness": 0.6}}


def asset(name: str) -> Path:
    for p in (ROOT / "output" / "pbr" / name / f"{name}.usda", ROOT / "output" / "gso_assets" / name / f"{name}.usda",
              FIT / "assets" / name / f"{name}.usda"):
        if p.exists():
            return p
    raise FileNotFoundError(name)


def static(prim: Usd.Prim) -> None:
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)


def place(stage, path: str, usd: Path, x: float, y: float, yaw: float) -> Usd.Prim:
    xf = UsdGeom.Xform.Define(stage, path)
    xf.GetPrim().GetReferences().AddReference(str(usd).replace("\\", "/"))
    api = UsdGeom.XformCommonAPI(xf)
    api.SetTranslate(Gf.Vec3d(x, y, 0.0))
    api.SetRotate(Gf.Vec3f(0.0, 0.0, yaw))
    static(xf.GetPrim())
    return xf.GetPrim()


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> Gf.Matrix4d:
    """Local -Z toward the target, +Y up (USD cameras and lights)."""
    f = np.subtract(target, eye)
    f = f / np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = x, y, -f, eye
    return Gf.Matrix4d(*m.flatten().tolist())


def catmull_rom(points: np.ndarray, n: int = 40) -> np.ndarray:
    p = np.vstack([points[0], points, points[-1]])
    out = []
    for i in range(1, len(p) - 2):
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p[i]) + (-p[i - 1] + p[i + 1]) * t
                              + (2 * p[i - 1] - 5 * p[i] + 4 * p[i + 1] - p[i + 2]) * t2
                              + (-p[i - 1] + 3 * p[i] - 3 * p[i + 1] + p[i + 2]) * t3))
    out.append(points[-1])
    return np.asarray(out)


def resample(curve: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    ts = np.arange(spacing / 2, s[-1], spacing)
    return np.c_[[np.interp(ts, s, curve[:, k]) for k in range(3)]].T


def tube(curve: np.ndarray, r: float, sides: int = 6):
    """Vertices and quads of a tube along a polyline (parallel-transport frames)."""
    t = np.gradient(curve, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.cross(t[0], [0, 0, 1.0] if abs(t[0][2]) < 0.9 else [1.0, 0, 0])
    n /= np.linalg.norm(n)
    verts = []
    for i, c in enumerate(curve):
        if i:
            n = n - np.dot(n, t[i]) * t[i]
            n /= np.linalg.norm(n)
        b = np.cross(t[i], n)
        for k in range(sides):
            a = 2 * np.pi * k / sides
            verts.append(c + r * (np.cos(a) * n + np.sin(a) * b))
    faces = []
    for i in range(len(curve) - 1):
        for k in range(sides):
            a, b_ = i * sides + k, i * sides + (k + 1) % sides
            faces.append([a, b_, b_ + sides, a + sides])
    return np.asarray(verts), np.asarray(faces)


def preview_material(stage, path: str, color, roughness: float, metallic: float = 0.0, emissive=None,
                     textures: dict | None = None) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if emissive is not None:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    if textures:
        st = UsdShade.Shader.Define(stage, f"{path}/stReader")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        for name, (file, slot, kind) in textures.items():
            tex = UsdShade.Shader.Define(stage, f"{path}/{name}")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file)
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            if kind == "normal":
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                sh.CreateInput(slot, Sdf.ValueTypeNames.Normal3f).ConnectToSource(out)
            elif kind == "scalar":
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                out = tex.CreateOutput("r", Sdf.ValueTypeNames.Float)
                sh.CreateInput(slot, Sdf.ValueTypeNames.Float).ConnectToSource(out)
            else:
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                sh.CreateInput(slot, Sdf.ValueTypeNames.Color3f).ConnectToSource(out)
    return mat


def table(stage) -> None:
    w, d, h = TABLE
    cx, cy = TABLE_CENTER
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2
    # A box with per-face UVs in meters (the texture covers ~1.2 m), so the grain isn't stretched.
    c = [(x0, y0, -h), (x1, y0, -h), (x1, y1, -h), (x0, y1, -h), (x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]
    quads = [(4, 5, 6, 7), (0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    pts, uvs, idx = [], [], []
    for q in quads:
        corners = np.array([c[i] for i in q])
        a, b = np.argsort(np.ptp(corners, axis=0))[::-1][:2]
        for p in corners:
            idx.append(len(pts))
            pts.append(p)
            uvs.append(((p[a] - x0) / 1.2, (p[b] - y0) / 1.2))
    mesh = UsdGeom.Mesh.Define(stage, "/World/table")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr("none")
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(*u) for u in uvs]))
    tex = "./textures/wood_table_001_{}.jpg"
    mat = preview_material(stage, "/World/Looks/Wood", (1, 1, 1), 0.5, textures={
        "diffuse": (tex.format("diff"), "diffuseColor", "color"),
        "normal": (tex.format("nor_gl"), "normal", "normal"),
        "rough": (tex.format("rough"), "roughness", "scalar")})
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def strand(stage, name: str, curve: np.ndarray, wire_mat, bulb_mat, intensity: float) -> list:
    v, f = tube(curve, WIRE_R)
    wire = UsdGeom.Mesh.Define(stage, f"/World/{name}/wire")
    wire.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in v]))
    wire.CreateFaceVertexCountsAttr([4] * len(f))
    wire.CreateFaceVertexIndicesAttr(f.flatten().tolist())
    wire.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(wire.GetPrim()).Bind(wire_mat)
    bulbs = resample(curve, BULB_SPACING)
    for i, b in enumerate(bulbs):
        pos = b + np.array([0, 0, BULB_R * 0.6])  # bulbs sit on the wire
        s = UsdGeom.Sphere.Define(stage, f"/World/{name}/bulb_{i:03d}")
        s.CreateRadiusAttr(BULB_R)
        UsdGeom.XformCommonAPI(s).SetTranslate(Gf.Vec3d(*pos))
        UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(bulb_mat)
        light = UsdLux.SphereLight.Define(stage, f"/World/{name}/light_{i:03d}")
        light.CreateRadiusAttr(BULB_R * 1.1)
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(*WARM))
        UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(True)
        UsdGeom.XformCommonAPI(light).SetTranslate(Gf.Vec3d(*pos))
    return bulbs.tolist()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "layout.usda"
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().documentation = "Scan2USD tabletop for the Unreal Engine render (scripts/unreal_layout.py)"

    table(stage)
    for name, (x, y, yaw) in LAYOUT.items():
        place(stage, f"/World/{name}", asset(name), x, y, yaw)
    for name, inputs in LOOK_OVERRIDES.items():
        shader = UsdShade.Shader(stage.GetPrimAtPath(f"/World/{name}/Looks/M_surface/PreviewSurface"))
        for k, v in inputs.items():
            shader.GetInput(k).Set(v)
    rel = json.loads((FIT / "state.json").read_text())["truth"]["controller_in_holder"]
    pose = UsdGeom.Xform.Define(stage, "/World/cod_holder/controller_pose")
    m = Gf.Matrix4d()
    m.SetRotateOnly(Gf.Matrix3d(*np.array(rel["R"]).T.flatten().tolist()))
    m.SetTranslateOnly(Gf.Vec3d(*rel["t"]))
    pose.AddTransformOp().Set(m)
    ctrl = UsdGeom.Xform.Define(stage, "/World/cod_holder/controller_pose/controller")
    ctrl.GetPrim().GetReferences().AddReference(str(asset("controller")).replace("\\", "/"))
    static(ctrl.GetPrim())

    wire_mat = preview_material(stage, "/World/Looks/Copper", (0.95, 0.64, 0.54), 0.35, metallic=1.0)
    bulb_mat = preview_material(stage, "/World/Looks/Bulb", (1.0, 0.9, 0.75), 0.3, emissive=(20.0, 11.0, 4.5))
    table_pts = np.array([(x, y, WIRE_R) for x, y in TABLE_STRAND])
    xs = np.linspace(*GARLAND_X, 9)
    u = (xs - GARLAND_X[0]) / (GARLAND_X[1] - GARLAND_X[0])
    zs = GARLAND_TOP - GARLAND_SAG * np.abs(np.sin(2 * np.pi * u))  # two swags
    garland_pts = np.c_[xs, GARLAND_Y + 0.04 * np.sin(xs * 5), zs]
    bulbs = {"table": strand(stage, "lights_table", catmull_rom(table_pts), wire_mat, bulb_mat, 8.0e3),
             "garland": strand(stage, "lights_garland", catmull_rom(garland_pts), wire_mat, bulb_mat, 8.0e3)}

    key = UsdLux.RectLight.Define(stage, "/World/key")
    key.CreateWidthAttr(0.45)
    key.CreateHeightAttr(0.30)
    key.CreateIntensityAttr(1.2e3)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.86, 0.70))
    key.AddTransformOp().Set(look_at((-0.75, -0.35, 0.55), (0.0, 0.05, 0.08)))

    cam = UsdGeom.Camera.Define(stage, "/World/camera")
    cam.AddTransformOp().Set(look_at(CAMERA["eye"], CAMERA["target"]))
    stage.Save()
    meta = {"camera": CAMERA, "bulbs": {k: len(v) for k, v in bulbs.items()}, "assets": list(LAYOUT) + ["controller"]}
    (OUT / "layout.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {path}: {len(LAYOUT) + 1} assets, {sum(len(v) for v in bulbs.values())} shadow-casting fairy lights")


if __name__ == "__main__":
    main()
