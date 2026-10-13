"""Isaac Sim 6 (PhysX): the scanned controller dropped into the scanned holder, vs. the real pose.

    python scripts/study_holder_fit.py            # assets + ground truth (main venv)
    set OMNI_KIT_ACCEPT_EULA=YES
    <isaac-sim>\\python.exe scripts\\isaac_holder_fit.py

Same protocol as the MuJoCo stage of study_holder_fit.py: the holder's USD asset is static, the
controller's USD asset (rigid body, mass, convex decomposition) is released 3 cm above its real
resting pose with the same ±8 mm / ±8 deg jitter, and after 3 s its pose is compared with the pose
measured in the combined scan (ADD: mean distance of controller surface points). Holder collision:
its convex decomposition, or one convex hull. Writes docs/figures/isaac_holder_fit.{json,jpg,webp}.
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
FIT = os.path.join(ROOT, "output", "holder_fit")
HOLDER = os.path.join(FIT, "assets", "cod_holder", "cod_holder.usda")
CTRL = os.path.join(FIT, "assets", "controller", "controller.usda")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

try:
    FONT = ImageFont.load_default(size=18)
except TypeError:
    FONT = ImageFont.load_default()

state = json.load(open(os.path.join(FIT, "state.json")))
rel = state["truth"]["controller_in_holder"]
R_TRUE, T_TRUE = np.array(rel["R"]), np.array(rel["t"])
rng = np.random.default_rng(0)  # same jitter as the MuJoCo stage
STARTS = [(np.zeros(3), 0.0)] + [(np.r_[rng.uniform(-0.008, 0.008, 2), 0.0], rng.uniform(-8, 8)) for _ in range(11)]


def surface_points(path, n=4000):
    stage = Usd.Stage.Open(path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/{stage.GetDefaultPrim().GetName()}/render"))
    pts = np.asarray(mesh.GetPointsAttr().Get(), float)
    return pts[np.random.default_rng(1).choice(len(pts), min(n, len(pts)), replace=False)]


CP = surface_points(CTRL)
TRUE_PTS = CP @ R_TRUE.T + T_TRUE


def scene(collision, start, ghost):
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    ps = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(ps.GetPrim()).CreateTimeStepsPerSecondAttr(240)
    floor = UsdGeom.Cube.Define(stage, "/World/floor")
    floor.CreateSizeAttr(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    floor.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.8, 0.78)])

    holder = UsdGeom.Xform.Define(stage, "/World/holder")
    holder.GetPrim().GetReferences().AddReference(HOLDER)
    for p in Usd.PrimRange(holder.GetPrim()):  # static: the holder stands on the floor
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
        if collision == "convex_hull" and p.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
    if collision == "convex_hull":
        render = UsdGeom.Mesh(stage.GetPrimAtPath("/World/holder/render"))
        UsdPhysics.CollisionAPI.Apply(render.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(render.GetPrim()).CreateApproximationAttr("convexHull")

    dxy, dyaw = start
    r0 = Rotation.from_euler("z", dyaw, degrees=True).as_matrix() @ R_TRUE
    t0 = T_TRUE + dxy + np.array([0, 0, 0.03])
    ctrl = UsdGeom.Xform.Define(stage, "/World/ctrl")
    ctrl.GetPrim().GetReferences().AddReference(CTRL)
    m = Gf.Matrix4d()
    m.SetRotateOnly(Gf.Matrix3d(*r0.T.flatten().tolist()))
    m.SetTranslateOnly(Gf.Vec3d(*t0))
    ctrl.AddTransformOp().Set(m)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(ctrl.GetPrim())
    rb.CreateSolverPositionIterationCountAttr(16)
    if ghost:
        g = UsdGeom.Xform.Define(stage, "/World/ghost")
        g.GetPrim().GetReferences().AddReference(CTRL)
        gm = Gf.Matrix4d()
        gm.SetRotateOnly(Gf.Matrix3d(*R_TRUE.T.flatten().tolist()))
        gm.SetTranslateOnly(Gf.Vec3d(*T_TRUE))
        g.AddTransformOp().Set(gm)
        for p in Usd.PrimRange(g.GetPrim()):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
        rm = UsdGeom.Mesh(stage.GetPrimAtPath("/World/ghost/render"))
        rm.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.8, 0.35)])
        rm.CreateDisplayOpacityAttr([0.35])
        UsdGeom.PrimvarsAPI(rm.GetPrim()).CreatePrimvar("doNotCastShadows", __import__("pxr").Sdf.ValueTypeNames.Bool).Set(True)
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(rm.GetPrim()).UnbindAllBindings()

    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(900)
    key = UsdLux.SphereLight.Define(stage, "/World/key")
    key.CreateRadiusAttr(0.3)
    key.CreateIntensityAttr(30000)
    key.AddTranslateOp().Set(Gf.Vec3d(0.6, -0.9, 1.4))
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    cam.CreateFocalLengthAttr(0.35)
    cam.CreateHorizontalApertureAttr(0.36)
    cam.CreateVerticalApertureAttr(0.27)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
    normal = R_TRUE @ np.array([0, 0, 1.0])
    az = np.arctan2(normal[1], normal[0]) + np.radians(35)
    look = Gf.Vec3d(T_TRUE[0], T_TRUE[1], 0.09)
    eye = look + 0.45 * Gf.Vec3d(np.cos(az) * np.cos(0.3), np.sin(az) * np.cos(0.3), np.sin(0.3))
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, look, Gf.Vec3d(0, 0, 1)).GetInverse())


def run(collision, start, record):
    scene(collision, start, ghost=record)
    for _ in range(10):
        app.update()
    rgb = rp = None
    if record:
        rp = rep.create.render_product("/World/cam", (640, 480))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([rp])
    app_utils.play()
    app.update()
    ctrl = RigidPrim("/World/ctrl")
    frames = []
    for i in range(180):  # 3 s at 60 Hz
        app.update()
        if record and i % 3 == 0:
            data = rgb.get_data()
            if data is not None and getattr(data, "size", 0):
                frames.append(Image.fromarray(np.asarray(data)[..., :3]))
    pos, quat = ctrl.get_world_poses()
    t1 = np.asarray(pos.numpy()).reshape(-1)[:3]
    w, x, y, z = np.asarray(quat.numpy()).reshape(-1)[:4]
    r1 = Rotation.from_quat([x, y, z, w]).as_matrix()
    app_utils.stop()
    if record:
        rgb.detach()
        rp.destroy()
    sim = CP @ r1.T + t1
    add = float(np.linalg.norm(sim - TRUE_PTS, axis=1).mean())
    ang = float(np.degrees(np.arccos(np.clip((np.trace(r1.T @ R_TRUE) - 1) / 2, -1, 1))))
    return {"add_mm": add * 1000, "rot_deg": ang, "min_z_mm": float(sim[:, 2].min() * 1000)}, frames


results, clips = {}, {}
for collision, title in (("decomposed", "convex decomposition"), ("convex_hull", "one convex hull")):
    rows = []
    for k, start in enumerate(STARTS):
        res, frames = run(collision, start, record=(k == 0))
        rows.append(res)
        if frames:
            clips[collision] = (frames, title)
        print(f"[scan2usd] {collision} drop {k}: ADD {res['add_mm']:.1f} mm, rot {res['rot_deg']:.1f} deg", flush=True)
    add = np.array([r["add_mm"] for r in rows])
    held = np.array([r["min_z_mm"] > 5 for r in rows])
    results[collision] = {"trials": rows, "held_in_holder": int(held.sum()), "of": len(rows),
                          "add_mm_median": float(np.median(add)),
                          "rot_deg_median": float(np.median([r["rot_deg"] for r in rows]))}
    print(f"[scan2usd] {collision}: held {held.sum()}/{len(rows)}, ADD median {np.median(add):.1f} mm", flush=True)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "isaac_holder_fit.json"), "w") as f:
    json.dump({"engine": "Isaac Sim 6 PhysX, 240 Hz", "mujoco": state.get("sim"), "results": results}, f, indent=2)
n = min(len(c[0]) for c in clips.values())
sheets = []
for i in range(n):
    s = Image.new("RGB", (1280, 480), "white")
    for k, key in enumerate(("decomposed", "convex_hull")):
        fr, title = clips[key]
        im = fr[i].copy()
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 640, 30], fill=(255, 255, 255))
        d.text((10, 5), f"Isaac Sim 6 PhysX · holder: {title}", fill=(30, 41, 59), font=FONT)
        s.paste(im, (k * 640, 0))
    sheets.append(s)
sheets[0].save(os.path.join(OUT, "isaac_holder_fit.webp"), save_all=True, append_images=sheets[1:] + [sheets[-1]] * 20,
               duration=50, loop=0, quality=65, method=4)
sheets[-1].save(os.path.join(OUT, "isaac_holder_fit.jpg"), quality=88)
app.close()
