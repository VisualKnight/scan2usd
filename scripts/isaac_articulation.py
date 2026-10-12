"""Isaac Sim check of an articulated Scan2USD asset: drive the hinge, verify the joint limit.

    set OMNI_KIT_ACCEPT_EULA=YES
    <isaac-sim>\\python.exe scripts\\isaac_articulation.py

The hinged keepsake box (scan2usd articulate) is referenced unmodified. The script adds an
angular drive to its revolute joint and commands: open to 90 deg, then 150 deg (past the
asset's 105 deg upper limit), then closed. PhysX must track the targets, stop at the limit,
and keep the base on the table. Writes docs/figures/isaac_articulation.{webp,jpg,json}.
"""

from __future__ import annotations

import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET = os.path.join(ROOT, "output", "articulated", "keepsake_box.usda")
OUT = os.path.join(ROOT, "docs", "figures")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx import get_physx_interface  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

carb.settings.get_settings().set("/rtx-transient/resourcemanager/texturestreaming/async", False)
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateTimeStepsPerSecondAttr(240)

table = UsdGeom.Cube.Define(stage, "/World/table")
table.CreateSizeAttr(1.0)
table.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
table.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.1))
UsdPhysics.CollisionAPI.Apply(table.GetPrim())
table.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.6, 0.57)])

box = UsdGeom.Xform.Define(stage, "/World/box")
box.GetPrim().GetReferences().AddReference(ASSET)
box.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.001))

joint = stage.GetPrimAtPath("/World/box/joints/hinge")
assert joint.IsValid(), "hinge joint not found"
rev = UsdPhysics.RevoluteJoint(joint)
limits = (rev.GetLowerLimitAttr().Get(), rev.GetUpperLimitAttr().Get())
drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
drive.CreateTypeAttr("force")
drive.CreateStiffnessAttr(0.25)  # N*m per degree
drive.CreateDampingAttr(0.02)
drive.CreateMaxForceAttr(5.0)
target_attr = drive.CreateTargetPositionAttr(0.0)

dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(800)
key = UsdLux.SphereLight.Define(stage, "/World/key")
key.CreateRadiusAttr(0.2)
key.CreateIntensityAttr(25000)
key.AddTranslateOp().Set(Gf.Vec3d(-0.5, -0.6, 0.9))

cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.CreateFocalLengthAttr(24.0)
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(27.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.24, -0.27, 0.23), Gf.Vec3d(0.0, 0.015, 0.075), Gf.Vec3d(0, 0, 1))
cam.AddTransformOp().Set(view.GetInverse())
rp = rep.create.render_product("/World/cam", (800, 600))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

for _ in range(20):
    app.update()

physx = get_physx_interface()
try:
    FONT = ImageFont.load_default(size=18)
except TypeError:  # Pillow < 10.1
    FONT = ImageFont.load_default()


def quat_xyzw(path):
    tr = physx.get_rigidbody_transformation(path)
    return np.array(tr["rotation"], dtype=float), np.array(tr["position"], dtype=float)


def lid_angle() -> float:
    qb, _ = quat_xyzw("/World/box/base")
    ql, _ = quat_xyzw("/World/box/lid")
    # relative rotation conj(qb) * ql (xyzw); its angle is the hinge angle.
    bx, by, bz, bw = -qb[0], -qb[1], -qb[2], qb[3]
    lx, ly, lz, lw = ql
    w = bw * lw - bx * lx - by * ly - bz * lz
    return math.degrees(2 * math.acos(min(1.0, abs(w))))


timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(30):  # settle on the table
    app.update()
_, base0 = quat_xyzw("/World/box/base")

# (target deg, seconds, ramp?)
phases = [("open to 90°", 90.0, 2.0), ("command 150° (limit 105°)", 150.0, 1.6), ("close", 0.0, 2.0)]
dt = 1.0 / 60.0
frames, log, results = [], [], {}
target = 0.0
for label, goal, dur in phases:
    start = target
    n = int(dur / dt)
    for i in range(n):
        s = min(1.0, (i + 1) / (0.6 * n))
        target = start + (goal - start) * (0.5 - 0.5 * math.cos(math.pi * s))
        target_attr.Set(float(target))
        app.update()
        ang = lid_angle()
        log.append((round(target, 2), round(ang, 2)))
        if i % 2 == 0:
            data = rgb.get_data()
            if data is not None and getattr(data, "size", 0):
                img = Image.fromarray(np.asarray(data)[..., :3])
                d = ImageDraw.Draw(img)
                d.rectangle([0, 0, 800, 34], fill=(255, 255, 255))
                d.text((10, 8), f"Isaac Sim 6 PhysX  |  {label}  |  target {target:5.1f}°  lid {ang:5.1f}°",
                       fill=(30, 41, 59), font=FONT)
                frames.append(img)
    results[label] = round(lid_angle(), 2)
    print(f"[scan2usd] {label}: lid at {results[label]} deg", flush=True)
_, base1 = quat_xyzw("/World/box/base")
timeline.stop()

open_err = abs(results[phases[0][0]] - 90.0)
limit_hold = results[phases[1][0]]
closed = results[phases[2][0]]
summary = {
    "asset": os.path.relpath(ASSET, ROOT).replace("\\", "/"), "limits_deg": list(limits),
    "open_target_deg": 90.0, "open_reached_deg": results[phases[0][0]],
    "over_limit_target_deg": 150.0, "over_limit_reached_deg": limit_hold,
    "closed_deg": closed, "base_drift_mm": float(np.linalg.norm(base1 - base0) * 1000),
    "pass": bool(open_err < 3.0 and abs(limit_hold - limits[1]) < 3.0 and closed < 3.0
                 and np.linalg.norm(base1 - base0) < 0.005),
}
print("[scan2usd] articulation:", json.dumps(summary), flush=True)
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "isaac_articulation.json"), "w") as f:
    json.dump(summary, f, indent=2)
if frames:
    frames[0].save(os.path.join(OUT, "isaac_articulation.webp"), save_all=True, append_images=frames[1:],
                   duration=66, loop=0, quality=65, method=4)
    best = max(range(len(frames)), key=lambda k: k if k < len(frames) * 0.6 else -1)
    frames[best].save(os.path.join(OUT, "isaac_articulation.jpg"), quality=88)
app.close()
