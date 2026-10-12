"""Isaac Sim check of the USD asset library: every collision variant in PhysX, with timing.

    set OMNI_KIT_ACCEPT_EULA=YES
    D:\\isaac\\v6\\Scripts\\python.exe scripts\\isaac_library.py

References output/library/scan2usd_library.usda (scan2usd library) into a stage on a table,
selects one collision variant for all assets, drops everything from 2 cm and simulates 2 s at
240 Hz. Per variant: collider count, PhysX time per step (no rendering), resting heights.
Also renders the library shelf. Writes docs/figures/isaac_library.{jpg,json}.
"""

from __future__ import annotations

import json
import math
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "output", "library", "scan2usd_library.usda")
OUT = os.path.join(ROOT, "docs", "figures")
VARIANTS = ("decomposed", "convex_hull", "bounding_box")
STEPS, DT = 480, 1.0 / 240.0

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx import get_physx_interface  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

carb.settings.get_settings().set("/rtx-transient/resourcemanager/texturestreaming/async", False)


def build_stage(variant: str):
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
    table.AddTranslateOp().Set(Gf.Vec3d(0.35, 0.25, -0.05))
    table.AddScaleOp().Set(Gf.Vec3f(1.6, 1.2, 0.1))
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    table.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.6, 0.57)])

    lib = UsdGeom.Xform.Define(stage, "/World/library")
    lib.GetPrim().GetReferences().AddReference(LIB)
    lib.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.02))
    assets = [p for p in lib.GetPrim().GetChildren()]
    for p in assets:
        p.GetVariantSet("collision").SetVariantSelection(variant)
    bodies = [p for p in Usd.PrimRange(lib.GetPrim()) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    colliders = sum(1 for p in Usd.PrimRange(lib.GetPrim()) if p.HasAPI(UsdPhysics.CollisionAPI))

    dome = UsdLux.DomeLight.Define(stage, "/World/dome")
    dome.CreateIntensityAttr(800)
    key = UsdLux.SphereLight.Define(stage, "/World/key")
    key.CreateRadiusAttr(0.3)
    key.CreateIntensityAttr(30000)
    key.AddTranslateOp().Set(Gf.Vec3d(-0.3, -0.8, 1.3))
    return stage, assets, bodies, colliders


results = {}
shelf = None
# Picture first (static layout, converged path tracing), then the timed physics runs.
stage, *_ = build_stage("decomposed")
cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.CreateFocalLengthAttr(24.0)
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(20.25)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.30, -0.55, 0.62), Gf.Vec3d(0.30, 0.17, 0.0), Gf.Vec3d(0, 0, 1))
cam.AddTransformOp().Set(view.GetInverse())
for _ in range(20):
    app.update()
rp = rep.create.render_product("/World/cam", (1280, 720))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])
for _ in range(3):
    rep.orchestrator.step(rt_subframes=32)
data = rgb.get_data()
if data is not None and getattr(data, "size", 0):
    shelf = Image.fromarray(np.asarray(data)[..., :3]).crop((200, 20, 1260, 520))  # trim empty table
rgb.detach()
rp.destroy()

for variant in VARIANTS:
    stage, assets, bodies, colliders = build_stage(variant)
    for _ in range(10):
        app.update()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app.update()
    timeline.pause()  # step PhysX directly below, without rendering
    physx = get_physx_interface()
    t0 = time.perf_counter()
    for i in range(STEPS):
        physx.update_simulation(DT, i * DT)
    elapsed = time.perf_counter() - t0
    physx.update_transformations(False, True)
    heights, finite = {}, True
    for b in bodies:
        tr = physx.get_rigidbody_transformation(str(b.GetPath()))
        pos = tr["position"] if tr and tr.get("ret_val") else None
        finite &= pos is not None and all(math.isfinite(v) for v in pos)
        if pos is not None:
            heights[str(b.GetPath()).split("/")[-1] if b.GetName() in ("base", "lid") else b.GetName()] = round(pos[2], 4)
    results[variant] = {"colliders": colliders, "bodies": len(bodies), "ms_per_step": round(1000 * elapsed / STEPS, 3),
                        "finite": bool(finite), "all_on_table": all(-0.02 < z < 0.3 for z in heights.values()),
                        "resting_z": heights}
    print(f"[scan2usd] {variant}: {json.dumps(results[variant])}", flush=True)
    timeline.stop()

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "isaac_library.json"), "w") as f:
    json.dump({"library": "output/library/scan2usd_library.usda", "steps": STEPS, "dt": DT, "variants": results}, f,
              indent=2)
if shelf is not None:
    shelf.save(os.path.join(OUT, "isaac_library.jpg"), quality=86)
app.close()
