"""Isaac Sim demo: Scan2USD assets in PhysX + a Replicator synthetic dataset.

Run with Isaac Sim's Python (6.0+), e.g.:

    set OMNI_KIT_ACCEPT_EULA=YES
    D:\\isaac\\v6\\Scripts\\python.exe scripts\\isaac_demo.py --frames 12

1. References every built asset (output/gso_assets/*, the demo mug, the articulated box)
   into one stage, unmodified, and drops them onto a table with PhysX.
2. Checks the pile settles (finite poses, everything at rest above the floor).
3. Renders a domain-randomized dataset with Replicator: random camera viewpoints and
   lighting; RGB, semantic segmentation, tight 2D boxes and depth.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=12)
parser.add_argument("--out", default=os.path.join(ROOT, "output", "isaac"))
parser.add_argument("--settle-steps", type=int, default=240)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1024, "height": 768})

import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

t0 = time.time()
assets = sorted(glob.glob(os.path.join(ROOT, "output", "gso_assets", "*", "*.usda")))
for extra in ("output/demo/asset/ceramic_mug.usda", "data/synthetic/mug/work/asset/mug.usda",
              "output/articulated/keepsake_box.usda", "output/buddha/buddha_head.usda"):
    pth = os.path.join(ROOT, extra)
    if os.path.exists(pth):
        assets.append(pth)
print(f"[scan2usd] {len(assets)} assets", flush=True)

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

# Table top: a thin box collider with a neutral matte material.
table = UsdGeom.Cube.Define(stage, "/World/table")
table.CreateSizeAttr(1.0)
table.AddScaleOp().Set(Gf.Vec3f(1.2, 0.9, 0.02))
table.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.01))
UsdPhysics.CollisionAPI.Apply(table.GetPrim())
table.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.6, 0.57)])

rng = random.Random(4)
prims = []
cols = 4
for i, path in enumerate(assets):
    name = os.path.splitext(os.path.basename(path))[0]
    prim_path = f"/World/obj_{i:02d}_{name[:20]}"
    xf = UsdGeom.Xform.Define(stage, prim_path)
    xf.GetPrim().GetReferences().AddReference(path)
    r, c = divmod(i, cols)
    x = (c - (cols - 1) / 2) * 0.24 + rng.uniform(-0.02, 0.02)
    y = (r - 0.5) * 0.26 + rng.uniform(-0.02, 0.02)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.08 + 0.05 * i))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(rng.uniform(-25, 25), rng.uniform(-25, 25), rng.uniform(0, 360)))
    prims.append((prim_path, name))

dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(700)
key = UsdLux.RectLight.Define(stage, "/World/key")
key.CreateIntensityAttr(30000)
key.CreateWidthAttr(0.8)
key.CreateHeightAttr(0.6)
key.AddTranslateOp().Set(Gf.Vec3d(-0.4, -0.5, 1.2))
key.AddRotateXYZOp().Set(Gf.Vec3f(30, -25, 0))

for _ in range(10):
    app.update()

import carb  # noqa: E402

# Load textures synchronously so the first captured frames aren't untextured.
carb.settings.get_settings().set("/rtx-transient/resourcemanager/texturestreaming/async", False)

# --- 1. PhysX drop test -------------------------------------------------------------
from omni.physx import get_physx_interface  # noqa: E402

timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(args.settle_steps):
    app.update()
physx = get_physx_interface()
poses = {}
for prim_path, name in prims:
    prim = stage.GetPrimAtPath(prim_path)
    links = [prim] if prim.HasAPI(UsdPhysics.RigidBodyAPI) else \
        [c for c in prim.GetChildren() if c.HasAPI(UsdPhysics.RigidBodyAPI)]
    pos = None
    if links:
        tr = physx.get_rigidbody_transformation(str(links[0].GetPath()))
        if tr and tr.get("ret_val", False):
            pos = list(tr["position"])
    poses[name] = pos
timeline.pause()
finite = all(p is not None and all(math.isfinite(v) for v in p) for p in poses.values())
on_table = all(p is not None and -0.02 < p[2] < 0.3 and abs(p[0]) < 0.6 and abs(p[1]) < 0.45 for p in poses.values())
physics = {"assets": len(prims), "finite": finite, "all_on_table": on_table, "poses": poses,
           "settle_steps": args.settle_steps}
print("[scan2usd] physics:", json.dumps(physics), flush=True)

# --- 2. Replicator dataset ------------------------------------------------------------
os.makedirs(args.out, exist_ok=True)
for prim_path, name in prims:
    with rep.get.prims(path_pattern=prim_path):
        rep.modify.semantics([("class", name)])

camera = rep.create.camera(position=(0.0, -0.9, 0.7), look_at=(0, 0, 0.04), focal_length=24)
rp = rep.create.render_product(camera, (1024, 768))
for _ in range(60):  # let materials and textures finish loading
    app.update()
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=os.path.join(args.out, "dataset"), rgb=True, semantic_segmentation=True,
                  colorize_semantic_segmentation=True, bounding_box_2d_tight=True, distance_to_camera=True)
writer.attach([rp])

with rep.trigger.on_frame():
    with camera:
        rep.modify.pose(position=rep.distribution.uniform((-0.8, -0.95, 0.45), (0.8, -0.55, 0.95)),
                        look_at=(0, 0, 0.04))
    with rep.get.prims(path_pattern="/World/dome"):
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(250, 1400))
        rep.modify.attribute("inputs:color", rep.distribution.uniform((0.75, 0.75, 0.75), (1.0, 1.0, 1.0)))
    with rep.get.prims(path_pattern="/World/key"):
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(8000, 45000))

for i in range(args.frames):
    rep.orchestrator.step(rt_subframes=16)
    print(f"[scan2usd] frame {i + 1}/{args.frames}", flush=True)
rep.orchestrator.wait_until_complete()

summary = {"physics": physics, "frames": args.frames, "seconds": round(time.time() - t0, 1),
           "isaac_sim": "6.0", "dataset": os.path.join(args.out, "dataset")}
with open(os.path.join(args.out, "isaac_demo.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("[scan2usd] done", json.dumps({k: v for k, v in summary.items() if k != "physics"}), flush=True)
app.close()
