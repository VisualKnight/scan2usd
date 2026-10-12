"""Domain-randomized training data for object detection, rendered in Isaac Sim 6 (Replicator).

Run with Isaac Sim's Python:

    set OMNI_KIT_ACCEPT_EULA=YES
    D:\\isaac\\v6\\Scripts\\python.exe scripts\\isaac_synthdata.py --frames 600 --out output\\synth

Each frame re-samples, in plain Python on the USD stage: which assets are present and where
(non-overlapping footprints, random yaw), the table color and roughness, the lights (color,
intensity, key-light position) and the camera (distance, height, aim, small roll). Replicator's
BasicWriter saves RGB, instance segmentation and tight 2D boxes; convert with
scripts/replicator_to_coco.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=600)
parser.add_argument("--out", default=os.path.join(ROOT, "output", "synth"))
parser.add_argument("--width", type=int, default=896)
parser.add_argument("--height", type=int, default=672)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--distractors", type=int, default=0, help="unlabeled random primitives on the table")
parser.add_argument("--panels", type=int, default=0, help="unlabeled background panels around the table")
args = parser.parse_args()
args.out = os.path.abspath(args.out)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

carb.settings.get_settings().set("/rtx-transient/resourcemanager/texturestreaming/async", False)
rng = random.Random(args.seed)

assets = sorted(glob.glob(os.path.join(ROOT, "output", "gso_assets", "*", "*.usda")))
for extra in ("output/demo/asset/ceramic_mug.usda", "data/work/mug_scan/asset/speckled_mug.usda",
              "output/articulated/keepsake_box.usda", "output/buddha/buddha_head.usda"):
    p = os.path.join(ROOT, extra)
    if os.path.exists(p):
        assets.append(p)
print(f"[scan2usd] {len(assets)} assets", flush=True)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")

# Table: a slab with its own material so its color can be randomized.
table = UsdGeom.Cube.Define(stage, "/World/table")
table.CreateSizeAttr(1.0)
table.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
table.AddScaleOp().Set(Gf.Vec3f(2.4, 2.0, 0.1))
mat = UsdShade.Material.Define(stage, "/World/Looks/table")
shader = UsdShade.Shader.Define(stage, "/World/Looks/table/surface")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(table.GetPrim()).Bind(mat)

dome = UsdLux.DomeLight.Define(stage, "/World/dome")
key = UsdLux.SphereLight.Define(stage, "/World/key")
key.CreateRadiusAttr(0.25)
key.AddTranslateOp()

# Assets, each with a translate + yaw we can resample, and its footprint radius.
objects = []
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
for i, path in enumerate(assets):
    name = os.path.splitext(os.path.basename(path))[0]
    xf = UsdGeom.Xform.Define(stage, f"/World/obj_{i:02d}_{name[:20]}")
    xf.GetPrim().GetReferences().AddReference(path)
    xf.AddTranslateOp()
    xf.AddRotateZOp()
    r = cache.ComputeLocalBound(xf.GetPrim()).ComputeAlignedRange()
    size = r.GetMax() - r.GetMin()
    objects.append({"path": str(xf.GetPath()), "name": name,
                    "radius": 0.5 * math.hypot(size[0], size[1])})

for obj in objects:
    with rep.get.prims(path_pattern=obj["path"]):
        rep.modify.semantics([("class", obj["name"])])

# Unlabeled distractors (never annotated): small primitives / flat cards on the table and large
# panels around it, so the detector learns that "unfamiliar stuff" is background.
def _material(name: str) -> str:
    path = f"/World/Looks/{name}"
    m = UsdShade.Material.Define(stage, path)
    s = UsdShade.Shader.Define(stage, path + "/surface")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    s.CreateInput("metallic", Sdf.ValueTypeNames.Float)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    return path


distractors = []
kinds = [UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Cylinder, UsdGeom.Cone]
for k in range(args.distractors + args.panels):
    panel = k >= args.distractors
    gprim = (UsdGeom.Cube if panel else kinds[k % len(kinds)]).Define(stage, f"/World/distractors/d{k:02d}")
    if isinstance(gprim, UsdGeom.Cube):
        gprim.CreateSizeAttr(1.0)
    elif isinstance(gprim, UsdGeom.Sphere):
        gprim.CreateRadiusAttr(0.5)
    else:
        gprim.CreateRadiusAttr(0.5)
        gprim.CreateHeightAttr(1.0)
    gprim.AddTransformOp()
    UsdShade.MaterialBindingAPI.Apply(gprim.GetPrim()).Bind(UsdShade.Material.Get(stage, _material(f"d{k:02d}")))
    distractors.append({"path": str(gprim.GetPath()), "panel": panel, "look": f"/World/Looks/d{k:02d}/surface"})

cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.CreateFocalLengthAttr(24.0)
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(36.0 * args.height / args.width)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
cam.AddTransformOp()

rp = rep.create.render_product("/World/cam", (args.width, args.height))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=args.out, rgb=True, instance_segmentation=True, colorize_instance_segmentation=False,
                  semantic_segmentation=False, bounding_box_2d_tight=True)
writer.attach([rp])


def set_pose(prim, x: float, y: float, yaw_deg: float) -> None:
    """Position on the table and yaw, whatever xform ops Replicator standardized the prim to."""
    half = math.radians(yaw_deg) / 2
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        kind = op.GetOpType()
        if kind == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(x, y, 0.0) if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Vec3f(x, y, 0.0))
        elif kind == UsdGeom.XformOp.TypeOrient:
            q = (Gf.Quatd if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Quatf)(
                math.cos(half), 0.0, 0.0, math.sin(half))
            op.Set(q)
        elif kind == UsdGeom.XformOp.TypeRotateZ:
            op.Set(yaw_deg)
        elif kind == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(Gf.Vec3f(0.0, 0.0, yaw_deg))


def place_objects() -> list[dict]:
    """Random subset, non-overlapping on the table, random yaw; the rest hidden."""
    placed = []
    stage = ctx.get_stage()  # re-fetch: Replicator steps can invalidate cached handles
    for obj in rng.sample(objects, len(objects)):
        prim = stage.GetPrimAtPath(obj["path"])
        visible = rng.random() < 0.8
        if visible:
            for _ in range(40):
                x, y = rng.uniform(-0.38, 0.38), rng.uniform(-0.3, 0.3)
                if all(math.hypot(x - p["x"], y - p["y"]) > obj["radius"] + p["radius"] + 0.01 for p in placed):
                    set_pose(prim, x, y, rng.uniform(0, 360))
                    placed.append({"x": x, "y": y, "radius": obj["radius"]})
                    break
            else:
                visible = False
        img = UsdGeom.Imageable(prim)
        img.MakeVisible() if visible else img.MakeInvisible()
    place_distractors(list(placed))
    return placed


def _color() -> Gf.Vec3f:
    if rng.random() < 0.3:  # near-black / near-white, like markers, cables, paper
        v = rng.choice([rng.uniform(0.01, 0.08), rng.uniform(0.8, 0.98)])
        return Gf.Vec3f(v, v, v)
    return Gf.Vec3f(*[rng.uniform(0.02, 0.95) for _ in range(3)])


def place_distractors(placed: list[dict]) -> None:
    stage = ctx.get_stage()
    n_on = rng.randint(0, args.distractors)
    for k, d in enumerate(distractors):
        prim = stage.GetPrimAtPath(d["path"])
        shader = UsdShade.Shader(stage.GetPrimAtPath(d["look"]))
        shader.GetInput("diffuseColor").Set(_color())
        shader.GetInput("roughness").Set(rng.uniform(0.2, 1.0))
        shader.GetInput("metallic").Set(rng.choice([0.0, 0.0, 0.0, 1.0]))
        visible = True
        if d["panel"]:
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(1.3, 2.2)
            s = Gf.Vec3d(rng.uniform(0.3, 1.2), rng.uniform(0.05, 0.4), rng.uniform(0.3, 1.6))
            m = (Gf.Matrix4d().SetScale(s) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(a) + 90))
                 * Gf.Matrix4d().SetTranslate(Gf.Vec3d(r * math.cos(a), r * math.sin(a), s[2] / 2 - 0.1)))
        else:
            visible = k < n_on
            size = rng.uniform(0.02, 0.12)
            flat = rng.random() < 0.35  # thin cards: printed markers, phones, coasters
            s = Gf.Vec3d(size, size * rng.uniform(0.4, 1.5), size * (rng.uniform(0.02, 0.08) if flat else rng.uniform(0.5, 2.0)))
            rad = 0.5 * math.hypot(s[0], s[1])
            for _ in range(30):
                x, y = rng.uniform(-0.6, 0.6), rng.uniform(-0.5, 0.5)
                if all(math.hypot(x - p["x"], y - p["y"]) > rad + p["radius"] + 0.01 for p in placed):
                    break
            else:
                visible = False
            tilt = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), rng.uniform(0, 360)))
            m = Gf.Matrix4d().SetScale(s) * tilt * Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, s[2] / 2))
            if visible:
                placed.append({"x": x, "y": y, "radius": rad})
        UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(m)
        img = UsdGeom.Imageable(prim)
        img.MakeVisible() if visible else img.MakeInvisible()


def randomize_look() -> None:
    stage = ctx.get_stage()
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/table/surface"))
    shader.GetInput("diffuseColor").Set(Gf.Vec3f(*[rng.uniform(0.15, 0.95) for _ in range(3)]))
    shader.GetInput("roughness").Set(rng.uniform(0.3, 0.95))
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/dome"))
    key = UsdLux.SphereLight(stage.GetPrimAtPath("/World/key"))
    warm = rng.uniform(0.0, 0.25)
    dome.GetIntensityAttr().Set(rng.uniform(250, 1300))
    dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0 - warm * 0.5, 1.0 - warm))
    key.GetIntensityAttr().Set(rng.uniform(0, 60000))
    key.GetColorAttr().Set(Gf.Vec3f(1.0, rng.uniform(0.8, 1.0), rng.uniform(0.65, 1.0)))
    a = rng.uniform(0, 2 * math.pi)
    key.GetOrderedXformOps()[0].Set(Gf.Vec3d(1.6 * math.cos(a), 1.6 * math.sin(a), rng.uniform(0.8, 2.0)))


def aim_camera() -> None:
    az = rng.uniform(0, 2 * math.pi)
    el = math.radians(rng.uniform(20, 65))
    dist = rng.uniform(0.35, 1.1)
    target = Gf.Vec3d(rng.uniform(-0.12, 0.12), rng.uniform(-0.1, 0.1), rng.uniform(0.0, 0.06))
    eye = target + dist * Gf.Vec3d(math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
    roll = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), rng.uniform(-6, 6)))
    cam = UsdGeom.Camera(ctx.get_stage().GetPrimAtPath("/World/cam"))
    cam.GetOrderedXformOps()[0].Set(roll * view.GetInverse())


for _ in range(30):
    app.update()

t0 = time.time()
counts = []
for i in range(args.frames):
    counts.append(len(place_objects()))
    randomize_look()
    aim_camera()
    rep.orchestrator.step(rt_subframes=8)
    if (i + 1) % 50 == 0:
        print(f"[scan2usd] frame {i + 1}/{args.frames} ({time.time() - t0:.0f} s)", flush=True)
rep.orchestrator.wait_until_complete()
with open(os.path.join(args.out, "synthdata.json"), "w") as f:
    json.dump({"frames": args.frames, "distractors": args.distractors, "panels": args.panels, "assets": [o["name"] for o in objects], "seed": args.seed,
               "objects_per_frame": sum(counts) / max(1, len(counts)), "seconds": round(time.time() - t0, 1)},
              f, indent=2)
print(f"[scan2usd] done in {time.time() - t0:.0f} s", flush=True)
app.close()
