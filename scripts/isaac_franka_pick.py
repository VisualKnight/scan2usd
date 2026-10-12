"""Isaac Sim 6: replay the Panda pick of the scanned mug with PhysX (sim-to-sim check).

    python scripts/export_panda_plan.py                      # plan once (MuJoCo IK), main venv
    set OMNI_KIT_ACCEPT_EULA=YES
    <isaac-sim>\\python.exe scripts\\isaac_franka_pick.py
    # hybrid: same pick, the mug drawn by its Gaussian splat (scan2usd splat), RTX real-time PT
    # (set PXR_PLUGINPATH_NAME=src\\scan2usd\\schema so the stored grasps' schema is recognized)
    ... isaac_franka_pick.py --asset speckled_mug --plan output/panda_plan_speckled.json \\
        --splat output/hybrid/speckled_mug/speckled_mug_splat.usdc --name isaac_franka_pick_splat

NVIDIA's Franka (Isaac assets, standard fingers) gets the same joint waypoints the MuJoCo
Menagerie Panda used in scripts/demo_panda.py, interpolated the same way, as joint position
targets. The scanned mug USD comes from the asset library, once with its convex decomposition and
once with the `convex_hull` variant. Same asset, same motion, a different physics engine: does
the pick still work, and does the single-hull failure reproduce?
Writes docs/figures/isaac_franka_pick.{webp,jpg,json}.
"""

from __future__ import annotations

import argparse
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "output", "library", "scan2usd_library.usda")
OUT = os.path.join(ROOT, "docs", "figures")
DT = 1.0 / 60.0

parser = argparse.ArgumentParser()
parser.add_argument("--plan", default=os.path.join(ROOT, "output", "panda_plan.json"))
parser.add_argument("--asset", default="ceramic_mug", help="library asset to pick")
parser.add_argument("--splat", help="splat .usdc from `scan2usd splat` (<name>_splat.usdc): draw the mug with "
                                    "its Gaussian splat; physics stays on the convex decomposition")
parser.add_argument("--name", default="isaac_franka_pick", help="output figure name")
args = parser.parse_args()
PLAN = args.plan
CASES = ([("decomposed", "Gaussian-splat look, convex-hull physics")] if args.splat
         else [("decomposed", "convex decomposition"), ("convex_hull", "one convex hull")])

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import carb  # noqa: E402

if args.splat:
    import omni.kit.app  # noqa: E402

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "omni.usd.schema.usd_particle_field", True)
    carb.settings.get_settings().set("/rtx/rendermode", "RealTimePathTracing")  # splats render in RT path tracing

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

carb.settings.get_settings().set("/rtx-transient/resourcemanager/texturestreaming/async", False)
try:
    FONT = ImageFont.load_default(size=18)
except TypeError:
    FONT = ImageFont.load_default()

with open(PLAN) as f:
    plan = json.load(f)
MUG_POS = np.array(plan["mug_pos"])
TARGET = MUG_POS + np.array(plan["place_shift"])


def scene(collision: str):
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
    floor.AddTranslateOp().Set(Gf.Vec3d(0.3, 0, -0.05))
    floor.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.78, 0.78, 0.76)])
    target = UsdGeom.Cylinder.Define(stage, "/World/target")
    target.CreateRadiusAttr(0.05)
    target.CreateHeightAttr(0.002)
    target.AddTranslateOp().Set(Gf.Vec3d(TARGET[0], TARGET[1], 0.001))
    target.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.75, 0.35)])

    robot = stage_utils.add_reference_to_stage(
        usd_path=get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", path="/World/robot")
    vsets = robot.GetVariantSets()
    variants = {n: list(vsets.GetVariantSet(n).GetVariantNames()) for n in vsets.GetNames()}
    selected = {n: vsets.GetVariantSet(n).GetVariantSelection() for n in vsets.GetNames()}

    mug = UsdGeom.Xform.Define(stage, "/World/mug")
    mug.GetPrim().GetReferences().AddReference(LIB, f"/Library/{args.asset}")
    mug.GetPrim().GetVariantSet("collision").SetVariantSelection(collision)
    if args.splat:
        # The splat prim carries its own reconstruction->asset transform, so it can ride on any copy
        # of the asset: payload it under the library mug and hide the textured mesh.
        stage.DefinePrim("/World/mug/splat").GetPayloads().AddPayload(os.path.abspath(args.splat))
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/mug/render")).CreateVisibilityAttr("invisible")
    # The library lays assets out on a grid; put this one where the plan expects it.
    UsdGeom.XformCommonAPI(mug.GetPrim()).SetTranslate(Gf.Vec3d(MUG_POS[0], MUG_POS[1], 0.0))
    # More solver iterations for the grasped body: PhysX's default 4 lets thin-wall grasps creep.
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(mug.GetPrim())
    rb.CreateSolverPositionIterationCountAttr(32)
    rb.CreateSolverVelocityIterationCountAttr(4)

    # Rubber finger pads, as in the MuJoCo demo (friction 1.5, strongest coefficient wins there too).
    pad = UsdShade.Material.Define(stage, "/World/Looks/finger_pad")
    pm = UsdPhysics.MaterialAPI.Apply(pad.GetPrim())
    pm.CreateStaticFrictionAttr(1.5)
    pm.CreateDynamicFrictionAttr(1.5)
    pm.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(pad.GetPrim()).CreateFrictionCombineModeAttr("max")
    pads = 0
    for name in ("panda_leftfinger", "panda_rightfinger"):
        link = stage.GetPrimAtPath(f"/World/robot/{name}")
        # Collision meshes live inside instanced prims (not editable), so bind on the link and
        # let the binding inherit (strongerThanDescendants).
        UsdShade.MaterialBindingAPI.Apply(link).Bind(pad, UsdShade.Tokens.strongerThanDescendants, "physics")
        pads += sum(1 for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI))
    drives = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/robot")):
        if p.GetName().startswith("panda_finger_joint") and p.HasAPI(UsdPhysics.DriveAPI, "linear"):
            d = UsdPhysics.DriveAPI(p, "linear")
            drives[p.GetName()] = {"stiffness": d.GetStiffnessAttr().Get(), "damping": d.GetDampingAttr().Get(),
                                   "max_force": d.GetMaxForceAttr().Get()}
            # Stock: 400 N/m (about 2 N when pinching a 5 mm cup wall). Same fix as the MuJoCo demo:
            # 2000 N/m (about 10 N here), force cap at the real Franka Hand's 70 N continuous.
            d.GetStiffnessAttr().Set(2000.0)
            d.GetMaxForceAttr().Set(70.0)
    variants["finger_pad_colliders"] = pads
    variants["finger_drives"] = drives
    g = stage.GetPrimAtPath("/World/mug/grasps")
    kids = g.GetChildren() if g else []
    variants["usd_grasps"] = {"count": len(kids),
                              "schema_recognized": bool(kids) and all(k.HasAPI("Scan2UsdGraspAPI") for k in kids)}

    dome = UsdLux.DomeLight.Define(stage, "/World/dome")
    dome.CreateIntensityAttr(900)
    key = UsdLux.SphereLight.Define(stage, "/World/key")
    key.CreateRadiusAttr(0.3)
    key.CreateIntensityAttr(30000)
    key.AddTranslateOp().Set(Gf.Vec3d(0.9, -1.0, 1.6))
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    cam.CreateFocalLengthAttr(32.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(27.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
    # Roughly the MuJoCo demo's viewpoint (lookat, distance 1.15, elevation -22 deg, azimuth 205 deg).
    look = Gf.Vec3d(0.48, 0.0, 0.16)
    az, el = math.radians(205), math.radians(22)
    eye = look - 1.15 * Gf.Vec3d(math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el))
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, look, Gf.Vec3d(0, 0, 1)).GetInverse())
    return variants, selected


def run(collision: str, title: str):
    variants, selected = scene(collision)
    for _ in range(10):
        app.update()
    rp = rep.create.render_product("/World/cam", (640, 480))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])

    app_utils.play()
    app.update()
    robot = Articulation("/World/robot")
    mug = RigidPrim("/World/mug")
    home = np.array(plan["home"])
    f0 = plan["finger_start_m"]
    robot.set_dof_positions([[*home, f0, f0]])
    robot.set_dof_position_targets([[*home, f0, f0]])
    for _ in range(30):
        app.update()
    z0 = float(mug.get_world_poses()[0].numpy().reshape(-1)[2])

    frames, max_z, q_from, g_from = [], z0, home, f0
    for w in plan["waypoints"]:
        q_to, g_to = np.array(w["q"]), w["finger_m"]
        n = max(1, int(round(w["seconds"] / DT)))
        for i in range(n):
            s = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / n)
            g = g_from + (g_to - g_from) * min(1.0, 2 * (i + 1) / n)
            robot.set_dof_position_targets([[*(q_from + (q_to - q_from) * s), g, g]])
            app.update()
            max_z = max(max_z, float(mug.get_world_poses()[0].numpy().reshape(-1)[2]))
            data = rgb.get_data()
            if data is not None and getattr(data, "size", 0):
                img = Image.fromarray(np.asarray(data)[..., :3])
                d = ImageDraw.Draw(img)
                d.rectangle([0, 0, 640, 30], fill=(255, 255, 255))
                d.text((10, 5), f"Isaac Sim 6 PhysX · {title}", fill=(30, 41, 59), font=FONT)
                frames.append(img)
        q_from, g_from = q_to, g_to
        if os.environ.get("SCAN2USD_DEBUG"):
            fp = np.asarray(robot.get_dof_positions().numpy()).reshape(-1)[7:]
            stage = omni.usd.get_context().get_stage()
            cache = UsdGeom.XformCache()
            tips = {n: np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/World/robot/{n}")).ExtractTranslation())
                    for n in ("panda_hand", "panda_leftfinger", "panda_rightfinger") if stage.GetPrimAtPath(f"/World/robot/{n}")}
            mp = mug.get_world_poses()[0].numpy().reshape(-1)
            print(f"[scan2usd] wp {plan['waypoints'].index(w)}: fingers {fp.round(4)} mug {mp.round(4)} "
                  + " ".join(f"{k} {v.round(4)}" for k, v in tips.items()), flush=True)
    for _ in range(40):
        app.update()
    pos, quat = mug.get_world_poses()
    pos = pos.numpy().reshape(-1)
    w_, x_, y_, z_ = np.asarray(quat.numpy()).reshape(-1)[:4]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x_ * x_ + y_ * y_)))))
    q_final = np.asarray(robot.get_dof_positions().numpy()).reshape(-1)
    app_utils.stop()
    rgb.detach()
    rp.destroy()
    lift = max_z - z0
    place = float(np.hypot(pos[0] - TARGET[0], pos[1] - TARGET[1]))
    res = {"collision": collision, "lifted_m": round(lift, 4), "place_error_m": round(place, 4),
           "tilt_deg": round(tilt, 1), "success": bool(lift > 0.1 and place < 0.03 and tilt < 10),
           "joint_tracking_error_rad": round(float(np.abs(q_final[:7] - q_from).max()), 4),
           "robot_variants": variants, "robot_variant_selection": selected}
    print("[scan2usd] pick:", json.dumps(res), flush=True)
    return res, frames


results, videos = [], []
for collision, title in CASES:
    res, frames = run(collision, title)
    results.append(res)
    videos.append(frames)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, f"{args.name}.json"), "w") as f:
    json.dump({"plan": os.path.relpath(PLAN, ROOT).replace(os.sep, "/") + " (scripts/export_panda_plan.py)",
               "asset": args.asset, "splat": bool(args.splat), "results": results}, f, indent=2)
n = min(len(v) for v in videos)
if n:
    sheets = []
    for i in range(0, n, 2):
        tiles = [v[i].copy() for v in videos]
        for t, res in zip(tiles, results):
            if i > n * 0.88:
                ImageDraw.Draw(t).text((12, t.height - 32), "picked and placed" if res["success"] else "failed",
                                       fill=(21, 128, 61) if res["success"] else (185, 28, 28), font=FONT)
        s = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height), "white")
        for k, t in enumerate(tiles):
            s.paste(t, (k * t.width, 0))
        sheets.append(s)
    sheets[0].save(os.path.join(OUT, f"{args.name}.webp"), save_all=True, append_images=sheets[1:],
                   duration=66, loop=0, quality=60, method=4)
    sheets[int(len(sheets) * 0.55)].save(os.path.join(OUT, f"{args.name}.jpg"), quality=86)
app.close()
