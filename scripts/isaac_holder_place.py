"""Isaac Sim 6 (PhysX): NVIDIA's Franka places the scanned controller into the scanned holder.

    python scripts/demo_holder_place.py --export-plans output/holder_place_plans.json   # main venv
    set OMNI_KIT_ACCEPT_EULA=YES
    <isaac-sim>\\python.exe scripts\\isaac_holder_place.py [--trials 12] [--probe]

A sim-to-sim check of scripts/demo_holder_place.py: the same joint plans (MuJoCo IK), the same tray,
holder and controller poses, replayed on Isaac Sim's Franka with PhysX contacts. The holder and the
controller are the Scan2USD USD assets, unmodified (the holder static, with its convex decomposition
or one convex hull). As in MuJoCo the finger pads are rubber (friction 1.5) and the gripper squeezes
like a real Franka Hand (force capped at its 70 N). Joint targets are set from a PhysX pre-step
callback, every substep (as isaac_hand_place.py; per-frame targets move the arm in jolts). Score:
the controller's pose 1.5 s after release vs the pose measured in the scan of the real pair.
Writes docs/figures/isaac_holder_place.{json,jpg,webp}.

Result: the plan that holds 12/12 in MuJoCo holds about 1/12 here. The pinch closes as in MuJoCo
(fingers stop at ~20 mm on the edge), but on the lift the controller, hanging 6 cm off the pinch,
twists out of the two pads; pad friction 2.0, 16 velocity iterations or 960 Hz don't fix it (0-1 of 4).
MuJoCo needed its no-slip solver for the same grip. The Shadow Hand, which grasps level with the
center of mass, places it in both engines (isaac_hand_place.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
FIT = os.path.join(ROOT, "output", "holder_fit")
HOLDER = os.path.join(FIT, "assets", "cod_holder", "cod_holder.usda")
CTRL = os.path.join(FIT, "assets", "controller", "controller.usda")
FPS = 60
# MuJoCo Menagerie's Panda finger pads (left finger frame: center xyz, half extents, m). Isaac's Franka collides
# with a convex hull of the whole tapered finger, which wedges the controller's rounded edge out of the pinch;
# MuJoCo grips with these flat pads. The right finger's frame is turned 180 deg about z relative to MuJoCo's.
FINGER_PADS = [(0.0, 0.0055, 0.0445, 0.0085, 0.004, 0.0085), (0.0055, 0.002, 0.05, 0.003, 0.002, 0.003), (-0.0055, 0.002, 0.05, 0.003, 0.002, 0.003), (0.0055, 0.002, 0.0395, 0.003, 0.002, 0.0035), (-0.0055, 0.002, 0.0395, 0.003, 0.002, 0.0035)]
PAD_MU = 1.5
VEL_ITERS = 4
ARM_GAIN = 4.0  # stiffer arm servos: tracking within ~1 mm at the fingertips, as in MuJoCo

parser = argparse.ArgumentParser()
parser.add_argument("--plans", default=os.path.join(ROOT, "output", "holder_place_plans.json"))
parser.add_argument("--trials", type=int, default=12)
parser.add_argument("--physics-hz", type=int, default=480)
parser.add_argument("--video-trial", type=int, default=0)
parser.add_argument("--probe", action="store_true", help="one trial, decomposed, with a per-waypoint trace")
parser.add_argument("--no-video", action="store_true")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import carb  # noqa: E402

carb.settings.get_settings().set("/rtx-transient/resourcemanager/texturestreaming/async", False)

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.physics.tensors as tensors  # noqa: E402
import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

try:
    FONT = ImageFont.load_default(size=18)
except TypeError:
    FONT = ImageFont.load_default()

with open(args.plans) as f:
    PLANS = json.load(f)


def surface_points(path, n=3000):
    stage = Usd.Stage.Open(path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/{stage.GetDefaultPrim().GetName()}/render"))
    pts = np.asarray(mesh.GetPointsAttr().Get(), float)
    return pts[np.random.default_rng(1).choice(len(pts), min(n, len(pts)), replace=False)]


CP = surface_points(CTRL)


def gf(T) -> Gf.Matrix4d:
    return Gf.Matrix4d(*np.asarray(T, float).T.flatten().tolist())


def static(prim):
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)


def preview(stage, path, color, roughness):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def scene(trial, collision):
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
    psx = PhysxSchema.PhysxSceneAPI.Apply(ps.GetPrim())
    psx.CreateTimeStepsPerSecondAttr(args.physics_hz)
    psx.CreateEnableGPUDynamicsAttr(False)  # CPU PhysX: one robot, and the numpy tensor frontend
    psx.CreateBroadphaseTypeAttr("MBP")
    floor = UsdGeom.Cube.Define(stage, "/World/floor")
    floor.CreateSizeAttr(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0.3, 0, -0.05))
    floor.AddScaleOp().Set(Gf.Vec3f(24.0, 24.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.86, 0.86, 0.85)])

    hx, hy, hz = trial["stand_half_extents"]
    T_tray = np.array(trial["stand"], float)
    tray = UsdGeom.Cube.Define(stage, "/World/tray")
    tray.CreateSizeAttr(2.0)
    tray.AddTransformOp().Set(gf(T_tray))
    tray.AddScaleOp().Set(Gf.Vec3f(hx, hy, hz))
    UsdPhysics.CollisionAPI.Apply(tray.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(tray.GetPrim()).Bind(preview(stage, "/World/Looks/tray", (0.70, 0.69, 0.67), 0.32))

    holder = UsdGeom.Xform.Define(stage, "/World/holder")
    holder.GetPrim().GetReferences().AddReference(HOLDER)
    holder.AddTransformOp().Set(gf(trial["holder"]))
    static(holder.GetPrim())
    if collision == "convex_hull":
        for p in Usd.PrimRange(holder.GetPrim()):
            if p.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
        render = stage.GetPrimAtPath("/World/holder/render")
        UsdPhysics.CollisionAPI.Apply(render)
        UsdPhysics.MeshCollisionAPI.Apply(render).CreateApproximationAttr("convexHull")

    ctrl = UsdGeom.Xform.Define(stage, "/World/ctrl")
    ctrl.GetPrim().GetReferences().AddReference(CTRL)
    T0 = np.array(trial["controller_start"])
    T0[2, 3] += 0.001
    ctrl.AddTransformOp().Set(gf(T0))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(ctrl.GetPrim())
    rb.CreateSolverPositionIterationCountAttr(32)
    rb.CreateSolverVelocityIterationCountAttr(VEL_ITERS)

    stage_utils.add_reference_to_stage(
        usd_path=get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", path="/World/robot")
    # Rubber finger pads (friction 1.5, the stronger coefficient of a pair wins), as in MuJoCo.
    pad = UsdShade.Material.Define(stage, "/World/Looks/finger_pad")
    pm = UsdPhysics.MaterialAPI.Apply(pad.GetPrim())
    pm.CreateStaticFrictionAttr(PAD_MU)
    pm.CreateDynamicFrictionAttr(PAD_MU)
    pm.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(pad.GetPrim()).CreateFrictionCombineModeAttr("max")
    for name, sign in (("panda_leftfinger", 1.0), ("panda_rightfinger", -1.0)):
        link = stage.GetPrimAtPath(f"/World/robot/{name}")
        for k, (x, y, z, hx, hy, hz) in enumerate(FINGER_PADS):
            box = UsdGeom.Cube.Define(stage, f"/World/robot/{name}/pad_{k}")
            box.CreateSizeAttr(2.0)
            box.AddTranslateOp().Set(Gf.Vec3d(sign * x, sign * y, z))
            box.AddScaleOp().Set(Gf.Vec3f(hx, hy, hz))
            box.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.15, 0.16)])
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(link).Bind(pad, UsdShade.Tokens.strongerThanDescendants, "physics")
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/robot")):
        # Gravity compensation, as MuJoCo's arm servos have: without it the stock drives sag a few cm
        # at the fingertips, and the pinch misses the controller's edge.
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
        if p.GetName().startswith("panda_joint") and p.HasAPI(UsdPhysics.DriveAPI, "angular"):
            d = UsdPhysics.DriveAPI(p, "angular")
            if args.probe:
                print("  drive", p.GetName(), d.GetStiffnessAttr().Get(), d.GetDampingAttr().Get(), d.GetMaxForceAttr().Get())
            d.GetStiffnessAttr().Set(d.GetStiffnessAttr().Get() * ARM_GAIN)
            d.GetDampingAttr().Set(d.GetDampingAttr().Get() * ARM_GAIN ** 0.5)
        if p.GetName().startswith("panda_finger_joint") and p.HasAPI(UsdPhysics.DriveAPI, "linear"):
            d = UsdPhysics.DriveAPI(p, "linear")
            # The MuJoCo grip (stiff position servo), force capped at a real Franka Hand's 70 N.
            d.GetStiffnessAttr().Set(PLANS["grip_stiffness_n_per_m"])
            d.GetDampingAttr().Set(100.0)
            d.GetMaxForceAttr().Set(70.0)
            PhysxSchema.PhysxJointAPI.Apply(p).CreateMaxJointVelocityAttr(0.3)

    if args.probe:
        for prim in ("/World/robot/panda_leftfinger", "/World/robot/panda_rightfinger", "/World/robot/panda_hand"):
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(prim)).CreateThresholdAttr(0.0)
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(800)
    key = UsdLux.SphereLight.Define(stage, "/World/key")
    key.CreateRadiusAttr(0.35)
    key.CreateIntensityAttr(26000)
    key.AddTranslateOp().Set(Gf.Vec3d(0.9, -1.0, 1.6))
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(27.0)
    cam.CreateFocalLengthAttr(28.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 60.0))
    # The MuJoCo video's wide view, a little farther back.
    look = Gf.Vec3d(0.49, 0.10, 0.20)
    az, el = math.radians(235), math.radians(18)
    eye = look - 1.25 * Gf.Vec3d(math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el))
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, look, Gf.Vec3d(0, 0, 1)).GetInverse())


class Plan:
    """Piecewise-linear joint-space tracking, as in MuJoCo, stepped from a PhysX pre-step callback.
    The fingers reach their target in the first half of each segment (MuJoCo: same)."""

    def __init__(self, view, trial, settle_s=1.5, trace=False):
        self.v, self.trial, self.trace = view, trial, trace
        self.q_from = np.array(trial["home"], float)
        self.g_from = float(trial["finger_start_m"])
        self.k, self.tm, self.t, self.wait, self.done = 0, 0.0, 0.0, settle_s, False
        self.carried = False  # controller within 3 cm of its goal when the fingers open
        self.set(self.q_from, self.g_from)

    def set(self, q, g):
        self.v.art.set_dof_position_targets(np.r_[q, g, g].astype(np.float32)[None], self.v.i0)

    def step(self, dt):
        self.t += dt
        if self.done or self.t < self.wait:
            return
        w = self.trial["waypoints"][self.k]
        q_to, g_to, dur = np.array(w["q"], float), float(w["finger_m"]), float(w["seconds"])
        self.tm += dt
        s = min(1.0, self.tm / dur)
        self.set(self.q_from + (q_to - self.q_from) * s, self.g_from + (g_to - self.g_from) * min(1.0, 2 * s))
        if s >= 1.0:
            if self.g_from < 0.005 and g_to > 0.02:  # this segment opened the fingers: the release
                goal = np.asarray(self.trial["target"])[:3, 3]
                self.carried = bool(np.linalg.norm(self.v.ctrl_pose()[:3, 3] - goal) < 0.03)
            if self.trace:
                q = self.v.q()
                print(f"  wp {self.k:2d} t {self.t:5.2f}: fingers {q[7] * 1000:5.1f}/{q[8] * 1000:5.1f} mm "
                      f"(target {g_to * 1000:4.1f}), arm err {np.abs(q[:7] - q_to).max():.3f} rad, "
                      f"ctrl {self.v.ctrl_pose()[:3, 3].round(3)}", flush=True)
            if self.trace and self.k in (4, 5, 6):
                for name, cv in self.v.contacts.items():
                    fm = np.asarray(cv.get_contact_force_matrix(1.0 / args.physics_hz)).reshape(-1, 3)
                    print(f"     {name}: N vs [ctrl, tray, holder] =", np.round(np.linalg.norm(fm, axis=1), 2), flush=True)
            self.q_from, self.g_from, self.tm = q_to, g_to, 0.0
            self.k += 1
            self.done = self.k >= len(self.trial["waypoints"])


class View:
    def __init__(self, trial):
        for _ in range(10):
            app.update()
        app_utils.play()
        app.update()
        self.sim = tensors.create_simulation_view("numpy", stage_id=omni.usd.get_context().get_stage_id())
        self.sim.set_subspace_roots("/")
        self.art = self.sim.create_articulation_view("/World/robot")
        self.ctrl = self.sim.create_rigid_body_view("/World/ctrl")
        self.i0 = np.array([0])
        self.contacts = {}
        if args.probe:
            for f in ("panda_leftfinger", "panda_rightfinger", "panda_hand"):
                self.contacts[f] = self.sim.create_rigid_contact_view(
                    f"/World/robot/{f}", filter_patterns=["/World/ctrl", "/World/tray", "/World/holder"])
        q0 = np.r_[trial["home"], trial["finger_start_m"], trial["finger_start_m"]].astype(np.float32)[None]
        self.art.set_dof_positions(q0, self.i0)
        self.art.set_dof_position_targets(q0, self.i0)

    def q(self):
        return np.asarray(self.art.get_dof_positions()).reshape(-1)

    def ctrl_pose(self):
        p = np.asarray(self.ctrl.get_transforms()).reshape(-1)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rotation.from_quat(p[3:7]).as_matrix(), p[:3]
        return T


def run(trial, collision, record=False, title="", trace=False, settle=1.5):
    scene(trial, collision)
    v = View(trial)
    plan = Plan(v, trial, trace=trace)
    sub = omni.physx.get_physx_interface().subscribe_physics_on_step_events(plan.step, True, 0)
    rgb = rp = None
    frames = []
    if record:
        import omni.replicator.core as rep

        rp = rep.create.render_product("/World/cam", (560, 420))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([rp])
    snap = None
    if trace:  # close-ups of the pinch
        import omni.replicator.core as rep

        stage = omni.usd.get_context().get_stage()
        c = UsdGeom.Camera.Define(stage, "/World/snapcam")
        c.CreateFocalLengthAttr(35.0)
        c.CreateHorizontalApertureAttr(36.0)
        c.CreateVerticalApertureAttr(27.0)
        c.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
        look = Gf.Vec3d(*np.array(trial["controller_start"])[:3, 3])
        eye = look + Gf.Vec3d(0.10, -0.32, 0.12)
        c.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, look, Gf.Vec3d(0, 0, 1)).GetInverse())
        srp = rep.create.render_product("/World/snapcam", (640, 480))
        snap = {"rgb": rep.AnnotatorRegistry.get_annotator("rgb"), "done": set()}
        snap["rgb"].attach([srp])
    t, t_end, n = 0.0, None, 0
    while t_end is None or t < t_end:
        app.update()
        t += 1.0 / FPS
        n += 1
        if record and n % 3 == 0 and t > 1.2:
            data = rgb.get_data()
            if data is not None and getattr(data, "size", 0):
                img = Image.fromarray(np.asarray(data)[..., :3])
                d = ImageDraw.Draw(img)
                d.rectangle([0, 0, img.width, 30], fill=(255, 255, 255))
                d.text((10, 5), f"Isaac Sim 6 PhysX \u00b7 {title}", fill=(30, 41, 59), font=FONT)
                frames.append(img)
        if trace and snap is not None and plan.k in (4, 5, 6, 7) and plan.k not in snap["done"]:
            snap["done"].add(plan.k)
            for _ in range(4):
                app.update()
            Image.fromarray(np.asarray(snap["rgb"].get_data())[..., :3]).save(
                os.path.join(ROOT, "output", f"ihp_wp{plan.k}.png"))
        if plan.done and t_end is None:
            t_end = t + settle
    del sub
    T1 = v.ctrl_pose()
    carried = plan.carried
    T = np.asarray(trial["target"])
    true_pts = CP @ T[:3, :3].T + T[:3, 3]
    end_pts = CP @ T1[:3, :3].T + T1[:3, 3]
    add = float(np.linalg.norm(end_pts - true_pts, axis=1).mean())
    rot = float(np.degrees(np.arccos(np.clip((np.trace(T1[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1))))
    held = bool(end_pts[:, 2].min() > 0.02)
    app_utils.stop()
    if record:
        rgb.detach()
        rp.destroy()
    res = {"seed": trial["seed"], "collision": collision, "add_mm": add * 1000, "rot_deg": rot, "held": held,
           "placed": bool(held and add < 0.02), "carried": carried}
    print("[scan2usd] place:", json.dumps({k: (round(x, 2) if isinstance(x, float) else x) for k, x in res.items()}),
          flush=True)
    return res, frames


def video(videos):
    (fa, ra), (fb, rb_) = videos["decomposed"], videos["convex_hull"]
    n = min(len(fa), len(fb))
    sheets = []
    for i in range(n):
        tiles = [fa[i].copy(), fb[i].copy()]
        if i > n * 0.85:
            for tile, r in zip(tiles, (ra, rb_)):
                ok = r["held"] and r["placed"]
                msg = (f"held: {r['add_mm']:.1f} mm / {r['rot_deg']:.0f}\u00b0 from the real pose" if ok
                       else "slides off the hands" if r.get("carried") else "dropped on the way")
                d = ImageDraw.Draw(tile)
                d.rectangle([4, tile.height - 36, 14 + d.textlength(msg, font=FONT), tile.height - 6], fill=(255, 255, 255))
                d.text((10, tile.height - 32), msg, fill=(21, 128, 61) if ok else (185, 28, 28), font=FONT)
        s = Image.new("RGB", (2 * tiles[0].width + 10, tiles[0].height), "white")
        s.paste(tiles[0], (0, 0))
        s.paste(tiles[1], (tiles[0].width + 10, 0))
        sheets.append(s)
    sheets[0].save(os.path.join(OUT, "isaac_holder_place.webp"), save_all=True, append_images=sheets[1:],
                   duration=100, loop=0, quality=50, method=6)
    sheets[int(n * 0.92)].save(os.path.join(OUT, "isaac_holder_place.jpg"), quality=88)
    print("[scan2usd] wrote isaac_holder_place.webp", flush=True)


if args.probe:
    run(PLANS["trials"][0], "decomposed", trace=True)
else:
    results = {"decomposed": [], "convex_hull": []}
    videos = {}
    for trial in PLANS["trials"][: args.trials]:
        for collision, title in (("decomposed", "holder: convex decomposition"), ("convex_hull", "holder: one convex hull")):
            record = trial["seed"] == args.video_trial and not args.no_video
            res, frames = run(trial, collision, record, title)
            results[collision].append(res)
            if record:
                videos[collision] = (frames, res)
    summary = {}
    for k, rows in results.items():
        summary[k] = {"trials": len(rows), "held": sum(r["held"] for r in rows), "placed": sum(r["placed"] for r in rows),
                      "add_mm_median": float(np.median([r["add_mm"] for r in rows])),
                      "rot_deg_median": float(np.median([r["rot_deg"] for r in rows]))}
        print(f"[scan2usd] {k}: held {summary[k]['held']}/{len(rows)}, median {summary[k]['add_mm_median']:.1f} mm / "
              f"{summary[k]['rot_deg_median']:.1f} deg", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "isaac_holder_place.json"), "w") as f:
        json.dump({"plans": "output/holder_place_plans.json (demo_holder_place.py --export-plans)",
                   "physics_hz": args.physics_hz, "summary": summary, "results": results}, f, indent=2)
    if len(videos) == 2:
        video(videos)
app.close()
