"""Isaac Sim 6 (PhysX): the Shadow Hand places the scanned controller into the scanned holder.

    python scripts/demo_hand_pinch.py --export-plans output/hand_pinch_plans.json   # main venv
    set OMNI_KIT_ACCEPT_EULA=YES
    <isaac-sim>\\python.exe scripts\\isaac_hand_place.py --plans output/hand_pinch_plans.json [--trials 12] [--video]

The MuJoCo run (demo_hand_pinch.py, the overview's; or demo_hand_place.py's span grasp, the default
--plans) replayed as a plan, not as a recording: the same Menagerie hand
(imported with Isaac Sim's MJCF importer), the same scanned USD assets, the same per-seed scene poses
and waypoints, and the same motion code (hand_motion.py). PhysX simulates every contact; the
re-aimed variant measures the controller in the hand in PhysX and re-plans the insertion from it (on the
fly when the plan passes through its pre-place pose; from the plan's place_template when it has one).

Mapping of the hand: the floating forearm is welded (fixed joint) to a kinematic body that follows
the planned path, as the mocap weld does in MuJoCo. Finger servos get MuJoCo's stiffness, damping,
armature and torque limits (1 N m for most finger joints). MuJoCo drives each finger's J1 + J2 through
one tendon; PhysX gets an independent drive on each, J1 = J2, at twice the stiffness (a tendon error
on both joints pushes both). Writes docs/figures/isaac_hand_place.{json,mp4,jpg}.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "docs", "figures")
FIT = os.path.join(ROOT, "output", "holder_fit")
HOLDER = os.path.join(FIT, "assets", "cod_holder", "cod_holder.usda")
CTRL = os.path.join(FIT, "assets", "controller", "controller.usda")
MJCF = os.path.join(ROOT, "data", "external", "menagerie", "shadow_hand", "right_hand.xml")
HAND_DIR = os.path.join(ROOT, "output", "isaac_hand")
HAND_USD = os.path.join(HAND_DIR, "right_hand", "right_hand.usda")
FPS = 60  # control and app updates; PhysX substeps at PHYSICS_HZ

parser = argparse.ArgumentParser()
parser.add_argument("--plans", default=os.path.join(ROOT, "output", "hand_place_plans.json"))
parser.add_argument("--trials", type=int, default=12)
parser.add_argument("--physics-hz", type=int, default=480)
parser.add_argument("--video", action="store_true", help="also render the re-aimed run of seed 0 (RTX)")
parser.add_argument("--video-only", action="store_true", help="only the video (keeps the benchmark JSON)")
parser.add_argument("--stills", action="store_true",
                    help="path-traced stills of the re-aimed run (grasp, place, result) for make_hand_hero.py")
parser.add_argument("--probe", action="store_true", help="one open-loop trial with a per-waypoint trace")
parser.add_argument("--probe-reaimed", action="store_true", help="probe: the re-aimed variant")
parser.add_argument("--skin", type=float, nargs=2, default=None, metavar=("STIFFNESS", "DAMPING"),
                    help="compliant contact on the hand (N/m, N s/m), as MuJoCo's soft contacts")
parser.add_argument("--sweep", type=str, default="", help="probe: semicolon-separated 'stiffness,damping' pairs")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)
carb.settings.get_settings().set("/rtx-transient/resourcemanager/texturestreaming/async", False)

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.physics.tensors as tensors  # noqa: E402
import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import hand_motion  # noqa: E402

plans = json.load(open(args.plans))
HAND_Q = plans["hand_q"]
SKIN = args.skin
SERVO = plans["servo"]


def import_hand() -> str:
    if os.path.exists(HAND_USD):
        return HAND_USD
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

    cfg = MJCFImporterConfig(mjcf_path=MJCF, usd_path=HAND_DIR, import_scene=False, fix_base=False)
    path = MJCFImporter(cfg).import_mjcf()
    print("imported the Shadow Hand:", path, flush=True)
    return path


def gf(T) -> Gf.Matrix4d:
    return Gf.Matrix4d(*np.asarray(T, float).T.flatten().tolist())


def np44(m) -> np.ndarray:
    return np.array(m, float).reshape(4, 4).T


def pose7(T) -> np.ndarray:
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x y z w
    return np.r_[T[:3, 3], q]


def from7(p) -> np.ndarray:
    return hand_motion.pose(Rotation.from_quat(p[3:7]).as_matrix(), p[:3])


GHOST_OPACITY = 0.3
# The green ghost of the real resting pose is off in the Isaac renders: 6 mm / 5 deg from the placed
# scan it pokes through one side and reads as broken geometry. The result is given as a number; the
# MuJoCo video keeps the ghost, with a caption that explains it.
SHOW_GHOST = False
# (color, roughness, metallic): glossy light grey shells, a slightly darker forearm, metal knuckles.
LOOKS = {"shell": ((0.78, 0.78, 0.79), 0.28, 0.0), "forearm": ((0.62, 0.62, 0.64), 0.3, 0.0),
         "metal": ((0.55, 0.56, 0.58), 0.3, 0.8), "tray": ((0.70, 0.69, 0.67), 0.32, 0.0)}


def preview(stage, path, color, roughness, metallic):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def surface_points(path, n=3000):
    stage = Usd.Stage.Open(path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/{stage.GetDefaultPrim().GetName()}/render"))
    pts = np.asarray(mesh.GetPointsAttr().Get(), float)
    return pts[np.random.default_rng(1).choice(len(pts), min(n, len(pts)), replace=False)]


CP = surface_points(CTRL)


def scene(trial, collision, ghost=False):
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
    psx.CreateEnableGPUDynamicsAttr(False)  # CPU PhysX: one hand, and the numpy tensor frontend
    psx.CreateBroadphaseTypeAttr("MBP")

    floor = UsdGeom.Cube.Define(stage, "/World/floor")
    floor.CreateSizeAttr(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    floor.AddScaleOp().Set(Gf.Vec3f(24.0, 24.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.86, 0.86, 0.85)])

    tray = UsdGeom.Cube.Define(stage, "/World/tray")
    tray.CreateSizeAttr(2.0)
    T = np.array(trial["tray"])
    tray.AddTransformOp().Set(gf(T))
    tray.AddScaleOp().Set(Gf.Vec3f(*trial["tray_half_extents"]))
    UsdPhysics.CollisionAPI.Apply(tray.GetPrim())
    tray.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.60, 0.57)])
    looks = {k: preview(stage, f"/World/Looks/{k}", *v) for k, v in LOOKS.items()}
    UsdShade.MaterialBindingAPI.Apply(tray.GetPrim()).Bind(looks["tray"])

    holder = UsdGeom.Xform.Define(stage, "/World/holder")
    holder.GetPrim().GetReferences().AddReference(HOLDER)
    holder.AddTransformOp().Set(gf(trial["holder"]))
    for p in Usd.PrimRange(holder.GetPrim()):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
        if collision == "convex_hull" and p.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
    if collision == "convex_hull":
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
    rb.CreateSolverVelocityIterationCountAttr(4)

    if ghost:
        g = UsdGeom.Xform.Define(stage, "/World/ghost")
        g.GetPrim().GetReferences().AddReference(CTRL)
        g.AddTransformOp().Set(gf(trial["target"]))
        for p in Usd.PrimRange(g.GetPrim()):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
        rm = UsdGeom.Mesh(stage.GetPrimAtPath("/World/ghost/render"))
        UsdGeom.PrimvarsAPI(rm.GetPrim()).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        # RTX ignores displayOpacity: the ghost needs a material with real opacity (path tracing honors
        # it; RTX real-time also needs /rtx/raytracing/fractionalCutoutOpacity), else it draws opaque.
        gm = preview(stage, "/World/Looks/ghost", (0.1, 0.8, 0.35), 0.5, 0.0)
        UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/ghost/PreviewSurface")).CreateInput(
            "opacity", Sdf.ValueTypeNames.Float).Set(GHOST_OPACITY)
        UsdShade.MaterialBindingAPI.Apply(rm.GetPrim()).Bind(gm, UsdShade.Tokens.strongerThanDescendants)
        UsdGeom.Imageable(g.GetPrim()).MakeInvisible()

    # The hand: referenced as imported, placed so its forearm sits at the first waypoint.
    hand = UsdGeom.Xform.Define(stage, "/World/hand")
    hand.GetPrim().GetReferences().AddReference(import_hand())
    forearm = next(p for p in Usd.PrimRange(hand.GetPrim()) if p.GetName() == "rh_forearm")
    cache = UsdGeom.XformCache()
    F_local = np44(cache.GetLocalToWorldTransform(forearm))
    T_start = np.array(trial["waypoints"][0]["T"])
    hand.AddTransformOp().Set(gf(T_start @ np.linalg.inv(F_local)))
    # Look: glossy light grey like Isaac's Franka (the imported dark finish disappears against the
    # scene). The visual meshes are instances, so bind on each instance prim, stronger than its own.
    for p in Usd.PrimRange(hand.GetPrim()):
        if p.IsInstance():
            name = p.GetName()
            look = looks["metal"] if "knuckle" in name else looks["forearm"] if "forearm" in name else looks["shell"]
            UsdShade.MaterialBindingAPI.Apply(p).Bind(look, UsdShade.Tokens.strongerThanDescendants)

    # Kinematic body the forearm is welded to (MuJoCo: the mocap body and its weld).
    target = UsdGeom.Xform.Define(stage, "/World/target")
    target.AddTransformOp().Set(gf(T_start))
    trb = UsdPhysics.RigidBodyAPI.Apply(target.GetPrim())
    trb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(target.GetPrim()).CreateMassAttr(1.0)
    weld = UsdPhysics.FixedJoint.Define(stage, "/World/weld")
    weld.CreateBody0Rel().SetTargets([target.GetPath()])
    weld.CreateBody1Rel().SetTargets([forearm.GetPath()])
    weld.CreateExcludeFromArticulationAttr(True)  # a plain joint to the kinematic body, not a link of the hand
    for a in ("CreateLocalPos0Attr", "CreateLocalPos1Attr"):
        getattr(weld, a)(Gf.Vec3f(0, 0, 0))
    for a in ("CreateLocalRot0Attr", "CreateLocalRot1Attr"):
        getattr(weld, a)(Gf.Quatf(1, 0, 0, 0))

    # Finger servos as in MuJoCo (USD angular drives are per degree).
    q0 = HAND_Q[str(trial["waypoints"][0]["g"])]
    n_drive = 0
    for p in Usd.PrimRange(hand.GetPrim()):
        name = p.GetName()
        if not p.IsA(UsdPhysics.RevoluteJoint) or name not in SERVO:
            continue
        sv = SERVO[name]
        kp = sv["kp"] * (2.0 if sv["tendon"] else 1.0)
        d = UsdPhysics.DriveAPI.Apply(p, "angular")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(kp * math.pi / 180.0)
        d.CreateDampingAttr(sv["damping"] * math.pi / 180.0)
        d.CreateMaxForceAttr(max(abs(v) for v in sv["force_range"]))
        d.CreateTargetPositionAttr(math.degrees(q0.get(name, 0.0)))
        pj = PhysxSchema.PhysxJointAPI.Apply(p)
        pj.CreateArmatureAttr(sv["armature"])
        n_drive += 1
    # Skin friction 1.0 on the hand, the strongest coefficient wins (MuJoCo takes the max too).
    skin = UsdShade.Material.Define(stage, "/World/Looks/skin")
    pm = UsdPhysics.MaterialAPI.Apply(skin.GetPrim())
    pm.CreateStaticFrictionAttr(1.0)
    pm.CreateDynamicFrictionAttr(1.0)
    pm.CreateRestitutionAttr(0.0)
    pxm = PhysxSchema.PhysxMaterialAPI.Apply(skin.GetPrim())
    pxm.CreateFrictionCombineModeAttr("max")
    if SKIN is not None:
        pxm.CreateCompliantContactStiffnessAttr(SKIN[0])
        pxm.CreateCompliantContactDampingAttr(SKIN[1])
    for p in Usd.PrimRange(hand.GetPrim()):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(skin, UsdShade.Tokens.strongerThanDescendants, "physics")

    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(800)
    key = UsdLux.SphereLight.Define(stage, "/World/key")
    key.CreateRadiusAttr(0.35)
    key.CreateIntensityAttr(26000)
    key.AddTranslateOp().Set(Gf.Vec3d(0.9, -0.7, 1.5))
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(20.25)
    cam.CreateFocalLengthAttr(hand_motion.focal_mm(20.25))  # the shared lens (hand_motion.FOVY_DEG)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60.0))
    cam.AddTransformOp()
    return forearm.GetPath().pathString, n_drive


class Sim:
    def __init__(self, forearm_path, q_start):
        for _ in range(10):
            app.update()
        app_utils.play()
        app.update()
        self.view = tensors.create_simulation_view("numpy", stage_id=omni.usd.get_context().get_stage_id())
        self.view.set_subspace_roots("/")
        self.art = self.view.create_articulation_view(forearm_path)
        self.target = self.view.create_rigid_body_view("/World/target")
        self.ctrl = self.view.create_rigid_body_view("/World/ctrl")
        self.names = list(self.art.shared_metatype.dof_names)
        self.links = list(self.art.shared_metatype.link_names)
        self.i0 = np.array([0])
        q = self.q(q_start)
        self.art.set_dof_positions(q[None], self.i0)
        self.art.set_dof_position_targets(q[None], self.i0)

    def q(self, qd: dict) -> np.ndarray:
        return np.array([qd.get(n, 0.0) for n in self.names], np.float32)

    def set(self, T, q):
        self.target.set_kinematic_targets(pose7(T)[None].astype(np.float32), self.i0)
        self.art.set_dof_position_targets(q[None], self.i0)

    def ctrl_pose(self):
        return from7(np.asarray(self.ctrl.get_transforms()).reshape(-1))

    def forearm_pose(self):
        lt = np.asarray(self.art.get_link_transforms()).reshape(len(self.links), 7)
        return from7(lt[self.links.index("rh_forearm")])


def score(T_end, T_target):
    end = CP @ T_end[:3, :3].T + T_end[:3, 3]
    true = CP @ T_target[:3, :3].T + T_target[:3, 3]
    add = float(np.linalg.norm(end - true, axis=1).mean())
    R = T_end[:3, :3].T @ T_target[:3, :3]
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    held = bool(end[:, 2].min() > 0.02)
    return {"add_mm": add * 1000, "rot_deg": rot, "held": held, "placed": bool(held and add < 0.02)}


class Plan:
    """Runs the waypoints from a PhysX pre-step callback: every substep gets its own kinematic target
    and finger targets (MuJoCo: every 2 ms step), not just every app update."""

    def __init__(self, sim, trial, wps, correct, wait=0.5, trace=False):
        self.sim, self.trial, self.wps, self.correct, self.trace = sim, trial, wps, correct, trace
        self.t, self.wait, self.done = 0.0, wait, False
        self.T_prev, self.g_prev = wps[0]["T"], wps[0]["g"]
        self.k, self.mv, self.rel0 = 1, None, None
        sim.set(self.T_prev, sim.q(HAND_Q[str(self.g_prev)]))

    def _start(self):
        self.a, self.b = hand_motion.moves(self.wps, self.k)[0]
        self.mv = hand_motion.Move(self.T_prev, self.g_prev, self.wps[self.a:self.b + 1])
        self.qs = [self.sim.q(HAND_Q[str(g)]) for g in self.mv.g]
        self.tm = 0.0
        # A pre-place waypoint passed through (via) instead of stopped at: re-aim on the fly as the hand passes.
        self.reaim = (next((j for j, w in enumerate(self.mv.legs[:-1]) if w["tag"] == "preplace"), None)
                      if self.correct else None)

    def replan(self, H_rel):
        """The placement waypoints (pre-place on) re-aimed at the forearm pose H_rel in the controller frame."""
        if "place_template" in self.trial:  # each waypoint a controller pose C: the hand at C @ H_rel
            return [{**w, "T": np.array(w["C"]) @ H_rel} for w in self.trial["place_template"]]
        return hand_motion.place_waypoints(np.array(self.trial["c_goal"]), self.trial["approach"], H_rel,
                                           float(plans["open_key"]), plans["transport"])

    def _finish(self):
        w = self.wps[self.b]
        self.T_prev, self.g_prev = w["T"], w["g"]
        sim = self.sim
        if self.trace:
            C, F = sim.ctrl_pose(), sim.forearm_pose()
            msg = ""
            if self.rel0 is not None:
                d = np.linalg.inv(self.rel0) @ (np.linalg.inv(C) @ F)
                ang = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
                msg = f"  in-hand slip {np.linalg.norm(d[:3, 3]) * 1000:5.1f} mm {ang:5.1f} deg"
            print(f"  {w['tag']:9s} t {self.t:5.2f}  ctrl z {C[2, 3] * 100:5.1f} cm  forearm err "
                  f"{np.linalg.norm(F[:3, 3] - w['T'][:3, 3]) * 1000:5.1f} mm{msg}", flush=True)
        if w["tag"] == "settle":
            self.rel0 = np.linalg.inv(sim.ctrl_pose()) @ sim.forearm_pose()
        if self.correct and w["tag"] == "preplace":
            H_rel = np.linalg.inv(sim.ctrl_pose()) @ sim.forearm_pose()
            self.wps[self.b + 1:] = self.replan(H_rel)[1:]
        self.k = self.b + 1
        self.mv = None
        if self.k >= len(self.wps):
            self.done = True

    def step(self, dt):
        self.t += dt
        if self.done or self.t < self.wait:
            return
        if self.mv is None:
            self._start()
        self.tm += dt
        T, leg, u = self.mv.at(min(self.tm, self.mv.duration))
        if self.reaim is not None and leg > self.reaim:
            H_rel = np.linalg.inv(self.sim.ctrl_pose()) @ self.sim.forearm_pose()
            j = self.a + self.reaim + 1
            self.wps[j:] = self.replan(H_rel)[1:]
            self.mv.retarget(self.reaim + 1, self.wps[j:self.b + 1])
            self.reaim = None
            T, leg, u = self.mv.at(min(self.tm, self.mv.duration))
        self.sim.set(T, self.qs[leg] + (self.qs[leg + 1] - self.qs[leg]) * u)
        if self.trace and self.rel0 is not None and int(self.t / 0.25) != int((self.t - dt) / 0.25):
            C, F = self.sim.ctrl_pose(), self.sim.forearm_pose()
            dd = np.linalg.inv(self.rel0) @ (np.linalg.inv(C) @ F)
            print(f"    t {self.t:5.2f} leg {self.mv.legs[leg]['tag']:8s} ctrl z {C[2, 3] * 100:5.1f} cm  in-hand slip "
                  f"{np.linalg.norm(dd[:3, 3]) * 1000:5.1f} mm", flush=True)
        if self.tm >= self.mv.duration - 1e-9:
            self._finish()


def run(trial, collision, correct, record=None, trace=False, settle=1.5):
    wps = [{**w, "T": np.array(w["T"])} for w in trial["waypoints"]]
    forearm, n_drive = scene(trial, collision, ghost=record is not None and SHOW_GHOST)
    sim = Sim(forearm, HAND_Q[str(wps[0]["g"])])
    plan = Plan(sim, trial, wps, correct, trace=trace)
    sub = omni.physx.get_physx_interface().subscribe_physics_on_step_events(plan.step, True, 0)
    t, t_end = 0.0, None
    while t_end is None or t < t_end:
        app.update()
        t += 1.0 / FPS
        if record is not None and t >= plan.wait:
            record(t - plan.wait, wps)
        if plan.done and t_end is None:
            t_end = t + settle
    sub.unsubscribe()
    res = score(sim.ctrl_pose(), np.array(trial["target"]))
    app_utils.stop()
    return res, n_drive


def encode(frames, path_mp4, fps=30):
    """H.264 via ffmpeg (Isaac Sim's Python has no imageio ffmpeg backend): ffmpeg on PATH, or the one
    imageio-ffmpeg installed in the project venv."""
    import glob
    import shutil
    import subprocess
    import tempfile

    from PIL import Image

    exe = shutil.which("ffmpeg") or next(iter(glob.glob(os.path.join(
        ROOT, ".venv", "Lib", "site-packages", "imageio_ffmpeg", "binaries", "ffmpeg*.exe"))), None)
    tmp = tempfile.mkdtemp(prefix="isaac_hand_")
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp, f"f{i:05d}.png"))
    if exe is None:
        print(f"no ffmpeg found; frames are in {tmp}", flush=True)
        return
    subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path_mp4], check=True)
    shutil.rmtree(tmp, ignore_errors=True)


def video(trial, path_mp4):
    import omni.replicator.core as rep
    from PIL import Image, ImageDraw, ImageFont

    try:
        font, small = ImageFont.load_default(size=26), ImageFont.load_default(size=18)
    except TypeError:
        font = small = ImageFont.load_default()
    frames, state = [], {}
    W, H = 1280, 720

    def record(t, wps):
        if "shots" not in state:
            c0, g = np.array(trial["controller_start"])[:3, 3], np.array(trial["target"])[:3, 3]
            keys = hand_motion.shot_keys(c0, g, wps)
            if plans.get("camera") == "continuous":  # the two shots as one move (demo_hand_pinch.shot_keys)
                keys = [(0.0, keys[0][1][:-1] + keys[1][1])]
            state["shots"] = [(t0, hand_motion.camera_track(k)) for t0, k in keys]
            state["ev"] = hand_motion.event_times(wps)
            stage = omni.usd.get_context().get_stage()
            state["cam"] = UsdGeom.Xformable(stage.GetPrimAtPath("/World/cam")).GetOrderedXformOps()[0]
            state["ghost"] = stage.GetPrimAtPath("/World/ghost")
            rp = rep.create.render_product("/World/cam", (W, H))
            state["rgb"] = rep.AnnotatorRegistry.get_annotator("rgb")
            state["rgb"].attach([rp])
            state["rp"] = rp
        eye, look = hand_motion.orbit_eye(hand_motion.camera_at(state["shots"], t))
        state["cam"].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1)).GetInverse())
        if state["ghost"] and t >= state["ev"]["release"] and not state.get("ghost_on"):
            UsdGeom.Imageable(state["ghost"]).MakeVisible()
            state["ghost_on"] = True
        if len(frames) < t * 30:  # 30 fps out of the 60 Hz updates
            data = state["rgb"].get_data()
            if data is not None and getattr(data, "size", 0):
                frames.append((t, Image.fromarray(np.asarray(data)[..., :3])))

    res, _ = run(trial, "decomposed", True, record=record, settle=3.0)
    state["rgb"].detach()
    state["rp"].destroy()
    out = []
    for t, im in frames[2:]:  # the first frames can be blank while RTX warms up
        d = ImageDraw.Draw(im)
        d.text((28, 22), "Isaac Sim 6 (PhysX)  |  same plan, same hand, same scanned USD assets",
               fill=(30, 41, 59), font=small)
        if t > frames[-1][0] - 1.6:
            msg = f"Placed: {res['add_mm']:.1f} mm / {res['rot_deg']:.0f}\u00b0 from the real pose"
            tw = d.textlength(msg, font=font)
            d.rounded_rectangle([28, H - 74, 28 + tw + 36, H - 28], 10, fill=(255, 255, 255))
            d.text((46, H - 66), msg, fill=(21, 128, 61), font=font)
        out.append(np.asarray(im))
    encode(out, path_mp4)
    Image.fromarray(out[min(len(out) - 1, int((state["ev"]["away"] + 0.8) * 30))]).save(
        path_mp4.replace(".mp4", ".jpg"), quality=90)
    print(f"wrote {path_mp4} ({len(out)} frames): {res}", flush=True)
    return res


def stills(trial, out_dir, size=1024, spp=64, total_spp=2048):
    """Path-traced stills of the re-aimed run (seed 0): the timeline pauses at each moment (no physics
    steps while paused) while the path tracer accumulates, then the run continues."""
    import omni.replicator.core as rep
    import omni.timeline
    from PIL import Image

    settings = carb.settings.get_settings()
    tl = omni.timeline.get_timeline_interface()
    os.makedirs(out_dir, exist_ok=True)
    # (event, offset s, name, lookat offset from the controller (start or goal), distance, azimuth, elevation)
    shots = [("lift", 0.0, "grasp", "start", (0.0, 0.0, 0.06), 0.85, 160, -26),
             ("seat", -0.45, "place", "goal", (0.0, -0.03, 0.06), 0.85, 140, -28),
             ("away", 1.4, "result", "goal", (0.0, 0.0, 0.01), 0.52, 100, -10)]
    state = {"done": set()}

    def record(t, wps):
        if "ev" not in state:
            state["ev"] = hand_motion.event_times(wps)
            stage = omni.usd.get_context().get_stage()
            cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/cam"))
            cam.GetHorizontalApertureAttr().Set(24.0)
            cam.GetVerticalApertureAttr().Set(24.0)
            cam.GetFocalLengthAttr().Set(45.0)  # 30 deg square view
            state["cam"] = cam.GetPrim()
            state["op"] = UsdGeom.Xformable(cam.GetPrim()).GetOrderedXformOps()[0]
            state["ghost"] = stage.GetPrimAtPath("/World/ghost")
            state["rp"] = rep.create.render_product("/World/cam", (size, size))
            state["rgb"] = rep.AnnotatorRegistry.get_annotator("rgb")
            state["rgb"].attach([state["rp"]])
        for ev, dt, name, ref, off, dist, az, el in shots:
            if name in state["done"] or t < state["ev"][ev] + dt:
                continue
            state["done"].add(name)
            base = np.array(trial["controller_start" if ref == "start" else "target"])[:3, 3]
            eye, look = hand_motion.orbit_eye([*(base + off), dist, az, el])
            state["op"].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1)).GetInverse())
            if name == "result" and state["ghost"]:
                UsdGeom.Imageable(state["ghost"]).MakeVisible()
            tl.pause()
            settings.set("/rtx/rendermode", "PathTracing")
            settings.set("/rtx/pathtracing/spp", spp)
            settings.set("/rtx/pathtracing/totalSpp", total_spp)
            for _ in range(total_spp // spp + 12):
                app.update()
            img = np.asarray(state["rgb"].get_data())[..., :3]
            Image.fromarray(img).save(os.path.join(out_dir, f"{name}.png"))
            print(f"still {name} at t {t:.2f} s", flush=True)
            settings.set("/rtx/rendermode", "RaytracedLighting")
            tl.play()

    res, _ = run(trial, "decomposed", True, record=record, settle=2.0)
    state["rgb"].detach()
    state["rp"].destroy()
    with open(os.path.join(out_dir, "stills.json"), "w") as f:
        json.dump({"seed": trial["seed"], "result": res, "shots": [s[2] for s in shots]}, f, indent=1)
    print("stills:", res, flush=True)


trials = plans["trials"][: args.trials]
if args.probe and args.sweep:
    for pair in args.sweep.split(";"):
        SKIN = [float(v) for v in pair.split(",")] if pair != "rigid" else None
        for correct in (False, True):
            rows = [run(tr, "decomposed", correct)[0] for tr in trials]
            print(f"SWEEP skin {pair} {'reaimed' if correct else 'open'}: held {sum(r['held'] for r in rows)}/"
                  f"{len(rows)}", " ".join(f"{r['add_mm']:.0f}/{r['rot_deg']:.0f}" for r in rows), flush=True)
elif args.probe:
    res, n_drive = run(trials[0], "decomposed", args.probe_reaimed, trace=True)
    print("drives set:", n_drive, "result:", res, flush=True)
elif args.video_only:
    video(trials[0], os.path.join(OUT, "isaac_hand_place.mp4"))
elif args.stills:
    stills(trials[0], os.path.join(ROOT, "output", "isaac_hand_stills"))
else:
    results = {}
    for collision, correct in (("decomposed", False), ("decomposed", True), ("convex_hull", True)):
        key = f"{collision}_{'reaimed' if correct else 'open_loop'}"
        rows = []
        for tr in trials:
            res, _ = run(tr, collision, correct)
            rows.append({"seed": tr["seed"], **res})
            print(key, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()}),
                  flush=True)
        results[key] = rows
    summary = {k: {"trials": len(v), "held": sum(r["held"] for r in v), "placed": sum(r["placed"] for r in v),
                   "add_mm_median": float(np.median([r["add_mm"] for r in v])),
                   "rot_deg_median": float(np.median([r["rot_deg"] for r in v]))} for k, v in results.items()}
    print(json.dumps(summary, indent=1), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "isaac_hand_place.json"), "w") as f:
        json.dump({"summary": summary, "trials": results, "physics_hz": args.physics_hz}, f, indent=2)
    if args.video:
        video(trials[0], os.path.join(OUT, "isaac_hand_place.mp4"))
app.close()
