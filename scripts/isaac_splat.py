"""Hybrid splat + mesh assets in Isaac Sim 6 (RTX real-time path tracing).

    set OMNI_KIT_ACCEPT_EULA=YES
    D:\\isaac\\v6\\Scripts\\python.exe scripts\\isaac_splat.py --shots heldout,showcase,drop

The hybrid asset (scan2usd splat) references the physics asset and adds a
UsdVol.ParticleField3DGaussianSplat under its rigid body, with an `appearance` variant set
(mesh | splat | both). Shots, written to output/splat/isaac/:
  heldout   each held-out photo's solved camera (scripts/lookdev_camera.py): the asset rendered
            with appearance=mesh and appearance=splat, plus the object mask, for
            scripts/compare_splat.py (Buddha: real photos; mug: synthetic photos, studio lighting
            differs from the renders). The photos were not used to train the splat.
  showcase  the hybrid Buddha and mug on a table, same camera and lights: mesh, splat, and
            splat_proxy (the splat seen by the camera, the mesh hidden from camera rays but casting
            shadows, primvars:hideForCamera).
  drop      physics: the splat Buddha and two splat mugs (rigid bodies, convex hulls) dropped onto the
            table; the Gaussians ride on the rigid bodies. Frames -> drop_frames/ (video: compare_splat.py).

Splats render in RTX real-time path tracing ('RealTimePathTracing'); the offline path tracer in
Isaac Sim 6.0 draws them as dark absorbers, so every shot here uses the real-time mode.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "splat", "isaac")

parser = argparse.ArgumentParser()
parser.add_argument("--shots", default="heldout,showcase,drop")
parser.add_argument("--buddha", default=os.path.join(ROOT, "output", "hybrid", "buddha_head", "buddha_head.usda"))
parser.add_argument("--mug", default=os.path.join(ROOT, "output", "hybrid", "speckled_mug", "speckled_mug.usda"))
parser.add_argument("--cams", default=os.path.join(ROOT, "output", "splat", "cams", "camera_*.json"))
parser.add_argument("--mug-cams", default=os.path.join(ROOT, "output", "splat", "cams_mug", "camera_*.json"))
parser.add_argument("--heldout-objects", default="buddha,mug")
parser.add_argument("--subframes", type=int, default=16)
parser.add_argument("--tag", default="")
args = parser.parse_args()
shots = args.shots.split(",")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "omni.usd.schema.usd_particle_field", True)

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

settings = carb.settings.get_settings()
settings.set("/rtx/rendermode", "RealTimePathTracing")
settings.set("/rtx/resourcemanager/texturestreaming/async", False)
settings.set("/rtx-transient/resourcemanager/texturestreaming/async", False)
ASSETS_ROOT = get_assets_root_path()
os.makedirs(OUT, exist_ok=True)
MM = 0.01  # USD lens/filmback units are tenths of a scene unit; this stage is in meters


def new_stage(physics=False):
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    if physics:
        ps = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        ps.CreateGravityMagnitudeAttr(9.81)
        PhysxSchema.PhysxSceneAPI.Apply(ps.GetPrim()).CreateTimeStepsPerSecondAttr(240)
    return stage


def place(stage, path, usd, pos=(0, 0, 0), rot=(0, 0, 0), appearance=None, dynamic=False):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.GetPrim().GetReferences().AddReference(usd)
    if appearance:
        xf.GetPrim().GetVariantSet("appearance").SetVariantSelection(appearance)
    api = UsdGeom.XformCommonAPI(xf.GetPrim())
    api.SetTranslate(Gf.Vec3d(*pos))
    api.SetRotate(Gf.Vec3f(*rot))
    if not dynamic:
        for p in Usd.PrimRange(xf.GetPrim()):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
    return xf


def camera(stage, path, eye, target, focal=50.0, aperture=36.0, aspect=16 / 9):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal * MM)
    cam.CreateHorizontalApertureAttr(aperture * MM)
    cam.CreateVerticalApertureAttr(aperture / aspect * MM)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    cam.AddTransformOp().Set(m.GetInverse())
    return cam


def photo_camera(stage, path, cfg):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(cfg["focal_length_mm"] * MM)
    cam.CreateHorizontalApertureAttr(cfg["horizontal_aperture_mm"] * MM)
    cam.CreateVerticalApertureAttr(cfg["vertical_aperture_mm"] * MM)
    cam.CreateHorizontalApertureOffsetAttr(cfg["horizontal_aperture_offset_mm"] * MM)
    cam.CreateVerticalApertureOffsetAttr(cfg["vertical_aperture_offset_mm"] * MM)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
    cam.AddTransformOp().Set(Gf.Matrix4d(cfg["camera_to_world_usd"]))
    return cam


def lights(stage, hdri="/NVIDIA/Assets/Skies/Indoor/hospital_room_4k.hdr", dome=900.0, key=20000.0):
    d = UsdLux.DomeLight.Define(stage, "/World/dome")
    d.CreateIntensityAttr(dome)
    d.CreateTextureFileAttr().Set(ASSETS_ROOT + hdri)
    d.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    r = UsdLux.RectLight.Define(stage, "/World/ceiling")
    r.CreateIntensityAttr(key)
    r.CreateWidthAttr(1.2)
    r.CreateHeightAttr(0.6)
    r.AddTranslateOp().Set(Gf.Vec3d(0, 0, 2.2))


def table(stage, size=(1.6, 1.0), color=(0.42, 0.3, 0.2), collide=False):
    t = UsdGeom.Cube.Define(stage, "/World/table")
    t.CreateSizeAttr(1.0)
    t.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.02))
    t.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], 0.04))
    t.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(t.GetPrim())
    return t


class Camera:
    def __init__(self, cam_path, size, semantic=False):
        self.rp = rep.create.render_product(cam_path, size)
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([self.rp])
        self.seg = None
        if semantic:
            self.seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
            self.seg.attach([self.rp])

    def shoot(self, name=None, subframes=None, steps=3):
        for _ in range(steps):
            rep.orchestrator.step(rt_subframes=subframes or args.subframes)
        img = np.asarray(self.rgb.get_data())[..., :3].copy()
        if name:
            Image.fromarray(img).save(os.path.join(OUT, f"{name}.png"))
        mask = None
        if self.seg is not None:
            data = self.seg.get_data()
            ids = np.asarray(data["data"])
            obj = [int(k) for k, v in data["info"]["idToLabels"].items() if "object" in str(v)]
            mask = np.isin(ids, obj)
            if name:
                Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(OUT, f"{name}_mask.png"))
        return img, mask

    def close(self):
        self.rgb.detach()
        if self.seg is not None:
            self.seg.detach()
        self.rp.destroy()


def settle(n=20):
    for _ in range(n):
        app.update()


results = {"render_mode": "RealTimePathTracing"}

if "heldout" in shots:
    jobs = {"buddha": (args.buddha, args.cams, "heldout_"), "mug": (args.mug, args.mug_cams, "mug_heldout_")}
    for key in args.heldout_objects.split(","):
        asset, cam_glob, prefix = jobs[key]
        cams = sorted(glob.glob(cam_glob))
        if not cams or not os.path.exists(asset):
            continue
        stage = new_stage()
        lights(stage)
        obj = place(stage, "/World/object", asset, appearance="mesh")
        with rep.get.prims(path_pattern="/World/object"):
            rep.modify.semantics([("class", "object")])
        vset = obj.GetPrim().GetVariantSet("appearance")
        for i, c in enumerate(cams):
            cfg = json.load(open(c))
            stem = os.path.splitext(os.path.basename(cfg["photo"]))[0].replace("._c", "")
            cam_path = f"/World/cam_{i}"
            photo_camera(stage, cam_path, cfg)
            size = (1280, round(cfg["height"] * 1280 / cfg["width"]))
            shooter = Camera(cam_path, size, semantic=True)
            for look in ("mesh", "splat"):
                vset.SetVariantSelection(look)
                settle(5)
                shooter.shoot(f"{prefix}{stem}_{look}{args.tag}")
            shooter.close()
            results.setdefault("heldout", []).append({"object": key, "photo": cfg["photo"], "stem": stem})
            print(f"[scan2usd] held-out view {key} {stem}", flush=True)

if "showcase" in shots:
    stage = new_stage()
    lights(stage)
    table(stage)
    buddha = place(stage, "/World/buddha", args.buddha, pos=(0.0, 0.05, 0.0), rot=(0, 0, 100))
    mug = place(stage, "/World/mug", args.mug, pos=(0.17, -0.08, 0.0), rot=(0, 0, 200)) if os.path.exists(args.mug) else None
    camera(stage, "/World/cam", (0.2, -0.62, 0.26), (0.06, 0.0, 0.09), focal=50)
    shooter = Camera("/World/cam", (1280, 720))
    for look in ("mesh", "splat", "splat_proxy"):
        for x in (buddha, mug):
            if x is not None:
                x.GetPrim().GetVariantSet("appearance").SetVariantSelection(look)
        settle(5)
        shooter.shoot(f"showcase_{look}{args.tag}", subframes=32)
    shooter.close()
    results["showcase"] = {"mug": mug is not None, "looks": ["mesh", "splat", "splat_proxy"]}

if "drop" in shots:
    stage = new_stage(physics=True)
    lights(stage)
    table(stage, size=(2.0, 2.0), collide=True)
    place(stage, "/World/buddha", args.buddha, pos=(0.0, 0.0, 0.16), rot=(18, -10, 100), appearance="splat_proxy",
          dynamic=True)
    if os.path.exists(args.mug):
        place(stage, "/World/mug", args.mug, pos=(0.16, -0.05, 0.12), rot=(70, 0, 150), appearance="splat_proxy",
              dynamic=True)
        place(stage, "/World/mug2", args.mug, pos=(-0.15, -0.09, 0.0), rot=(0, 0, 230), appearance="splat_proxy",
              dynamic=True)
    camera(stage, "/World/cam", (0.36, -0.78, 0.36), (0.0, 0.0, 0.07), focal=35)
    settle(10)
    shooter = Camera("/World/cam", (960, 540))
    import omni.timeline

    tl = omni.timeline.get_timeline_interface()
    tl.play()
    frames = []
    for i in range(120):
        app.update()
        img, _ = shooter.shoot(None, subframes=2, steps=1)
        frames.append(img)
    tl.stop()
    shooter.close()
    frame_dir = os.path.join(OUT, f"drop_frames{args.tag}")
    os.makedirs(frame_dir, exist_ok=True)
    for i, img in enumerate(frames):  # scripts/compare_splat.py assembles drop.mp4 (no ffmpeg in Isaac's python)
        Image.fromarray(img).save(os.path.join(frame_dir, f"{i:04d}.png"))
    results["drop"] = {"frames": len(frames), "dir": frame_dir}

with open(os.path.join(OUT, f"isaac_splat{args.tag}.json"), "w") as f:
    json.dump(results, f, indent=2)
app.close()
