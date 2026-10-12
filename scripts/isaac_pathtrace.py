"""Path-traced renders of Scan2USD assets in Isaac Sim 6 (RTX).

    set OMNI_KIT_ACCEPT_EULA=YES
    D:\\isaac\\v6\\Scripts\\python.exe scripts\\isaac_pathtrace.py [--shots beauty,normals,lookdev] [--spp 64 --subframes 8]

Shots (written to output/pathtrace/):
  beauty   scanned assets on a table, HDRI dome light + key light, camera depth of field;
           rendered with RTX real-time path tracing and with offline path tracing.
  normals  the Buddha under grazing light: 50k-triangle asset without vs. with the baked normal map.
  lookdev  the Buddha from a real photo's solved camera (scripts/lookdev_camera.py), with an object
           mask, for photo-vs-render comparison (scripts/compare_lookdev.py).

Assets: output/pbr/* (scan2usd build --bake-texture 2048 with look presets) and the GSO assets.
HDRIs: NVIDIA's Isaac asset server (/NVIDIA/Assets/Skies).
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "pathtrace")

parser = argparse.ArgumentParser()
parser.add_argument("--shots", default="beauty,normals,lookdev")
parser.add_argument("--spp", type=int, default=64, help="path-traced samples per pixel per subframe")
parser.add_argument("--subframes", type=int, default=8, help="accumulated subframes (total spp = spp x subframes)")
parser.add_argument("--hdri", default="/NVIDIA/Assets/Skies/Indoor/wooden_lounge_4k.hdr")
parser.add_argument("--fstop", type=float, default=5.6, help="lens f-number (physical: see camera())")
parser.add_argument("--tag", default="")
args = parser.parse_args()
shots = args.shots.split(",")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "PathTracing", "samples_per_pixel_per_frame": args.spp,
                     "denoiser": True, "max_bounces": 6})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

settings = carb.settings.get_settings()
settings.set("/rtx/resourcemanager/texturestreaming/async", False)
settings.set("/rtx-transient/resourcemanager/texturestreaming/async", False)
ASSETS_ROOT = get_assets_root_path()
os.makedirs(OUT, exist_ok=True)

PBR = {name: os.path.join(ROOT, "output", "pbr", name, f"{name}.usda") for name in ("speckled_mug", "buddha_head")}
GSO = lambda n: os.path.join(ROOT, "output", "gso_assets", n, f"{n}.usda")  # noqa: E731


def new_stage():
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def preview_material(stage, path, color, roughness, metallic=0.0, clearcoat=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if clearcoat:
        sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(clearcoat)
        sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.1)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def mdl_material(stage, url, name, path):
    """An MDL material from the asset server (e.g. NVIDIA's Base/Wood/Oak), or None if unavailable."""
    import omni.client
    import omni.kit.commands

    if omni.client.stat(url)[0] != omni.client.Result.OK:
        return None
    omni.kit.commands.execute("CreateMdlMaterialPrimCommand", mtl_url=url, mtl_name=name, mtl_path=path)
    return UsdShade.Material(stage.GetPrimAtPath(path))


def place(stage, path, usd, x, y, yaw=0.0, prim_path=None):
    xf = UsdGeom.Xform.Define(stage, path)
    if prim_path:
        xf.GetPrim().GetReferences().AddReference(usd, prim_path)
    else:
        xf.GetPrim().GetReferences().AddReference(usd)
    api = UsdGeom.XformCommonAPI(xf.GetPrim())
    api.SetTranslate(Gf.Vec3d(x, y, 0.0))
    api.SetRotate(Gf.Vec3f(0, 0, yaw))
    # Static for rendering: Replicator steps play the timeline, and rigid bodies would fall while
    # path-traced samples accumulate (smeared, misplaced objects).
    for p in Usd.PrimRange(xf.GetPrim()):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
    return xf


MM = 0.001 / (0.1 * 1.0)  # USD lens/filmback units are tenths of a scene unit; this stage is in meters


def camera(stage, path, eye, target, focal=50.0, aperture=36.0, aspect=16 / 9, fstop=0.0, focus=None):
    """USD camera with a physical lens: focal length and aperture given in millimeters.

    UsdGeomCamera measures focal length and filmback in *tenths of a scene unit* (millimeters only on a
    centimeter stage). On this meters stage, writing 50 for a 50 mm lens makes a 5 m lens: the field of
    view is unchanged (it depends only on ratios), but the aperture diameter, and so the depth-of-field
    blur, is 100x too big. focusDistance is in scene units (meters).
    """
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal * MM)
    cam.CreateHorizontalApertureAttr(aperture * MM)
    cam.CreateVerticalApertureAttr(aperture / aspect * MM)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
    if fstop:
        cam.CreateFStopAttr(fstop)
        cam.CreateFocusDistanceAttr(focus or (Gf.Vec3d(*eye) - Gf.Vec3d(*target)).GetLength())
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    cam.AddTransformOp().Set(m.GetInverse())
    return cam


def render(cam_path, size, name, subframes=None, mode="PathTracing", semantic=False):
    """Render one frame; returns the RGB array (and the mask if semantic)."""
    settings.set("/rtx/rendermode", mode)
    settings.set("/rtx/pathtracing/spp", args.spp)
    settings.set("/rtx/pathtracing/totalSpp", args.spp * (subframes or args.subframes))
    rp = rep.create.render_product(cam_path, size)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])
    seg = None
    if semantic:
        seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
        seg.attach([rp])
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=subframes or args.subframes)
    img = np.asarray(rgb.get_data())[..., :3]
    Image.fromarray(img).save(os.path.join(OUT, f"{name}.png"))
    mask = None
    if seg is not None:
        data = seg.get_data()
        ids = np.asarray(data["data"])
        labels = data["info"]["idToLabels"]
        obj_ids = [int(k) for k, v in labels.items() if "object" in str(v)]
        mask = np.isin(ids, obj_ids)
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(OUT, f"{name}_mask.png"))
        seg.detach()
    rgb.detach()
    rp.destroy()
    print(f"[scan2usd] wrote {name}.png ({mode})", flush=True)
    return img, mask


def dome(stage, hdri, intensity=1.0, rotate_z=0.0):
    d = UsdLux.DomeLight.Define(stage, "/World/dome")
    d.CreateIntensityAttr(intensity)
    d.CreateTextureFileAttr().Set(ASSETS_ROOT + hdri)
    d.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    UsdGeom.XformCommonAPI(d.GetPrim()).SetRotate(Gf.Vec3f(0, 0, rotate_z))
    return d


def rect_light(stage, path, pos, target, intensity, size=(0.5, 0.5), color=(1, 1, 1)):
    light = UsdLux.RectLight.Define(stage, path)
    light.CreateIntensityAttr(intensity)
    light.CreateWidthAttr(size[0])
    light.CreateHeightAttr(size[1])
    light.CreateColorAttr(Gf.Vec3f(*color))
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    light.AddTransformOp().Set(m.GetInverse())  # rect lights emit along -Z, like cameras look
    return light


def table(stage, size=(1.6, 1.0), color=(0.42, 0.3, 0.2)):
    t = UsdGeom.Cube.Define(stage, "/World/table")
    t.CreateSizeAttr(1.0)
    t.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.02))
    t.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], 0.04))
    mat = mdl_material(stage, ASSETS_ROOT + "/NVIDIA/Materials/Base/Wood/Oak.mdl", "Oak", "/World/Looks/Oak")
    if mat is None:
        mat = preview_material(stage, "/World/Looks/table", color, 0.55)
    UsdShade.MaterialBindingAPI.Apply(t.GetPrim()).Bind(mat)
    return t


def settle(n=30):
    for _ in range(n):
        app.update()


results = {}
if "beauty" in shots:
    stage = new_stage()
    table(stage)
    dome(stage, args.hdri, intensity=900.0, rotate_z=40)
    rect_light(stage, "/World/key", (-0.5, -0.6, 0.7), (0, 0, 0.05), 25000, size=(0.6, 0.4), color=(1.0, 0.95, 0.88))
    place(stage, "/World/mug", PBR["speckled_mug"], 0.0, 0.0, yaw=-30)
    place(stage, "/World/buddha", PBR["buddha_head"], 0.21, 0.09, yaw=100)
    place(stage, "/World/printed", GSO("ace_coffee_mug_kristen_16_oz_cup"), -0.19, 0.1, yaw=60)
    place(stage, "/World/lion", GSO("schleich_lion_action_figure"), -0.13, -0.1, yaw=30)
    place(stage, "/World/android", GSO("great_dinos_triceratops_toy"), 0.12, -0.12, yaw=-40)
    place(stage, "/World/bowl", GSO("cole_hardware_bowl_scirocco_yellowblue"), 0.42, 0.3, yaw=0)
    eye, target = (0.06, -0.66, 0.24), (0.06, 0.0, 0.06)
    camera(stage, "/World/cam", eye, target, focal=50, fstop=args.fstop,
           focus=float(np.linalg.norm(np.subtract(eye, (0.0, 0.0, 0.05)))))
    settle()
    render("/World/cam", (1920, 1080), "beauty_rt" + args.tag, subframes=1, mode="RealTimePathTracing")
    render("/World/cam", (1920, 1080), "beauty_pt" + args.tag)
    # Square hero shot for the README strip: the real-photo Buddha in front, the other scans behind.
    hero_eye = (0.45, -0.32, 0.2)
    camera(stage, "/World/hero_cam", hero_eye, (0.13, 0.03, 0.08), focal=50, aspect=1.0, fstop=args.fstop,
           focus=float(np.linalg.norm(np.subtract(hero_eye, (0.21, 0.09, 0.1)))))
    render("/World/hero_cam", (1024, 1024), "hero_pt" + args.tag)
    results["beauty"] = {"hdri": args.hdri, "spp": args.spp * args.subframes, "fstop": args.fstop}

if "normals" in shots:
    for name, usd in (("normals_off", os.path.join(ROOT, "output", "buddha", "buddha_head.usda")),
                      ("normals_on", PBR["buddha_head"])):
        stage = new_stage()
        preview_material(stage, "/World/Looks/bg", (0.5, 0.5, 0.5), 0.8)
        t = UsdGeom.Cube.Define(stage, "/World/floor")
        t.CreateSizeAttr(1.0)
        t.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.01))
        t.AddScaleOp().Set(Gf.Vec3f(2, 2, 0.02))
        UsdShade.MaterialBindingAPI.Apply(t.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Looks/bg")))
        d = UsdLux.DomeLight.Define(stage, "/World/dome")
        d.CreateIntensityAttr(80)
        # Grazing light from the side: relief detail shows up as light and shadow.
        rect_light(stage, "/World/key", (0.45, -0.1, 0.16), (0, 0, 0.11), 9000, size=(0.15, 0.3))
        place(stage, "/World/buddha", usd, 0, 0, yaw=-100)
        camera(stage, "/World/cam", (0.05, -0.42, 0.16), (0, 0, 0.105), focal=70, aspect=1.0)
        settle()
        render("/World/cam", (1024, 1024), name)
    results["normals"] = {"off": "output/buddha/buddha_head.usda (vertex colors)", "on": PBR["buddha_head"]}

if "lookdev" in shots:
    with open(os.path.join(ROOT, "output", "lookdev", "camera.json")) as f:
        cam_cfg = json.load(f)
    stage = new_stage()
    d = dome(stage, "/NVIDIA/Assets/Skies/Indoor/hospital_room_4k.hdr", intensity=900.0)
    rect_light(stage, "/World/ceiling", (0.0, 0.0, 2.2), (0.0, 0.0, 0.0), 20000, size=(1.2, 0.6))
    xf = place(stage, "/World/object", cam_cfg["asset"], 0, 0)
    with rep.get.prims(path_pattern="/World/object"):
        rep.modify.semantics([("class", "object")])
    w, h = cam_cfg["width"], cam_cfg["height"]
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    cam.CreateFocalLengthAttr(cam_cfg["focal_length_mm"] * MM)
    cam.CreateHorizontalApertureAttr(cam_cfg["horizontal_aperture_mm"] * MM)
    cam.CreateVerticalApertureAttr(cam_cfg["vertical_aperture_mm"] * MM)
    cam.CreateHorizontalApertureOffsetAttr(cam_cfg["horizontal_aperture_offset_mm"] * MM)
    cam.CreateVerticalApertureOffsetAttr(cam_cfg["vertical_aperture_offset_mm"] * MM)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
    cam.AddTransformOp().Set(Gf.Matrix4d(cam_cfg["camera_to_world_usd"]))
    settle()
    scale = 1280 / w
    render("/World/cam", (1280, round(h * scale)), "lookdev", semantic=True)
    results["lookdev"] = {"photo": cam_cfg["photo"]}

with open(os.path.join(OUT, "pathtrace.json"), "w") as f:
    json.dump(results, f, indent=2)
app.close()
