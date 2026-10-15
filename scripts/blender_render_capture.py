"""Render a synthetic phone-photogrammetry capture of a ground-truth mesh in Blender.

    blender -b --factory-startup --python scripts/blender_render_capture.py -- \
        --mesh gt_mesh.ply --out data/synthetic/mug [--rings 15,35,60 --per-ring 24 --top 6]

Scene: the object on a newspaper-like table under soft, warm "overcast window" light,
with a ColorChecker Classic Mini lying next to it. A handheld phone camera (26 mm
equivalent) circles the object in rings. Writes:

    images/IMG_####.jpg      the capture
    chart_photo.jpg          a separate close-up of the chart in the same light
    chart_corners.json       its 6x4 patch-grid corners in pixels (TL, TR, BR, BL)
    cameras.json             ground-truth intrinsics and camera-to-world poses (OpenCV)

The mesh must be Z-up, in meters, resting on z = 0.
"""

import argparse
import json
import math
import os
import random
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

# Same values as scan2usd.color.COLORCHECKER_SRGB8 (Blender can't import scan2usd).
CHART_SRGB8 = [
    (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67), (133, 128, 177), (103, 189, 170),
    (214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46),
    (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31), (187, 86, 149), (8, 133, 161),
    (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52),
]
CHART_CELL = 0.016   # m, patch pitch (Classic Mini is ~2.5 x 4.3 in overall)
CHART_PATCH = 0.013  # m, patch size inside each cell
CHART_YAW = math.radians(25)
WARM_LIGHT = (1.0, 0.86, 0.68)  # tungsten-ish cast the calibration must remove


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rings", default="15,35,60", help="camera elevations in degrees")
    p.add_argument("--per-ring", type=int, default=24)
    p.add_argument("--top", type=int, default=6, help="extra near-top-down views")
    p.add_argument("--distance", type=float, default=0.24, help="camera distance in meters")
    p.add_argument("--width", type=int, default=2016)
    p.add_argument("--height", type=int, default=1512)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--keep-materials", action="store_true", help="use the mesh's own (scanned) materials")
    p.add_argument("--no-chart", action="store_true", help="leave the color chart out of the scene")
    p.add_argument("--keep-position", action="store_true",
                   help="keep the mesh where its file puts it (Z-up, m) instead of centering it on the table")
    p.add_argument("--resolution-scale", type=float, default=1.0, help="render at this fraction of width x height")
    p.add_argument("--poses", help="JSON list of {name, location, target} or {name, camera_to_world} (world, m): "
                                   "render only these views "
                                   "(the scene is identical to the full capture's), write poses_out.json")
    a = p.parse_args(argv)
    a.mesh, a.out = os.path.abspath(a.mesh), os.path.abspath(a.out)
    return a


def srgb_to_linear(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def principled(name, color=(0.8, 0.8, 0.8), roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat, bsdf


def speckled_glaze():
    """Speckled stoneware: oatmeal glaze with iron flecks and a dipped blue band.

    A plain dark glaze barely reconstructs (photogrammetry needs surface texture), so the
    synthetic mug uses a common real-world look that also gives features to match.
    """
    mat, bsdf = principled("glaze", roughness=0.5)
    nt = mat.node_tree
    coord = nt.nodes.new("ShaderNodeTexCoord")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 140.0
    vor2 = nt.nodes.new("ShaderNodeTexVoronoi")
    vor2.inputs["Scale"].default_value = 420.0
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 40.0
    noise.inputs["Detail"].default_value = 8.0
    flecks = nt.nodes.new("ShaderNodeMath")
    flecks.operation = "MINIMUM"
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.03, 0.022, 0.015, 1)
    ramp.color_ramp.elements[1].position = 0.22
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    oatmeal = tuple(srgb_to_linear(v) for v in (214, 198, 170))
    blue = tuple(srgb_to_linear(v) for v in (46, 92, 140))
    # Blue band on the upper part (dipped glaze), with a wavy edge.
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    wave = nt.nodes.new("ShaderNodeMath")
    wave.operation = "MULTIPLY_ADD"
    wave.inputs[1].default_value = 0.012
    band = nt.nodes.new("ShaderNodeMath")
    band.operation = "GREATER_THAN"
    band.inputs[1].default_value = 0.055
    base = nt.nodes.new("ShaderNodeMix")
    base.data_type = "RGBA"
    base.inputs[6].default_value = (*oatmeal, 1)
    base.inputs[7].default_value = (*blue, 1)
    mottle = nt.nodes.new("ShaderNodeMix")
    mottle.data_type = "RGBA"
    mottle.blend_type = "OVERLAY"
    mottle.inputs["Factor"].default_value = 0.45
    speck = nt.nodes.new("ShaderNodeMix")
    speck.data_type = "RGBA"
    speck.blend_type = "MULTIPLY"
    speck.inputs["Factor"].default_value = 1.0
    links = nt.links
    links.new(coord.outputs["Object"], vor.inputs["Vector"])
    links.new(coord.outputs["Object"], vor2.inputs["Vector"])
    links.new(coord.outputs["Object"], noise.inputs["Vector"])
    links.new(coord.outputs["Object"], sep.inputs["Vector"])
    links.new(noise.outputs["Fac"], wave.inputs[0])
    links.new(sep.outputs["Z"], wave.inputs[2])
    links.new(wave.outputs[0], band.inputs[0])
    links.new(band.outputs[0], base.inputs["Factor"])
    links.new(vor.outputs["Distance"], flecks.inputs[0])
    links.new(vor2.outputs["Distance"], flecks.inputs[1])
    links.new(flecks.outputs[0], ramp.inputs["Fac"])
    links.new(base.outputs[2], mottle.inputs[6])
    links.new(noise.outputs["Fac"], mottle.inputs[7])
    links.new(mottle.outputs[2], speck.inputs[6])
    links.new(ramp.outputs["Color"], speck.inputs[7])
    links.new(speck.outputs[2], bsdf.inputs["Base Color"])
    return mat


def newspaper():
    """Off-white paper with rows of 'text' and photo blocks, so the camera solver has features."""
    mat, bsdf = principled("newspaper", roughness=0.9)
    nt = mat.node_tree
    coord = nt.nodes.new("ShaderNodeTexCoord")
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.inputs["Scale"].default_value = 90.0
    brick.inputs["Mortar Size"].default_value = 0.012
    brick.inputs["Color1"].default_value = (0.08, 0.08, 0.08, 1)
    brick.inputs["Color2"].default_value = (0.25, 0.25, 0.24, 1)
    brick.inputs["Mortar"].default_value = (0.72, 0.70, 0.64, 1)
    brick.offset = 0.37
    brick.squash = 3.0
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 9.0
    noise.inputs["Detail"].default_value = 8.0
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    mix.inputs["Factor"].default_value = 0.4
    nt.links.new(coord.outputs["Object"], brick.inputs["Vector"])
    nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(brick.outputs["Color"], mix.inputs[6])
    nt.links.new(noise.outputs["Fac"], mix.inputs[7])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    return mat


def add_chart(obj_dims):
    """ColorChecker Classic Mini: 6x4 patches on a black board, lying beside the object."""
    board_w, board_h = CHART_CELL * 6 + 0.008, CHART_CELL * 4 + 0.008
    bpy.ops.mesh.primitive_plane_add(size=1)
    board = bpy.context.active_object
    board.name = "chart_board"
    board.scale = (board_w, board_h, 1)
    board.location = (0, 0, 0.0004)
    board.data.materials.append(principled("chart_black", (0.03, 0.03, 0.03), 0.8)[0])
    patches = []
    for i, rgb in enumerate(CHART_SRGB8):
        r, c = divmod(i, 6)
        bpy.ops.mesh.primitive_plane_add(size=CHART_PATCH)
        p = bpy.context.active_object
        p.name = f"chart_patch_{i:02d}"
        # "dark skin" (index 0) at top-left when viewed with the board's +Y pointing up.
        p.location = ((c + 0.5 - 3) * CHART_CELL, (2 - (r + 0.5)) * CHART_CELL, 0.0007)
        p.data.materials.append(principled(f"patch_{i:02d}", tuple(srgb_to_linear(v) for v in rgb), 0.85)[0])
        patches.append(p)
    root = bpy.data.objects.new("chart", None)
    bpy.context.scene.collection.objects.link(root)
    for o in [board] + patches:
        o.parent = root
    root.location = (obj_dims.x / 2 + 0.065, -(obj_dims.y / 2 + 0.045), 0)
    root.rotation_euler.z = CHART_YAW
    bpy.context.view_layer.update()
    # Patch-grid corners in world space: TL, TR, BR, BL.
    local = [(-3, 2), (3, 2), (3, -2), (-3, -2)]
    return [root.matrix_world @ Vector((x * CHART_CELL, y * CHART_CELL, 0.0007)) for x, y in local]


def setup_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            if any(d.type == backend for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == backend
                sc.cycles.device = "GPU"
                break
        except TypeError:
            continue
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = args.width, args.height
    sc.render.resolution_percentage = int(round(100 * args.resolution_scale))
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 93
    # No tone mapping: pixel values stay proportional to reflectance x light (needed for calibration).
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = -1.2  # keep the chart's white patch below clipping

    world = bpy.data.worlds.new("overcast")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*WARM_LIGHT, 1)
    bg.inputs["Strength"].default_value = 0.18
    light = bpy.data.lights.new("window", "AREA")
    light.shape = "RECTANGLE"
    light.size, light.size_y = 1.2, 0.8
    light.energy = 38.0
    light.color = WARM_LIGHT
    lo = bpy.data.objects.new("window", light)
    sc.collection.objects.link(lo)
    lo.location = (-0.6, -0.4, 0.9)
    lo.rotation_euler = (Vector((0, 0, 0.05)) - lo.location).to_track_quat("-Z", "Y").to_euler()

    bpy.ops.mesh.primitive_plane_add(size=1.4)
    table = bpy.context.active_object
    table.name = "table"
    table.data.materials.append(newspaper())

    if args.mesh.lower().endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=args.mesh)
    elif args.mesh.lower().endswith((".glb", ".gltf")):  # glTF is Y-up: the importer converts to Blender's Z-up
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.import_scene.gltf(filepath=args.mesh)
        meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        for o in bpy.context.selected_objects:
            if o.type != "MESH":
                o.select_set(False)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    else:
        bpy.ops.wm.obj_import(filepath=args.mesh, forward_axis="Y", up_axis="Z")
    parts = list(bpy.context.selected_objects)
    bpy.context.view_layer.objects.active = parts[0]
    if len(parts) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "object"
    if not args.keep_materials:
        obj.data.materials.clear()
        obj.data.materials.append(speckled_glaze())
    if args.keep_position:
        bpy.ops.object.shade_smooth()
        return sc, obj
    # Center on the table: bounding-box center at x = y = 0, resting on z = 0.
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    obj.location -= Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    try:
        bpy.ops.object.shade_auto_smooth(angle=math.radians(40))
    except (AttributeError, RuntimeError, TypeError):
        bpy.ops.object.shade_smooth()
    return sc, obj


def look_at(cam_obj, target, roll):
    direction = target - cam_obj.location
    quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = (quat @ Matrix.Rotation(roll, 4, "Z").to_quaternion()).to_euler()


def opencv_pose(cam_obj):
    """Camera-to-world 4x4 in OpenCV convention (x right, y down, z forward)."""
    flip = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    m = cam_obj.matrix_world @ flip
    return [list(row) for row in m]


def pixel(sc, cam_obj, p):
    co = world_to_camera_view(sc, cam_obj, p)
    return [co.x * sc.render.resolution_x, (1.0 - co.y) * sc.render.resolution_y]


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    sc, obj = setup_scene(args)
    bpy.context.view_layer.update()
    corners_world = None if args.no_chart else add_chart(obj.dimensions)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    cam = bpy.data.cameras.new("phone")
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = 36.0
    cam.lens = 26.0
    cam.clip_start = 0.01
    cam_obj = bpy.data.objects.new("phone", cam)
    sc.collection.objects.link(cam_obj)
    sc.camera = cam_obj

    fx = cam.lens / cam.sensor_width * args.width
    if args.poses:  # explicit extra views (e.g. capture guidance): no chart photo, no ring views
        with open(args.poses) as f:
            poses = json.load(f)
        out_meta = []
        for pz in poses:
            if "camera_to_world" in pz:  # an exact pose (OpenCV axes, as cameras.json): reproduce a photo
                flip = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
                cam_obj.matrix_world = Matrix(pz["camera_to_world"]) @ flip
            else:
                cam_obj.location = Vector(pz["location"])
                look_at(cam_obj, Vector(pz["target"]), 0.0)
            bpy.context.view_layer.update()
            sc.render.filepath = os.path.join(args.out, "images", pz["name"])
            bpy.ops.render.render(write_still=True)
            out_meta.append({"name": pz["name"], "camera_to_world": opencv_pose(cam_obj)})
            print(f"rendered {pz['name']}", flush=True)
        with open(os.path.join(args.out, "poses_out.json"), "w") as f:
            json.dump(out_meta, f, indent=1)
        return

    dims = obj.dimensions
    target = Vector((0, 0, dims.z * 0.45))
    views = []
    elevations = [float(e) for e in args.rings.split(",")]
    for ring, elev in enumerate(elevations):
        for k in range(args.per_ring):
            az = 360.0 * (k + 0.5 * (ring % 2)) / args.per_ring
            views.append((az, elev))
    for k in range(args.top):
        views.append((360.0 * k / max(args.top, 1), 80.0))

    meta = {
        "intrinsics": {"model": "PINHOLE", "width": args.width, "height": args.height,
                       "fx": fx, "fy": fx, "cx": args.width / 2, "cy": args.height / 2},
        "convention": "camera_to_world, OpenCV (x right, y down, z forward), meters, Z-up world",
        "light_rgb": WARM_LIGHT,
        "images": [],
    }
    for i, (az, elev) in enumerate(views):
        az_j = math.radians(az + rng.uniform(-3, 3))
        el_j = math.radians(elev + rng.uniform(-3, 3))
        dist = args.distance * rng.uniform(0.9, 1.1)
        cam_obj.location = target + dist * Vector((math.cos(el_j) * math.cos(az_j),
                                                   math.cos(el_j) * math.sin(az_j), math.sin(el_j)))
        aim = target + Vector((rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01)))
        look_at(cam_obj, aim, math.radians(rng.uniform(-3, 3)))
        bpy.context.view_layer.update()
        name = f"IMG_{i:04d}.jpg"
        sc.render.filepath = os.path.join(args.out, "images", name)
        bpy.ops.render.render(write_still=True)
        meta["images"].append({"name": name, "camera_to_world": opencv_pose(cam_obj)})
        print(f"rendered {name} ({i + 1}/{len(views)})", flush=True)

    if corners_world is None:
        with open(os.path.join(args.out, "cameras.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"done: {len(views)} images in {args.out} (no chart)")
        return
    # Close-up of the chart, like the separate reference photo you'd take.
    chart_center = sum(corners_world, Vector()) / 4
    cam_obj.location = chart_center + Vector((-0.02, -0.06, 0.16))
    look_at(cam_obj, chart_center, 0.0)
    bpy.context.view_layer.update()
    sc.render.filepath = os.path.join(args.out, "chart_photo.jpg")
    bpy.ops.render.render(write_still=True)
    corners_px = [pixel(sc, cam_obj, p) for p in corners_world]
    with open(os.path.join(args.out, "chart_corners.json"), "w") as f:
        json.dump(corners_px, f)
    with open(os.path.join(args.out, "cameras.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done: {len(views)} images in {args.out}")


main()
