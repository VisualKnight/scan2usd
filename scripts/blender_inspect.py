"""Build a Blender scene to inspect a Scan2USD build: raw input | asset | collision hulls.

Runs inside Blender (Scan2USD itself isn't needed there):

    blender -b --factory-startup --python scripts/blender_inspect.py -- \
        --usd output/demo/asset/ceramic_mug.usda \
        --raw output/demo/inputs/mug_scan.ply --raw-units cm --raw-up y \
        --save output/demo/inspect.blend

Collections: "raw_input" (the scan as exported, shown in meters / Z-up but otherwise
untouched), "asset" (the USD render mesh), "collision" (the USD's convex hulls,
offset to the right so both are visible).
"""

import argparse
import colorsys
import math
import os
import sys

import bpy

UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--usd", required=True)
    p.add_argument("--raw")
    p.add_argument("--raw-units", default="m", choices=sorted(UNIT))
    p.add_argument("--raw-up", default="z", choices=["z", "y"])
    p.add_argument("--save", required=True)
    args = p.parse_args(argv)
    for key in ("usd", "raw", "save"):
        if getattr(args, key):
            setattr(args, key, os.path.abspath(getattr(args, key)))
    return args


def collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def relink(objs, col):
    for obj in objs:
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        col.objects.link(obj)


def color_attribute_material(name, attr):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    node = nodes.new("ShaderNodeAttribute")
    node.attribute_name = attr
    mat.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.45
    return mat


def descendants(obj):
    out = []
    for child in obj.children:
        out.append(child)
        out.extend(descendants(child))
    return out


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = "METRIC"

    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=args.usd, import_guide=True, read_mesh_colors=True, import_usd_preview=True)
    imported = set(bpy.data.objects) - before
    hull_root = next(o for o in imported if o.name.startswith("collision"))
    hulls = [hull_root] + descendants(hull_root)
    relink(hulls, collection("collision"))
    relink(imported - set(hulls), collection("asset"))

    render = [o for o in imported - set(hulls) if o.type == "MESH"]
    width = max(o.dimensions.x for o in render)
    hull_root.location.x += 1.4 * width
    for i, obj in enumerate(o for o in hulls if o.type == "MESH"):
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.55, 0.95)
        mat = bpy.data.materials.new(f"hull_{i:03d}")
        mat.diffuse_color = (r, g, b, 1.0)
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (r, g, b, 1.0)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    for obj in render:
        if obj.data.color_attributes and not obj.data.materials:
            obj.data.materials.append(color_attribute_material("asset_color", obj.data.color_attributes[0].name))

    if args.raw:
        before = set(bpy.data.objects)
        bpy.ops.wm.ply_import(filepath=args.raw)
        raws = list(set(bpy.data.objects) - before)
        relink(raws, collection("raw_input"))
        s = UNIT[args.raw_units]
        for obj in raws:
            obj.scale = (s, s, s)
            if args.raw_up == "y":
                obj.rotation_euler.x = math.pi / 2
            if obj.data.color_attributes:
                obj.data.materials.append(color_attribute_material("raw_color", obj.data.color_attributes[0].name))
        bpy.context.view_layer.update()
        # Keep the raw scan's orientation and debris, but park it to the left of the asset.
        hi_x = max(max((obj.matrix_world @ v.co).x for v in obj.data.vertices) for obj in raws)
        ys = [(obj.matrix_world @ v.co).y for obj in raws for v in obj.data.vertices]
        cy = (min(ys) + max(ys)) / 2
        for obj in raws:
            obj.location.x += -0.7 * width - hi_x
            obj.location.y -= cy

    bpy.ops.wm.save_as_mainfile(filepath=args.save)
    print(f"saved {args.save}")


main()
