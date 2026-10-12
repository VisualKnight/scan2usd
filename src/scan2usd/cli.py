"""Command-line interface: `scan2usd build | validate | calibrate | demo | materials`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _corners(text: str) -> list[list[float]]:
    p = Path(text)
    if p.suffix.lower() == ".json" and p.exists():
        pts = json.loads(p.read_text())
    else:
        pts = [[float(v) for v in pair.split(",")] for pair in text.split(";")]
    if len(pts) != 4 or any(len(q) != 2 for q in pts):
        raise argparse.ArgumentTypeError("need 4 corners: 'x,y;x,y;x,y;x,y' (TL;TR;BR;BL) or a JSON file")
    return pts


def _box(text: str) -> list[float]:
    vals = [float(v) for v in text.replace(";", ",").split(",")]
    if len(vals) != 4 or vals[2] <= vals[0] or vals[3] <= vals[1]:
        raise argparse.ArgumentTypeError("box must be 'x0,y0,x1,y1' in pixels with x1>x0, y1>y0")
    return vals


def _reflectance(text: str) -> float | None:
    if text.lower() in ("none", "wb", "auto"):
        return None
    v = float(text)
    if not 0.0 < v < 1.0:
        raise argparse.ArgumentTypeError("reflectance must be between 0 and 1 (e.g. 0.18), or 'none'")
    return v


def _print_validation(rep) -> None:
    c = rep.counts
    print(f"validation: {rep.status.upper()}  ({c['pass']} passed, {c['warn']} warnings, {c['fail']} failed)")
    for chk in rep.checks:
        if chk.status != "pass":
            print(f"  [{chk.status}] {chk.category}: {chk.name}  {chk.detail}")


def _add_build_args(p: argparse.ArgumentParser) -> None:
    from .physics import MATERIALS

    p.add_argument("--material", default="plastic", choices=sorted(MATERIALS), help="sets density and friction")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--density", type=float, help="override density (kg/m^3)")
    g.add_argument("--mass", type=float, help="measured mass in kg (most accurate)")
    p.add_argument("--up-axis", default="z", choices=["z", "y", "-y", "-z"], help="up axis of the input mesh")
    p.add_argument("--units", default="m", choices=["m", "cm", "mm", "in"], help="units of the input mesh")
    p.add_argument("--scale-to", type=float, help="measured real size in meters (overrides --units)")
    p.add_argument("--scale-axis", default="max", choices=["max", "x", "y", "z"], help="dimension --scale-to refers to (after Z-up)")
    p.add_argument("--target-faces", type=int, default=50_000)
    p.add_argument("--collision", default="coacd", choices=["coacd", "hull"])
    p.add_argument("--hull-threshold", type=float, default=0.05, help="CoACD concavity tolerance")
    p.add_argument("--max-hulls", type=int, default=32)
    p.add_argument("--roughness", type=float, help="override the material's look preset (scan2usd materials)")
    p.add_argument("--chart-image", type=Path, help="photo containing a ColorChecker Classic")
    p.add_argument("--chart-corners", type=_corners, help="chart corners TL;TR;BR;BL as 'x,y;...' or JSON file")
    p.add_argument("--gray-card-image", type=Path, help="photo containing a gray card or white paper")
    p.add_argument("--gray-card-box", type=_box, help="neutral region 'x0,y0,x1,y1' in pixels")
    p.add_argument("--gray-reflectance", type=_reflectance, default=0.18,
                   help="reflectance of the neutral target (0.18 gray card), or 'none' for white balance only")
    p.add_argument("--ccm", type=Path, help="previously fitted color_calibration.json")
    p.add_argument("--usda", action="store_true", help="write human-readable .usda instead of binary .usd")
    p.add_argument("--no-gltf", action="store_true")
    p.add_argument("--no-sim", action="store_true", help="skip MuJoCo drop tests")
    p.add_argument("--no-preview", action="store_true", help="skip preview.png")
    p.add_argument("--bake-texture", type=int, default=0, metavar="PX",
                   help="bake the dense scan's vertex colors into a PX x PX UV texture on the decimated mesh")
    p.add_argument("--no-normal-map", action="store_true", help="with --bake-texture: skip the normal map bake")


def _options(args, name: str):
    from .pipeline import BuildOptions

    return BuildOptions(
        name=name, material=args.material, density=args.density, mass=args.mass,
        up_axis=args.up_axis, units=args.units, scale_to=args.scale_to, scale_axis=args.scale_axis,
        target_faces=args.target_faces, collision=args.collision, hull_threshold=args.hull_threshold,
        max_hulls=args.max_hulls, roughness=args.roughness, chart_image=args.chart_image,
        chart_corners=args.chart_corners, gray_card_image=args.gray_card_image,
        gray_card_box=args.gray_card_box, gray_reflectance=args.gray_reflectance, ccm_path=args.ccm, usda=args.usda,
        gltf=not args.no_gltf, run_sim=not args.no_sim, preview=not args.no_preview,
        bake_texture=args.bake_texture, bake_normals=not args.no_normal_map,
    )


def _report_build(res) -> None:
    s = res.summary
    mp = s["mass_properties"]
    print(f"wrote {res.usd_path}")
    if res.gltf_path:
        print(f"wrote {res.gltf_path}")
    print(f"  extents: {' x '.join(f'{v * 100:.1f}' for v in s['extents_m'])} cm, "
          f"{s['render_triangles']:,} triangles, {s['collision_hulls']} collision hulls")
    print(f"  mass: {mp['mass'] * 1000:.1f} g ({mp['volume_source']}), density {mp['density']:.0f} kg/m^3")
    if s["color_calibration"]:
        b, a = s["color_calibration"]["delta_e_before"], s["color_calibration"]["delta_e_after"]
        print(f"  color chart dE2000 mean: {b['mean']:.2f} -> {a['mean']:.2f} (max {a['max']:.2f})")
    for note in s["clean"]["notes"]:
        print(f"  note: {note}")
    print(f"  time: {s['timings_s']['total']:.1f} s")
    _print_validation(res.validation)
    print(f"report: {res.report_path}")


def cmd_build(args) -> int:
    from .pipeline import build_asset

    name = args.name or Path(args.input).stem
    res = build_asset(args.input, args.output, _options(args, name))
    _report_build(res)
    return 0 if res.validation.status != "fail" else 2


def cmd_validate(args) -> int:
    from .validate import validate_usd

    rep = validate_usd(args.usd, run_sim=not args.no_sim)
    _print_validation(rep)
    if args.json:
        rep.save_json(args.json)
    if args.markdown:
        Path(args.markdown).write_text(rep.to_markdown(), encoding="utf-8")
    return 0 if rep.status != "fail" else 2


def cmd_calibrate(args) -> int:
    from PIL import Image

    from .color import fit_ccm, fit_neutral, load_reference_csv, sample_chart, sample_region

    img = np.asarray(Image.open(args.image).convert("RGB"))
    if args.chart_corners:
        ref = load_reference_csv(args.reference) if args.reference else None
        cal = fit_ccm(sample_chart(img, np.asarray(args.chart_corners)), ref)
    elif args.neutral_box:
        cal = fit_neutral(sample_region(img, args.neutral_box), args.reflectance)
    else:
        print("error: give --chart-corners (ColorChecker) or --neutral-box (gray card / white paper)")
        return 1
    cal.save(args.output)
    print(f"{cal.method}: dE2000 mean {cal.delta_e_before['mean']:.2f} -> {cal.delta_e_after['mean']:.2f} "
          f"(max {cal.delta_e_after['max']:.2f}, {cal.patches_used} patch(es))")
    print(f"wrote {args.output}")
    return 0


def cmd_demo(args) -> int:
    from .demo import run_demo

    res, glaze = run_demo(args.output, target_faces=args.target_faces, run_sim=not args.no_sim)
    print(f"synthetic capture written to {Path(args.output) / 'inputs'}")
    _report_build(res)
    print(f"  mug glaze vs ground truth dE2000: {glaze['captured']:.2f} (as captured) -> "
          f"{glaze['calibrated']:.2f} (calibrated)")
    print(f"preview: {Path(args.output) / 'preview.png'}")
    return 0 if res.validation.status != "fail" else 2


def cmd_preview(args) -> int:
    from .preview import preview_file

    out = args.output or Path(args.input).with_suffix(".preview.png")
    preview_file(args.input, size=args.size).save(out)
    print(f"wrote {out}")
    return 0


def cmd_scan(args) -> int:
    """Photos -> COLMAP -> Scan2USD asset in one go."""
    import trimesh

    from .pipeline import build_asset
    from .reconstruct import IMAGE_EXTS, reconstruct

    use_chart = args.auto_chart or args.chart_pitch_mm is not None
    if args.scale_to is None and args.chart_pitch_mm is None:
        print("note: no --scale-to or --chart-pitch-mm; COLMAP's scale is arbitrary, so the asset size will be wrong")
    out = Path(args.output)
    mesh_path, rep = reconstruct(args.photos, out / "colmap", quality=args.quality)
    print(f"reconstructed: {rep.registered_images}/{rep.images} photos registered, "
          f"{rep.object_points:,} object points, {'watertight' if rep.watertight else 'open'} mesh")
    opts = _options(args, args.name or Path(args.photos).resolve().name)
    opts.up_axis, opts.units = "z", "m"
    if args.scale_to is not None and args.scale_axis == "max":
        opts.scale_axis = "z"  # the table plane is known, so "height" is unambiguous

    if use_chart:
        from .chart import best_chart_image, chart_metric_scale

        photos = sorted(p for p in Path(args.photos).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if args.chart_image is None and args.gray_card_image is None and args.ccm is None:
            best = best_chart_image(photos)
            if best is None:
                print("note: no ColorChecker found in the photos; skipping color calibration")
            else:
                opts.chart_image, opts.chart_corners = best.image, best.lattice_corners().tolist()
                print(f"color chart: detected in {best.image.name} ({best.pitch_px:.0f} px per patch)")
        if args.chart_pitch_mm is not None and args.scale_to is None:
            s = chart_metric_scale(args.photos, out / "colmap" / "sparse" / "0", args.chart_pitch_mm / 1000.0)
            # Hand the build a height, exactly like a ruler measurement (pre-scaling the raw mesh
            # would shrink its tiniest triangles below the vertex-merge tolerance).
            height_units = float(trimesh.load(str(mesh_path), force="mesh").extents[2])
            opts.scale_to, opts.scale_axis = height_units * s["meters_per_unit"], "z"
            print(f"metric scale from the chart: {s['inlier_views']}/{s['views']} views agree, "
                  f"patch-grid spread {s['pitch_spread'] * 100:.2f}% -> height {opts.scale_to * 1000:.1f} mm")

    res = build_asset(mesh_path, out / "asset", opts)
    _report_build(res)
    return 0 if res.validation.status != "fail" else 2


def cmd_reconstruct(args) -> int:
    from .reconstruct import reconstruct

    mesh, rep = reconstruct(args.images, args.output, quality=args.quality, poisson_depth=args.depth)
    print(f"registered {rep.registered_images}/{rep.images} images, {rep.dense_points:,} dense points, "
          f"{rep.object_points:,} on the object")
    print(f"wrote {mesh} ({rep.mesh_faces:,} faces, {'watertight' if rep.watertight else 'open'}; "
          f"Z-up, table at z=0, arbitrary scale)")
    for note in rep.notes:
        print(f"  note: {note}")
    print(f"  time: {sum(rep.timings_s.values()):.0f} s {rep.timings_s}")
    print("next: scan2usd build", mesh, "-o <out> --scale-to <measured height in m> --scale-axis z --material <m>")
    return 0


def _floats(n: int):
    def parse(text: str) -> list[float]:
        vals = [float(v) for v in text.replace(";", ",").split(",")]
        if len(vals) != n:
            raise argparse.ArgumentTypeError(f"expected {n} comma-separated numbers")
        return vals
    return parse


def cmd_articulate(args) -> int:
    from .articulation import Hinge, build_parts, write_articulated_usd
    from .mesh_ops import clean_mesh, load_mesh, orient_and_scale
    from .physics import MATERIALS
    from .usd_io import PhysicsMaterial, SurfaceMaterial

    mesh, _ = clean_mesh(load_mesh(args.input))
    mesh, _ = orient_and_scale(mesh, args.up_axis, args.units, args.scale_to, args.scale_axis)
    sp, hg = args.split, args.hinge
    hinge = Hinge(origin=tuple(hg[:3]), axis=tuple(hg[3:]), lower_deg=args.limits[0], upper_deg=args.limits[1])
    parts = build_parts(mesh, sp[:3], sp[3:], material=args.material, names=tuple(args.parts.split(",")),
                        collision=args.collision)
    m = MATERIALS[args.material]
    name = args.name or Path(args.input).stem
    out = Path(args.output) / f"{name}.usda"
    write_articulated_usd(out, name, parts, hinge, SurfaceMaterial(),
                          PhysicsMaterial(m["static_friction"], m["dynamic_friction"], m["restitution"]))
    for p in parts:
        print(f"  {p.name}: {len(p.mesh.faces):,} triangles, {len(p.hulls)} hulls, {p.mass.mass * 1000:.1f} g")
    print(f"wrote {out} (revolute joint {args.limits[0]:g}..{args.limits[1]:g} deg)")
    return 0


def cmd_library(args) -> int:
    from .library import build_library

    files: list[Path] = []
    for p in args.assets:
        files += sorted(p.rglob("*.usda")) if p.is_dir() else [p]
    entries = build_library(files, args.output, spacing=args.spacing)
    for e in entries:
        print(f"  {e.name}: {e.mass_kg * 1000:.0f} g, colliders " +
              ", ".join(f"{k} {v}" for k, v in e.colliders.items()))
    print(f"wrote {args.output} ({len(entries)} assets; variant sets: collision, friction)")
    return 0


def cmd_grasps(args) -> int:
    from .schemas import read_grasps

    grasps = read_grasps(args.usd, args.prim, args.gripper)
    if not grasps:
        print(f"no grasps in {args.usd} (annotate with scripts/annotate_grasps.py)")
        return 1
    print(f"{'#':>3} {'gripper':<12} {'type':<7} {'success':>8} {'trials':>6}  position (m)            by variant")
    for k, g in enumerate(grasps):
        pos = " ".join(f"{v:+.3f}" for v in g.position)
        by = ", ".join(f"{n} {r:.2f}" for n, r in g.by_variant.items())
        print(f"{k:>3} {g.gripper:<12} {g.type:<7} {g.success_rate:>8.2f} {g.trials:>6}  {pos}  {by}")
    return 0


def cmd_materials(_args) -> int:
    from .physics import MATERIALS

    from .usd_io import LOOKS

    print(f"{'material':<10} {'density':>8} {'mu_s':>5} {'mu_d':>5} {'rest':>5}  look (UsdPreviewSurface)")
    for k, v in MATERIALS.items():
        look = ", ".join(f"{n} {x:g}" for n, x in LOOKS.get(k, {}).items())
        print(f"{k:<10} {v['density']:>8.0f} {v['static_friction']:>5.2f} {v['dynamic_friction']:>5.2f} "
              f"{v['restitution']:>5.2f}  {look}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scan2usd", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="convert a scanned mesh into a physics-ready USD asset")
    b.add_argument("input", type=Path, help="mesh file (.obj, .ply, .glb, .gltf, .stl)")
    b.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    b.add_argument("--name", help="asset name (default: input file stem)")
    _add_build_args(b)
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="run quality checks on a USD asset")
    v.add_argument("usd", type=Path)
    v.add_argument("--no-sim", action="store_true")
    v.add_argument("--json", type=Path)
    v.add_argument("--markdown", type=Path)
    v.set_defaults(func=cmd_validate)

    c = sub.add_parser("calibrate", help="fit a color correction from a ColorChecker or gray-card photo")
    c.add_argument("image", type=Path)
    c.add_argument("--chart-corners", type=_corners, help="ColorChecker corners TL;TR;BR;BL")
    c.add_argument("--reference", type=Path, help="CSV of measured reference patches (sRGB 0..1)")
    c.add_argument("--neutral-box", type=_box, help="gray card / white paper region 'x0,y0,x1,y1'")
    c.add_argument("--reflectance", type=_reflectance, default=0.18, help="neutral reflectance, or 'none'")
    c.add_argument("-o", "--output", type=Path, default=Path("color_calibration.json"))
    c.set_defaults(func=cmd_calibrate)

    d = sub.add_parser("demo", help="run the full pipeline on a synthetic mug capture")
    d.add_argument("-o", "--output", type=Path, default=Path("output/demo"))
    d.add_argument("--target-faces", type=int, default=20_000)
    d.add_argument("--no-sim", action="store_true")
    d.set_defaults(func=cmd_demo)

    m = sub.add_parser("materials", help="list built-in material presets")
    m.set_defaults(func=cmd_materials)

    pv = sub.add_parser("preview", help="render a PNG preview of a mesh or USD asset (no 3D app needed)")
    pv.add_argument("input", type=Path)
    pv.add_argument("-o", "--output", type=Path, help="PNG path (default: <input>.preview.png)")
    pv.add_argument("--size", type=int, default=420)
    pv.set_defaults(func=cmd_preview)

    r = sub.add_parser("reconstruct", help="photos -> object mesh with COLMAP (table removed, Z-up)")
    r.add_argument("images", type=Path, help="folder of photos")
    r.add_argument("-o", "--output", type=Path, required=True, help="workspace folder")
    r.add_argument("--quality", default="medium", choices=["low", "medium", "high", "extreme"])
    r.add_argument("--depth", type=int, help="Poisson octree depth (default: from point spacing)")
    r.set_defaults(func=cmd_reconstruct)

    sc = sub.add_parser("scan", help="photos -> physics-ready USD in one step (reconstruct + build)")
    sc.add_argument("photos", type=Path, help="folder of photos")
    sc.add_argument("-o", "--output", type=Path, required=True)
    sc.add_argument("--name", help="asset name (default: photo folder name)")
    sc.add_argument("--quality", default="medium", choices=["low", "medium", "high", "extreme"])
    sc.add_argument("--auto-chart", action="store_true",
                    help="find the ColorChecker in the photos and calibrate color from it")
    sc.add_argument("--chart-pitch-mm", type=float,
                    help="center-to-center patch spacing of your ColorChecker (mm): enables --auto-chart and "
                         "metric scale from the chart (no ruler needed)")
    _add_build_args(sc)
    sc.set_defaults(func=cmd_scan)

    a = sub.add_parser("articulate", help="split a scan into two parts joined by a hinge (articulated USD)")
    a.add_argument("input", type=Path)
    a.add_argument("-o", "--output", type=Path, required=True)
    a.add_argument("--name")
    a.add_argument("--split", type=_floats(6), required=True,
                   help="seam plane 'ox,oy,oz,nx,ny,nz' in the asset frame (m, Z-up, origin at bottom center)")
    a.add_argument("--hinge", type=_floats(6), required=True, help="hinge line 'ox,oy,oz,ax,ay,az'")
    a.add_argument("--limits", type=_floats(2), default=[0.0, 105.0], help="joint limits in degrees")
    a.add_argument("--parts", default="base,lid", help="names for the parts below/above the plane")
    a.add_argument("--material", default="wood")
    a.add_argument("--collision", default="coacd", choices=["coacd", "hull"])
    a.add_argument("--up-axis", default="z", choices=["z", "y", "-y", "-z"])
    a.add_argument("--units", default="m", choices=["m", "cm", "mm", "in"])
    a.add_argument("--scale-to", type=float)
    a.add_argument("--scale-axis", default="max", choices=["max", "x", "y", "z"])
    a.set_defaults(func=cmd_articulate)

    lb = sub.add_parser("library", help="compose assets into one USD library (payloads + collision/friction variants)")
    lb.add_argument("assets", type=Path, nargs="+", help="asset .usd(a) files, or folders to search")
    lb.add_argument("-o", "--output", type=Path, required=True, help="library .usda path")
    lb.add_argument("--spacing", type=float, default=0.1, help="gap between assets in the layout (m)")
    lb.set_defaults(func=cmd_library)

    gr = sub.add_parser("grasps", help="list the grasps stored in an asset (Scan2UsdGraspAPI)")
    gr.add_argument("usd", type=Path)
    gr.add_argument("--prim", help="asset prim inside a larger stage (e.g. /Library/ceramic_mug)")
    gr.add_argument("--gripper", choices=["panda", "shadow_hand", "generic_parallel"])
    gr.set_defaults(func=cmd_grasps)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
