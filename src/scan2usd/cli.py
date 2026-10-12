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
    p.add_argument("--roughness", type=float, default=0.5)
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


def cmd_materials(_args) -> int:
    from .physics import MATERIALS

    print(f"{'material':<10} {'density':>8} {'mu_s':>5} {'mu_d':>5} {'rest':>5}")
    for k, v in MATERIALS.items():
        print(f"{k:<10} {v['density']:>8.0f} {v['static_friction']:>5.2f} {v['dynamic_friction']:>5.2f} {v['restitution']:>5.2f}")
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
