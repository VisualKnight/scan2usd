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
    p.add_argument("--ccm", type=Path, help="previously fitted color_calibration.json")
    p.add_argument("--usda", action="store_true", help="write human-readable .usda instead of binary .usd")
    p.add_argument("--no-gltf", action="store_true")
    p.add_argument("--no-sim", action="store_true", help="skip MuJoCo drop tests")


def _options(args, name: str):
    from .pipeline import BuildOptions

    return BuildOptions(
        name=name, material=args.material, density=args.density, mass=args.mass,
        up_axis=args.up_axis, units=args.units, scale_to=args.scale_to, scale_axis=args.scale_axis,
        target_faces=args.target_faces, collision=args.collision, hull_threshold=args.hull_threshold,
        max_hulls=args.max_hulls, roughness=args.roughness, chart_image=args.chart_image,
        chart_corners=args.chart_corners, ccm_path=args.ccm, usda=args.usda,
        gltf=not args.no_gltf, run_sim=not args.no_sim,
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

    from .color import fit_ccm, load_reference_csv, sample_chart

    img = np.asarray(Image.open(args.chart_image).convert("RGB"))
    ref = load_reference_csv(args.reference) if args.reference else None
    cal = fit_ccm(sample_chart(img, np.asarray(args.chart_corners)), ref)
    cal.save(args.output)
    print(f"dE2000 mean {cal.delta_e_before['mean']:.2f} -> {cal.delta_e_after['mean']:.2f} "
          f"(max {cal.delta_e_after['max']:.2f}, {cal.patches_used}/24 patches)")
    print(f"wrote {args.output}")
    return 0


def cmd_demo(args) -> int:
    from .demo import run_demo

    res, glaze = run_demo(args.output, target_faces=args.target_faces, run_sim=not args.no_sim)
    print(f"synthetic capture written to {Path(args.output) / 'inputs'}")
    _report_build(res)
    print(f"  mug glaze vs ground truth dE2000: {glaze['captured']:.2f} (as captured) -> "
          f"{glaze['calibrated']:.2f} (calibrated)")
    return 0 if res.validation.status != "fail" else 2


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

    c = sub.add_parser("calibrate", help="fit a color-correction matrix from a ColorChecker photo")
    c.add_argument("chart_image", type=Path)
    c.add_argument("--chart-corners", type=_corners, required=True)
    c.add_argument("--reference", type=Path, help="CSV of measured reference patches (sRGB 0..1)")
    c.add_argument("-o", "--output", type=Path, default=Path("color_calibration.json"))
    c.set_defaults(func=cmd_calibrate)

    d = sub.add_parser("demo", help="run the full pipeline on a synthetic mug capture")
    d.add_argument("-o", "--output", type=Path, default=Path("output/demo"))
    d.add_argument("--target-faces", type=int, default=20_000)
    d.add_argument("--no-sim", action="store_true")
    d.set_defaults(func=cmd_demo)

    m = sub.add_parser("materials", help="list built-in material presets")
    m.set_defaults(func=cmd_materials)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
