"""SimReady conformance of the asset library: every asset as built, and as a SimReady package.

    pip install simready-validate simready-foundation-tier-core   (Python 3.12, its own venv)
    python scripts/study_simready.py --out D:/local_projects/scan2usd_simready --validator <venv>/Scripts/simready-validate.exe

Profile: NVIDIA SimReady Foundation ``Robotics-Prop`` 4.0.0, validated with ``simready-validate``.
"As built" is the asset file and its textures copied as they are to the (space-free) output root, so the
check judges the asset and not this machine's folder names (NP.003 checks every folder of the absolute
path). "Package" is ``scan2usd.simready.package``. Writes docs/figures/simready.json and simready.png.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scan2usd.simready import (PROFILE, PROFILE_VERSION, SimReadyInfo, bad_path_parts, package,  # noqa: E402
                               validate)

FIG = ROOT / "docs" / "figures"
GSO, AV = "CC-BY-4.0", "CC-BY-4.0"
# (asset, short label, category, Wikidata Q-code (checked on wikidata.org), material label, license, note)
ASSETS = [
    ("output/gso_assets/ace_coffee_mug_kristen_16_oz_cup/ace_coffee_mug_kristen_16_oz_cup.usda", "coffee mug",
     "mug", "Q386215", "ceramic", GSO, "Google Scanned Objects, benchmark capture"),
    ("output/gso_assets/cole_hardware_bowl_scirocco_yellowblue/cole_hardware_bowl_scirocco_yellowblue.usda", "bowl",
     "bowl", "Q153988", "ceramic", GSO, "Google Scanned Objects, benchmark capture"),
    ("output/gso_assets/great_dinos_triceratops_toy/great_dinos_triceratops_toy.usda", "Android toy",
     "action figure", "Q343566", "plastic", GSO, "Google Scanned Objects, benchmark capture"),
    ("output/gso_assets/schleich_lion_action_figure/schleich_lion_action_figure.usda", "lion figure",
     "figurine", "Q1066288", "plastic", GSO, "Google Scanned Objects, benchmark capture"),
    ("output/gso_assets/womens_bluefish_2eye_boat_shoe_in_tan/womens_bluefish_2eye_boat_shoe_in_tan.usda",
     "boat shoe", "boat shoe", "Q893193", "rubber", GSO, "Google Scanned Objects, benchmark capture"),
    ("output/demo/asset/ceramic_mug.usda", "ceramic mug", "mug", "Q386215", "ceramic", "Apache-2.0",
     "synthetic demo capture"),
    ("data/work/mug_scan/asset/speckled_mug.usda", "speckled mug", "mug", "Q386215", "ceramic", "Apache-2.0",
     "procedural benchmark mug"),
    ("output/articulated/keepsake_box.usda", "keepsake box", "box", "Q639460", "wood", "Apache-2.0",
     "articulated: base + lid"),
    ("output/buddha/buddha_head.usda", "Buddha head", "sculpture", "Q860861", "concrete", AV,
     "AliceVision Buddha photos"),
    ("output/holder_fit/assets/cod_holder/cod_holder.usda", "holder (scan)", "figurine", "Q1066288", "plastic",
     "proprietary", "own phone scan (a commercial product)"),
    ("output/holder_fit/assets/controller/controller.usda", "controller (scan)", "gamepad", "Q273140", "plastic",
     "proprietary", "own phone scan (a commercial product)"),
]
NAMES = {  # the profile's own feature titles (robotics_prop.toml), shortened
    "FET_000_STANDARD": "Core", "FET_001_STANDARD": "Minimal placeable visual", "FET_003_STANDARD": "Rigid-body physics",
    "FET_006_STANDARD": "Materials: UsdPreviewSurface", "FET_011_STANDARD": "Semantic labels",
    "FET_011_RTX": "Semantic labels: materials", "FET_046_STANDARD": "Wikidata Q-code",
    "FET_031_STANDARD": "Self-contained package", "FET_033_STANDARD": "Metadata + thumbnail",
    "FET_004_STANDARD": "Multi-body physics", "FET_005_STANDARD": "Grasp vectors",
    "FET_006_OPENPBR": "Materials: OpenPBR", "FET_006_MDL": "Materials: MDL", "FET_010_STANDARD": "Display color",
    "FET_007_STANDARD": "Non-visual (sensor) materials", "FET_100_ISAAC": "Isaac Sim composition",
    "FET_000_PHYSX": "PhysX runtime: core", "FET_003_PHYSX": "PhysX runtime: rigid body",
    "FET_004_PHYSX": "PhysX runtime: multi-body + SDF", "FET_000_NEWTON": "Newton runtime: core",
    "FET_003_NEWTON": "Newton runtime: rigid body", "FET_004_NEWTON": "Newton runtime: multi-body",
    "FET_000_MUJOCO": "MuJoCo runtime: core", "FET_003_MUJOCO": "MuJoCo runtime: rigid body",
    "FET_004_MUJOCO": "MuJoCo runtime: multi-body",
}


def copy_as_built(src: Path, dest_root: Path) -> Path:
    """The asset file and everything it references, at the same relative paths, under dest_root/<stem>/."""
    from pxr import UsdUtils

    dest = dest_root / src.stem
    if dest.exists():
        shutil.rmtree(dest)
    layers, assets, _ = UsdUtils.ComputeAllDependencies(str(src))
    for f in [Path(lyr.realPath) for lyr in layers] + [Path(a) for a in assets]:
        rel = f.resolve().relative_to(src.parent.resolve())
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / rel)
    return dest / src.name


def slim(report: dict) -> dict:
    return {"passed": report["passed"],
            "features": {k: {"met": f["passed"], "blocking": f["blocking"], "optional": f["optional"],
                             "unmet": f["failing"]} for k, f in report["features"].items()}}


def figure(rows: list[dict], order: list[str], optional: dict, out: Path) -> None:
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font

    cw, ch, lw, top = 30, 26, 300, 150
    n = len(rows)
    pw = lw + cw * n
    img = Image.new("RGB", (pw * 2 + 40 + 90, top + ch * (len(order) + 2) + 70), "white")
    d = ImageDraw.Draw(img)
    green, red, grey, ink, soft = (34, 139, 34), (200, 40, 40), (226, 230, 236), (30, 41, 59), (100, 116, 139)
    for p, (title, key) in enumerate((("As built", "as_built"), ("SimReady package", "package"))):
        x0 = p * (pw + 40)
        passed = sum(r[key]["passed"] for r in rows)
        d.text((x0, 10), f"{title}: {passed}/{n} pass {PROFILE} {PROFILE_VERSION}", fill=ink, font=_font(22))
        for j, r in enumerate(rows):
            lbl = Image.new("RGBA", (130, 22), (255, 255, 255, 0))
            ImageDraw.Draw(lbl).text((0, 2), r["label"], fill=soft, font=_font(14))
            img.paste(lbl.rotate(60, expand=True), (x0 + lw + j * cw - 20, 34), lbl.rotate(60, expand=True))
        y = top
        for i, fid in enumerate(order):
            if i == 0 or optional[fid] != optional[order[i - 1]]:
                d.text((x0, y + 4), "required" if not optional[fid] else "optional", fill=soft, font=_font(13))
                y += ch
            d.text((x0 + 8, y + 4), NAMES.get(fid, fid), fill=ink, font=_font(15))
            for j, r in enumerate(rows):
                f = r[key]["features"].get(fid, {})
                c = green if f.get("met") else (red if f.get("blocking") and not optional[fid] else grey)
                d.rectangle([x0 + lw + j * cw + 3, y + 3, x0 + lw + (j + 1) * cw - 3, y + ch - 3], fill=c)
            y += ch
    d.text((10, img.height - 40), "green: met   red: required and failing   grey: optional, not met  "
           "(simready-validate, SimReady Foundation core tier)", fill=soft, font=_font(15))
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "simready",
                    help="package root; no spaces in its path (SimReady NP.003)")
    ap.add_argument("--validator", help="simready-validate executable (default: $SIMREADY_VALIDATE or PATH)")
    ap.add_argument("--figure-only", action="store_true", help="redraw docs/figures/simready.png from simready.json")
    args = ap.parse_args()
    if args.figure_only:
        data = json.loads((FIG / "simready.json").read_text())
        s = data["summary"]
        optional = {**{f: False for f in s["required_features"]}, **{f: True for f in s["optional_features"]}}
        figure(data["assets"], s["required_features"] + s["optional_features"], optional, FIG / "simready.png")
        return
    if bad_path_parts(args.out):
        raise SystemExit(f"{args.out}: SimReady NP.003 rejects the folder names {bad_path_parts(args.out)}; "
                         "pass --out with a path of letters, digits, '.', '-', '_'")
    rows, t0 = [], time.time()
    for rel, label, category, qcode, material, lic, note in ASSETS:
        src = ROOT / rel
        info = SimReadyInfo(category=category, qcode=qcode, material=material, license=lic)
        built = validate(copy_as_built(src, args.out / "_as_built"), exe=args.validator)
        main_usd = package(src, args.out, info)
        pkg = validate(main_usd, exe=args.validator)
        rows.append({"name": main_usd.parent.parent.name, "label": label, "source": rel, "category": category,
                     "qcode": qcode, "material": material, "license": lic, "note": note, "package": slim(pkg),
                     "as_built": slim(built), "package_file": main_usd.relative_to(args.out).as_posix()})
        blocking = [k for k, f in pkg["features"].items() if f["blocking"] and not f["optional"]]
        print(f"{rows[-1]['name']:40} as built {'PASS' if built['passed'] else 'fail'} -> package "
              f"{'PASS' if pkg['passed'] else 'fail ' + str(blocking)}", flush=True)
    from scan2usd.simready import find_validator, profile_features

    optional = profile_features(find_validator(args.validator))
    order = sorted(optional, key=lambda f: (optional[f], list(NAMES).index(f) if f in NAMES else 99))
    summary = {"profile": PROFILE, "version": PROFILE_VERSION, "validator": pkg["validator"], "tiers": pkg["tiers"],
               "assets": len(rows), "as_built_passed": sum(r["as_built"]["passed"] for r in rows),
               "package_passed": sum(r["package"]["passed"] for r in rows),
               "required_features": [f for f in order if not optional[f]],
               "optional_features": [f for f in order if optional[f]], "minutes": round((time.time() - t0) / 60, 1)}
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "simready.json").write_text(json.dumps({"summary": summary, "assets": rows}, indent=1))
    figure(rows, order, optional, FIG / "simready.png")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
