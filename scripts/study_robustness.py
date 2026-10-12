"""Robustness study: how accurate does a scanned asset need to be for a robot pick to work?

    python scripts/study_robustness.py [--workers 16]

The Franka Panda (scripts/demo_panda.py) plans a rim grasp from the *nominal* Scan2USD mug,
taken from the USD asset library (scan2usd library), while the simulated "real" mug differs:

  A. collision geometry (library `collision` variant) x friction coefficient
  B. mass (x0.5 .. x5) x friction: when does the grip stop holding?
  C. size error of the scan (-15% .. +15%) x grasp position error (perception, -8 .. +8 mm)

The mug's colliders get contact priority in MuJoCo, so its friction (not the rubber finger
pads') sets the grip, as with a real slippery or grippy object. Success = lifted > 10 cm,
placed within 3 cm, upright within 10 degrees. Writes docs/figures/robustness.{png,json}.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "output" / "library" / "scan2usd_library.usda"
PRIM = "/Library/ceramic_mug"

FRICTION = [0.15, 0.25, 0.35, 0.5, 0.7, 0.9]
COLLISION = ["decomposed", "convex_hull", "bounding_box"]
MASS = [0.5, 1.0, 2.0, 3.0, 5.0]
SCALE = [-15, -10, -5, -2, 0, 2, 5, 10, 15]
OFFSET = [-8, -4, 0, 4, 8]
NOMINAL_MU = 0.5

_cache: dict = {}


def physical(collision: str, scale_pct: float, mass_scale: float):
    """The simulated 'real' mug: a library collision variant, scaled about its base, re-weighed."""
    from scan2usd.usd_io import read_usd

    if collision not in _cache:
        _cache[collision] = read_usd(LIB, PRIM, {"collision": collision})
    a = copy.deepcopy(_cache[collision])
    s = 1.0 + scale_pct / 100.0
    if s != 1.0:
        a.render_mesh.apply_scale(s)
        for h in a.hulls:
            h.apply_scale(s)
        a.center_of_mass = a.center_of_mass * s
        a.mass *= s ** 3  # same material, bigger object
        a.diagonal_inertia = a.diagonal_inertia * s ** 5
    a.mass *= mass_scale
    a.diagonal_inertia = a.diagonal_inertia * mass_scale
    return a


def trial(p: dict) -> dict:
    import mujoco
    from demo_panda import HOME, MUG_POS, PLACE_SHIFT, Arm, build_scene, execute, plan_pick, rim_grasp

    from scan2usd.mjcf import asset_body

    t0 = time.time()
    nominal = physical("decomposed", 0.0, 1.0)
    real = physical(p["collision"], p["scale_pct"], p["mass_scale"])
    body = asset_body(real, "mug", pos=tuple(MUG_POS), collision="hulls", friction=p["mu"])
    body = replace(body, body_xml=body.body_xml.replace('solref="0.005 1"/>', 'solref="0.005 1" priority="1"/>'))
    model = build_scene(nominal, "hulls", ROOT / "data" / "work" / "robust" / p["id"], mug_body=body)
    arm = Arm(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mug_free = model.joint(model.body("mug").jntadr[0]).qposadr[0]
    data.qpos[mug_free: mug_free + 3] = MUG_POS
    for fj in ("finger_joint1", "finger_joint2"):
        data.qpos[model.joint(fj).qposadr[0]] = 0.022
    data.ctrl[:7] = HOME
    mujoco.mj_forward(model, data)

    grasp, radial = rim_grasp(nominal.render_mesh, MUG_POS)  # the robot only knows the nominal scan
    grasp = grasp + radial * p["offset_mm"] / 1000.0
    plan = plan_pick(model, arm, grasp, radial)
    _, lifted = execute(model, arm, data, plan)
    mug = model.body("mug").id
    final = data.xpos[mug]
    target = MUG_POS + PLACE_SHIFT
    up = data.xmat[mug].reshape(3, 3)[:, 2]
    lift = lifted - real.center_of_mass[2]
    place = float(np.linalg.norm(final[:2] - target[:2]))
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
    return {**p, "lifted_m": round(float(lift), 4), "place_error_m": round(place, 4), "tilt_deg": round(tilt, 1),
            "mass_kg": round(float(real.mass), 4), "colliders": len(real.hulls),
            "success": bool(lift > 0.1 and place < 0.03 and tilt < 10), "seconds": round(time.time() - t0, 1)}


def grid() -> list[dict]:
    out = []
    for c in COLLISION:
        for mu in FRICTION:
            out.append({"study": "A", "collision": c, "mu": mu, "mass_scale": 1.0, "scale_pct": 0, "offset_mm": 0})
    for m in MASS:
        for mu in FRICTION:
            out.append({"study": "B", "collision": "decomposed", "mu": mu, "mass_scale": m, "scale_pct": 0,
                        "offset_mm": 0})
    for s in SCALE:
        for o in OFFSET:
            out.append({"study": "C", "collision": "decomposed", "mu": NOMINAL_MU, "mass_scale": 1.0,
                        "scale_pct": s, "offset_mm": o})
    for k, p in enumerate(out):
        p["id"] = f"t{k:03d}"
    return out


# --- figure --------------------------------------------------------------------------------
def heatmap(title, rows, cols, row_label, col_label, cell, note=""):
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font

    cw, ch, left, top = 74, 46, 150, 92
    w, h = left + cw * len(cols) + 20, top + ch * len(rows) + 64
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((12, 10), title, fill=(30, 41, 59), font=_font(21))
    d.text((12, 40), note, fill=(100, 116, 139), font=_font(15))
    d.text((left, top - 26), col_label, fill=(71, 85, 105), font=_font(15))
    for j, c in enumerate(cols):
        d.text((left + j * cw + 8, top + ch * len(rows) + 6), str(c), fill=(71, 85, 105), font=_font(15))
    for i, r in enumerate(rows):
        d.text((12, top + i * ch + 13), str(r), fill=(71, 85, 105), font=_font(15))
        for j, c in enumerate(cols):
            ok, label = cell(r, c)
            fill = (187, 247, 208) if ok else (254, 202, 202)
            x0, y0 = left + j * cw, top + i * ch
            d.rectangle([x0 + 2, y0 + 2, x0 + cw - 2, y0 + ch - 2], fill=fill)
            d.text((x0 + 10, y0 + 13), label, fill=(21, 128, 61) if ok else (185, 28, 28), font=_font(15))
    d.text((left, top + ch * len(rows) + 30), row_label, fill=(71, 85, 105), font=_font(15))
    return img


def figure(results: list[dict], out: Path) -> None:
    from PIL import Image

    by = {(r["study"], r["collision"], r["mu"], r["mass_scale"], r["scale_pct"], r["offset_mm"]): r for r in results}

    def mark(r):
        return r["success"], "pick" if r["success"] else ("slip" if r["lifted_m"] < 0.1 else "miss")

    a = heatmap("A · collider vs. friction", COLLISION, FRICTION, "rows: library collision variant",
                "friction μ →", lambda r, c: mark(by[("A", r, c, 1.0, 0, 0)]), "nominal mass 372 g, exact size")
    b = heatmap("B · mass vs. friction", [f"×{m:g}" for m in MASS], FRICTION, "rows: mass (372 g × k)", "friction μ →",
                lambda r, c: mark(by[("B", "decomposed", c, float(r[1:]), 0, 0)]), "convex decomposition")
    c = heatmap("C · scan size error vs. grasp position error", [f"{s:+d}%" for s in SCALE],
                [f"{o:+d}" for o in OFFSET], "rows: real mug size vs. scan", "grasp offset along radius (mm) →",
                lambda r, o: mark(by[("C", "decomposed", NOMINAL_MU, 1.0, int(r[:-1]), int(o))]),
                f"μ = {NOMINAL_MU}, robot plans with the nominal scan")
    w = a.width + b.width + c.width
    sheet = Image.new("RGB", (w, max(a.height, b.height, c.height) + 34), "white")
    x = 0
    for im in (a, b, c):
        sheet.paste(im, (x, 0))
        x += im.width
    from PIL import ImageDraw

    from scan2usd.preview import _font

    ImageDraw.Draw(sheet).text(
        (12, sheet.height - 30), "pick = lifted > 10 cm and placed within 3 cm, upright  ·  slip = never lifted "
        "(slid out of the grip or fingers missed the wall)  ·  miss = lifted, then dropped or tilted before placing",
        fill=(71, 85, 105), font=_font(15))
    sheet.save(out, quality=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--figure-only", action="store_true", help="redraw from docs/figures/robustness.json")
    args = ap.parse_args()
    out = ROOT / "docs" / "figures"
    if args.figure_only:
        figure(json.loads((out / "robustness.json").read_text())["trials"], out / "robustness.png")
        return
    if not LIB.exists():
        raise SystemExit(f"{LIB} not found: run `scan2usd library ...` first (see README)")
    params = grid()
    t0 = time.time()
    with Pool(args.workers) as pool:
        results = []
        for r in pool.imap_unordered(trial, params):
            results.append(r)
            print(f"[{len(results)}/{len(params)}] {r['study']} {r['collision']:<12} mu {r['mu']:<4} "
                  f"m x{r['mass_scale']:<3} size {r['scale_pct']:+d}% off {r['offset_mm']:+d}mm -> "
                  f"{'OK ' if r['success'] else 'FAIL'} lift {r['lifted_m'] * 100:5.1f} cm ({r['seconds']} s)", flush=True)
    results.sort(key=lambda r: r["id"])
    summary = {"asset": PRIM, "library": "output/library/scan2usd_library.usda", "trials": results,
               "minutes": round((time.time() - t0) / 60, 1)}
    (out / "robustness.json").write_text(json.dumps(summary, indent=2))
    figure(results, out / "robustness.png")
    print(f"{sum(r['success'] for r in results)}/{len(results)} succeeded in {summary['minutes']} min")


if __name__ == "__main__":
    main()
