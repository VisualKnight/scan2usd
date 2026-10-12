"""Uncertainty calibration on the six benchmark objects: do the stored intervals contain the truth?

    python scripts/study_uncertainty.py [--objects ...] [--reuse]

For each synthetic capture (data/synthetic/<object>, scripts/run_benchmark.py) the built asset is
analyzed (scan2usd.uncertainty.analyze_asset: observation confidence, carved hypothesis, intervals),
then compared with the ground-truth mesh, which stands in for the real object:
  * coverage: true scale / volume / mass / center of mass inside the stored intervals?
  * hypotheses: F-score and volume error of the filled vs carved shapes;
  * confidence vs error: reconstruction error at observed / inferred / sealed vertices.
Writes docs/figures/uncertainty.{json,png}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_synthetic import load_ground_truth  # noqa: E402
from run_benchmark import OBJECTS  # noqa: E402

from scan2usd.evaluate import align, compare  # noqa: E402
from scan2usd.uncertainty import LABEL_NAMES, analyze_asset  # noqa: E402
from scan2usd.usd_io import read_usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TITLES = {"mug": "Speckled mug", "ace_coffee_mug_kristen_16_oz_cup": "Printed mug",
          "cole_hardware_bowl_scirocco_yellowblue": "Bowl", "great_dinos_triceratops_toy": "Android figure",
          "schleich_lion_action_figure": "Lion", "womens_bluefish_2eye_boat_shoe_in_tan": "Boat shoe"}


def asset_path(folder: str) -> Path:
    d = ROOT / "data" / "synthetic" / folder / "work" / "asset"
    return next(p for p in d.glob("*.usda") if not p.stem.endswith("_carved"))


def gt_properties(gt: trimesh.Trimesh, density: float) -> dict:
    return {"volume": float(gt.volume), "mass": float(gt.volume * density), "com": np.asarray(gt.center_mass)}


def run(folder: str, material: str, reuse: bool) -> dict:
    cap = ROOT / "data" / "synthetic" / folder
    usd = asset_path(folder)
    t0 = time.time()
    if reuse and (usd.parent / "uncertainty.json").exists():
        r = json.loads((usd.parent / "uncertainty.json").read_text())
    else:
        res = analyze_asset(usd, cap / "work" / "colmap", material=material, scale_source="ruler", scale_precision=0.001)
        r = {"uncertainty": res.uncertainty.to_dict(), "observed_fraction": res.field.observed_fraction,
             "label_area": res.field.label_area, "sealed_fraction": res.sealed_fraction,
             "carved": res.carved_path.name if res.carved_path else None, "near_m": res.field.near, "far_m": res.field.far}
        (usd.parent / "uncertainty.json").write_text(json.dumps(r, indent=2))
    u = r["uncertainty"]
    asset = read_usd(usd)
    density = asset.mass / asset.render_mesh.volume
    gt = load_ground_truth(cap)
    truth = gt_properties(gt, density)
    out = {"object": folder, "title": TITLES.get(folder, folder), "material": material, "seconds": round(time.time() - t0),
           "observed_fraction": r["observed_fraction"], "label_area": r["label_area"],
           "sealed_volume_fraction": r["sealed_fraction"], "carved": r["carved"]}

    # Hypotheses vs truth.
    hyps = {"filled": asset}
    if r["carved"]:
        hyps["carved"] = read_usd(usd.parent / r["carved"])
    out["hypotheses"] = {}
    for name, a in hyps.items():
        m = compare(a.render_mesh, gt)
        vol = a.render_mesh.volume if a.render_mesh.is_watertight else float("nan")
        out["hypotheses"][name] = {"fscore_2mm": m["fscore_2mm"], "accuracy_mean_mm": m["accuracy_mm"]["mean"],
                                   "volume_cm3": vol * 1e6, "volume_error_pct": (vol / truth["volume"] - 1) * 100,
                                   "mass_kg": a.mass}

    # Coverage: same density for the truth (the material is known; mass error is geometric).
    true_scale = float(gt.extents[2] / asset.render_mesh.extents[2])
    m_lo, m_hi = u["mass"]
    out["coverage"] = {
        "scale": {"interval": u["scale"], "truth": true_scale, "inside": u["scale"][0] <= true_scale <= u["scale"][1]},
        "mass": {"interval": [m_lo, m_hi], "nominal": asset.mass, "truth": truth["mass"],
                 "inside": m_lo <= truth["mass"] <= m_hi},
    }
    # Center of mass in the asset frame: align the truth to the asset (yaw + ICP) first.
    tf = np.linalg.inv(align(asset.render_mesh, gt))
    com_gt = trimesh.transform_points(truth["com"][None], tf)[0]
    lo, hi = np.asarray(u["com_min"]), np.asarray(u["com_max"])
    out["coverage"]["center_of_mass"] = {"box": [lo.tolist(), hi.tolist()], "truth": com_gt.tolist(),
                                         "nominal": np.asarray(asset.center_of_mass).tolist(),
                                         "inside": bool(np.all(com_gt >= lo - 1e-4) and np.all(com_gt <= hi + 1e-4)),
                                         "nominal_error_mm": float(np.linalg.norm(np.asarray(asset.center_of_mass) - com_gt) * 1000)}

    # Does low confidence flag real errors? Error of each render vertex vs the aligned truth.
    from scan2usd.uncertainty import observation_field
    from scan2usd.uncertainty import workspace_points

    report = json.loads((usd.parent / "build_report.json").read_text())
    pts, _ = workspace_points(asset, cap / "work" / "colmap", float(report["scale_applied"]))
    fld = observation_field(asset.render_mesh, pts)
    gt_in_asset = gt.copy()
    gt_in_asset.apply_transform(tf)
    v = asset.render_mesh.vertices
    sel = np.random.default_rng(0).choice(len(v), min(30000, len(v)), replace=False)
    err = np.abs(trimesh.proximity.closest_point(gt_in_asset, v[sel])[1]) * 1000
    out["error_by_label_mm"] = {LABEL_NAMES[k]: (float(np.median(err[fld.labels[sel] == k])) if np.any(fld.labels[sel] == k) else None)
                                for k in LABEL_NAMES}
    out["error_by_label_p90_mm"] = {LABEL_NAMES[k]: (float(np.percentile(err[fld.labels[sel] == k], 90)) if np.any(fld.labels[sel] == k) else None)
                                    for k in LABEL_NAMES}
    return out


def figure(results: list[dict], out: Path) -> None:
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font

    W, row, left = 1500, 54, 170
    img = Image.new("RGB", (W, 150 + row * len(results)), "white")
    d = ImageDraw.Draw(img)
    d.text((16, 12), "Do the asset's mass intervals contain the truth?", fill=(30, 41, 59), font=_font(22))
    d.text((16, 44), "bar: stored mass interval (filled shape … carved shape, ± density spread) · ● nominal (single-value asset) "
           "· ◆ ground truth · right: observed surface and F-score of each geometry hypothesis", fill=(100, 116, 139), font=_font(15))
    span = 620
    for i, r in enumerate(results):
        y = 90 + i * row
        c = r["coverage"]["mass"]
        lo, hi, nom, tru = c["interval"][0], c["interval"][1], c["nominal"], c["truth"]
        vmax = max(hi, nom, tru) * 1.05

        def x(v):
            return left + int(span * v / vmax)
        d.text((16, y + 10), r["title"], fill=(51, 65, 85), font=_font(16))
        d.line([(left, y + 20), (left + span, y + 20)], fill=(226, 232, 240), width=2)
        ok = c["inside"]
        d.rectangle([x(lo), y + 12, x(hi), y + 28], fill=(187, 247, 208) if ok else (254, 202, 202),
                    outline=(21, 128, 61) if ok else (185, 28, 28))
        d.ellipse([x(nom) - 6, y + 14, x(nom) + 6, y + 26], fill=(100, 116, 139))
        d.polygon([(x(tru), y + 9), (x(tru) + 8, y + 20), (x(tru), y + 31), (x(tru) - 8, y + 20)], fill=(37, 99, 235))
        d.text((left + span + 14, y + 10), f"{lo * 1000:.0f}–{hi * 1000:.0f} g (truth {tru * 1000:.0f}, nominal {nom * 1000:.0f})",
               fill=(51, 65, 85), font=_font(14))
        h = r["hypotheses"]
        txt = f"seen {r['observed_fraction'] * 100:.0f}% · F filled {h['filled']['fscore_2mm']:.2f}"
        if "carved" in h:
            txt += f" / carved {h['carved']['fscore_2mm']:.2f}"
        d.text((left + span + 330, y + 10), txt, fill=(51, 65, 85), font=_font(14))
    n_in = sum(r["coverage"]["mass"]["inside"] for r in results)
    n_nom = sum(abs(r["coverage"]["mass"]["nominal"] / r["coverage"]["mass"]["truth"] - 1) < 0.1 for r in results)
    d.text((16, 100 + row * len(results)), f"truth inside the interval: {n_in}/{len(results)} objects · "
           f"single-value nominal within 10% of the truth: {n_nom}/{len(results)}", fill=(30, 41, 59), font=_font(16))
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=",".join(OBJECTS))
    ap.add_argument("--reuse", action="store_true", help="reuse uncertainty.json next to each asset")
    args = ap.parse_args()
    results = []
    for key in args.objects.split(","):
        folder, material, _ = OBJECTS[key]
        r = run(folder, material, args.reuse)
        results.append(r)
        cm = r["coverage"]["mass"]
        print(f"{r['title']:<15} seen {r['observed_fraction'] * 100:4.0f}%  mass {cm['interval'][0] * 1000:5.0f}-{cm['interval'][1] * 1000:5.0f} g "
              f"truth {cm['truth'] * 1000:5.0f} nominal {cm['nominal'] * 1000:5.0f} inside={cm['inside']}  "
              f"scale inside={r['coverage']['scale']['inside']} com inside={r['coverage']['center_of_mass']['inside']}  "
              f"F {', '.join(k + ' ' + format(v['fscore_2mm'], '.3f') for k, v in r['hypotheses'].items())}  "
              f"err by label {r['error_by_label_mm']}  ({r['seconds']} s)", flush=True)
    out = ROOT / "docs" / "figures"
    (out / "uncertainty.json").write_text(json.dumps(results, indent=2, default=float))
    figure(results, out / "uncertainty.png")


if __name__ == "__main__":
    main()
