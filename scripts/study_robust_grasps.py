"""Does the asset's uncertainty help a robot? Robust grasp choice evaluated on the true object.

    python scripts/study_robust_grasps.py [--samples 12] [--workers 16]

For each reconstructed cup (the benchmark captures, where the ground truth is known):
  1. Candidate Panda rim grasps are sampled around the rim of the *reconstructed* asset.
  2. Each candidate is scored three ways, the way a planner would choose before ever touching the
     real object:
       nominal      success on the single-value asset (what every real-to-sim pipeline ships);
       uniform DR   success averaged over hand-tuned randomization (scale ±5%, mass ±20%, friction ±30%);
       uncertainty  success averaged over samples drawn from the asset's own stored uncertainty,
                    including the "carved" geometry hypothesis for sealed regions.
  3. The top grasp of each strategy is executed on the **ground-truth** object (the stand-in for
     reality), with several execution perturbations.
Writes docs/figures/robust_grasps.{json,png}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
CUPS = {"ace_coffee_mug_kristen_16_oz_cup": "Printed mug", "mug": "Speckled mug"}
YAWS = np.radians(np.arange(0, 360, 30))
EXEC_OFFSETS = (-0.002, 0.0, 0.002)  # radial execution error (m) when testing on the truth
STRATEGIES = ("nominal", "uniform", "uncertainty")

_state: dict = {}


def _load(folder: str):
    if folder in _state:
        return _state[folder]
    from evaluate_synthetic import load_ground_truth

    from scan2usd.evaluate import align
    from scan2usd.physics import convex_decompose, estimate_mass_properties
    from scan2usd.uncertainty import load_hypotheses
    from scan2usd.schemas import read_uncertainty
    from scan2usd.usd_io import AssetData, read_usd

    usd = next(p for p in (ROOT / "data" / "synthetic" / folder / "work" / "asset").glob("*.usda")
               if not p.stem.endswith("_carved"))
    asset = read_usd(usd)
    hyps = load_hypotheses(usd)
    unc = read_uncertainty(usd)
    gt_mesh = load_ground_truth(ROOT / "data" / "synthetic" / folder)
    gt_mesh.apply_transform(np.linalg.inv(align(asset.render_mesh, gt_mesh)))  # truth in the asset frame
    gt_mesh.apply_translation([0, 0, -gt_mesh.bounds[0][2]])
    cache = ROOT / "data" / "work" / "robust_gt" / f"{folder}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        hulls = [trimesh.Trimesh(v, f) for v, f in zip(z["hv"], z["hf"])]
    else:
        hulls = convex_decompose(gt_mesh, 0.04, 32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, hv=np.array([h.vertices for h in hulls], dtype=object),
                 hf=np.array([h.faces for h in hulls], dtype=object))
    density = asset.mass / asset.render_mesh.volume
    mp = estimate_mass_properties(gt_mesh, hulls, density=density)
    w, x, y, zq = mp.principal_axes_quat_wxyz
    truth = AssetData(name="truth", render_mesh=gt_mesh, hulls=hulls, mass=mp.mass,
                      center_of_mass=np.asarray(mp.center_of_mass), diagonal_inertia=np.asarray(mp.principal_moments),
                      principal_axes_wxyz=np.array([w, x, y, zq]), physics_material=dict(asset.physics_material))
    _state[folder] = (asset, hyps, unc, truth)
    return _state[folder]


def _uniform_sample(asset, rng):
    from scan2usd.uncertainty import scale_asset

    a = scale_asset(asset, rng.uniform(0.95, 1.05))
    k = rng.uniform(0.8, 1.2)
    a.mass *= k
    a.diagonal_inertia = np.asarray(a.diagonal_inertia) * k
    a.physics_material = dict(a.physics_material)
    a.physics_material["dynamic_friction"] = (a.physics_material.get("dynamic_friction") or 0.4) * rng.uniform(0.7, 1.3)
    return a


def job(p: dict) -> dict:
    import demo_panda as dp

    from scan2usd.uncertainty import sample_physical

    asset, hyps, unc, truth = _load(p["folder"])
    rng = np.random.default_rng(p["seed"])
    if p["kind"] == "nominal":
        phys = asset
    elif p["kind"] == "uniform":
        phys = _uniform_sample(asset, rng)
    elif p["kind"] == "uncertainty":
        phys, _ = sample_physical(asset, unc, rng, hyps)
    else:  # truth
        phys = truth
    mu = (phys.physics_material or {}).get("dynamic_friction")
    r = dp.pick_trial(asset, physical=phys, yaw=p["yaw"], radial_offset=p.get("offset", 0.0), friction=mu,
                      work=ROOT / "data" / "work" / "robust" / p["id"])
    return {**p, "success": r["success"], "lifted_m": r["lifted_m"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12, help="randomized samples per candidate per strategy")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    jobs = []
    for folder in CUPS:
        for k, yaw in enumerate(YAWS):
            jobs.append({"folder": folder, "yaw": float(yaw), "cand": k, "kind": "nominal", "seed": 0})
            for kind in ("uniform", "uncertainty"):
                for s in range(args.samples):
                    jobs.append({"folder": folder, "yaw": float(yaw), "cand": k, "kind": kind, "seed": 1000 * k + s})
            for off in EXEC_OFFSETS:  # every candidate on the truth (so any strategy's pick can be looked up)
                jobs.append({"folder": folder, "yaw": float(yaw), "cand": k, "kind": "truth", "seed": 0, "offset": off})
    for i, j in enumerate(jobs):
        j["id"] = f"r{i:05d}"
    with Pool(args.workers) as pool:
        res = pool.map(job, jobs, chunksize=4)
    print(f"{len(jobs)} trials in {time.time() - t0:.0f} s", flush=True)

    summary = {}
    for folder, title in CUPS.items():
        r = [x for x in res if x["folder"] == folder]
        score = {kind: {k: float(np.mean([x["success"] for x in r if x["kind"] == kind and x["cand"] == k]))
                        for k in range(len(YAWS))} for kind in STRATEGIES}
        real = {k: float(np.mean([x["success"] for x in r if x["kind"] == "truth" and x["cand"] == k]))
                for k in range(len(YAWS))}
        picks = {}
        for kind in STRATEGIES:
            best = max(score[kind].values())
            tied = [k for k, v in score[kind].items() if v == best]
            # A planner can't break ties by peeking at the truth: report the expected real success
            # over all tied top candidates.
            picks[kind] = {"predicted": best, "tied_candidates": len(tied),
                           "real_success": float(np.mean([real[k] for k in tied]))}
        summary[folder] = {"title": title, "scores": score, "real": real, "picks": picks,
                           "real_best": max(real.values()), "real_mean": float(np.mean(list(real.values())))}
        print(f"{title}: " + ", ".join(f"{k} predicts {v['predicted']:.2f} -> real {v['real_success']:.2f} "
                                      f"({v['tied_candidates']} tied)" for k, v in picks.items())
              + f" | real best {summary[folder]['real_best']:.2f}, mean {summary[folder]['real_mean']:.2f}", flush=True)
    out = ROOT / "docs" / "figures"
    (out / "robust_grasps.json").write_text(json.dumps({"samples": args.samples, "results": summary}, indent=2))
    figure(summary, out / "robust_grasps.png")


def figure(summary: dict, out: Path) -> None:
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font

    labels = {"nominal": "nominal asset", "uniform": "hand-tuned randomization", "uncertainty": "asset's own uncertainty"}
    colors = {"nominal": (148, 163, 184), "uniform": (234, 179, 8), "uncertainty": (37, 99, 235)}
    W, H = 980, 120 + 150 * len(summary)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((16, 12), "Grasp chosen in simulation, then executed on the true object", fill=(30, 41, 59), font=_font(22))
    d.text((16, 44), "bars: success on the ground-truth object (open bar: what the strategy predicted)",
           fill=(100, 116, 139), font=_font(15))
    for i, (folder, s) in enumerate(summary.items()):
        y0 = 84 + i * 150
        d.text((16, y0), s["title"], fill=(30, 41, 59), font=_font(18))
        for j, kind in enumerate(STRATEGIES):
            y = y0 + 30 + j * 36
            p = s["picks"][kind]
            d.text((16, y + 4), labels[kind], fill=(71, 85, 105), font=_font(15))
            x0, span = 250, 520
            d.rectangle([x0, y, x0 + span, y + 24], outline=(226, 232, 240))
            d.rectangle([x0, y, x0 + int(span * p["real_success"]), y + 24], fill=colors[kind])
            d.rectangle([x0, y + 9, x0 + int(span * p["predicted"]), y + 15], outline=(30, 41, 59))
            d.text((x0 + span + 12, y + 3), f"{p['real_success'] * 100:.0f}% real (predicted {p['predicted'] * 100:.0f}%)",
                   fill=(51, 65, 85), font=_font(14))
    img.save(out)


if __name__ == "__main__":
    main()
