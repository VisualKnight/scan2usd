"""Annotate every library asset with simulated grasps (Scan2UsdGraspAPI), stored in the asset file.

    python scripts/annotate_grasps.py [--workers 16]

Shadow Hand: the successful power grasps from scripts/demo_hand_grasp.py (docs/figures/hand_grasp.json)
are re-tested under 6 execution perturbations (±4 mm, ±8° yaw) on both collision variants; the best
are written with successRate (decomposition), trials and per-variant rates.
Franka Panda: rim grasps on the cups are sampled at 12 directions around the rim and tested with
3 radial offsets (−3, 0, +3 mm) per variant.

Grasps are written to the asset files the USD library payloads, so they show up in the library
(and in Isaac Sim) under /Library/<asset>/grasps. Results: docs/figures/grasps.json.
Rebuilding an asset (e.g. `scan2usd demo` rewrites output/demo/asset/ceramic_mug.usda) drops its
grasps: re-run this script afterwards (about 1 min).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "output" / "library" / "scan2usd_library.usda"
VARIANTS = ("decomposed", "convex_hull")
HAND_PERTURB = [(0.004, 0, 0), (-0.004, 0, 0), (0, 0.004, 0), (0, -0.004, 0), (0, 0, 8), (0, 0, -8)]
PANDA_CUPS = ("ceramic_mug", "speckled_mug", "ace_coffee_mug_kristen_16_oz_cup")
PANDA_YAWS = np.radians(np.arange(0, 360, 30))
PANDA_OFFSETS = (-0.003, 0.0, 0.003)
KEEP = 8


def hand_job(p: dict) -> dict:
    import demo_hand_grasp as h

    r = h.trial(p)
    return {"key": p["key"], "variant": p["variant"], "success": r["success"]}


def panda_job(p: dict) -> dict:
    import demo_panda as dp

    a = _asset(p["prim"], p["variant"])
    nominal = _asset(p["prim"], "decomposed")
    r = dp.pick_trial(nominal, physical=a, yaw=p["yaw"], radial_offset=p["offset"],
                      work=ROOT / "data" / "work" / "annotate" / p["id"])
    return {"key": p["key"], "variant": p["variant"], "success": r["success"], "grasp": r["grasp"], "radial": r["radial"]}


_cache: dict = {}


def _asset(prim, variant):
    from scan2usd.usd_io import read_usd

    if (prim, variant) not in _cache:
        _cache[(prim, variant)] = read_usd(LIB, prim, {"collision": variant})
    return _cache[(prim, variant)]


def hand_candidates(results: list[dict]) -> dict[str, list[dict]]:
    """Successful decomposed-collider grasps per asset, spread over approach directions."""
    out: dict[str, list[dict]] = {}
    for r in results:
        if r["variant"] == "decomposed" and r["success"]:
            out.setdefault(r["prim"], []).append(r)
    for prim, rs in out.items():
        seen, picked = set(), []
        for r in sorted(rs, key=lambda r: -r["held_m"]):
            key = (r["mode"], r["yaw"])
            if key not in seen:
                seen.add(key)
                picked.append(r)
        out[prim] = picked[:12]
    return out


def main() -> None:
    import demo_hand_grasp as h

    from scan2usd.library import asset_files
    from scan2usd.schemas import Grasp, write_grasps

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    files = asset_files(LIB)
    t0 = time.time()

    # --- Shadow Hand ---------------------------------------------------------------------------
    hand = json.loads((ROOT / "docs" / "figures" / "hand_grasp.json").read_text())["trials"]
    cands = hand_candidates(hand)
    jobs = []
    for prim, rs in cands.items():
        for c in rs:
            base = {k: c[k] for k in ("prim", "mode", "yaw", "a", "b")}
            key = f"{prim}|{c['mode']}|{c['yaw']}|{c['a']}|{c['b']}"
            for v in VARIANTS:
                for k, (dx, dy, dyaw) in enumerate(HAND_PERTURB):
                    jobs.append({**base, "variant": v, "dx": dx, "dy": dy, "dyaw": dyaw, "key": key,
                                 "id": f"ah{len(jobs):05d}"})
    with Pool(args.workers) as pool:
        hand_res = pool.map(hand_job, jobs)
    print(f"hand: {len(jobs)} trials in {time.time() - t0:.0f} s", flush=True)

    # --- Panda rim grasps ----------------------------------------------------------------------
    pjobs = []
    for name in PANDA_CUPS:
        prim = f"/Library/{name}"
        for yaw in PANDA_YAWS:
            key = f"{prim}|{yaw:.4f}"
            for v in VARIANTS:
                for off in PANDA_OFFSETS:
                    pjobs.append({"prim": prim, "yaw": float(yaw), "offset": off, "variant": v, "key": key,
                                  "id": f"ap{len(pjobs):05d}"})
    with Pool(args.workers) as pool:
        panda_res = pool.map(panda_job, pjobs)
    print(f"panda: {len(pjobs)} trials ({time.time() - t0:.0f} s total)", flush=True)

    # --- aggregate and write --------------------------------------------------------------------
    def rates(res, key):
        by = {}
        for v in VARIANTS:
            s = [r["success"] for r in res if r["key"] == key and r["variant"] == v]
            by[v] = float(np.mean(s)) if s else 0.0
        n = sum(1 for r in res if r["key"] == key and r["variant"] == "decomposed")
        return by, n

    summary = {}
    for name, path in files.items():
        prim = f"/Library/{name}"
        grasps = []
        for c in cands.get(prim, []):
            key = f"{prim}|{c['mode']}|{c['yaw']}|{c['a']}|{c['b']}"
            by, n = rates(hand_res, key)
            site, r_palm = h.grasp_frame(_asset(prim, "decomposed"), c)
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = r_palm, site
            grasps.append(Grasp(pose=pose, gripper="shadow_hand", type="power", success_rate=by["decomposed"],
                                trials=n, evaluated_with="mujoco:decomposed", by_variant=by))
        grasps = sorted(grasps, key=lambda g: -g.success_rate)[:KEEP]
        if name in PANDA_CUPS:
            pg = []
            for yaw in PANDA_YAWS:
                key = f"{prim}|{yaw:.4f}"
                by, n = rates(panda_res, key)
                r0 = next(r for r in panda_res if r["key"] == key)
                radial = np.array(r0["radial"])
                z = np.array([0.0, 0.0, -1.0])
                pose = np.eye(4)
                pose[:3, :3] = np.c_[np.cross(radial, z), radial, z]
                pose[:3, 3] = np.array(r0["grasp"]) - radial * PANDA_OFFSETS[0]  # the zero-offset grasp
                pg.append(Grasp(pose=pose, gripper="panda", type="rim", width=0.044, success_rate=by["decomposed"],
                                trials=n, evaluated_with="mujoco:decomposed", by_variant=by))
            grasps += sorted(pg, key=lambda g: -g.success_rate)[:KEEP]
        if grasps:
            write_grasps(path, grasps)
        summary[name] = [{"gripper": g.gripper, "successRate": g.success_rate, "byVariant": g.by_variant}
                         for g in grasps]
        best = {gr: max((g.success_rate for g in grasps if g.gripper == gr), default=None)
                for gr in ("shadow_hand", "panda")}
        print(f"  {name}: {len(grasps)} grasps, best {best}", flush=True)
    (ROOT / "docs" / "figures" / "grasps.json").write_text(json.dumps(summary, indent=2))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
