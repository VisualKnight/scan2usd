"""Does physics-aware capture guidance beat random extra photos? A controlled synthetic experiment.

    python scripts/study_capture_guide.py [--objects ace_coffee_mug_kristen_16_oz_cup,...]

Per object (Blender synthetic captures with ground truth, scripts/make_synthetic_capture.py):
  weak     48 photos: the 15° and 35° rings of the full 78-photo capture (no high views)
  +guided  weak + 8 views chosen by `scan2usd guide` on the weak reconstruction
  +random  weak + 8 views drawn at random from the same reachable candidate positions
  +ring    weak + 8 photos from the full capture's 60° ring (a sensible human heuristic)
  full     the original 78-photo capture (reference)
Each set is reconstructed with COLMAP and built with Scan2USD, then scored against the ground truth:
F-score @2 mm, volume error, and the area the cameras never saw. Guided/random views are rendered
with the same Blender scene (scripts/blender_render_capture.py --poses). Their positions are mapped
from the asset frame to Blender's world with a similarity fitted on the weak capture's cameras.
Writes docs/figures/capture_guide.{json,png}.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_synthetic import load_ground_truth  # noqa: E402

from scan2usd.evaluate import compare  # noqa: E402
from scan2usd.guide import capture_views, hemisphere_views, run_guide  # noqa: E402
from scan2usd.pipeline import BuildOptions, build_asset  # noqa: E402
from scan2usd.reconstruct import reconstruct  # noqa: E402
from scan2usd.uncertainty import observation_field, workspace_points  # noqa: E402
from scan2usd.usd_io import read_usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLENDER = os.environ.get("BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
OBJECTS = {"ace_coffee_mug_kristen_16_oz_cup": ("Printed mug", "ceramic"),
           "cole_hardware_bowl_scirocco_yellowblue": ("Bowl", "ceramic"),
           "womens_bluefish_2eye_boat_shoe_in_tan": ("Boat shoe", "rubber")}
K = 8


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) minimizing |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    r = u @ d @ vt
    s = np.trace(np.diag(sig) @ d) / (a ** 2).sum(1).mean()
    return s, r, mu_d - s * r @ mu_s


def build(images: Path, work: Path, cap: Path, material: str, gt) -> dict:
    t0 = time.time()
    mesh_path, rec = reconstruct(images, work / "colmap", quality="medium")
    opts = BuildOptions(name=cap.name, material=material, scale_to=round(float(gt.extents[2]), 3), scale_axis="z",
                        usda=True, run_sim=False, preview=False, gltf=False)
    res = build_asset(mesh_path, work / "asset", opts)
    asset = read_usd(res.usd_path)
    m = compare(asset.render_mesh, gt)
    pts, _ = workspace_points(asset, work / "colmap", float(res.summary["scale_applied"]))
    f = observation_field(asset.render_mesh, pts)
    return {"photos": len(list(images.glob("*.jpg"))), "registered": rec.registered_images,
            "fscore_2mm": m["fscore_2mm"], "accuracy_mm": m["accuracy_mm"]["mean"],
            "completeness_mm": m["completeness_mm"]["mean"],
            "volume_error_pct": m.get("volume_cm3", {}).get("error_pct"),
            "observed_fraction": f.observed_fraction, "sealed_area": f.label_area["sealed"],
            "minutes": round((time.time() - t0) / 60, 1), "usd": Path(os.path.relpath(res.usd_path, ROOT)).as_posix()}


def run_object(folder: str, title: str, material: str, reuse: bool) -> dict:
    cap = ROOT / "data" / "synthetic" / folder
    root = cap / "guide"
    gt = load_ground_truth(cap)
    meta = json.loads((cap / "cameras.json").read_text())
    names = [im["name"] for im in meta["images"]]
    c2w = {im["name"]: np.array(im["camera_to_world"]) for im in meta["images"]}
    weak_names, ring60 = names[:48], names[48:72]
    results = {}

    def image_set(name, extra_files):
        d = root / name / "images"
        if d.exists() and reuse:
            return d
        shutil.rmtree(root / name, ignore_errors=True)
        d.mkdir(parents=True)
        for n in weak_names:
            shutil.copy2(cap / "images" / n, d / n)
        for src in extra_files:
            shutil.copy2(src, d / Path(src).name)
        return d

    def cached(name, images):
        f = root / name / "result.json"
        if reuse and f.exists():
            return json.loads(f.read_text())
        r = build(images, root / name, cap, material, gt)
        f.write_text(json.dumps(r, indent=2))
        return r

    results["weak"] = cached("weak", image_set("weak", []))
    print(f"[{title}] weak: {results['weak']}", flush=True)

    # Guidance on the weak reconstruction; random views from the same candidates.
    weak_usd = ROOT / results["weak"]["usd"]
    sugg, stats = run_guide(weak_usd, root / "weak" / "colmap", root / "shotlist", k=K)
    asset = read_usd(weak_usd)
    lo, hi = asset.render_mesh.bounds
    center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + 0.4 * (hi[2] - lo[2])])
    views_weak = capture_views(root / "weak" / "colmap", weak_usd)
    dist = float(np.median([np.linalg.norm(v.center - center) for v in views_weak]))
    cands = hemisphere_views(center, dist)
    rng = np.random.default_rng(0)
    random_views = [cands[i] for i in rng.choice(len(cands), K, replace=False)]

    # asset frame -> Blender world, from the weak capture's solved vs. true camera centers.
    src = np.array([v.center for v in views_weak])
    dst = np.array([c2w[v.name][:3, 3] for v in views_weak])
    s, r, t = umeyama(src, dst)
    fit_mm = float(np.median(np.linalg.norm((s * src @ r.T + t) - dst, axis=1)) * 1000)

    def to_world(p):
        return (s * r @ p + t).tolist()

    poses = []
    for tag, views in (("guided", [g.view for g in sugg]), ("random", random_views)):
        for i, v in enumerate(views):
            poses.append({"name": f"{tag.upper()}_{i:02d}.jpg", "location": to_world(v.center),
                          "target": to_world(v.center + v.forward * dist)})
    extra = root / "extra"
    if not (reuse and (extra / "poses_out.json").exists()):
        shutil.rmtree(extra, ignore_errors=True)
        (extra / "images").mkdir(parents=True)
        (extra / "poses.json").write_text(json.dumps(poses, indent=1))
        rel = (cap / (cap / "gt_path.txt").read_text().strip())
        cmd = [BLENDER, "-b", "--factory-startup", "--python", str(ROOT / "scripts" / "blender_render_capture.py"), "--",
               "--mesh", str(rel), "--out", str(extra), "--poses", str(extra / "poses.json")]
        if "gso" in str(rel).lower() or (cap / "gt").exists():
            cmd.append("--keep-materials")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    guided_files = sorted((extra / "images").glob("GUIDED_*.jpg"))
    random_files = sorted((extra / "images").glob("RANDOM_*.jpg"))
    ring_files = [cap / "images" / n for n in ring60[::3]]
    results["+guided"] = cached("guided", image_set("guided", guided_files))
    print(f"[{title}] +guided: {results['+guided']}", flush=True)
    results["+random"] = cached("random", image_set("random", random_files))
    print(f"[{title}] +random: {results['+random']}", flush=True)
    results["+ring"] = cached("ring", image_set("ring", ring_files))
    print(f"[{title}] +ring: {results['+ring']}", flush=True)
    ev = json.loads((cap / "work" / "evaluation.json").read_text())
    results["full"] = {"photos": 78, "fscore_2mm": ev["fscore_2mm"],
                       "volume_error_pct": ev.get("volume_cm3", {}).get("error_pct")}
    return {"title": title, "results": results, "frame_fit_mm": fit_mm,
            "suggestions": [{"azimuth": g.azimuth_deg, "elevation": g.elevation_deg, "gain": g.gain} for g in sugg],
            "guide_covered_fraction": stats["covered_fraction"]}


def figure(summary: dict, out: Path) -> None:
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font

    conds = ["weak", "+ring", "+random", "+guided", "full"]
    labels = {"weak": "48 photos (2 low rings)", "+ring": "+8 from a higher ring", "+random": "+8 random views",
              "+guided": "+8 guided views", "full": "all 78 photos"}
    colors = {"weak": (203, 213, 225), "+ring": (234, 179, 8), "+random": (148, 163, 184), "+guided": (37, 99, 235),
              "full": (100, 116, 139)}
    W = 1500
    img = Image.new("RGB", (W, 120 + 40 * len(conds) * len(summary) + 30 * len(summary)), "white")
    d = ImageDraw.Draw(img)
    d.text((16, 12), "Capture guidance: what 8 extra photos buy", fill=(30, 41, 59), font=_font(22))
    d.text((16, 44), "left: |volume error| vs ground truth (shorter is better) · right: F-score @2 mm (longer is better)",
           fill=(100, 116, 139), font=_font(15))
    y = 84
    vmax = max(abs(r.get("volume_error_pct") or 0) for s in summary.values() for r in s["results"].values()) * 1.05 or 1
    for s in summary.values():
        d.text((16, y), s["title"], fill=(30, 41, 59), font=_font(18))
        y += 30
        for c in conds:
            r = s["results"].get(c)
            if not r:
                continue
            d.text((16, y + 2), labels[c], fill=(71, 85, 105), font=_font(14))
            ve = r.get("volume_error_pct")
            if ve is not None:
                w = int(420 * abs(ve) / vmax)
                d.rectangle([230, y, 230 + w, y + 22], fill=colors[c])
                d.text((236 + w, y + 2), f"{ve:+.0f}%", fill=(51, 65, 85), font=_font(14))
            else:
                d.text((236, y + 2), "n/a (mesh not closed)", fill=(148, 163, 184), font=_font(14))
            f = r["fscore_2mm"]
            d.rectangle([820, y, 820 + int(420 * f), y + 22], fill=colors[c])
            d.text((826 + int(420 * f), y + 2), f"{f:.3f}", fill=(51, 65, 85), font=_font(14))
            y += 40
    img = img.crop((0, 0, W, y + 10))
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=",".join(OBJECTS))
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    out = ROOT / "docs" / "figures"
    summary_path = out / "capture_guide.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for folder in args.objects.split(","):
        title, material = OBJECTS[folder]
        summary[folder] = run_object(folder, title, material, args.reuse)
        summary_path.write_text(json.dumps(summary, indent=2))
        figure(summary, out / "capture_guide.png")


if __name__ == "__main__":
    main()
