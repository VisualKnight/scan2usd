"""Robustness study: how does reconstruction quality degrade with fewer photos?

    python scripts/study_photo_count.py [--capture data/synthetic/schleich_lion_action_figure]

Re-runs photos -> COLMAP -> Scan2USD on evenly spaced subsets of an existing synthetic
capture and scores each against ground truth. Answers a practical capture question:
how many photos are enough? Appends a section to docs/benchmark.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_synthetic import load_ground_truth  # noqa: E402

from scan2usd.evaluate import compare  # noqa: E402
from scan2usd.pipeline import BuildOptions, build_asset  # noqa: E402
from scan2usd.reconstruct import reconstruct  # noqa: E402
from scan2usd.usd_io import read_usd  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--capture", type=Path, default=ROOT / "data" / "synthetic" / "schleich_lion_action_figure")
    p.add_argument("--counts", default="12,20,39,78")
    p.add_argument("--material", default="plastic")
    args = p.parse_args()
    cap = args.capture
    gt = load_ground_truth(cap)
    height = round(float(gt.extents[2]), 3)
    photos = sorted((cap / "images").glob("*.jpg"))
    rows = []
    for n in [int(v) for v in args.counts.split(",")]:
        full_eval = cap / "work" / "evaluation.json"
        if n == len(photos) and full_eval.exists():  # the benchmark already ran the full capture
            m = json.loads(full_eval.read_text())
            rec = m["reconstruction"]
            row = {"photos": n, "registered": rec["registered_images"], "accuracy_mm": m["accuracy_mm"]["mean"],
                   "completeness_mm": m["completeness_mm"]["mean"], "fscore_2mm": m["fscore_2mm"],
                   "volume_error_pct": m.get("volume_cm3", {}).get("error_pct"),
                   "seconds": round(sum(rec["timings_s"].values()), 0)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        work = cap / "work_subsets" / f"n{n:03d}"
        imgs = work / "images"
        if not imgs.exists():
            imgs.mkdir(parents=True)
            step = len(photos) / n
            for k in range(n):
                shutil.copy2(photos[int(k * step)], imgs / photos[int(k * step)].name)
        try:
            mesh, rec = reconstruct(imgs, work / "colmap")
            res = build_asset(mesh, work / "asset", BuildOptions(
                name=f"n{n}", material=args.material, scale_to=height, scale_axis="z",
                run_sim=False, preview=False, usda=True))
            m = compare(read_usd(res.usd_path).render_mesh, gt)
            row = {"photos": n, "registered": rec.registered_images, "accuracy_mm": m["accuracy_mm"]["mean"],
                   "completeness_mm": m["completeness_mm"]["mean"], "fscore_2mm": m["fscore_2mm"],
                   "volume_error_pct": m.get("volume_cm3", {}).get("error_pct"),
                   "seconds": round(sum(rec.timings_s.values()), 0)}
        except Exception as exc:  # report failures as data points
            row = {"photos": n, "error": f"{type(exc).__name__}: {exc}"[:120]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (cap / "work_subsets" / "photo_count.json").write_text(json.dumps(rows, indent=2))

    lines = ["", f"## How many photos? ({cap.name})", "",
             "Evenly spaced subsets of the same capture, each reconstructed from scratch.", "",
             "| Photos | Registered | Accuracy (mm) | Completeness (mm) | F-score @2 mm | Volume error | COLMAP + mesh (s) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['photos']} | – | failed | | | | |")
            continue
        vol = "n/a" if r["volume_error_pct"] is None else f"{r['volume_error_pct']:+.1f}%"
        lines.append(f"| {r['photos']} | {r['registered']} | {r['accuracy_mm']:.2f} | {r['completeness_mm']:.2f} | "
                     f"{r['fscore_2mm']:.3f} | {vol} | {r['seconds']:.0f} |")
    md = ROOT / "docs" / "benchmark.md"
    marker = f"\n## How many photos? ({cap.name})"
    parts = md.read_text(encoding="utf-8").split(marker)
    tail = ""
    if len(parts) > 1 and "\n## " in parts[1]:
        tail = "\n## " + parts[1].split("\n## ", 1)[1]
    md.write_text(parts[0].rstrip("\n") + "\n" + "\n".join(lines) + "\n" + tail, encoding="utf-8")


if __name__ == "__main__":
    main()
