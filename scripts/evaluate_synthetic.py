"""End-to-end check on a synthetic capture: photos -> COLMAP -> Scan2USD asset -> compare to truth.

    python scripts/evaluate_synthetic.py [--capture data/synthetic/mug] [--material ceramic]

Uses only what a real user would have: the photos, the chart photo + corners, and one
ruler measurement (the true height rounded to 1 mm). The ground-truth mesh is used for
scoring only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import trimesh

from scan2usd.evaluate import compare
from scan2usd.mesh_ops import orient_and_scale
from scan2usd.pipeline import BuildOptions, build_asset
from scan2usd.reconstruct import reconstruct
from scan2usd.usd_io import read_usd

ROOT = Path(__file__).resolve().parents[1]


def load_ground_truth(cap: Path) -> trimesh.Trimesh:
    rel = (cap / "gt_path.txt").read_text().strip() if (cap / "gt_path.txt").exists() else "gt_mesh.ply"
    gt = trimesh.load(str(cap / rel), force="mesh")
    gt = trimesh.Trimesh(gt.vertices, gt.faces, process=True)  # drop UV seams so the solid is closed
    gt.merge_vertices()
    gt, _ = orient_and_scale(gt)
    return gt


def evaluate_capture(cap: Path, material: str = "plastic", quality: str = "medium") -> dict:
    work = cap / "work"
    gt = load_ground_truth(cap)
    height = round(float(gt.extents[2]), 3)  # what a ruler would give

    mesh_path, rec = reconstruct(cap / "images", work / "colmap", quality=quality)
    opts = BuildOptions(name=cap.name, material=material, up_axis="z", units="m",
                        scale_to=height, scale_axis="z", chart_image=cap / "chart_photo.jpg",
                        chart_corners=json.loads((cap / "chart_corners.json").read_text()), usda=True)
    res = build_asset(mesh_path, work / "asset", opts)

    recon = read_usd(res.usd_path).render_mesh
    metrics = compare(recon, gt)
    metrics["object"] = cap.name
    metrics["ruler_height_m"] = height
    metrics["mass_g"] = res.summary["mass_properties"]["mass"] * 1000
    metrics["mass_source"] = res.summary["mass_properties"]["volume_source"]
    metrics["color_chart_delta_e2000"] = {k: res.summary["color_calibration"][k]["mean"]
                                          for k in ("delta_e_before", "delta_e_after")}
    metrics["reconstruction"] = {"registered_images": rec.registered_images, "images": rec.images,
                                 "object_points": rec.object_points, "poisson_depth": rec.poisson_depth,
                                 "watertight": rec.watertight, "timings_s": rec.timings_s, "notes": rec.notes}
    metrics["validation"] = res.validation.status
    metrics["build_time_s"] = res.summary["timings_s"]["total"]
    (work / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def summarize(m: dict) -> str:
    a, c = m["accuracy_mm"], m["completeness_mm"]
    lines = [
        f"[{m['object']}] registered {m['reconstruction']['registered_images']}/{m['reconstruction']['images']}, "
        f"validation {m['validation']}",
        f"  accuracy  mean {a['mean']:.2f} mm, p95 {a['p95']:.2f} mm | completeness mean {c['mean']:.2f} mm",
        f"  F-score   @1mm {m['fscore_1mm']:.3f}, @2mm {m['fscore_2mm']:.3f}",
    ]
    if "volume_cm3" in m:
        v = m["volume_cm3"]
        lines.append(f"  volume    {v['reconstruction']:.1f} vs {v['ground_truth']:.1f} cm^3 ({v['error_pct']:+.1f}%)")
    ce = m["color_chart_delta_e2000"]
    lines.append(f"  mass {m['mass_g']:.1f} g ({m['mass_source']}), chart dE {ce['delta_e_before']:.2f} -> "
                 f"{ce['delta_e_after']:.2f}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capture", type=Path, default=ROOT / "data" / "synthetic" / "mug")
    p.add_argument("--material", default="ceramic")
    p.add_argument("--quality", default="medium")
    args = p.parse_args()
    print(summarize(evaluate_capture(args.capture, args.material, args.quality)))


if __name__ == "__main__":
    main()
