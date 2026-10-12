"""How accurate is metric scale from the ColorChecker? Checked on the synthetic benchmark captures.

    python scripts/eval_chart_scale.py

True scale comes from aligning COLMAP's camera centers to the ground-truth Blender
cameras (similarity transform); the chart estimate uses only the photos and the chart's
16 mm patch pitch. Appends a section to docs/benchmark.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scan2usd.chart import chart_metric_scale, read_colmap_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_benchmark import OBJECTS  # noqa: E402

PITCH_M = 0.016  # the synthetic Classic Mini's patch pitch (scripts/blender_render_capture.py)


def true_scale(cap: Path) -> float:
    _, images = read_colmap_model(cap / "work" / "colmap" / "sparse" / "0")
    gt = {d["name"]: np.array(d["camera_to_world"])[:3, 3]
          for d in json.loads((cap / "cameras.json").read_text())["images"]}
    names = sorted(set(gt) & set(images))
    c = np.array([-images[n][1].T @ images[n][2] for n in names])
    g = np.array([gt[n] for n in names])
    cc, gc = c - c.mean(0), g - g.mean(0)
    _, s, _ = np.linalg.svd(gc.T @ cc)
    return float(s.sum() / (cc ** 2).sum())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for key, (folder, _, display) in OBJECTS.items():
        cap = ROOT / "data" / "synthetic" / folder
        if not (cap / "work" / "colmap" / "sparse" / "0").exists():
            continue
        truth = true_scale(cap)
        try:
            est = chart_metric_scale(cap / "images", cap / "work" / "colmap" / "sparse" / "0", PITCH_M)
            err = (est["meters_per_unit"] / truth - 1) * 100
            rows.append((display, est["inlier_views"], est["views"], err))
            print(f"{display}: {err:+.2f}% ({est['inlier_views']}/{est['views']} views)")
        except ValueError as exc:
            rows.append((display, 0, 0, None))
            print(f"{display}: failed ({exc})")
    lines = ["", "## Metric scale from the ColorChecker (no ruler)", "",
             "Scale estimated by triangulating the chart's patch centers in the photos (RANSAC over views), "
             "compared with the true scale from the ground-truth cameras.", "",
             "| Object | Chart views used | Scale error |", "|---|---|---|"]
    for display, inl, n, err in rows:
        lines.append(f"| {display} | {inl}/{n} | {'failed' if err is None else f'{err:+.2f}%'} |")
    errs = [abs(r[3]) for r in rows if r[3] is not None]
    if errs:
        lines += ["", f"Mean absolute scale error: **{np.mean(errs):.2f}%** (max {np.max(errs):.2f}%)."]
    md = ROOT / "docs" / "benchmark.md"
    text = md.read_text(encoding="utf-8").split("\n## Metric scale from the ColorChecker")[0].rstrip("\n")
    md.write_text(text + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
