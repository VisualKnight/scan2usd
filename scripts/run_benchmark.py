"""Benchmark: synthetic captures of several objects through the full pipeline.

    python scripts/run_benchmark.py [--objects mug,Schleich_Lion_Action_Figure,...] [--skip-render]

For each object: render the capture with Blender if needed (make_synthetic_capture.py),
then photos -> COLMAP -> Scan2USD -> compare with ground truth (evaluate_synthetic.py).
Writes data/synthetic/benchmark.json and docs/benchmark.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_synthetic import evaluate_capture, summarize  # noqa: E402

# object -> (capture folder, material preset, display name)
OBJECTS = {
    "mug": ("mug", "ceramic", "Speckled stoneware mug (procedural)"),
    "ACE_Coffee_Mug_Kristen_16_oz_cup": ("ace_coffee_mug_kristen_16_oz_cup", "ceramic", "Printed coffee mug"),
    "Cole_Hardware_Bowl_Scirocco_YellowBlue": ("cole_hardware_bowl_scirocco_yellowblue", "ceramic", "Ceramic bowl"),
    "Great_Dinos_Triceratops_Toy": ("great_dinos_triceratops_toy", "plastic", "Triceratops toy"),
    "Schleich_Lion_Action_Figure": ("schleich_lion_action_figure", "plastic", "Lion figure"),
    "Womens_Bluefish_2Eye_Boat_Shoe_in_Tan": ("womens_bluefish_2eye_boat_shoe_in_tan", "rubber", "Boat shoe"),
}


def table(rows: list[dict]) -> str:
    lines = [
        "| Object | Photos registered | Accuracy mean / p95 (mm) | Completeness mean (mm) | F-score @2 mm | "
        "Volume error | Chart ΔE2000 | Validation | Time (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['display']} | – | failed: {r['error']} | | | | | | |")
            continue
        vol = f"{r['volume_cm3']['error_pct']:+.1f}%" if "volume_cm3" in r else "n/a (open mesh)"
        rec = r["reconstruction"]
        ce = r["color_chart_delta_e2000"]
        t = sum(rec["timings_s"].values()) + r["build_time_s"]
        lines.append(
            f"| {r['display']} | {rec['registered_images']}/{rec['images']} | "
            f"{r['accuracy_mm']['mean']:.2f} / {r['accuracy_mm']['p95']:.2f} | {r['completeness_mm']['mean']:.2f} | "
            f"{r['fscore_2mm']:.3f} | {vol} | {ce['delta_e_before']:.1f} → {ce['delta_e_after']:.2f} | "
            f"{r['validation']} | {t:.0f} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--objects", default=",".join(OBJECTS))
    p.add_argument("--skip-render", action="store_true")
    args = p.parse_args()
    rows = []
    for key in args.objects.split(","):
        folder, material, display = OBJECTS[key]
        cap = ROOT / "data" / "synthetic" / folder
        t0 = time.time()
        if not (cap / "images").exists() and not args.skip_render:
            cmd = [sys.executable, str(ROOT / "scripts" / "make_synthetic_capture.py")]
            if key != "mug":
                cmd += ["--gso", key]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[{key}] rendered in {time.time() - t0:.0f} s", flush=True)
        try:
            m = evaluate_capture(cap, material)
            m["display"] = display
            print(summarize(m), flush=True)
        except Exception as exc:  # keep the benchmark going; report the failure in the table
            m = {"object": folder, "display": display, "error": f"{type(exc).__name__}: {exc}"[:160]}
            print(f"[{key}] FAILED: {m['error']}", flush=True)
        rows.append(m)
        (ROOT / "data" / "synthetic" / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = ("# Scan2USD synthetic benchmark\n\n"
          "Each object is rendered as a 78-photo phone-style capture in Blender (3 rings + top views, "
          "warm light, ColorChecker in the scene), reconstructed with COLMAP, and built by Scan2USD using one "
          "ruler measurement (height, rounded to 1 mm). Scores compare the final USD render mesh against the "
          "ground-truth mesh after rigid alignment.\n\n"
          "- **Accuracy**: distance from the reconstruction to the true surface. "
          "**Completeness**: distance from the true surface to the reconstruction.\n"
          "- **F-score @2 mm**: share of both surfaces within 2 mm of each other (1.0 = perfect).\n\n"
          + table(rows) + "\n\nGround-truth meshes: Google Scanned Objects (CC-BY 4.0) and a procedural mug.\n")
    (ROOT / "docs" / "benchmark.md").write_text(md, encoding="utf-8")
    print(table(rows))


if __name__ == "__main__":
    main()
