"""The vision policy's results: one chart and the numbers for the page (vision_eval.py's outputs).

Reads output/rl/vision/eval_*.json and train_vision.json; writes docs/figures/vision_policy.png and
vision_policy.json (the numbers, without the per-cell lists).

    .venv\\Scripts\\python.exe scripts\\rl\\vision_summary.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw  # noqa: E402

from plot_learning import BLUE, FIG, GREY, ROOT, bars  # noqa: E402

VIS = os.path.join(ROOT, "output", "rl", "vision")
AMBER = (217, 119, 6)
# (key, eval file, mode): every run is the same 400 fresh scenes (seed 5000) under random looks.
# first: the first pose networks (in-hand data under 5 lightings); final: the in-hand network trained with sensor
# noise and 23 lightings; dagger: the policy retrained (DAgger) on the first networks' noisy estimates.
RUNS = {
    "state": ("scene_only", "exact/exact"),
    "state_x2": ("frozen_x2", "exact/exact"),
    "first": ("frozen", "vision/vision"),
    "first_no_hand": ("frozen", "vision/mean"),
    "first_noisy": ("frozen_noisy", "vision/vision"),
    "first_x2": ("frozen_x2", "vision/vision"),
    "robust5": ("robust", "vision/vision"),
    "robust5_noisy": ("robust_noisy", "vision/vision"),
    "robust5_x2": ("robust_x2", "vision/vision"),
    "final": ("final", "vision/vision"),
    "final_no_hand": ("final", "vision/mean"),
    "final_noisy": ("final_noisy", "vision/vision"),
    "final_x2": ("final_x2", "vision/vision"),
    "final_x2_no_hand": ("final_x2_mean", "vision/mean"),
    "dagger": ("dagger", "vision/vision"),
    "dagger_no_hand": ("dagger", "vision/mean"),
    "dagger_noisy": ("dagger_noisy", "vision/vision"),
    "dagger_x2": ("dagger_x2", "vision/vision"),
}


def load() -> dict:
    out = {}
    for key, (name, mode) in RUNS.items():
        p = os.path.join(VIS, f"eval_{name}.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))["modes"].get(mode)
        if m is not None:
            out[key] = {k: m[k] for k in ("envs", "held", "placed", "add_mm_median", "rot_deg_median")}
            out[key]["estimates"] = m.get("estimates", {})
    return out


def main():
    r = load()
    tv = json.load(open(os.path.join(VIS, "train_vision.json")))
    nets = {k: v["val"] for k, v in tv.items() if "val" in v}
    rows = [("simulator\nstate", "state", GREY, "clean cameras"), ("cameras", "final", BLUE, "clean cameras"),
            ("cameras,\nno in-hand", "final_no_hand", (147, 197, 253), "clean cameras"), None,
            ("first\nnetworks", "first_noisy", (147, 197, 253), "noisier sensor"),
            ("cameras", "final_noisy", BLUE, "noisier sensor"),
            ("retrained\npolicy", "dagger_noisy", AMBER, "noisier sensor"), None,
            ("simulator\nstate", "state_x2", GREY, "2\u00d7 wider scenes"), ("cameras", "final_x2", BLUE, "2\u00d7 wider scenes"),
            ("cameras,\nno in-hand", "final_x2_no_hand", (147, 197, 253), "2\u00d7 wider scenes"),
            ("retrained\npolicy", "dagger_x2", AMBER, "2\u00d7 wider scenes")]
    bar_rows = [None if x is None else (x[0], r[x[1]]["placed"], r[x[1]]["envs"], x[2], x[3])
                for x in rows if x is None or x[1] in r]
    im = Image.new("RGB", (1500, 620), (255, 255, 255))
    d = ImageDraw.Draw(im)
    bars(d, (30, 20, 1470, 600), "Placed out of 400: the hand from two RGB-D cameras",
         "The same 400 fresh scenes in every run, under a lighting no network trained on. Grey: the policy reads the "
         "simulator. Blue: pose networks on the camera images. Amber: the policy retrained on noisy estimates.",
         bar_rows)
    os.makedirs(FIG, exist_ok=True)
    im.save(os.path.join(FIG, "vision_policy.png"))
    json.dump({"runs": r, "nets": nets}, open(os.path.join(FIG, "vision_policy.json"), "w"), indent=1)
    print(json.dumps({k: v["placed"] for k, v in r.items()}))


if __name__ == "__main__":
    main()
