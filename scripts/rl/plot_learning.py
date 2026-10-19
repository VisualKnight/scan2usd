"""The learned hand policy, before and beyond: placed / 400 per training stage, and vs scene jitter.

Reads output/rl/sweep_curve.json and sweep_jitter.json (eval_sweep.py) and writes
docs/figures/learned_progress.png + learned_progress.json (the numbers, without the per-cell lists).

    .venv\\Scripts\\python.exe scripts\\rl\\plot_learning.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from PIL import Image, ImageDraw  # noqa: E402

from scan2usd.preview import _font  # noqa: E402

RL = os.path.join(ROOT, "output", "rl")
FIG = os.path.join(ROOT, "docs", "figures")
INK, SUB, GRID = (30, 41, 59), (71, 85, 105), (226, 232, 240)
GREEN, RED, BLUE, GREY = (21, 128, 61), (185, 28, 28), (37, 99, 235), (148, 163, 184)
BC_STEPS = {"bc100": 100, "bc300": 300, "bc1000": 1000, "bc3000": 3000, "bc10000": 10000, "bc40000": 40000}


def slim(j):
    return {k: j[k] for k in ("label", "scale", "envs", "held", "placed", "add_mm_median", "rot_deg_median")}


def bars(d, box, title, note, rows):
    """rows: (label, placed, envs, colour, group); a gap row is None."""
    x0, y0, x1, y1 = box
    d.text((x0, y0), title, fill=INK, font=_font(22))
    d.text((x0, y0 + 32), note, fill=SUB, font=_font(15))
    top, bot = y0 + 80, y1 - 90
    for v in (0, 100, 200, 300, 400):
        y = bot - (bot - top) * v / 400
        d.line([x0 + 40, y, x1, y], fill=GRID, width=1)
        d.text((x0, y - 9), str(v), fill=SUB, font=_font(14))
    slot = (x1 - x0 - 50) / len(rows)
    groups = {}
    for i, r in enumerate(rows):
        if r is None:
            continue
        label, placed, envs, col, group = r
        cx = x0 + 50 + slot * (i + 0.5)
        groups.setdefault(group, []).append(cx)
        y = bot - (bot - top) * placed / envs
        d.rectangle([cx - slot * 0.32, y, cx + slot * 0.32, bot], fill=col)
        t = str(placed)
        d.text((cx - d.textlength(t, font=_font(14)) / 2, y - 20), t, fill=INK, font=_font(14))
        for k, ln in enumerate(label.split("\n")):
            d.text((cx - d.textlength(ln, font=_font(13)) / 2, bot + 8 + 17 * k), ln, fill=SUB, font=_font(13))
    for g, cs in groups.items():
        a, b = cs[0] - slot * 0.4, cs[-1] + slot * 0.4
        d.line([a, bot + 50, b, bot + 50], fill=GREY, width=2)
        d.text(((a + b - d.textlength(g, font=_font(15))) / 2, bot + 56), g, fill=INK, font=_font(15))


def lines(d, box, title, note, series, xs):
    x0, y0, x1, y1 = box
    d.text((x0, y0), title, fill=INK, font=_font(22))
    d.text((x0, y0 + 32), note, fill=SUB, font=_font(15))
    top, bot, lft, rgt = y0 + 80, y1 - 90, x0 + 50, x1 - 20
    for v in (0, 100, 200, 300, 400):
        y = bot - (bot - top) * v / 400
        d.line([lft, y, rgt, y], fill=GRID, width=1)
        d.text((x0, y - 9), str(v), fill=SUB, font=_font(14))
    lo, hi = xs[0], xs[-1]

    def px(s):
        return lft + (rgt - lft) * (s - lo) / (hi - lo)

    for s in xs:
        t = f"{s:g}\u00d7"
        d.text((px(s) - d.textlength(t, font=_font(14)) / 2, bot + 8), t, fill=SUB, font=_font(14))
    d.text((lft, bot + 32), "scene jitter (1\u00d7 = training range: \u00b12 cm, \u00b110\u00b0)", fill=SUB, font=_font(14))
    for name, col, pts in series:
        xy = [(px(s), bot - (bot - top) * p / 400) for s, p in pts]
        d.line(xy, fill=col, width=4)
        up = col == GREY
        for (x, y), (_, p) in zip(xy, pts):
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col)
            t = str(p)
            d.text((x - d.textlength(t, font=_font(14)) / 2, y - 28 if up else y + 12), t, fill=col, font=_font(14))
        x, y = xy[-1]
        d.text((x - d.textlength(name, font=_font(16)) - 16, y - 54 if up else y + 36), name, fill=col, font=_font(16))


def main():
    curve = json.load(open(os.path.join(RL, "sweep_curve.json")))["jobs"]
    jitter = json.load(open(os.path.join(RL, "sweep_jitter.json")))["jobs"]
    by = {j["label"]: j for j in curve}

    def col(p, n):
        return GREEN if p >= 0.99 * n else (RED if p < 0.5 * n else (217, 119, 6))

    rerun, used = "a re-run: imitation only", "the network in use"
    rows = [(f"{BC_STEPS[k]:,}\nsteps" if k != "bc40000" else "40,000\n(done)", by[k]["placed"], by[k]["envs"],
             col(by[k]["placed"], by[k]["envs"]), rerun) for k in BC_STEPS]
    rows += [None, ("imitation\n(40,000)", by["r0"]["placed"], by["r0"]["envs"], col(by["r0"]["placed"], by["r0"]["envs"]),
                    used)]
    rows += [(f"DAgger\nround {k}", by[f"r{k}"]["placed"], by[f"r{k}"]["envs"],
              col(by[f"r{k}"]["placed"], by[f"r{k}"]["envs"]), used) for k in (1, 2, 5, 10)]
    xs = sorted({1.0} | {j["scale"] for j in jitter})
    series = []
    for name, label, c in (("scripted teacher", "teacher", GREY), ("neural network", "final", BLUE)):
        pts = [(1.0, 400 if label == "teacher" else by["r10"]["placed"])]
        pts += [(j["scale"], j["placed"]) for j in jitter if j["label"] == label]
        series.append((name, c, sorted(pts)))
    W, H = 1800, 620
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    bars(d, (20, 20, 1040, H - 10), "How it learned: placed / 400 fresh scenes",
         "imitation (behavior cloning) needs all 40,000 training steps, and is luck of the draw; DAgger fixes it",
         rows)
    lines(d, (1110, 20, W - 20, H - 10), "Beyond its training: wider random scenes",
          "placed / 400; the script re-plans every scene, the network has only seen 1\u00d7", series, xs)
    img.save(os.path.join(FIG, "learned_progress.png"), optimize=True)
    out = {"curve": [slim(j) for j in curve], "jitter": [slim(j) for j in jitter],
           "rerun_bc_steps": BC_STEPS, "note": "eval_sweep.py: 400 fresh scenes per job (seed 7000), the same scenes per jitter"}
    films = {}
    for k in ("stage_bc3000", "stage_bc_only", "stage_jitter3"):
        p = os.path.join(RL, f"rollout_{k}.json")
        if os.path.exists(p):
            s = json.load(open(p))["summary"]
            films[k] = {key: s[key] for key in ("envs", "held", "placed", "add_mm_median", "rot_deg_median", "policy")}
    out["films"] = films
    with open(os.path.join(FIG, "learned_progress.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(FIG, "learned_progress.png"))


if __name__ == "__main__":
    main()
