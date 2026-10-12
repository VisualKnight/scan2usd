"""Docs figure from an Isaac Sim Replicator dataset: RGB + 2D boxes, and segmentation.

    python scripts/make_isaac_figure.py [output/isaac/dataset] [--frames 0,2,3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scan2usd.preview import _font

ROOT = Path(__file__).resolve().parents[1]
NICE = {
    "ace_coffee_mug_kristen_16_oz_cup": "printed mug", "cole_hardware_bowl_scirocco_yellowblue": "bowl",
    "great_dinos_triceratops_toy": "figure", "schleich_lion_action_figure": "lion",
    "womens_bluefish_2eye_boat_shoe_in_tan": "shoe", "ceramic_mug": "demo mug", "mug": "reconstructed mug",
    "keepsake_box": "hinged box", "buddha_head": "Buddha (real photos)",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path, nargs="?", default=ROOT / "output" / "isaac" / "dataset")
    p.add_argument("--frames", default="0,2,3")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "isaac_dataset.jpg")
    args = p.parse_args()
    ids = [int(v) for v in args.frames.split(",")]
    w, h = 640, 480
    sheet = Image.new("RGB", (w * len(ids), h * 2), "white")
    font = _font(15)
    for col, i in enumerate(ids):
        rgb = Image.open(args.dataset / f"rgb_{i:04d}.png").convert("RGB")
        sx, sy = w / rgb.width, h / rgb.height
        rgb = rgb.resize((w, h))
        d = ImageDraw.Draw(rgb)
        boxes = np.load(args.dataset / f"bounding_box_2d_tight_{i:04d}.npy")
        labels = json.loads((args.dataset / f"bounding_box_2d_tight_labels_{i:04d}.json").read_text())
        merged: dict[int, list[float]] = {}
        for b in boxes:  # multi-body assets (e.g. box + lid) share one class: merge their boxes
            sid = int(b["semanticId"])
            x0, y0, x1, y1 = float(b["x_min"]), float(b["y_min"]), float(b["x_max"]), float(b["y_max"])
            m = merged.get(sid)
            merged[sid] = [x0, y0, x1, y1] if m is None else [min(m[0], x0), min(m[1], y0), max(m[2], x1), max(m[3], y1)]
        for sid, (bx0, by0, bx1, by1) in merged.items():
            name = labels[str(sid)]["class"]
            x0, y0, x1, y1 = bx0 * sx, by0 * sy, bx1 * sx, by1 * sy
            d.rectangle([x0, y0, x1, y1], outline=(234, 88, 12), width=2)
            text = NICE.get(name, name)
            tw = d.textlength(text, font=font)
            d.rectangle([x0, y0 - 19, x0 + tw + 8, y0], fill=(234, 88, 12))
            d.text((x0 + 4, y0 - 18), text, fill="white", font=font)
        sheet.paste(rgb, (col * w, 0))
        seg = Image.open(args.dataset / f"semantic_segmentation_{i:04d}.png").convert("RGB").resize((w, h))
        sheet.paste(seg, (col * w, h))
    sheet.save(args.out, quality=86)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
