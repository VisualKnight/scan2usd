"""Sim-to-real comparison strip: the same real photos through both detectors (docs/figures/sim2real_compare.jpg).

    python scripts/train_detector.py --data output/synth --out output/detector --tag _baseline
    python scripts/train_detector.py --data output/synth_dr --out output/detector_dr
    python scripts/make_sim2real_compare.py

Top row: trained on scanned objects only. Bottom row: with unlabeled distractors. Green is the best
Buddha box, red any other detection scoring at least 0.5. Most AliceVision Buddha photos are stored
sideways with no EXIF orientation tag, so the panels use upright photos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_detector as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
PHOTOS = ROOT / "data" / "external" / "buddha" / "buddha"
SHOWN = ("00017._c.png", "00040._c.png", "00036._c.png")  # upright views; many false detections without distractors
ROWS = (("Objects only", "synth", "detector", "sim2real_detection_baseline.json", (185, 28, 28)),
        ("+ distractors", "synth_dr", "detector_dr", "sim2real_detection.json", (21, 128, 61)))


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H, L = 512, 288, 230
    sheet = Image.new("RGB", (L + W * len(SHOWN), H * len(ROWS)), "white")
    d = ImageDraw.Draw(sheet)
    for r, (title, data, det, res_json, color) in enumerate(ROWS):
        cats = {c["id"]: c["name"] for c in json.loads((ROOT / "output" / data / "coco.json").read_text())["categories"]}
        buddha = next(c for c, n in cats.items() if n == "buddha_head")
        model = T.build_model(len(cats)).to(dev)
        model.load_state_dict(torch.load(ROOT / "output" / det / "fasterrcnn_scan2usd.pt", map_location=dev))
        model.eval()
        res = json.loads((FIG / res_json).read_text())
        y = r * H
        d.text((14, y + 90), title, fill=color, font=font(22))
        d.text((14, y + 130), f"{res['real_top1_rate'] * 100:.0f}% top-1", fill=(51, 65, 85), font=font(16))
        d.text((14, y + 152), f"{res['real_false_positives_per_photo']:.2g} false pos./photo", fill=(51, 65, 85),
               font=font(16))
        for k, name in enumerate(SHOWN):
            img = Image.open(PHOTOS / name).convert("RGB")
            s = 1024 / max(img.size)
            small = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
            boxes, scores, labels = T.predict(model, small, dev)
            b_idx = [i for i in range(len(scores)) if labels[i] == buddha]
            best = max(b_idx, key=lambda i: scores[i]) if b_idx else None
            dd = ImageDraw.Draw(small)
            for i in range(len(scores)):
                if scores[i] < 0.5:
                    continue
                c = (21, 128, 61) if i == best else (220, 38, 38)
                dd.rectangle(list(boxes[i]), outline=c, width=5)
                dd.text((boxes[i][0] + 6, boxes[i][1] + 4), f"{cats[int(labels[i])][:18]} {scores[i]:.2f}", fill=c)
            sheet.paste(small.resize((W, H), Image.LANCZOS), (L + k * W, y))
    sheet.save(FIG / "sim2real_compare.jpg", quality=88)
    print("wrote", FIG / "sim2real_compare.jpg")


if __name__ == "__main__":
    main()
