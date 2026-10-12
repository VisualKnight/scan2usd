"""Contact sheet of a COCO dataset: RGB frames with labeled boxes.

    python scripts/make_coco_figure.py output/synth --frames 3,57,142,388,611,905
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("--frames", default="0,1,2,3,4,5")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "isaac_synth_dataset.jpg")
    args = p.parse_args()

    coco = json.loads((args.dataset / "coco.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    colors = {c: tuple(int(255 * v) for v in colorsys.hsv_to_rgb((i * 0.61) % 1, 0.85, 0.95))
              for i, c in enumerate(sorted(cats))}
    tiles = []
    for fid in (int(f) for f in args.frames.split(",")):
        im = coco["images"][fid]
        img = Image.open(args.dataset / im["file_name"]).convert("RGB")
        d = ImageDraw.Draw(img)
        for a in coco["annotations"]:
            if a["image_id"] != im["id"]:
                continue
            x, y, w, h = a["bbox"]
            c = colors[a["category_id"]]
            d.rectangle([x, y, x + w, y + h], outline=c, width=3)
            d.rectangle([x, y - 14, x + 7 * len(cats[a["category_id"]]) + 6, y], fill=c)
            d.text((x + 3, y - 13), cats[a["category_id"]], fill=(0, 0, 0))
        s = args.width / img.width
        tiles.append(img.resize((args.width, round(img.height * s))))
    rows = (len(tiles) + args.cols - 1) // args.cols
    th = tiles[0].height
    sheet = Image.new("RGB", (args.width * args.cols, th * rows), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % args.cols) * args.width, (i // args.cols) * th))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=86)
    print(f"wrote {args.out} ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
