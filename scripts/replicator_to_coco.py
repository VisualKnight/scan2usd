"""Convert an Isaac Sim Replicator dataset (BasicWriter) to COCO instance segmentation.

    python scripts/replicator_to_coco.py [output/isaac/dataset] [-o output/isaac/coco.json]

One COCO "instance" per object prim: multi-part assets (e.g. a box and its lid) are
merged by their asset root, so each scanned object is one annotation with a polygon
mask, a bounding box and its area. The result loads in pycocotools / Detectron2 /
torchvision-style detection pipelines.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def asset_root(prim_path: str) -> str:
    """'/World/obj_07_keepsake_box/lid/render' -> '/World/obj_07_keepsake_box'."""
    parts = prim_path.strip("/").split("/")
    return "/" + "/".join(parts[:2]) if len(parts) >= 2 else prim_path


def mask_to_polygons(mask: np.ndarray, min_area: float = 4.0) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) >= min_area and len(c) >= 3:
            polys.append([float(v) for v in c.reshape(-1)])
    return polys


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-9, a[2] * a[3] + b[2] * b[3] - inter)


def _annotator_boxes(dataset: Path, idx: str) -> dict[str, list[float]] | None:
    """Tight 2D boxes per class from Replicator's bbox annotator (parts of one asset are merged)."""
    npy = dataset / f"bounding_box_2d_tight_{idx}.npy"
    if not npy.exists():
        return None
    labels = json.loads((dataset / f"bounding_box_2d_tight_labels_{idx}.json").read_text())
    boxes: dict[str, list[float]] = {}
    for b in np.load(npy):
        name = labels[str(int(b["semanticId"]))]["class"]
        x0, y0, x1, y1 = float(b["x_min"]), float(b["y_min"]), float(b["x_max"]) + 1, float(b["y_max"]) + 1
        if name in boxes:
            p = boxes[name]
            x0, y0, x1, y1 = min(p[0], x0), min(p[1], y0), max(p[0] + p[2], x1), max(p[1] + p[3], y1)
        boxes[name] = [x0, y0, x1 - x0, y1 - y0]
    return boxes


def convert(dataset: Path) -> dict:
    frames = sorted(dataset.glob("rgb_*.png"))
    categories: dict[str, int] = {}
    images, annotations = [], []
    for img_id, rgb_path in enumerate(frames):
        idx = re.search(r"_(\d+)\.png$", rgb_path.name).group(1)
        inst = np.asarray(Image.open(dataset / f"instance_segmentation_{idx}.png"))
        paths = json.loads((dataset / f"instance_segmentation_mapping_{idx}.json").read_text())
        classes = json.loads((dataset / f"instance_segmentation_semantics_mapping_{idx}.json").read_text())
        h, w = inst.shape[:2]
        images.append({"id": img_id, "file_name": rgb_path.name, "width": w, "height": h})
        groups: dict[str, tuple[str, np.ndarray]] = {}
        for key, path in paths.items():
            name = classes.get(key, {}).get("class", "")
            if not name or name in ("BACKGROUND", "UNLABELLED"):
                continue
            m = inst == int(key)
            if not m.any():
                continue
            root = asset_root(path) if path else f"id{key}"
            prev = groups.get(root)
            groups[root] = (name, m if prev is None else (prev[1] | m))
        masks = {}
        for root, (name, m) in groups.items():
            ys, xs = np.nonzero(m)
            box = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
            masks[name] = (root, m, box)

        # Boxes: the bbox annotator is authoritative (instance masks occasionally drop an object);
        # masks are attached only where they agree with the box.
        boxes = _annotator_boxes(dataset, idx)
        if boxes is None:
            boxes = {name: box for name, (_, _, box) in masks.items()}
        for name, box in boxes.items():
            if box[2] < 2 or box[3] < 2:
                continue
            cat = categories.setdefault(name, len(categories) + 1)
            seg, area, prim = [], box[2] * box[3], None
            if name in masks and _box_iou(masks[name][2], box) > 0.7:
                prim, m, _ = masks[name]
                seg, area = mask_to_polygons(m), float(m.sum())
            annotations.append({
                "id": len(annotations) + 1, "image_id": img_id, "category_id": cat, "bbox": box,
                "area": area, "segmentation": seg, "iscrowd": 0, "attributes": {"prim": prim},
            })
    return {
        "info": {"description": "Scan2USD assets rendered with Isaac Sim Replicator", "version": "1.0"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n, "supercategory": "object"} for n, i in categories.items()],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path, nargs="?", default=ROOT / "output" / "isaac" / "dataset")
    p.add_argument("-o", "--output", type=Path)
    args = p.parse_args()
    coco = convert(args.dataset)
    out = args.output or args.dataset.parent / "coco.json"
    out.write_text(json.dumps(coco))
    per_img = len(coco["annotations"]) / max(1, len(coco["images"]))
    print(f"wrote {out}: {len(coco['images'])} images, {len(coco['annotations'])} instances "
          f"({per_img:.1f}/image), {len(coco['categories'])} categories")


if __name__ == "__main__":
    main()
