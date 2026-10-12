"""Train an object detector on Scan2USD synthetic data, then test it on real photos.

    python scripts/train_detector.py [--data output/synth] [--epochs 6]

Faster R-CNN (ResNet-50 FPN v2, COCO-pretrained backbone) fine-tuned on the Isaac Sim
Replicator dataset (scripts/isaac_synthdata.py -> scripts/replicator_to_coco.py).

Evaluation:
  * synthetic held-out split: mAP@0.5 (VOC-style, per class, averaged)
  * sim-to-real: the 67 real AliceVision Buddha photos, which the detector never saw. The
    "buddha_head" asset was built from those photos, but the detector only saw renders of
    it in random scenes. Reports how often the top detection is the Buddha and how well
    it covers the object.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parents[1]


class CocoBoxes(torch.utils.data.Dataset):
    def __init__(self, root: Path, coco: dict, image_ids: list[int], train: bool):
        self.root, self.train = root, train
        self.images = {im["id"]: im for im in coco["images"]}
        self.ids = image_ids
        self.anns: dict[int, list] = {i: [] for i in image_ids}
        for a in coco["annotations"]:
            if a["image_id"] in self.anns:
                self.anns[a["image_id"]].append(a)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        im = self.images[self.ids[k]]
        img = Image.open(self.root / im["file_name"]).convert("RGB")
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                 for a in self.anns[self.ids[k]]]
        labels = [a["category_id"] for a in self.anns[self.ids[k]]]
        t = F.to_tensor(img)
        b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.train:
            if random.random() < 0.5:  # horizontal flip
                t = t.flip(-1)
                b[:, [0, 2]] = t.shape[-1] - b[:, [2, 0]]
            t = F.adjust_brightness(t, random.uniform(0.7, 1.3))
            t = F.adjust_contrast(t, random.uniform(0.7, 1.3))
            t = F.adjust_saturation(t, random.uniform(0.7, 1.3))
            if random.random() < 0.3:
                t = F.gaussian_blur(t, kernel_size=5, sigma=random.uniform(0.3, 1.5))
            t = (t + torch.randn_like(t) * random.uniform(0.0, 0.02)).clamp(0, 1)
        return t, {"boxes": b, "labels": torch.tensor(labels, dtype=torch.int64)}


def build_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def average_precision(dets, gts, thr=0.5) -> float:
    """dets: [(image, score, box)], gts: {image: [box]} for one class."""
    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        return float("nan")
    used = {k: [False] * len(v) for k, v in gts.items()}
    tp = []
    for img, _, box in sorted(dets, key=lambda d: -d[1]):
        best, j = 0.0, -1
        for k, g in enumerate(gts.get(img, [])):
            o = iou(box, g)
            if o > best:
                best, j = o, k
        if best >= thr and not used[img][j]:
            used[img][j] = True
            tp.append(1)
        else:
            tp.append(0)
    tp = np.array(tp)
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    rec = ctp / npos
    prec = ctp / np.arange(1, len(tp) + 1)
    # All-point interpolation.
    mrec = np.r_[0, rec, 1]
    mpre = np.r_[0, prec, 0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


@torch.no_grad()
def predict(model, img: Image.Image, device):
    out = model([F.to_tensor(img).to(device)])[0]
    return out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "output" / "synth")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--val", type=int, default=100)
    p.add_argument("--real", type=Path, default=ROOT / "data" / "external" / "buddha" / "buddha")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "detector")
    p.add_argument("--tag", default="", help="suffix for docs/figures/sim2real_detection{tag}.*")
    p.add_argument("--eval-only", action="store_true", help="load OUT/fasterrcnn_scan2usd.pt instead of training")
    args = p.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    coco = json.loads((args.data / "coco.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    ids = [im["id"] for im in coco["images"]]
    random.shuffle(ids)
    val_ids, train_ids = sorted(ids[: args.val]), ids[args.val:]
    train = CocoBoxes(args.data, coco, train_ids, train=True)
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0,
                                         collate_fn=lambda b: tuple(zip(*b)))

    model = build_model(len(cats)).to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    t0 = time.time()
    model.train()
    for epoch in range(0 if args.eval_only else args.epochs):
        total = 0.0
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = sum(model(imgs, targets).values())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss)
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / len(loader):.3f} ({time.time() - t0:.0f} s)", flush=True)
    if args.eval_only:
        model.load_state_dict(torch.load(args.out / "fasterrcnn_scan2usd.pt", map_location=device))
    else:
        torch.save(model.state_dict(), args.out / "fasterrcnn_scan2usd.pt")

    # --- synthetic held-out mAP ----------------------------------------------------------
    model.eval()
    val = CocoBoxes(args.data, coco, val_ids, train=False)
    dets = {c: [] for c in cats}
    gts = {c: {} for c in cats}
    for k in range(len(val)):
        img_t, tgt = val[k]
        for b, l in zip(tgt["boxes"].tolist(), tgt["labels"].tolist()):
            gts[l].setdefault(k, []).append(b)
        with torch.no_grad():
            out = model([img_t.to(device)])[0]
        for b, s, l in zip(out["boxes"].cpu().tolist(), out["scores"].cpu().tolist(), out["labels"].cpu().tolist()):
            dets[l].append((k, s, b))
    aps = {cats[c]: average_precision(dets[c], gts[c]) for c in cats}
    aps75 = {cats[c]: average_precision(dets[c], gts[c], 0.75) for c in cats}
    m_ap = float(np.nanmean(list(aps.values())))
    m_ap75 = float(np.nanmean(list(aps75.values())))
    print(f"synthetic val mAP@0.5: {m_ap:.3f}, mAP@0.75: {m_ap75:.3f}")

    # --- sim-to-real: real Buddha photos (the only known object in them) ----------------
    buddha_id = next(c for c, n in cats.items() if n == "buddha_head")
    photos = sorted(q for q in args.real.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg"))
    shown = {photos[i].name for i in range(0, len(photos), max(1, len(photos) // 6))}
    hits, top_scores, false_pos, examples = 0, [], [], []
    for q in photos:
        img = ImageOps.exif_transpose(Image.open(q)).convert("RGB")
        scale = 1024 / max(img.size)
        small = img.resize((round(img.width * scale), round(img.height * scale)))
        boxes, scores, labels = predict(model, small, device)
        b_idx = [i for i in range(len(scores)) if labels[i] == buddha_id]
        best = max(b_idx, key=lambda i: scores[i]) if b_idx else None
        top_is_buddha = len(scores) > 0 and labels[int(np.argmax(scores))] == buddha_id and scores.max() > 0.5
        hits += int(top_is_buddha)
        top_scores.append(float(scores[best]) if best is not None else 0.0)
        # Anything confidently detected other than the single best Buddha box is a false positive.
        false_pos.append(sum(1 for i in range(len(scores)) if scores[i] >= 0.5 and i != best))
        if len(examples) < 6 and q.name in shown:
            d = ImageDraw.Draw(small)
            for i in range(len(scores)):
                if scores[i] < 0.5:
                    continue
                x0, y0, x1, y1 = boxes[i]
                color = (21, 128, 61) if i == best else (220, 38, 38)
                d.rectangle([x0, y0, x1, y1], outline=color, width=5)
                d.text((x0 + 6, y0 + 4), f"{cats[int(labels[i])][:18]} {scores[i]:.2f}", fill=color)
            examples.append(small)
    rate = hits / len(photos)
    fp = float(np.mean(false_pos))
    print(f"real photos: top detection is the Buddha in {hits}/{len(photos)} ({rate * 100:.0f}%), "
          f"median Buddha score {np.median(top_scores):.2f}, false positives/photo {fp:.2f}")

    if examples:
        w, h = 512, round(512 * examples[0].height / examples[0].width)
        sheet = Image.new("RGB", (w * 3, h * 2), "white")
        for i, e in enumerate(examples[:6]):
            sheet.paste(e.resize((w, h)), ((i % 3) * w, (i // 3) * h))
        sheet.save(ROOT / "docs" / "figures" / f"sim2real_detection{args.tag}.jpg", quality=86)
    result = {"data": args.data.name, "train_images": len(train_ids), "val_images": len(val_ids),
              "epochs": args.epochs, "train_seconds": None if args.eval_only else round(time.time() - t0),
              "synthetic_val_map50": m_ap, "synthetic_val_map75": m_ap75,
              "per_class_ap50": aps, "real_buddha_photos": len(photos), "real_top1_hits": hits,
              "real_top1_rate": rate, "real_median_score": float(np.median(top_scores)),
              "real_false_positives_per_photo": fp}
    (args.out / "results.json").write_text(json.dumps(result, indent=2))
    (ROOT / "docs" / "figures" / f"sim2real_detection{args.tag}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
