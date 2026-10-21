"""Train the vision policy's pose networks (vision_policy.py) on the camera data (vision_data.py).

    tools\\il\\venv\\Scripts\\python.exe scripts\\rl\\train_vision.py [scene|hand|both] [--steps 15000]

scene  the holder's pose and the controller's settled pose, from the snapshot frames (snap_*.pt)
hand   the controller's pose relative to the forearm, from the in-hand frames (hand_*.pt), given the forearm

Augmented on the GPU: colour gain, contrast, tint and noise; depth noise, dropped pixels and missing patches
(as a phone's LiDAR or a RealSense); a +-1 px image shift (camera calibration error). Held out: the last
snapshot batches, and a share of the cells of every in-hand episode (whole episodes, not neighbouring
frames). Reported against the truth, beside the error of always predicting the average (what vision adds).
Writes output/rl/vision/{scene_net,hand_net}.pt and train_vision.json. No Isaac: plain torch.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import hand_teacher as T  # noqa: E402
import vision_policy as VP  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("which", nargs="?", default="both", choices=["scene", "hand", "both"])
parser.add_argument("--data", default=os.path.join(T.ROOT, "output", "rl", "vision", "data"))
parser.add_argument("--out", default=os.path.join(T.ROOT, "output", "rl", "vision"))
parser.add_argument("--steps", type=int, default=15000)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--val_snaps", type=int, default=3, help="snapshot batches held out")
parser.add_argument("--val_cells", type=float, default=0.1, help="share of each in-hand episode's cells held out")
parser.add_argument("--no_aug", action="store_true")
parser.add_argument("--strong", type=float, default=0.0,
                    help="share of samples also given the stress test's sensor noise (vision_policy.corrupt)")
parser.add_argument("--tag", default="", help="suffix for the saved networks (scene_net<tag>.pt, hand_net<tag>.pt)")
args = parser.parse_args()

DEV = torch.device("cuda")
REL = torch.tensor(T.Spec().rel, dtype=torch.float32, device=DEV)


def load(pattern: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(args.data, pattern)))
    if not files:
        raise SystemExit(f"no data: {os.path.join(args.data, pattern)}")
    return [torch.load(f, weights_only=False) for f in files]


def cat(chunks, key):
    return torch.cat([c[key] for c in chunks])


# --- augmentation -----------------------------------------------------------------------------------------

def augment(img: torch.Tensor) -> torch.Tensor:
    """(n, V, 4, H, W) colour in [0, 1], depth in m."""
    if args.no_aug:
        return img
    n, v, _, h, w = img.shape
    rgb, dep = img[:, :, :3], img[:, :, 3:4]
    g = torch.empty((n, v, 1, 1, 1), device=DEV).uniform_(0.7, 1.3)
    tint = torch.empty((n, v, 3, 1, 1), device=DEV).uniform_(0.9, 1.1)
    mean = rgb.mean((2, 3, 4), keepdim=True)
    con = torch.empty((n, v, 1, 1, 1), device=DEV).uniform_(0.75, 1.25)
    rgb = ((rgb - mean) * con + mean) * g * tint + 0.02 * torch.randn_like(rgb)
    rgb = rgb.clamp(0, 1)
    valid = dep > 0
    dep = dep + (0.002 + 0.004 * torch.rand((n, v, 1, 1, 1), device=DEV)) * torch.randn_like(dep)
    drop = torch.rand_like(dep) < 0.02 * torch.rand((n, v, 1, 1, 1), device=DEV)
    # missing patches (dark or shiny surfaces, edges): a few rectangles per view
    holes = torch.zeros_like(dep, dtype=torch.bool)
    for _ in range(3):
        cy = torch.randint(0, h, (n, v), device=DEV)
        cx = torch.randint(0, w, (n, v), device=DEV)
        ry = torch.randint(0, 10, (n, v), device=DEV)
        rx = torch.randint(0, 14, (n, v), device=DEV)
        yy = torch.arange(h, device=DEV)[None, None, :, None]
        xx = torch.arange(w, device=DEV)[None, None, None, :]
        m = ((yy - cy[..., None, None]).abs() <= ry[..., None, None]) & ((xx - cx[..., None, None]).abs() <= rx[..., None, None])
        holes |= (m & (torch.rand((n, v, 1, 1), device=DEV) < 0.5))[:, :, None]
    dep = torch.where(valid & ~drop & ~holes, dep, torch.zeros_like(dep))
    out = torch.cat([rgb, dep], 2)
    if args.strong > 0:  # some samples as noisy as the stress test (and noisier: a random level up to 1.5x)
        k = torch.rand(n, device=DEV) < args.strong
        if k.any():
            s = float(np.random.uniform(0.5, 1.5))
            out[k] = VP.corrupt(out[k], depth_sigma=0.006 * s, drop=0.05 * s, holes=int(round(4 * s)), rgb_sigma=0.04 * s)
    # +-1 px shift per view (calibration error), edge-padded
    sx, sy = np.random.randint(-1, 2), np.random.randint(-1, 2)
    if sx or sy:
        out = F.pad(out.flatten(0, 1), (1, 1, 1, 1), mode="replicate")[:, :, 1 + sy:1 + sy + h, 1 + sx:1 + sx + w].view(n, v, 4, h, w)
    return out


# --- training -------------------------------------------------------------------------------------------------

def batches(rgb, dep, idx, bs):
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield j, VP.from_storage(rgb[j].to(DEV, non_blocking=True), dep[j].to(DEV, non_blocking=True))


def fit(net, rgb, dep, y, extra, train_idx, log, kind):
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    yn = (y - net.out_mean.cpu()) / net.out_std.cpu()
    ckpt = os.path.join(args.out, f"{kind}{args.tag}_resume.pt")
    start = 0
    if os.path.exists(ckpt):  # a paused run: pick up where it stopped
        ck = torch.load(ckpt, map_location=DEV, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start = ck["step"]
        print(f"[vision] resuming {kind} at step {start}", flush=True)
    net.train()
    t0, losses = time.time(), []
    for s in range(start, args.steps):
        j = train_idx[torch.randint(0, len(train_idx), (args.batch,))]
        img = augment(VP.from_storage(rgb[j].to(DEV, non_blocking=True), dep[j].to(DEV, non_blocking=True)))
        ex = extra[j].to(DEV) if extra is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net.forward_norm(VP.normalize(img), ex)
        loss = F.smooth_l1_loss(pred.float(), yn[j].to(DEV), beta=0.5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        losses.append(float(loss))
        if s % 1000 == 0 or s == args.steps - 1:
            print(f"[vision] step {s}: loss {np.mean(losses[-200:]):.4f} ({time.time() - t0:.0f} s)", flush=True)
        if (s + 1) % 1000 == 0 and s + 1 < args.steps:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": s + 1},
                       ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    net.eval()
    log.update(steps=args.steps, train_samples=len(train_idx), loss_last=float(np.mean(losses[-500:])),
               train_min=(time.time() - t0) / 60)


@torch.no_grad()
def predict(net, rgb, dep, extra, idx, noisy=False):
    out = []
    torch.manual_seed(0)
    for j, img in batches(rgb, dep, idx, 256):
        if noisy:
            img = VP.corrupt(img)
        out.append(net(img, extra[j].to(DEV) if extra is not None else None).float())
    return torch.cat(out)


def stats(mm, deg) -> dict:
    return {"mm_median": float(mm.median()), "mm_p95": float(mm.quantile(0.95)),
            "deg_median": float(deg.median()), "deg_p95": float(deg.quantile(0.95))}


def train_scene(log):
    chunks = load("snap_*.pt")
    rgb, dep = cat(chunks, "rgb"), cat(chunks, "depth")
    holder7, ctrl7 = cat(chunks, "holder7"), cat(chunks, "ctrl7")
    y = torch.cat([VP.holder4(holder7), VP.pose9(ctrl7)], -1)
    sizes = [len(c["ctrl7"]) for c in chunks]
    n_val = sum(sizes[-args.val_snaps:])
    idx = torch.arange(len(y))
    train_idx, val_idx = idx[:-n_val], idx[-n_val:]
    net = VP.PoseNet(VP.SCENE_OUT).to(DEV)
    net.set_stats(y[train_idx])
    net.out_std.clamp_(min=0.005)
    print(f"[vision] scene: {len(train_idx)} train, {len(val_idx)} val", flush=True)
    fit(net, rgb, dep, y, None, train_idx, log, "scene")
    out = predict(net, rgb, dep, None, val_idx)
    yv = y[val_idx].to(DEV)
    t_true = VP.holder_target7(yv[:, :4], REL)
    res = {}
    for name, o in (("vision", out), ("average", net.out_mean[None].expand(len(out), -1))):
        hxy = (o[:, :2] - yv[:, :2]).norm(dim=-1) * 1000
        hyaw = torch.rad2deg((torch.atan2(o[:, 3], o[:, 2]) - torch.atan2(yv[:, 3], yv[:, 2] )).remainder(2 * np.pi))
        hyaw = torch.minimum(hyaw, 360 - hyaw)
        tgt = stats(*VP.pose_errors(VP.holder_target7(o[:, :4], REL), t_true))
        st = stats(*VP.pose_errors(VP.from9(o[:, 4:13]), VP.from9(yv[:, 4:13])))
        res[name] = {"holder_xy_mm_median": float(hxy.median()), "holder_yaw_deg_median": float(hyaw.median()),
                     "target": tgt, "start": st}
    log["val"] = res
    VP.save_net(net, os.path.join(args.out, f"scene_net{args.tag}.pt"), {"kind": "scene", "val": res})
    print(f"[vision] scene val: {json.dumps(res)}", flush=True)


def train_hand(log):
    chunks = load("hand_*.pt")
    rgb, dep = cat(chunks, "rgb"), cat(chunks, "depth")
    c7, f7 = cat(chunks, "ctrl7"), cat(chunks, "forearm7")
    y = VP.pose9(VP.relative7(c7, f7))
    extra = VP.pose9(f7)
    g = torch.Generator().manual_seed(0)
    val_mask = torch.cat([(torch.rand(4096, generator=g) < args.val_cells)[c["env"]] for c in chunks])
    idx = torch.arange(len(y))
    train_idx, val_idx = idx[~val_mask], idx[val_mask]
    net = VP.PoseNet(VP.HAND_OUT, VP.HAND_EXTRA).to(DEV)
    net.set_stats(y[train_idx], extra[train_idx])
    net.out_std.clamp_(min=0.005)
    print(f"[vision] hand: {len(train_idx)} train, {len(val_idx)} val", flush=True)
    fit(net, rgb, dep, y, extra, train_idx, log, "hand")
    out = predict(net, rgb, dep, extra, val_idx)
    noisy = predict(net, rgb, dep, extra, val_idx, noisy=True)
    fv = f7[val_idx].to(DEV)
    true7 = c7[val_idx].to(DEV)
    res = {}
    for name, o in (("vision", out), ("vision_noisy_sensor", noisy), ("average", net.out_mean[None].expand(len(out), -1))):
        res[name] = stats(*VP.pose_errors(VP.compose7(fv, VP.from9(o)), true7))
    log["val"] = res
    VP.save_net(net, os.path.join(args.out, f"hand_net{args.tag}.pt"), {"kind": "hand", "val": res})
    print(f"[vision] hand val: {json.dumps(res)}", flush=True)


def main():
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "train_vision.json")
    log = json.load(open(path)) if os.path.exists(path) else {}
    for which, fn in (("scene", train_scene), ("hand", train_hand)):
        if args.which in (which, "both"):
            log[which + args.tag] = {"args": vars(args)}
            fn(log[which + args.tag])
            json.dump(log, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
