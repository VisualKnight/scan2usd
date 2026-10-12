"""Perception-in-the-loop pick: detect the scanned mug, estimate its pose, then pick it.

    python scripts/demo_perception_pick.py [--trials 5] [--out docs/figures]

Closes the loop from scan to robot behavior, using only Scan2USD outputs:

1. The scanned mug is dropped at a random position and yaw among other scanned objects.
2. Two fixed scene cameras render RGB + depth (MuJoCo).
3. The detector trained only on Isaac Sim synthetic data (scripts/train_detector.py) finds the mug.
4. Depth pixels inside the box are back-projected to 3D, and the scanned mesh is fitted to them
   (yaw grid search + translation ICP, mug upright on the table): that's the pose estimate.
5. The Franka Panda (scripts/demo_panda.py) rim-grasps at the *estimated* pose and places the mug.

Reports detection, pose error (mm, degrees) and pick success per trial.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_panda import HOME, MUG_POS, PANDA, PLACE_SHIFT, Arm, build_scene, execute, plan_pick, rim_grasp  # noqa: E402
from train_detector import build_model  # noqa: E402

from scan2usd.mjcf import asset_body  # noqa: E402
from scan2usd.preview import _font  # noqa: E402
from scan2usd.usd_io import read_usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIZE = (720, 960)  # (h, w)
CAMERAS = [  # (lookat, distance, elevation, azimuth): two fixed cell cameras ~90 degrees apart
    ((0.5, -0.1, 0.0), 0.8, -45, 150),
    ((0.5, -0.1, 0.0), 0.8, -45, 240),
]


def yaw_quat(yaw: float):
    return (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def free_camera(spec) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.elevation, cam.azimuth = spec
    return cam


def observe(model, data, renderer, cam):
    """RGB, depth (m) and world-space points per pixel for one camera."""
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render().copy()
    # Mono rendering uses the average of the two (stereo) eye cameras, which sit ±ipd/2 apart.
    gl = mujoco.mjv_averageCamera(renderer.scene.camera[0], renderer.scene.camera[1])
    pos, fwd, up = np.array(gl.pos), np.array(gl.forward), np.array(gl.up)
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    h, w = depth.shape
    f = 0.5 * h / np.tan(np.radians(model.vis.global_.fovy) / 2)
    right = np.cross(fwd, up)
    u, v = np.meshgrid(np.arange(w) + 0.5 - w / 2, np.arange(h) + 0.5 - h / 2)
    rays = fwd[None, None] + (u / f)[..., None] * right + (-v / f)[..., None] * up
    pts = pos + depth[..., None] * rays
    return rgb, depth, pts


def _icp_xy(tree, observed, t0, iters):
    t = t0.copy()
    for _ in range(iters):
        d, idx = tree.query(observed - t)
        keep = d < np.percentile(d, 90)
        delta = np.mean((observed - t)[keep] - tree.data[idx[keep]], axis=0)
        delta[2] = 0.0
        t += delta
        if np.linalg.norm(delta) < 5e-5:
            break
    d, _ = tree.query(observed - t)
    return t, float(np.mean(np.minimum(d, 0.01)))


def fit_pose(observed: np.ndarray, model_pts: np.ndarray, init_xy: np.ndarray | None = None):
    """Upright mug: yaw grid search, translation-only ICP in xy. Returns (x, y, yaw, residual m).

    Each yaw starts with the model's median aligned to the observed median, so the origin
    offset from the cup axis (the handle shifts the bounding-box center) doesn't bias the search.
    """
    o_med = np.median(observed, axis=0)
    best = None
    for yaw in np.radians(np.arange(0, 360, 5)):
        m = model_pts @ rot_z(yaw).T
        tree = cKDTree(m)
        t0 = o_med - np.median(m, axis=0)
        t0[2] = 0.0
        t, score = _icp_xy(tree, observed, t0, 30)
        if best is None or score < best[3]:
            best = (t[0], t[1], yaw, score)
    x, y, yaw0, _ = best
    for yaw in yaw0 + np.radians(np.arange(-4, 5, 1)):
        tree = cKDTree(model_pts @ rot_z(yaw).T)
        t, score = _icp_xy(tree, observed, np.array([x, y, 0.0]), 30)
        if score < best[3]:
            best = (t[0], t[1], yaw, score)
    return best


def detect(det_model, cats, rgb, device, target="ceramic_mug", min_score=0.5):
    from torchvision.transforms import functional as F
    with torch.no_grad():
        out = det_model([F.to_tensor(rgb).to(device)])[0]
    boxes, scores, labels = (out[k].cpu().numpy() for k in ("boxes", "scores", "labels"))
    dets = [(cats[int(l)], float(s), b) for b, s, l in zip(boxes, scores, labels) if s >= min_score]
    mug = [d for d in dets if d[0] == target]
    return (max(mug, key=lambda d: d[1]) if mug else None), dets


def trial(seed, asset, others, det_model, cats, device, render_video=False):
    rng = np.random.default_rng(seed)
    mug_xy = MUG_POS[:2] + rng.uniform([-0.06, -0.06], [0.06, 0.04])
    mug_yaw = rng.uniform(-np.pi, np.pi)
    mug_pos = np.array([mug_xy[0], mug_xy[1], 0.0])
    extras, placed = [], [mug_xy]
    for k, other in enumerate(others):
        for _ in range(200):
            xy = np.array([rng.uniform(0.3, 0.72), rng.uniform(-0.34, 0.0)])
            # Clear of the mug and each other, and out of the carry corridor (+y of the mug).
            in_corridor = abs(xy[0] - mug_xy[0]) < 0.14 and xy[1] > mug_xy[1]
            if not in_corridor and all(np.linalg.norm(xy - p) > 0.17 for p in placed):
                break
        placed.append(xy)
        extras.append(asset_body(other, f"obj{k}", pos=(xy[0], xy[1], 0.0),
                                 quat=yaw_quat(rng.uniform(-np.pi, np.pi)), collision="hulls"))
    work = ROOT / "data" / "work" / "perception_pick"
    model = build_scene(asset, "hulls", work, mug_pos=mug_pos, mug_quat=yaw_quat(mug_yaw), extras=extras)
    arm = Arm(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for fj in ("finger_joint1", "finger_joint2"):
        data.qpos[model.joint(fj).qposadr[0]] = 0.022
    for _ in range(300):  # let everything settle on the floor
        data.ctrl[:7] = HOME
        data.qfrc_applied[arm.dadr] = data.qfrc_bias[arm.dadr]
        mujoco.mj_step(model, data)
    mug = model.body("mug").id
    true_xy = data.xpos[mug][:2].copy()
    r = data.xmat[mug].reshape(3, 3)
    true_yaw = float(np.arctan2(r[1, 0], r[0, 0]))

    # --- perception -------------------------------------------------------------------
    renderer = mujoco.Renderer(model, *SIZE)
    observed, views = [], []
    for spec in CAMERAS:
        rgb, depth, pts = observe(model, data, renderer, free_camera(spec))
        best, dets = detect(det_model, cats, rgb, device)
        views.append((rgb, best, dets))
        if best is None:
            continue
        x0, y0, x1, y1 = (int(round(c)) for c in best[2])
        p = pts[y0:y1, x0:x1].reshape(-1, 3)
        p = p[(p[:, 2] > 0.004) & (p[:, 2] < 0.3)]  # above the table, below the arm
        observed.append(p)
    result = {"seed": seed, "detected_views": sum(v[1] is not None for v in views)}
    if not observed or sum(len(o) for o in observed) < 50:
        result.update(success=False, reason="mug not detected")
        return result, views, None
    obs = np.concatenate(observed)
    # Keep the biggest cluster near the median (a box can include a neighbor's pixels).
    med = np.median(obs, axis=0)
    obs = obs[np.linalg.norm((obs - med)[:, :2], axis=1) < 0.09]
    if len(obs) > 6000:
        obs = obs[np.random.default_rng(0).choice(len(obs), 6000, replace=False)]
    mesh = asset.render_mesh
    model_pts = np.asarray(mesh.sample(6000, seed=0)) if hasattr(mesh, "sample") else mesh.vertices
    ex, ey, eyaw, resid = fit_pose(obs, model_pts, med[:2])
    yaw_err = float(np.degrees(np.angle(np.exp(1j * (eyaw - true_yaw)))))
    result.update(pos_error_mm=float(np.hypot(ex - true_xy[0], ey - true_xy[1]) * 1000),
                  yaw_error_deg=abs(yaw_err), fit_residual_mm=resid * 1000, points=int(len(obs)))

    # --- act on the estimate -------------------------------------------------------------
    est_pos = np.array([ex, ey, 0.0])
    est_mesh = mesh.copy()
    est_mesh.vertices = est_mesh.vertices @ rot_z(eyaw).T
    grasp, radial = rim_grasp(est_mesh, est_pos)
    plan = plan_pick(model, arm, grasp, radial)
    data.time = 0.0
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.48, 0.0, 0.16)
    cam.distance, cam.elevation, cam.azimuth = 1.15, -22, 205
    small = mujoco.Renderer(model, 480, 640) if render_video else None

    def render(d):
        small.update_scene(d, camera=cam)
        return small.render().copy()

    frames, lifted = execute(model, arm, data, plan, render if render_video else None)
    target = true_xy + PLACE_SHIFT[:2]
    final = data.xpos[mug]
    up_axis = data.xmat[mug].reshape(3, 3)[:, 2]
    lift = lifted - asset.center_of_mass[2]
    placed_err = float(np.linalg.norm(final[:2] - target))
    upright = float(np.degrees(np.arccos(np.clip(up_axis[2], -1, 1))))
    result.update(lifted_m=lift, place_error_mm=placed_err * 1000, upright_deg=upright,
                  success=bool(lift > 0.1 and placed_err < 0.03 and upright < 10))
    return result, views, frames


def detection_panel(views, cats_seen=None, width=480):
    tiles = []
    for rgb, best, dets in views:
        img = Image.fromarray(rgb)
        d = ImageDraw.Draw(img)
        for name, s, b in dets:
            ok = best is not None and name == best[0] and s == best[1]
            color = (21, 128, 61) if ok else (100, 116, 139)
            d.rectangle(list(b), outline=color, width=5 if ok else 3)
            d.text((b[0] + 6, b[1] - 28), f"{name} {s:.2f}", fill=color, font=_font(22))
        s = width / img.width
        tiles.append(img.resize((width, round(img.height * s))))
    return tiles


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asset", type=Path, default=ROOT / "output" / "demo" / "asset" / "ceramic_mug.usda")
    p.add_argument("--detector", type=Path, default=ROOT / "output" / "detector_dr")
    p.add_argument("--data", type=Path, default=ROOT / "output" / "synth_dr")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--video-seed", type=int, default=1, help="trial to animate")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = p.parse_args()
    if not PANDA.exists():
        raise SystemExit(f"{PANDA} not found; fetch mujoco_menagerie/franka_emika_panda first (see README)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = json.loads((args.data / "coco.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    det_model = build_model(len(cats))
    det_model.load_state_dict(torch.load(args.detector / "fasterrcnn_scan2usd.pt", map_location=device))
    det_model.to(device).eval()

    asset = read_usd(args.asset)
    gso = {Path(q).stem: q for q in glob.glob(str(ROOT / "output" / "gso_assets" / "*" / "*.usda"))}
    others = [read_usd(gso[n]) for n in ("schleich_lion_action_figure", "great_dinos_triceratops_toy",
                                          "cole_hardware_bowl_scirocco_yellowblue") if n in gso]

    results, first = [], None
    for seed in range(args.trials):
        res, views, frames = trial(seed, asset, others, det_model, cats, device, render_video=(seed == args.video_seed))
        results.append(res)
        print(json.dumps(res), flush=True)
        if seed == args.video_seed:
            first = (views, frames, res)

    views, frames, res = first
    args.out.mkdir(parents=True, exist_ok=True)
    panel = detection_panel(views)
    ph = sum(t.height for t in panel)
    if frames:
        out = []
        label = (f"detected → pose error {res.get('pos_error_mm', float('nan')):.0f} mm, "
                 f"{res.get('yaw_error_deg', float('nan')):.0f}° → "
                 + ("picked and placed" if res["success"] else "failed"))
        for i, fr in enumerate(frames):
            main_img = Image.fromarray(fr).resize((round(640 * ph / 480), ph))
            sheet = Image.new("RGB", (panel[0].width + main_img.width, ph), "white")
            y = 0
            for t in panel:
                sheet.paste(t, (0, y))
                y += t.height
            sheet.paste(main_img, (panel[0].width, 0))
            d = ImageDraw.Draw(sheet)
            d.rectangle([0, 0, panel[0].width, 30], fill=(255, 255, 255))
            d.text((10, 5), "Scene cameras + detector (synthetic-trained)", fill=(30, 41, 59), font=_font(17))
            d.rectangle([panel[0].width, 0, sheet.width, 30], fill=(255, 255, 255))
            d.text((panel[0].width + 10, 5), label if i > len(frames) * 0.25 else "Panda acts on the estimate",
                   fill=(21, 128, 61) if res["success"] else (185, 28, 28), font=_font(17))
            out.append(sheet)
        out[0].save(args.out / "perception_pick.webp", save_all=True, append_images=out[1::2], duration=66,
                    loop=0, quality=60, method=4)
        out[int(len(out) * 0.5)].save(args.out / "perception_pick.jpg", quality=86)
    ok = [r for r in results if "pos_error_mm" in r]
    summary = {
        "trials": len(results), "successes": sum(r["success"] for r in results),
        "detected": len(ok),
        "median_pos_error_mm": float(np.median([r["pos_error_mm"] for r in ok])) if ok else None,
        "median_yaw_error_deg": float(np.median([r["yaw_error_deg"] for r in ok])) if ok else None,
        "results": results,
    }
    (args.out / "perception_pick.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
