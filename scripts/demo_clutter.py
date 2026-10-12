"""Robotics demo: drop a pile of Scan2USD assets into a bin in MuJoCo.

    python scripts/demo_clutter.py [--count 14] [--out docs/figures]

The classic stress test for simulation assets before bin-picking work: many objects with
many convex colliders, stacked contacts, and random poses. Reports whether the pile
settles stably (no explosions, everything ends in the bin, shallow penetration).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from scan2usd.mjcf import asset_body
from scan2usd.preview import _font
from scan2usd.usd_io import read_usd

ROOT = Path(__file__).resolve().parents[1]
BIN = (0.42, 0.32, 0.14)  # inner size (m)
WALL = 0.012


def find_assets() -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(str(ROOT / "output" / "gso_assets" / "*" / "*.usda")))
    demo = ROOT / "output" / "demo" / "asset" / "ceramic_mug.usda"
    if demo.exists():
        paths.append(demo)
    return paths


def bin_xml() -> str:
    lx, ly, h = BIN[0] / 2, BIN[1] / 2, BIN[2] / 2
    t = WALL / 2
    walls = [
        (0, 0, -t, lx + WALL, ly + WALL, t),
        (lx + t, 0, h, t, ly + WALL, h), (-lx - t, 0, h, t, ly + WALL, h),
        (0, ly + t, h, lx, t, h), (0, -ly - t, h, lx, t, h),
    ]
    return "".join(f'<geom type="box" pos="{x} {y} {z}" size="{a} {b} {c}" rgba="0.33 0.45 0.62 1" '
                   f'friction="0.6 0.005 0.0001"/>' for x, y, z, a, b, c in walls)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=14)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    library = [read_usd(pth) for pth in find_assets()]
    if not library:
        raise SystemExit("no assets found; build some into output/gso_assets/ first")

    bodies, files, assets_xml = [], {}, []
    drop_order = []
    for i in range(args.count):
        a = library[i % len(library)]
        name = f"obj{i:02d}"
        x, y = rng.uniform(-0.12, 0.12), rng.uniform(-0.08, 0.08)
        z = 0.12 + 0.11 * i
        quat = Rotation.random(random_state=rng.integers(1 << 31)).as_quat()[[3, 0, 1, 2]]
        b = asset_body(a, name, pos=(x, y, z), quat=quat)
        bodies.append(b.body_xml)
        assets_xml.append(b.assets_xml)
        files.update(b.files)
        drop_order.append((name, a.name))

    xml = f"""<mujoco model="clutter">
  <option timestep="0.001" integrator="implicitfast" cone="elliptic"><flag energy="enable"/></option>
  <visual><global offwidth="960" offheight="720"/><quality shadowsize="4096"/></visual>
  <asset>{''.join(assets_xml)}
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.80 0.84 0.90" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".82 .82 .80" rgb2=".74 .74 .72" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.6 1.5" dir="-0.3 0.4 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light pos="-0.6 0.6 1.2" dir="0.4 -0.4 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom type="plane" size="1.5 1.5 0.05" material="grid" pos="0 0 -0.0121"/>
    {bin_xml()}
    {''.join(bodies)}
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml, files)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 540, 720)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0, 0, 0.12)
    cam.distance, cam.elevation, cam.azimuth = 1.05, -38, 125
    ids = [model.body(n).id for n, _ in drop_order]

    duration, fps = 5.0, 30
    frames, max_pen, peak_speed = [], 0.0, 0.0
    steps = int(duration / model.opt.timestep)
    for step in range(steps):
        mujoco.mj_step(model, data)
        if data.ncon:
            max_pen = max(max_pen, float(-data.contact.dist[: data.ncon].min()))
        if step % int(1 / (fps * model.opt.timestep)) == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
    speeds = np.linalg.norm(data.cvel[ids][:, 3:], axis=1)
    pos = data.xpos[ids]
    inside = (np.abs(pos[:, 0]) < BIN[0] / 2) & (np.abs(pos[:, 1]) < BIN[1] / 2) & (pos[:, 2] < BIN[2] + 0.2)
    result = {
        "objects": len(ids),
        "unique_assets": len(library),
        "colliders": int(sum(1 for g in range(model.ngeom) if model.geom_contype[g] and model.geom_bodyid[g] in ids)),
        "finite": bool(np.isfinite(data.qpos).all()),
        "all_in_bin": bool(inside.all()),
        "in_bin": int(inside.sum()),
        "max_final_speed_mps": float(speeds.max()),
        "settled": bool(speeds.max() < 0.02),
        "max_penetration_mm": max_pen * 1000,
        "sim_seconds": duration,
    }
    print(json.dumps(result, indent=2))

    out = []
    for i, f in enumerate(frames):
        img = Image.fromarray(f)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, img.width, 36], fill=(255, 255, 255))
        d.text((12, 7), f"{len(ids)} scanned objects ({result['colliders']} convex colliders) · MuJoCo",
               fill=(30, 41, 59), font=_font(18))
        out.append(img)
    args.out.mkdir(parents=True, exist_ok=True)
    out[0].save(args.out / "clutter_demo.webp", save_all=True, append_images=out[1:], duration=33, loop=0,
                quality=70, method=4)
    out[-1].save(args.out / "clutter_demo.jpg", quality=88)
    (args.out / "clutter_demo.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
