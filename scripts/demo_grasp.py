"""Robotics demo: a parallel gripper rim-grasps a Scan2USD mug in MuJoCo.

    python scripts/demo_grasp.py [--asset output/demo/asset/ceramic_mug.usda] [--out docs/figures]

One finger goes inside the cup, one outside. Runs twice:
  (a) Scan2USD collision: convex decomposition, the cavity stays open
  (b) naive collision: one convex hull of the whole mug, which "lids" the cup
and writes a side-by-side animation plus metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from scan2usd.mjcf import asset_body
from scan2usd.preview import _font
from scan2usd.usd_io import read_usd

ROOT = Path(__file__).resolve().parents[1]
FINGER_HALF = (0.0025, 0.009, 0.03)  # finger pad half-sizes (m)
OPEN, GRIP_FORCE = 0.022, 25.0


def rim_grasp_pose(mesh):
    """Wall point opposite the handle, at the rim: (x, y, top_z, yaw)."""
    v = mesh.vertices
    top = v[:, 2].max()
    ring = v[v[:, 2] > top - 0.003]
    a = np.c_[2 * ring[:, 0], 2 * ring[:, 1], np.ones(len(ring))]
    cx, cy, c = np.linalg.lstsq(a, (ring[:, :2] ** 2).sum(1), rcond=None)[0]
    radii = np.hypot(ring[:, 0] - cx, ring[:, 1] - cy)
    r_mid = 0.5 * (np.percentile(radii, 5) + np.percentile(radii, 95))
    band = v[(v[:, 2] > 0.25 * top) & (v[:, 2] < 0.75 * top)]
    far = band[np.argmax(np.hypot(band[:, 0] - cx, band[:, 1] - cy))]
    handle_yaw = np.arctan2(far[1] - cy, far[0] - cx)
    yaw = handle_yaw + np.pi
    return cx + r_mid * np.cos(yaw), cy + r_mid * np.sin(yaw), top, yaw


def build_model(asset, collision: str):
    body = asset_body(asset, "mug", collision=collision)
    gx, gy, top, yaw = rim_grasp_pose(asset.render_mesh)
    hand_z = top + 0.10
    q = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    fx, fy, fz = FINGER_HALF
    finger = (f'<geom type="box" size="{fx} {fy} {fz}" pos="0 0 {-fz}" rgba="0.15 0.15 0.18 1" '
              f'friction="1.2 0.02 0.001" solref="0.005 1" condim="4"/>')
    xml = f"""<mujoco model="rim_grasp">
  <option timestep="0.001" integrator="implicitfast" cone="elliptic"/>
  <visual><global offwidth="960" offheight="720"/><quality shadowsize="4096"/></visual>
  <asset>{body.assets_xml}
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.80 0.84 0.90" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".82 .82 .80" rgb2=".74 .74 .72" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="12 12"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.5 1.2" dir="-0.3 0.5 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light pos="-0.5 0.5 1" dir="0.5 -0.5 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="table" type="plane" size="1 1 0.05" material="grid" friction="0.6 0.005 0.0001"/>
    {body.body_xml}
    <body name="target" mocap="true" pos="{gx} {gy} {hand_z}" quat="{q[0]} 0 0 {q[3]}"/>
    <body name="hand" pos="{gx} {gy} {hand_z}" quat="{q[0]} 0 0 {q[3]}">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.6" diaginertia="0.001 0.001 0.001"/>
      <geom type="box" size="0.035 0.014 0.01" rgba="0.55 0.57 0.62 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.012 0.03" pos="0 0 0.04" rgba="0.55 0.57 0.62 1" contype="0" conaffinity="0"/>
      <body name="finger_in" pos="{-OPEN} 0 -0.01"><joint name="j_in" type="slide" axis="1 0 0" range="-0.005 0.03"/>
        <inertial pos="0 0 -0.03" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>{finger}</body>
      <body name="finger_out" pos="{OPEN} 0 -0.01"><joint name="j_out" type="slide" axis="-1 0 0" range="-0.005 0.03"/>
        <inertial pos="0 0 -0.03" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>{finger}</body>
    </body>
  </worldbody>
  <equality><weld body1="hand" body2="target" solref="0.02 1"/></equality>
  <actuator>
    <position joint="j_in" kp="400" forcerange="-{GRIP_FORCE} {GRIP_FORCE}"/>
    <position joint="j_out" kp="400" forcerange="-{GRIP_FORCE} {GRIP_FORCE}"/>
  </actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml, body.files)
    return model, (gx, gy, top, yaw)


def run(asset, collision: str, fps: int = 30, size=(480, 640)):
    model, (gx, gy, top, yaw) = build_model(asset, collision)
    data = mujoco.MjData(model)
    mug = model.body("mug").id
    tid = model.body("target").mocapid[0]
    renderer = mujoco.Renderer(model, *size)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1  # show collision geometry (translucent)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (gx, gy, top * 0.5 + 0.07)
    cam.distance, cam.elevation, cam.azimuth = 0.62, -14, np.degrees(yaw) + 70

    grasp_z = top + 0.01 + 0.02  # finger tips 2 cm below the rim
    lift_z = grasp_z + 0.12
    start_z = top + 0.10
    phases = [(0.4, start_z, OPEN), (1.6, grasp_z, OPEN), (2.2, grasp_z, -0.01), (3.8, lift_z, -0.01), (4.6, lift_z, -0.01)]
    frames, t_prev, z_prev, w_prev = [], 0.0, start_z, OPEN
    z0 = data.xpos[mug][2] if data.time else None
    for t_end, z_goal, w_goal in phases:
        n = int((t_end - t_prev) / model.opt.timestep)
        for i in range(n):
            s = (i + 1) / n
            data.mocap_pos[tid][2] = z_prev + (z_goal - z_prev) * s
            target = OPEN - (w_prev + (w_goal - w_prev) * s)
            data.ctrl[:] = target
            mujoco.mj_step(model, data)
            if z0 is None and data.time > 0.3:
                z0 = float(data.xipos[mug][2])
            if int(data.time * fps) > len(frames) - 1:
                renderer.update_scene(data, camera=cam, scene_option=opt)
                frames.append(renderer.render().copy())
        t_prev, z_prev, w_prev = t_end, z_goal, w_goal
    lift = float(data.xipos[mug][2] - z0)
    return frames, {"collision": collision, "geoms": int(sum(1 for g in range(model.ngeom)
                                                                if model.geom_bodyid[g] == mug) - 1),
                    "mug_lift_m": lift, "success": lift > 0.08}


def label(frame, text, ok):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 34], fill=(255, 255, 255))
    d.text((12, 6), text, fill=(30, 41, 59), font=_font(18))
    return img, ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asset", type=Path, default=ROOT / "output" / "demo" / "asset" / "ceramic_mug.usda")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = p.parse_args()
    asset = read_usd(args.asset)
    results, videos = [], []
    for collision, title in (("hulls", "Scan2USD: convex decomposition"), ("single", "Naive: one convex hull")):
        frames, res = run(asset, collision)
        results.append(res)
        videos.append((frames, title))
        print(f"{title}: {res['geoms']} collider(s), mug lifted {res['mug_lift_m'] * 100:.1f} cm "
              f"-> {'SUCCESS' if res['success'] else 'FAIL'}")
    n = min(len(v[0]) for v in videos)
    out_frames = []
    for i in range(n):
        tiles = []
        for (frames, title), res in zip(videos, results):
            img = Image.fromarray(frames[i])
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, img.width, 36], fill=(255, 255, 255))
            d.text((12, 7), title, fill=(30, 41, 59), font=_font(19))
            if i > n * 0.8:
                verdict = "lifted" if res["success"] else "failed to lift"
                color = (21, 128, 61) if res["success"] else (185, 28, 28)
                d.text((12, img.height - 34), verdict, fill=color, font=_font(22))
            tiles.append(img)
        sheet = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height))
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0))
            x += t.width
        out_frames.append(sheet)
    args.out.mkdir(parents=True, exist_ok=True)
    anim = args.out / "grasp_demo.webp"
    out_frames[0].save(anim, save_all=True, append_images=out_frames[1:], duration=33, loop=0, quality=70, method=4)
    out_frames[int(n * 0.9)].save(args.out / "grasp_demo.jpg", quality=88)
    (args.out / "grasp_demo.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {anim}")


if __name__ == "__main__":
    main()
