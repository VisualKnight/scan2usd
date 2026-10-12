"""Robotics demo: a Franka Panda (MuJoCo Menagerie) picks up and places a Scan2USD mug.

    python scripts/demo_panda.py [--asset output/demo/asset/ceramic_mug.usda] [--out docs/figures]

Rim grasp (one finger inside the cup), lift, carry 25 cm, place, release. Runs with the
asset's convex decomposition and with a single convex hull, side by side.

Robot model: mujoco_menagerie/franka_emika_panda (Apache-2.0), fetched into
data/external/menagerie (see README).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from scan2usd.mjcf import asset_body
from scan2usd.preview import _font
from scan2usd.usd_io import read_usd

ROOT = Path(__file__).resolve().parents[1]
PANDA = ROOT / "data" / "external" / "menagerie" / "franka_emika_panda"
TCP = np.array([0.0, 0.0, 0.1034])  # fingertip center in the hand frame
MUG_POS = np.array([0.52, -0.12, 0.0])
PLACE_SHIFT = np.array([0.0, 0.25, 0.0])
HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])


def rim_grasp(mesh, offset):
    """TCP target on the rim opposite the handle (world frame) and the radial closing direction."""
    v = mesh.vertices
    top = v[:, 2].max()
    ring = v[v[:, 2] > top - 0.003]
    a = np.c_[2 * ring[:, 0], 2 * ring[:, 1], np.ones(len(ring))]
    cx, cy, _ = np.linalg.lstsq(a, (ring[:, :2] ** 2).sum(1), rcond=None)[0]
    radii = np.hypot(ring[:, 0] - cx, ring[:, 1] - cy)
    r_mid = 0.5 * (np.percentile(radii, 5) + np.percentile(radii, 95))
    band = v[(v[:, 2] > 0.25 * top) & (v[:, 2] < 0.75 * top)]
    far = band[np.argmax(np.hypot(band[:, 0] - cx, band[:, 1] - cy))]
    yaw = np.arctan2(far[1] - cy, far[0] - cx) + np.pi
    radial = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    point = np.array([cx, cy, 0.0]) + r_mid * radial + offset
    point[2] = top - 0.016
    return point, radial


def hand_rotation(closing_dir):
    """Hand frame: z down, fingers close along y = closing_dir.

    The Panda's fingers sit 45 degrees from the hand frame's y axis (the hand body is
    rotated -45 degrees about z in the model), so the frame is pre-rotated to line them
    up radially with the cup wall.
    """
    z = np.array([0.0, 0.0, -1.0])
    y = closing_dir / np.linalg.norm(closing_dir)
    x = np.cross(y, z)
    return np.c_[x, y, z]


class Arm:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.hand = model.body("hand").id
        self.qadr = [model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]
        self.dadr = [model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)]
        self.lo = np.array([model.joint(f"joint{i}").range[0] for i in range(1, 8)])
        self.hi = np.array([model.joint(f"joint{i}").range[1] for i in range(1, 8)])

    def tcp(self, data):
        r = data.xmat[self.hand].reshape(3, 3)
        return data.xpos[self.hand] + r @ TCP, r

    def ik(self, pos, rot, q0, iters=400, damping=0.05):
        d = self.data
        q = np.array(q0, dtype=np.float64)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            p, r = self.tcp(d)
            ep = pos - p
            er = 0.5 * sum(np.cross(r[:, i], rot[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jac(self.model, d, jp, jr, p, self.hand)
            j = np.r_[jp[:, self.dadr], jr[:, self.dadr]]
            e = np.r_[ep, er]
            dq = j.T @ np.linalg.solve(j @ j.T + damping ** 2 * np.eye(6), e)
            q = np.clip(q + dq, self.lo, self.hi)
        return q, float(np.linalg.norm(ep))


def build_scene(asset, collision: str, work: Path) -> mujoco.MjModel:
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(PANDA, work)
    # Grippy fingertip pads (rubber-like friction, torsional friction via condim 4).
    panda_xml = work / "panda.xml"
    text = panda_xml.read_text().replace(
        '<geom type="mesh" group="3"/>',
        '<geom type="mesh" group="3" friction="1.5 0.02 0.002" condim="4" solref="0.005 1"/>', 1)
    # Menagerie's gripper servo is soft (100 N/m: ~0.5 N when pinching a 5 mm cup wall). A real
    # Panda hand squeezes with up to 70 N, so stiffen it to 2000 N/m (~9 N here).
    text = text.replace('gainprm="0.01568627451 0 0" biasprm="0 -100 -10"',
                        'gainprm="0.31372549 0 0" biasprm="0 -2000 -60"')
    panda_xml.write_text(text)
    body = asset_body(asset, "mug", pos=tuple(MUG_POS), collision=collision)
    for name, data in body.files.items():
        (work / "assets" / name).write_bytes(data)
    tx, ty = MUG_POS[:2] + PLACE_SHIFT[:2]
    scene = f"""<mujoco model="panda_pick">
  <include file="panda.xml"/>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="960" offheight="720"/><quality shadowsize="4096"/></visual>
  <asset>{body.assets_xml}
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.80 0.84 0.90" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".82 .82 .80" rgb2=".74 .74 .72" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="14 14"/>
  </asset>
  <worldbody>
    <light pos="0.6 -0.8 1.6" dir="-0.2 0.5 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light pos="-0.6 0.6 1.4" dir="0.4 -0.4 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="grid" solref="0.005 1" friction="0.8 0.005 0.0001"/>
    <site name="target" pos="{tx} {ty} 0.001" size="0.05 0.0015" type="cylinder" rgba="0.1 0.6 0.25 0.5"/>
    {body.body_xml}
  </worldbody>
</mujoco>"""
    (work / "pick_scene.xml").write_text(scene)
    return mujoco.MjModel.from_xml_path(str(work / "pick_scene.xml"))


def run(asset, collision: str, fps: int = 30, size=(480, 640)):
    model = build_scene(asset, collision, ROOT / "data" / "work" / f"panda_{collision}")
    arm = Arm(model)
    data = mujoco.MjData(model)
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    mug_free = model.joint(model.body("mug").jntadr[0]).qposadr[0]
    data.qpos[mug_free: mug_free + 3] = MUG_POS
    for fj in ("finger_joint1", "finger_joint2"):
        data.qpos[model.joint(fj).qposadr[0]] = 0.022
    mujoco.mj_forward(model, data)

    grasp, radial = rim_grasp(asset.render_mesh, MUG_POS)
    up = np.array([0, 0, 0.12])
    # Open just enough to straddle the wall (ctrl 0..255 maps to 0..40 mm per finger).
    straddle = 255 * 0.022 / 0.04
    waypoints = [  # (tcp position, gripper ctrl 0..255, duration s)
        (grasp + up, straddle, 1.6), (grasp, straddle, 1.0), (grasp, 0, 0.7),
        (grasp + np.array([0, 0, 0.15]), 0, 1.2), (grasp + PLACE_SHIFT + np.array([0, 0, 0.15]), 0, 1.4),
        (grasp + PLACE_SHIFT + np.array([0, 0, 0.004]), 0, 1.2), (grasp + PLACE_SHIFT, straddle, 0.6),
        (grasp + PLACE_SHIFT + up, straddle, 1.0),
    ]
    q_prev, plan = HOME.copy(), []
    # A parallel gripper is symmetric: try both hand yaws and keep the one the wrist can reach.
    candidates = []
    for sign in (1.0, -1.0):
        r_try = hand_rotation(sign * radial)
        q_try, _ = arm.ik(grasp, r_try, HOME, iters=1500, damping=0.02)
        d_try = mujoco.MjData(model)
        d_try.qpos[arm.qadr] = q_try
        mujoco.mj_kinematics(model, d_try)
        _, r_got = arm.tcp(d_try)
        candidates.append((np.arccos(np.clip((np.trace(r_got.T @ r_try) - 1) / 2, -1, 1)), r_try))
    rot = min(candidates, key=lambda c: c[0])[1]
    for pos, grip, dur in waypoints:
        q, err = arm.ik(pos, rot, q_prev, iters=1500, damping=0.02)
        plan.append((q, grip, dur, err))
        q_prev = q

    renderer = mujoco.Renderer(model, *size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.48, 0.0, 0.16)
    cam.distance, cam.elevation, cam.azimuth = 1.15, -22, 205
    mug = model.body("mug").id
    frames, q_from, grip_from = [], HOME.copy(), 255 * 0.022 / 0.04
    lifted = 0.0
    for q_to, grip_to, dur, _ in plan:
        n = int(dur / model.opt.timestep)
        for i in range(n):
            s = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / n)  # smooth start and stop
            data.ctrl[:7] = q_from + (q_to - q_from) * s
            data.ctrl[7] = grip_from + (grip_to - grip_from) * min(1.0, 2 * (i + 1) / n)
            data.qfrc_applied[arm.dadr] = data.qfrc_bias[arm.dadr]  # gravity compensation
            mujoco.mj_step(model, data)
            lifted = max(lifted, float(data.xipos[mug][2]))
            if len(frames) <= data.time * fps:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())
        q_from, grip_from = q_to, float(grip_to)
    for _ in range(int(0.6 / model.opt.timestep)):
        data.ctrl[:7] = q_from
        data.qfrc_applied[arm.dadr] = data.qfrc_bias[arm.dadr]
        mujoco.mj_step(model, data)
        if len(frames) <= data.time * fps:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    final = data.xpos[mug]
    target = MUG_POS + PLACE_SHIFT
    up_axis = data.xmat[mug].reshape(3, 3)[:, 2]
    placed = float(np.linalg.norm(final[:2] - target[:2]))
    com0 = asset.center_of_mass[2]
    result = {
        "collision": collision,
        "colliders": len(asset.hulls) if collision == "hulls" else 1,
        "max_com_height_m": lifted, "lifted_m": lifted - com0,
        "place_error_m": placed, "upright_deg": float(np.degrees(np.arccos(np.clip(up_axis[2], -1, 1)))),
        "ik_error_mm": max(p[3] for p in plan) * 1000,
    }
    result["success"] = bool(result["lifted_m"] > 0.1 and placed < 0.03 and result["upright_deg"] < 10)
    return frames, result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asset", type=Path, default=ROOT / "output" / "demo" / "asset" / "ceramic_mug.usda")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = p.parse_args()
    if not PANDA.exists():
        raise SystemExit(f"{PANDA} not found; fetch mujoco_menagerie/franka_emika_panda first (see README)")
    asset = read_usd(args.asset)
    results, videos = [], []
    for collision, title in (("hulls", "Scan2USD collision (convex decomposition)"),
                             ("single", "Naive collision (one convex hull)")):
        frames, res = run(asset, collision)
        results.append(res)
        videos.append((frames, title))
        print(json.dumps(res))
    n = min(len(v[0]) for v in videos)
    out = []
    for i in range(n):
        tiles = []
        for (frames, title), res in zip(videos, results):
            img = Image.fromarray(frames[i])
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, img.width, 36], fill=(255, 255, 255))
            d.text((12, 7), title, fill=(30, 41, 59), font=_font(18))
            if i > n * 0.88:
                ok = res["success"]
                d.text((12, img.height - 34), "picked and placed" if ok else "failed",
                       fill=(21, 128, 61) if ok else (185, 28, 28), font=_font(22))
            tiles.append(img)
        sheet = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height))
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0))
            x += t.width
        out.append(sheet)
    args.out.mkdir(parents=True, exist_ok=True)
    out[0].save(args.out / "panda_demo.webp", save_all=True, append_images=out[1:], duration=33, loop=0,
                quality=65, method=4)
    out[int(n * 0.55)].save(args.out / "panda_demo.jpg", quality=88)
    (args.out / "panda_demo.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out / 'panda_demo.webp'}")


if __name__ == "__main__":
    main()
