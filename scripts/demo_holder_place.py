"""A robot places the scanned controller into the scanned holder (MuJoCo, Franka Panda).

    python scripts/demo_holder_place.py [--trials 12] [--out docs/figures] [--no-video]

The drop test (study_holder_fit.py) releases the controller 3 cm above its real resting pose. Here a
robot does the whole job, and the answer key is the same: where the real controller rests in the
real holder, measured in the scan of the two together.

1. The controller lies face up on a raised tray, its top edge overhanging toward the robot's side.
2. The Panda pinches the top edge near one corner (the closing axis through the controller's face),
   the approach swung 20 degrees toward the middle: the figure stands right behind the middle of the
   edge, so only the corners leave room for the back finger, and the swing keeps the palm off its head.
3. It lifts, turns the controller upright in the air, and lowers it along its top edge's direction
   until the controller is 5 mm above the measured pose, then lets go and backs away.
4. Score: does it stay in the hands, and how far (mm, degrees) from the real pose does it settle?

Every trial moves the tray and the holder a little (the robot is told their poses, as from a
calibrated camera), and is run against the decomposed holder and against a single convex hull.

Simulation settings that matter (checked, not tuned to the result): the controller hangs 6 cm off a
corner pinch, so the grip needs a realistic force (6000 N/m servo, ~45 N per finger; a real Panda hand
reaches 70 N; at 2000 N/m it slips out) and MuJoCo's noslip solver (without it, soft-contact creep lets
the controller rotate out of the fingers within seconds, a simulation artifact rather than physics).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_panda import HOME, PANDA, Arm, prepare_panda  # noqa: E402

from scan2usd.mjcf import asset_body  # noqa: E402
from scan2usd.preview import _font  # noqa: E402
from scan2usd.register import rotation_angle_deg  # noqa: E402
from scan2usd.usd_io import read_usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / "output" / "holder_fit"
HOLDER_XY = np.array([0.50, 0.17])
STAND_XY = np.array([0.55, -0.15])
STAND_H = 0.15  # tray height: room under the overhanging edge for the lower finger and the hand
EDGE_DIR = np.array([0.0, -1.0])  # the controller's top edge faces -y on the tray: a comfortable side reach
RELEASE_GAP = 0.005  # release this far above the measured pose
GRIP_STIFFNESS = 6000.0  # N/m on the finger tendon: ~45 N per finger on the 3.6 cm edge (a real Panda: 70 N)


def yaw_matrix(deg: float) -> np.ndarray:
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


def quat_wxyz(R: np.ndarray) -> tuple:
    q = Rotation.from_matrix(R).as_quat()
    return (q[3], q[0], q[1], q[2])


class Controller:
    """The controller asset and its natural frame (right, top, face), from the measured pose in the holder."""

    def __init__(self, asset, R_rel: np.ndarray):
        self.asset = asset
        v = np.asarray(asset.render_mesh.vertices)
        self.v = v
        right = R_rel.T @ [1.0, 0, 0]
        right[2] = 0
        self.right = right / np.linalg.norm(right)
        top = R_rel.T @ [0, 0, 1.0]
        top[2] = 0
        top -= self.right * (top @ self.right)
        self.top = top / np.linalg.norm(top)
        self.face = np.array([0, 0, 1.0])
        self.lat, self.up = v @ self.right, v @ self.top

    def grasp_frame(self, lateral: float, side: int, inset: float = 0.012, tilt_deg: float = 0.0):
        """Hand pose (4x4, controller frame) for a pinch across the top edge at `lateral` (m along right).

        Hand z (approach) points into the edge, hand y (finger closing axis) is the face normal.
        `tilt_deg` swings the approach about the face normal toward the controller's center (+) or the
        outer side (-); a corner grasp needs it to keep the palm off the figure behind the edge.
        Returns the pose and the edge thickness there (m).
        """
        band = np.abs(self.lat - lateral) < 0.004
        edge = self.up[band].max()
        m = band & (np.abs(self.up - (edge - inset)) < 0.004)
        z0, z1 = self.v[m, 2].min(), self.v[m, 2].max()
        p = lateral * self.right + (edge - inset) * self.top + np.array([0, 0, 0.5 * (z0 + z1)])
        a = np.radians(tilt_deg) * np.sign(lateral)
        z = -(np.cos(a) * self.top + np.sin(a) * self.right)
        y = side * self.face
        x = np.cross(y, z)
        G = np.eye(4)
        G[:3, :3] = np.c_[x, y, z]
        G[:3, 3] = p
        return G, float(z1 - z0)


def pose(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def build(ctrl_asset, holder_asset, holder_pose, ctrl_pose, collision: str, work: Path, ghost=None,
          stand_pose=None, stand_size=(0.045, 0.045, STAND_H / 2)) -> mujoco.MjModel:
    prepare_panda(work, GRIP_STIFFNESS)
    hold = asset_body(holder_asset, "holder", pos=holder_pose[:3, 3], quat=quat_wxyz(holder_pose[:3, :3]),
                      collision=collision, free=False, rgba=(0.30, 0.30, 0.33, 1), visual_faces=60000)
    ctrl = asset_body(ctrl_asset, "ctrl", pos=ctrl_pose[:3, 3], quat=quat_wxyz(ctrl_pose[:3, :3]),
                      collision="hulls", rgba=(0.90, 0.90, 0.88, 1), visual_faces=40000)
    # Contacts with the controller take the larger friction of the pair (MuJoCo's default) and its
    # contact dimension; the controller's own plastic friction (0.3) stays what the holder sees, but a
    # grip by rubber pads needs condim 4 (torsional friction) on the controller side too.
    ctrl = replace(ctrl, body_xml=ctrl.body_xml.replace('solref="0.005 1"/>', 'solref="0.005 1" condim="4"/>'))
    for b in (hold, ctrl):
        for name, data in b.files.items():
            (work / "assets" / name).write_bytes(data)
    sp = stand_pose
    stand = (f'<body name="stand" pos="{sp[0, 3]:.5f} {sp[1, 3]:.5f} {STAND_H / 2:.5f}" '
             f'quat="{" ".join(f"{q:.6f}" for q in quat_wxyz(sp[:3, :3]))}">'
             f'<geom type="box" size="{stand_size[0]} {stand_size[1]} {stand_size[2]}" rgba="0.18 0.32 0.55 1" '
             f'friction="0.8 0.005 0.0001" solref="0.005 1"/></body>')
    ghost_xml = ""
    if ghost is not None:
        ghost_xml = (f'<body name="ghost" pos="{" ".join(f"{v:.6f}" for v in ghost[:3, 3])}" '
                     f'quat="{" ".join(f"{q:.6f}" for q in quat_wxyz(ghost[:3, :3]))}">'
                     f'<geom type="mesh" mesh="ctrl_visual" contype="0" conaffinity="0" rgba="0.1 0.8 0.35 0.35"/></body>')
    scene = f"""<mujoco model="holder_place">
  <include file="panda.xml"/>
  <!-- noslip: a post-pass that removes the creep MuJoCo's soft contacts allow under steady load, so a
       pinched object stays put in the fingers instead of slowly sliding out -->
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="8"/>
  <visual><global offwidth="1280" offheight="960"/><quality shadowsize="4096"/></visual>
  <asset>{hold.assets_xml}{ctrl.assets_xml}
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.80 0.84 0.90" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".82 .82 .80" rgb2=".74 .74 .72" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="14 14"/>
  </asset>
  <worldbody>
    <light pos="0.6 -0.8 1.6" dir="-0.2 0.5 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light pos="-0.6 0.6 1.4" dir="0.4 -0.4 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="grid" solref="0.005 1" friction="0.8 0.005 0.0001"/>
    {stand}{hold.body_xml}{ctrl.body_xml}{ghost_xml}
  </worldbody>
</mujoco>"""
    (work / "place_scene.xml").write_text(scene)
    return mujoco.MjModel.from_xml_path(str(work / "place_scene.xml"))


def body_pose(model, data, name: str) -> np.ndarray:
    b = model.body(name).id
    return pose(data.xmat[b].reshape(3, 3).copy(), data.xpos[b].copy())


def plan(arm: Arm, waypoints, q0=HOME):
    """IK through (tcp position, hand rotation, grip ctrl, duration) waypoints; continuous seeds.

    A parallel gripper is symmetric about its approach axis, so the whole plan may also be run with the
    hand turned 180 degrees, and the redundant arm can start its IK from different configurations. Every
    combination is solved and the plan with the smallest worst-case IK error is kept (the wrist never
    flips mid-motion within a plan).
    """
    flip = np.diag([-1.0, -1.0, 1.0])
    seeds = [np.array(q0, float)] + [np.array(q0, float) + d for d in
                                     (np.r_[0.6, 0, 0, 0, 0, 0, 0], np.r_[-0.6, 0, 0, 0, 0, 0, 0],
                                      np.r_[0, 0, 0, 0, 1.2, 0, 0], np.r_[0, 0, 0, 0, -1.2, 0, 0],
                                      np.r_[0.4, 0.3, 0, 0.3, 0.8, 0, 0])]
    best = None
    for use_flip in (False, True):
        for seed in seeds:
            out, q_prev = [], seed
            for k, (pos, rot, grip, dur) in enumerate(waypoints):
                q, err = arm.ik(pos, rot @ flip if use_flip else rot, q_prev, iters=1500 if k == 0 else 800,
                                damping=0.02)
                out.append((q, grip, dur, err))
                q_prev = q
            worst = max(o[3] for o in out)
            if best is None or worst < best[0]:
                best = (worst, out)
            if worst < 1e-3:
                return out
    return best[1]


def segment(p0, R0, p1, R1, n: int):
    """n poses from (p0, R0) to (p1, R1): straight line, slerped rotation."""
    sl = Slerp([0, 1], Rotation.from_matrix([R0, R1]))
    return [(p0 + (p1 - p0) * s, sl([s]).as_matrix()[0]) for s in np.linspace(0, 1, n + 1)[1:]]


def make_plan(arm, G, thickness, ctrl_start, ctrl_target):
    """Waypoints: approach the edge, pinch, lift, turn upright on the way over, descend, release, retreat."""
    open_ctrl = min(255.0, 255 * (0.5 * thickness + 0.014) / 0.04)
    H_pick, H_place = ctrl_start @ G, ctrl_target @ G
    pp, Rp = H_pick[:3, 3], H_pick[:3, :3]
    pl, Rl = H_place[:3, 3], H_place[:3, :3]
    a_pick, a_place = Rp[:, 2], Rl[:, 2]  # approach directions
    wp = []
    pre = pp - 0.08 * a_pick
    wp.append((pre + np.array([0, 0, 0.05]), Rp, open_ctrl, 2.0))
    wp.append((pre, Rp, open_ctrl, 0.8))
    wp += [(p, R, open_ctrl, 0.35) for p, R in segment(pre, Rp, pp, Rp, 4)]
    wp.append((pp, Rp, 0.0, 0.8))  # pinch
    lift = pp + np.array([0, 0, 0.07])
    wp.append((lift, Rp, 0.0, 1.0))
    high = pl - 0.13 * a_place
    high[2] = max(high[2], lift[2] + 0.04)
    wp += [(p, R, 0.0, 0.3) for p, R in segment(lift, Rp, high, Rl, 10)]
    pre_place = pl - 0.06 * a_place
    wp += [(p, Rl, 0.0, 0.3) for p, R in segment(high, Rl, pre_place, Rl, 4)]
    wp += [(p, Rl, 0.0, 0.35) for p, R in segment(pre_place, Rl, pl, Rl, 6)]  # slow final descent
    wp.append((pl, Rl, open_ctrl, 0.7))  # release
    back = pl - 0.07 * a_place
    wp += [(p, Rl, open_ctrl, 0.3) for p, R in segment(pl, Rl, back, Rl, 3)]
    wp.append((back + np.array([0, 0, 0.06]), Rl, open_ctrl, 0.8))
    return plan(arm, wp)


def execute(model, arm, data, steps, render=None, fps=30, settle=1.5, on_waypoint=None):
    """Track the joint-space plan (position servos, gravity compensated); returns rendered frames."""
    frames = []

    def step():
        mujoco.mj_step(model, data)
        if render is not None and len(frames) <= data.time * fps:
            frames.append(render(data))

    q_from, grip_from = data.ctrl[:7].copy(), float(data.ctrl[7])
    for k, (q_to, grip_to, dur, _) in enumerate(steps):
        n = int(dur / model.opt.timestep)
        for i in range(n):
            s = (i + 1) / n
            data.ctrl[:7] = q_from + (q_to - q_from) * s  # dense waypoints: piecewise linear in joint space
            data.ctrl[7] = grip_from + (grip_to - grip_from) * min(1.0, 2 * s)
            data.qfrc_applied[arm.dadr] = data.qfrc_bias[arm.dadr]  # gravity compensation
            step()
        q_from, grip_from = q_to, float(grip_to)
        if on_waypoint is not None:
            on_waypoint(k, data)
    for _ in range(int(settle / model.opt.timestep)):
        data.ctrl[:7] = q_from
        data.qfrc_applied[arm.dadr] = data.qfrc_bias[arm.dadr]
        step()
    return frames


class Setup:
    """One trial's scene: holder and tray poses (jittered by seed), the model, settled data, the plan."""

    def __init__(self, seed, ctrl, holder_asset, R_rel, t_rel, collision, lateral, side, tilt=20.0, ghost=False,
                 work_tag=None):
        rng = np.random.default_rng(seed)
        jitter = seed > 0
        hold_xy = HOLDER_XY + (rng.uniform(-0.02, 0.02, 2) if jitter else 0)
        to_base = -hold_xy / np.linalg.norm(hold_xy)
        # Face the holder's front (-y) toward the robot, +-10 degrees: R(yaw) @ (0, -1) = to_base.
        self.hold_yaw = np.degrees(np.arctan2(to_base[0], -to_base[1])) + (rng.uniform(-10, 10) if jitter else 0)
        T_hold = pose(yaw_matrix(self.hold_yaw), [*hold_xy, 0.0])
        self.T_target = T_hold @ pose(R_rel, t_rel)
        stand_xy = STAND_XY + (rng.uniform(-0.02, 0.02, 2) if jitter else 0)
        # Controller face up on the tray, its top edge toward EDGE_DIR (+-10 degrees).
        yaw0 = np.degrees(np.arctan2(EDGE_DIR[1], EDGE_DIR[0]) - np.arctan2(ctrl.top[1], ctrl.top[0]))
        yaw0 += rng.uniform(-10, 10) if jitter else 0
        R0 = yaw_matrix(yaw0)
        # Tray under the controller, aligned with its natural axes. It spans the grips (up to 4 cm below
        # the center) and stops 2.5 cm short of the top edge: the grasp corner overhangs with room under
        # it for the lower finger, and the center of mass (1.4 cm above center) is well inside.
        t_lo, t_hi = ctrl.up.min() + 0.005, ctrl.up.max() - 0.025
        half = (0.075, 0.5 * (t_hi - t_lo))
        t_local = 0.5 * (t_lo + t_hi)  # tray center along the controller's top axis
        stand_center = stand_xy + (R0[:2, :2] @ (t_local * ctrl.top[:2]))
        T_stand = pose(R0 @ np.c_[ctrl.right, ctrl.top, ctrl.face], [*stand_center, 0.0])
        T_ctrl0 = pose(R0, [stand_xy[0], stand_xy[1], STAND_H + 0.003])
        work = ROOT / "data" / "work" / f"holder_place_{work_tag or collision}"
        self.model = build(ctrl.asset, holder_asset, T_hold, T_ctrl0, "hulls" if collision == "hulls" else "single",
                           work, ghost=self.T_target if ghost else None, stand_pose=T_stand,
                           stand_size=(half[0], half[1], STAND_H / 2))
        model = self.model
        self.arm = Arm(model)
        data = self.data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        cadr = model.joint(model.body("ctrl").jntadr[0]).qposadr[0]
        data.qpos[cadr:cadr + 3] = T_ctrl0[:3, 3]
        data.qpos[cadr + 3:cadr + 7] = quat_wxyz(R0)
        data.ctrl[:7] = HOME
        data.ctrl[7] = 255
        for _ in range(1500):  # the controller settles on the tray
            data.qfrc_applied[self.arm.dadr] = data.qfrc_bias[self.arm.dadr]
            mujoco.mj_step(model, data)
        data.time = 0.0
        self.T_start = body_pose(model, data, "ctrl")  # the robot is told the controller's pose
        self.G, self.thick = ctrl.grasp_frame(lateral, side, tilt_deg=tilt)
        T_goal = self.T_target.copy()
        T_goal[2, 3] += RELEASE_GAP
        self.steps = make_plan(self.arm, self.G, self.thick, self.T_start, T_goal)


def score(ctrl, model, data, T_target) -> dict:
    T_end = body_pose(model, data, "ctrl")
    cp = ctrl.v[:: max(1, len(ctrl.v) // 3000)]
    true_pts = cp @ T_target[:3, :3].T + T_target[:3, 3]
    end_pts = cp @ T_end[:3, :3].T + T_end[:3, 3]
    add = float(np.linalg.norm(end_pts - true_pts, axis=1).mean())
    held = bool(end_pts[:, 2].min() > 0.02)  # not on the floor (the holder's hands are ~8 cm up)
    return {"add_mm": add * 1000, "rot_deg": rotation_angle_deg(T_end[:3, :3], T_target[:3, :3]), "held": held,
            "placed": bool(held and add < 0.02)}


def trial(seed, ctrl, holder_asset, R_rel, t_rel, collision, lateral, side, tilt=20.0, render_cam=None,
          size=(720, 960)):
    su = Setup(seed, ctrl, holder_asset, R_rel, t_rel, collision, lateral, side, tilt, ghost=render_cam is not None)
    renderer = mujoco.Renderer(su.model, *size) if render_cam is not None else None

    def render(d):
        renderer.update_scene(d, camera=render_cam)
        return renderer.render().copy()

    frames = execute(su.model, su.arm, su.data, su.steps, render if renderer else None)
    res = {"seed": seed, "collision": collision, **score(ctrl, su.model, su.data, su.T_target),
           "ik_error_mm": max(s[3] for s in su.steps) * 1000, "edge_thickness_mm": su.thick * 1000,
           "holder_yaw_deg": float(su.hold_yaw)}
    return res, frames, su.model


def video(ctrl, holder, R_rel, t_rel, lateral, side, out: Path, fps: int = 30):
    """Side by side: decomposed holder vs one convex hull, the same plan. Each panel: the cell from
    the front, with a close-up of the hands inset; the green ghost is the real resting pose."""
    wide = ((0.52, 0.02, 0.15), 0.95, -15, 200)
    close = ((0.50, 0.17, 0.13), 0.42, -12, 120)
    W, H, IW, IH = 560, 420, 220, 176
    panels = []
    for collision, title in (("hulls", "Holder: convex decomposition (64 pieces)"), ("single", "Holder: one convex hull")):
        su = Setup(0, ctrl, holder, R_rel, t_rel, collision, lateral, side, ghost=True, work_tag=f"video_{collision}")
        # MuJoCo renderers share the GL context of the last one created: render both views with one.
        rend = mujoco.Renderer(su.model, H, W)
        cams = []
        for la, dist, el, az in (wide, close):
            c = mujoco.MjvCamera()
            c.lookat[:], c.distance, c.elevation, c.azimuth = la, dist, el, az
            cams.append(c)

        def render(d, rend=rend, cams=cams):
            rend.update_scene(d, camera=cams[0])
            a = rend.render().copy()
            rend.update_scene(d, camera=cams[1])
            b = rend.render().copy()
            return a, b

        frames = execute(su.model, su.arm, su.data, su.steps, render)
        rend.close()
        res = score(ctrl, su.model, su.data, su.T_target)
        panels.append((title, frames, res))
    n = min(len(p[1]) for p in panels)
    out_frames = []
    for i in range(n):
        sheet = Image.new("RGB", (2 * W + 10, H + 44), "white")
        d = ImageDraw.Draw(sheet)
        for k, (title, frames, res) in enumerate(panels):
            x = k * (W + 10)
            wide_img, close_img = (Image.fromarray(a) for a in frames[i])
            close_img = close_img.crop(((W - IW * H // IH) // 2, 0, (W + IW * H // IH) // 2, H)).resize((IW, IH))
            sheet.paste(wide_img, (x, 44))
            sheet.paste(close_img, (x + W - IW - 8, 44 + 8))
            d.rectangle([x + W - IW - 9, 44 + 7, x + W - 8, 44 + 8 + IH], outline=(30, 41, 59), width=1)
            d.text((x + 8, 10), title, fill=(30, 41, 59), font=_font(19))
            if i > n * 0.82:
                ok = res["held"] and res["placed"]
                msg = (f"held: {res['add_mm']:.1f} mm / {res['rot_deg']:.0f}\u00b0 from the real pose" if ok
                       else "slides off the hands")
                d.text((x + 10, H + 44 - 34), msg, fill=(21, 128, 61) if ok else (185, 28, 28), font=_font(20))
        out_frames.append(sheet)
    out.mkdir(parents=True, exist_ok=True)
    out_frames[0].save(out / "holder_place.webp", save_all=True, append_images=out_frames[3::3], duration=100, loop=0,
                       quality=45, method=6)
    out_frames[int(n * 0.86)].save(out / "holder_place.jpg", quality=88)
    print(f"wrote {out / 'holder_place.webp'} ({len(out_frames[::3])} frames at 10 fps)")


def load():
    st = json.loads((FIT / "state.json").read_text())
    rel = st["truth"]["controller_in_holder"]
    R_rel, t_rel = np.array(rel["R"]), np.array(rel["t"])
    ctrl = Controller(read_usd(FIT / "assets" / "controller" / "controller.usda"), R_rel)
    holder = read_usd(FIT / "assets" / "cod_holder" / "cod_holder.usda")
    return ctrl, holder, R_rel, t_rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--lateral", type=float, default=0.055, help="grasp position along the top edge (m)")
    ap.add_argument("--side", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    if not PANDA.exists():
        raise SystemExit(f"{PANDA} not found; fetch mujoco_menagerie/franka_emika_panda first (see README)")
    ctrl, holder, R_rel, t_rel = load()
    results = {}
    for collision in ("hulls", "single"):
        rows = []
        for seed in range(args.trials):
            res, _, _ = trial(seed, ctrl, holder, R_rel, t_rel, collision, args.lateral, args.side)
            rows.append(res)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
        results[collision] = rows
    summary = {}
    for k, rows in results.items():
        add = [r["add_mm"] for r in rows]
        summary[k] = {"trials": len(rows), "held": sum(r["held"] for r in rows),
                      "placed": sum(r["placed"] for r in rows),
                      "add_mm_median": float(np.median(add)), "rot_deg_median": float(np.median([r["rot_deg"] for r in rows]))}
    print(json.dumps(summary, indent=1))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "holder_place.json").write_text(json.dumps({"summary": summary, "trials": results,
                                                            "grasp": {"lateral_m": args.lateral, "side": args.side},
                                                            "release_gap_m": RELEASE_GAP}, indent=2))
    if not args.no_video:
        video(ctrl, holder, R_rel, t_rel, args.lateral, args.side, args.out)


if __name__ == "__main__":
    main()
