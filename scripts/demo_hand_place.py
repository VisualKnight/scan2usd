"""A Shadow Hand places the scanned controller into the scanned holder (MuJoCo).

    python scripts/demo_hand_place.py [--trials 12] [--out docs/figures] [--no-video]

The five-finger version of scripts/demo_holder_place.py, with the same answer key: the controller
lies face up on a raised tray; the hand comes down over its face, spans its width (thumb on one
side, middle and ring fingers on the other, level with its center of mass), lifts it, turns it
upright, lowers it to 5 mm above the pose measured in the scan of the real pair, opens and backs away.

The hand floats: its forearm is welded to a mocap target that follows the planned path (no arm, so
reachability is not checked; demo_hand_grasp.py does the same). The span poses are designed by
optimization (span_poses): the thumb pad faces the middle and ring pads across a chosen width, with
the palm clear of the face.

What it took (each found by a failed trial):
- A span, not a pinch. A pinch near the edge (as the Panda does) holds the 243 g controller ~6 cm
  from its center of mass; the fingertip servos (1 N m) can't resist that moment.
- Load sharing. The controller is wider under the ring finger than under the middle finger, so the
  grip degenerates to two points (thumb, ring) that let it spin; the middle finger curls further.
- Two-stage turn. Yaw with the palm down first, then the 90-degree tilt upright about the
  controller's center. A one-shot slerp swung the hand palm-up and dropped it.
- Room for the forearm. Tilted upright, the forearm points back the way it came; the tray sits off
  to the side so it doesn't hit it.

Variants: open loop (the plan as made at the pick), and re-aimed: at the pre-place pose the
controller's pose in the hand is re-measured (read from the simulation, standing in for a wrist
camera running the perception pick's registration) and the insertion is re-planned from it.

Robot: mujoco_menagerie/shadow_hand (Apache-2.0).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_holder_place import RELEASE_GAP, body_pose, load, pose, quat_wxyz, yaw_matrix  # noqa: E402

from scan2usd.mjcf import asset_body  # noqa: E402
from scan2usd.register import rotation_angle_deg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HAND = ROOT / "data" / "external" / "menagerie" / "shadow_hand"
POSES = ROOT / "output" / "hand_place_span.json"
SPAN_JOINTS = ["rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1", "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_RFJ4",
               "rh_RFJ3", "rh_RFJ2"]
PALM_CLEAR = 0.055  # the palm hovers this far from the contact line: over the face (5.6 cm thick at the sticks)
GRASP_UP = 0.014  # contacts level with the controller's center of mass (1.4 cm above its center, along "top")
HOLDER_XY = np.array([0.0, 0.25])
STAND_XY = np.array([0.25, 0.05])  # off to the side: the forearm swings back over the line to the holder
STAND_H = 0.15
EDGE_DIR = np.array([1.0, 0.0])
FINGER_GAIN = 5.0  # finger servo stiffness x5 (as demo_hand_grasp.py); torque limits unchanged


# --- span grasp design ---------------------------------------------------------------------------------

def _tip(m, d, body, r):
    R = d.xmat[m.body(body).id].reshape(3, 3)
    return d.xpos[m.body(body).id] + R @ [0, -r, 0.022], R @ [0, -1.0, 0]  # pads on the distal's -y side


def _set_span(m, d, x, J):
    d.qpos[:] = 0
    for n, v in zip(SPAN_JOINTS, x):
        d.qpos[J[n]] = v
    for f in ("MF", "RF"):  # the J1/J2 pair is driven by one tendon
        d.qpos[J[f"rh_{f}J1"]] = d.qpos[J[f"rh_{f}J2"]]
    mujoco.mj_kinematics(m, d)
    pt, nt = _tip(m, d, "rh_thdistal", 0.008)
    pm, nm = _tip(m, d, "rh_mfdistal", 0.0065)
    pr, nr = _tip(m, d, "rh_rfdistal", 0.0065)
    pf = (pm + pr) / 2
    axis = (pf - pt) / np.linalg.norm(pf - pt)  # thumb -> fingers
    return pt, nt, pm, nm, pr, nr, pf, axis


def span_poses(gaps=(0.133, 0.145, 0.160, 0.170, 0.120, 0.110)) -> dict:
    """Span-grasp joint angles for each thumb-to-fingers distance (m).

    The thumb pad faces the middle and ring pads across the gap (holding a book by its sides), the palm
    stays at least PALM_CLEAR from the line between them (it hovers over the controller's face), and the
    middle and ring pads sit 2-3 cm apart across the axis (a stable two-point side). The first gap (the
    controller's width at its center of mass) is solved freely; the others continue from their nearest
    neighbour and keep its grasp center and axis in the hand frame, so the fingers open and close about
    the same line. Cached in output/.
    """
    if POSES.exists():
        return {float(k): np.array(v) for k, v in json.loads(POSES.read_text())["poses"].items()}
    m = mujoco.MjModel.from_xml_path(str(HAND / "right_hand.xml"))
    d = mujoco.MjData(m)
    J = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
    lo = np.array([m.jnt_range[m.joint(n).id][0] for n in SPAN_JOINTS])
    hi = np.array([m.jnt_range[m.joint(n).id][1] for n in SPAN_JOINTS])
    ref, poses = {}, {}

    def cost(x, gap):
        pt, nt, pm, nm, pr, nr, pf, axis = _set_span(m, d, x, J)
        palm = d.xpos[m.body("rh_palm").id] + d.xmat[m.body("rh_palm").id].reshape(3, 3) @ [0, -0.01, 0.05]
        center = (pt + pf) / 2
        off = palm - center
        off -= axis * (off @ axis)
        c = 1e4 * (np.linalg.norm(pf - pt) - gap) ** 2
        c += 3 * (1 - nt @ axis) + 3 * (2 + nm @ axis + nr @ axis)  # pads face each other across the gap
        c += 1e4 * max(0.0, PALM_CLEAR - np.linalg.norm(off)) ** 2
        spread = np.linalg.norm((pm - pr) - axis * ((pm - pr) @ axis))
        c += 1e3 * max(0.0, 0.02 - spread) ** 2 + 1e3 * ((pm - pr) @ axis) ** 2
        if ref:
            c += 1e4 * np.sum((center - ref["center"]) ** 2) + 300 * (1 - axis @ ref["axis"])
            near = min(poses, key=lambda g: abs(g - gap))
            c += 0.2 * np.sum((x - poses[near]) ** 2)
        c += 1e3 * np.sum(np.maximum(0, lo - x) ** 2 + np.maximum(0, x - hi) ** 2)
        return c

    for gap in gaps:
        if poses:
            starts = [poses[min(poses, key=lambda g: abs(g - gap))]]
        else:
            starts = [np.random.default_rng(s).uniform(lo, hi) for s in range(24)]
        best = min((minimize(cost, x0, args=(gap,), method="Nelder-Mead",
                             options={"maxiter": 12000, "xatol": 1e-6, "fatol": 1e-10}) for x0 in starts),
                   key=lambda r: r.fun)
        x = np.clip(best.x, lo, hi)
        poses[gap] = x
        pt, nt, pm, nm, pr, nr, pf, axis = _set_span(m, d, x, J)
        center = (pt + pf) / 2
        if not ref:
            ref["center"], ref["axis"] = center, axis
        print(f"span {gap * 1000:.0f} mm: actual {np.linalg.norm(pf - pt) * 1000:.1f} mm, facing "
              f"{nt @ axis:+.2f}/{-(nm @ axis):+.2f}/{-(nr @ axis):+.2f}, center drift "
              f"{np.linalg.norm(center - ref['center']) * 1000:.1f} mm, cost {best.fun:.3f}")
    POSES.parent.mkdir(parents=True, exist_ok=True)
    POSES.write_text(json.dumps({"joints": SPAN_JOINTS, "poses": {str(k): v.tolist() for k, v in poses.items()}},
                                indent=1))
    return poses


def span_frame(x) -> np.ndarray:
    """Grasp frame in the forearm frame (4x4): origin midway between the thumb pad and the middle/ring
    pads, y = thumb -> fingers, z = from the palm toward that line (the approach), x = y cross z."""
    m = mujoco.MjModel.from_xml_path(str(HAND / "right_hand.xml"))
    d = mujoco.MjData(m)
    J = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
    pt, nt, pm, nm, pr, nr, pf, axis = _set_span(m, d, x, J)
    center = (pt + pf) / 2
    palm = d.xpos[m.body("rh_palm").id] + d.xmat[m.body("rh_palm").id].reshape(3, 3) @ [0, -0.01, 0.05]
    z = center - palm
    z -= axis * (z @ axis)
    z /= np.linalg.norm(z)
    f = m.body("rh_forearm").id
    W = pose(np.c_[np.cross(axis, z), axis, z], center)
    return np.linalg.inv(pose(d.xmat[f].reshape(3, 3), d.xpos[f])) @ W


def hand_ctrl(model, x) -> np.ndarray:
    """Actuator targets for a span pose. The index and little fingers follow their neighbours (middle,
    ring) a little less flexed, so they hover over the face instead of pressing on it; wrist straight."""
    q = {n: v for n, v in zip(SPAN_JOINTS, x)}
    q.update({"rh_MFJ1": q["rh_MFJ2"], "rh_RFJ1": q["rh_RFJ2"],
              "rh_FFJ4": q["rh_MFJ4"] - 0.1, "rh_FFJ3": 0.7 * q["rh_MFJ3"], "rh_FFJ2": 0.3, "rh_FFJ1": 0.3,
              "rh_LFJ4": q["rh_RFJ4"] + 0.1, "rh_LFJ3": 0.7 * q["rh_RFJ3"], "rh_LFJ2": 0.3, "rh_LFJ1": 0.3})
    ctrl = np.zeros(model.nu)
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            ctrl[a] = q.get(model.joint(model.actuator_trnid[a][0]).name, 0.0)
        else:  # tendon: sum of its joints
            t = model.actuator_trnid[a][0]
            adr, num = model.tendon_adr[t], model.tendon_num[t]
            ctrl[a] = sum(model.wrap_prm[i] * q.get(model.joint(model.wrap_objid[i]).name, 0.0)
                          for i in range(adr, adr + num))
    return ctrl


# --- scene -------------------------------------------------------------------------------------------

def build(ctrl, holder, T_hold, T_ctrl0, T_stand, stand_half, collision, work: Path, ghost=None) -> mujoco.MjModel:
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(HAND, work)
    xml = (work / "right_hand.xml").read_text()
    xml = xml.replace('<material name="black" specular="0.5" shininess="0.25" rgba="0.16355 0.16355 0.16355 1"/>',
                      '<material name="black" specular="0.4" shininess="0.4" rgba="0.93 0.93 0.95 1"/>')
    xml = xml.replace('<material name="gray" specular="0.0" shininess="0.25" rgba="0.80848 0.80848 0.80848 1"/>',
                      '<material name="gray" specular="0.2" shininess="0.3" rgba="0.18 0.45 0.85 1"/>')
    xml = xml.replace('<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
                      '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1"><freejoint name="forearm_free"/>')
    (work / "right_hand.xml").write_text(xml)
    hold = asset_body(holder, "holder", pos=T_hold[:3, 3], quat=quat_wxyz(T_hold[:3, :3]), collision=collision,
                      free=False, rgba=(0.30, 0.30, 0.33, 1), visual_faces=60000)
    c = asset_body(ctrl.asset, "ctrl", pos=T_ctrl0[:3, 3], quat=quat_wxyz(T_ctrl0[:3, :3]), collision="hulls",
                   rgba=(0.90, 0.90, 0.88, 1), visual_faces=40000)
    for b in (hold, c):
        for name, data in b.files.items():
            (work / "assets" / name).write_bytes(data)
    ghost_xml = ""
    if ghost is not None:
        ghost_xml = (f'<body name="ghost" pos="{" ".join(f"{v:.6f}" for v in ghost[:3, 3])}" '
                     f'quat="{" ".join(f"{q:.6f}" for q in quat_wxyz(ghost[:3, :3]))}">'
                     f'<geom type="mesh" mesh="ctrl_visual" contype="0" conaffinity="0" rgba="0.1 0.8 0.35 0.35"/></body>')
    sp = T_stand
    scene = f"""<mujoco model="hand_place">
  <include file="right_hand.xml"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="8"/>
  <visual><global offwidth="1280" offheight="960"/><quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/></visual>
  <asset>{hold.assets_xml}{c.assets_xml}
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.80 0.84 0.90" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".82 .82 .80" rgb2=".74 .74 .72" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.6 1.2" dir="-0.2 0.5 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light pos="-0.5 0.5 1.0" dir="0.4 -0.4 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="grid" solref="0.005 1"/>
    <body name="stand" pos="{sp[0, 3]:.5f} {sp[1, 3]:.5f} {STAND_H / 2:.5f}" quat="{" ".join(f"{q:.6f}" for q in quat_wxyz(sp[:3, :3]))}">
      <geom type="box" size="{stand_half[0]} {stand_half[1]} {STAND_H / 2}" rgba="0.18 0.32 0.55 1" friction="0.8 0.005 0.0001"/>
    </body>
    <body name="target" mocap="true" quat="0 1 0 1"/>
    {hold.body_xml}{c.body_xml}{ghost_xml}
  </worldbody>
  <equality><weld body1="target" body2="rh_forearm" solref="0.005 1"/></equality>
</mujoco>"""
    (work / "place_scene.xml").write_text(scene)
    model = mujoco.MjModel.from_xml_path(str(work / "place_scene.xml"))
    for a in range(model.nu):
        if "WRJ" not in model.actuator(a).name:
            model.actuator_gainprm[a, 0] *= FINGER_GAIN
            model.actuator_biasprm[a, 1] *= FINGER_GAIN
    return model


def span_grasp(ctrl, up: float, side: int = 1):
    """Grasp frame (4x4, controller frame) across the controller's width at height `up` along its top
    axis: origin midway between its left and right sides at mid-thickness, y = thumb side -> finger side
    (side=+1: the thumb on the controller's left), z = into the face (the palm comes from the face side).
    Returns the frame and the width there (m)."""
    band = np.abs(ctrl.up - up) < 0.004
    lo, hi = ctrl.lat[band].min(), ctrl.lat[band].max()
    edge = band & ((ctrl.lat < lo + 0.006) | (ctrl.lat > hi - 0.006))
    z_mid = 0.5 * (ctrl.v[edge, 2].min() + ctrl.v[edge, 2].max())
    o = 0.5 * (lo + hi) * ctrl.right + up * ctrl.top + np.array([0, 0, z_mid])
    y = side * ctrl.right
    z = -ctrl.face
    return pose(np.c_[np.cross(y, z), y, z], o), float(hi - lo)


class Setup:
    def __init__(self, seed, ctrl, holder, R_rel, t_rel, collision, poses, ghost=False, work_tag=None, side=1,
                 squeeze=0.025, opening=0.03, mf_extra=(0.15, 0.25), transport=2.0, correct=False):
        rng = np.random.default_rng(seed)
        jit = seed > 0
        hold_xy = HOLDER_XY + (rng.uniform(-0.02, 0.02, 2) if jit else 0)
        self.hold_yaw = 0.0 + (rng.uniform(-10, 10) if jit else 0)  # front (-y) toward the tray
        T_hold = pose(yaw_matrix(self.hold_yaw), [*hold_xy, 0.0])
        self.T_target = T_hold @ pose(R_rel, t_rel)
        self.ctrl, self.correct = ctrl, correct
        self.C_goal = self.T_target.copy()
        self.C_goal[2, 3] += RELEASE_GAP
        self.al = -(self.C_goal[:3, :3] @ ctrl.face)  # approach at the place: into the face
        stand_xy = STAND_XY + (rng.uniform(-0.02, 0.02, 2) if jit else 0)
        yaw0 = np.degrees(np.arctan2(EDGE_DIR[1], EDGE_DIR[0]) - np.arctan2(ctrl.top[1], ctrl.top[0]))
        yaw0 += rng.uniform(-10, 10) if jit else 0
        R0 = yaw_matrix(yaw0)
        # Tray under the middle of the controller only (3 cm either side of center): its left and right
        # sides, where the fingertips go, overhang it.
        t_lo, t_hi = ctrl.up.min() + 0.005, ctrl.up.max() - 0.005
        half = (0.03, 0.5 * (t_hi - t_lo))
        stand_center = stand_xy + R0[:2, :2] @ (0.5 * (t_lo + t_hi) * ctrl.top[:2])
        T_stand = pose(R0 @ np.c_[ctrl.right, ctrl.top, ctrl.face], [*stand_center, 0.0])
        T_ctrl0 = pose(R0, [*stand_xy, STAND_H + 0.003])
        work = ROOT / "data" / "work" / f"hand_place_{work_tag or collision}"
        self.model = build(ctrl, holder, T_hold, T_ctrl0, T_stand, half, "hulls" if collision == "hulls" else "single",
                           work, ghost=self.T_target if ghost else None)
        m = self.model
        self.transport = transport
        d = self.data = mujoco.MjData(m)
        cadr = m.joint(m.body("ctrl").jntadr[0]).qposadr[0]
        d.qpos[cadr:cadr + 3] = T_ctrl0[:3, 3]
        d.qpos[cadr + 3:cadr + 7] = quat_wxyz(R0)
        # Park the hand out of the way while the controller settles.
        fadr = m.joint("forearm_free").qposadr[0]
        d.qpos[fadr:fadr + 3] = [0.0, -0.6, 0.6]
        d.mocap_pos[0] = [0.0, -0.6, 0.6]
        for _ in range(750):
            mujoco.mj_step(m, d)
        d.time = 0.0
        self.T_start = body_pose(m, d, "ctrl")
        # Span grasp in the controller frame: across its width at the center of mass's height (along
        # "top") and mid-thickness, thumb on one side, middle and ring on the other, palm over the face.
        G, self.width = span_grasp(ctrl, GRASP_UP, side)
        self.poses = dict(poses)
        gaps = np.array(sorted(poses))
        self.g_contact = float(gaps[np.argmin(np.abs(gaps - self.width))])
        self.g_open = float(gaps[np.argmin(np.abs(gaps - (self.width + opening)))])
        self.g_close = float(gaps[np.argmin(np.abs(gaps - (self.width - squeeze)))])
        F = span_frame(poses[self.g_contact])  # grasp frame in the forearm frame
        self.H = G @ np.linalg.inv(F)  # forearm pose in the controller frame
        # The controller is wider where the ring finger lands (lower on its sides) than under the middle
        # finger, so at the same span the ring finger takes most of the load and the grip degenerates to
        # two points (thumb, ring), which resists turning about the line between them only by torsional
        # friction. Curl the middle finger further to share the load: a three-point grip.
        grip = poses[self.g_close].copy()
        grip[SPAN_JOINTS.index("rh_MFJ3")] += mf_extra[0]
        grip[SPAN_JOINTS.index("rh_MFJ2")] += mf_extra[1]
        self.poses["grip"] = grip


def place_waypoints(su, H_rel):
    """Waypoints from the pre-place pose on, for a forearm pose H_rel in the controller frame: tilt
    upright in front of the holder, slide in along the approach, set down, open, back out, rise."""
    C = su.C_goal

    def at(v):  # forearm pose with the controller at its goal pose shifted by v (world)
        return pose(C[:3, :3], C[:3, 3] + v) @ H_rel

    al = su.al
    return [
        (at(-0.12 * al + [0, 0, 0.04]), "grip", su.transport),
        (at(-0.05 * al), "grip", 1.2),
        (at(np.zeros(3)), "grip", 1.2),
        (at(np.zeros(3)), su.g_open, 0.8),
        (at(-0.08 * al), su.g_open, 1.0),
        (at(-0.08 * al + [0, 0, 0.08]), su.g_open, 0.8),
    ]


def path(su):
    """Mocap waypoints: (forearm pose 4x4, span key, duration s).

    Approach the face from above with the fingers open around the sides, close, lift, then turn in two
    separate moves: a yaw with the palm still down (gravity stays along the approach, which the grip
    holds best), then the 90-degree tilt upright, which is a turn about the grasp axis itself (from face
    up on the tray to face forward on the holder there is no way around that) and is the one move the
    grip resists only by friction. The tilt is about the controller's center, so it doesn't swing.
    """
    A0 = su.T_start @ su.H
    F = span_frame(su.poses[su.g_contact])
    ap = (A0 @ F)[:3, 2]  # approach direction in the world (palm toward the face), at the pick
    C = su.C_goal
    # The goal pose tipped face-up about a horizontal axis: the same yaw as the goal, palm down.
    face = C[:3, :3] @ su.ctrl.face
    tip = Rotation.align_vectors([[0, 0, 1.0]], [face])[0].as_matrix()
    C_mid = pose(tip @ C[:3, :3], C[:3, 3] - 0.12 * su.al + [0, 0, 0.10])
    return [
        (A0 @ pose(np.eye(3), -0.10 * (np.linalg.inv(A0[:3, :3]) @ ap) + np.linalg.inv(A0[:3, :3]) @ [0, 0, 0.06]),
         su.g_open, 1.5),
        (pose(A0[:3, :3], A0[:3, 3] - 0.06 * ap), su.g_open, 1.0),
        (A0, su.g_open, 1.0),
        (A0, "grip", 1.0),
        (pose(A0[:3, :3], A0[:3, 3] + [0, 0, 0.08]), "grip", 1.2),
        (C_mid @ su.H, "grip", 1.8),
    ] + place_waypoints(su, su.H)


def execute(su, wps, render=None, fps=30, settle=1.5, on_waypoint=None):
    m, d = su.model, su.data
    frames = []

    def step():
        mujoco.mj_step(m, d)
        if render is not None and len(frames) <= d.time * fps:
            frames.append(render(d))

    # Start: put the hand at the first waypoint in the open pose, settled (the weld pulls the hand to the
    # mocap target; starting it there avoids a jolt across the scene).
    T0, g0, _ = wps[0]
    fadr = m.joint("forearm_free").qposadr[0]
    vadr = m.joint("forearm_free").dofadr[0]
    d.qpos[fadr:fadr + 3] = T0[:3, 3]
    d.qpos[fadr + 3:fadr + 7] = quat_wxyz(T0[:3, :3])
    d.qvel[vadr:vadr + 6] = 0
    d.mocap_pos[0], d.mocap_quat[0] = T0[:3, 3], quat_wxyz(T0[:3, :3])
    c_prev = hand_ctrl(m, su.poses[g0])
    d.ctrl[:] = c_prev
    mujoco.mj_forward(m, d)
    T_prev = T0
    wps = list(wps)
    k = 0
    while k < len(wps):
        T, g, dur = wps[k]
        c_to = hand_ctrl(m, su.poses[g])
        sl = Slerp([0, 1], Rotation.from_matrix([T_prev[:3, :3], T[:3, :3]]))
        n = max(1, int(dur / m.opt.timestep))
        for i in range(n):
            s = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / n)
            d.mocap_pos[0] = T_prev[:3, 3] + (T[:3, 3] - T_prev[:3, 3]) * s
            q = sl([s]).as_quat()[0]
            d.mocap_quat[0] = (q[3], q[0], q[1], q[2])
            d.ctrl[:] = c_prev + (c_to - c_prev) * min(1.0, (i + 1) / n)
            step()
        T_prev, c_prev = T, c_to
        if on_waypoint is not None:
            on_waypoint(k, d)
        if su.correct and k == len(wps) - 6:
            # At the pre-place pose: re-measure the controller in the hand (a wrist camera running the
            # perception pick's registration would; here it is read from the simulation) and re-aim the
            # rest of the placement at it.
            H_rel = np.linalg.inv(body_pose(m, d, "ctrl")) @ body_pose(m, d, "rh_forearm")
            wps[k + 1:] = place_waypoints(su, H_rel)[1:]
        k += 1
    for _ in range(int(settle / m.opt.timestep)):
        step()
    return frames


def score(ctrl, m, d, T_target):
    T_end = body_pose(m, d, "ctrl")
    cp = ctrl.v[:: max(1, len(ctrl.v) // 3000)]
    end_pts = cp @ T_end[:3, :3].T + T_end[:3, 3]
    true_pts = cp @ T_target[:3, :3].T + T_target[:3, 3]
    add = float(np.linalg.norm(end_pts - true_pts, axis=1).mean())
    held = bool(end_pts[:, 2].min() > 0.02)
    return {"add_mm": add * 1000, "rot_deg": rotation_angle_deg(T_end[:3, :3], T_target[:3, :3]), "held": held,
            "placed": bool(held and add < 0.02)}


VARIANTS = (("hulls", False), ("hulls", True), ("single", True))


def trial(seed, ctrl, holder, R_rel, t_rel, poses, collision, correct):
    su = Setup(seed, ctrl, holder, R_rel, t_rel, collision, poses, correct=correct)
    execute(su, path(su))
    return {"seed": seed, "collision": collision, "reaimed": correct, "holder_yaw_deg": float(su.hold_yaw),
            **score(ctrl, su.model, su.data, su.T_target)}


def video(ctrl, holder, R_rel, t_rel, poses, out: Path, fps: int = 30):
    """Side by side: open loop vs re-aimed from the in-hand pose, same seed. Each panel: the cell from
    the right, with a close-up of the holder inset in the bottom-right corner (over empty floor); the green ghost is
    the real resting pose."""
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font

    wide = ((0.05, 0.25, 0.15), 0.85, -30, 170)
    close = ((-0.03, 0.22, 0.12), 0.42, -10, 20)
    W, Hh, IW, IH = 560, 420, 220, 176
    panels = []
    for correct, title in ((False, "Open loop"), (True, "Re-aimed from the in-hand pose")):
        su = Setup(0, ctrl, holder, R_rel, t_rel, "hulls", poses, ghost=True, correct=correct,
                   work_tag=f"video_{int(correct)}")
        rend = mujoco.Renderer(su.model, Hh, W)
        cams = []
        for la, dist, el, az in (wide, close):
            c = mujoco.MjvCamera()
            c.lookat[:], c.distance, c.elevation, c.azimuth = la, dist, el, az
            cams.append(c)

        def render(d, rend=rend, cams=cams):
            rend.update_scene(d, camera=cams[0])
            a = rend.render().copy()
            rend.update_scene(d, camera=cams[1])
            return a, rend.render().copy()

        frames = execute(su, path(su), render=render, fps=fps)
        rend.close()
        panels.append((title, frames, score(ctrl, su.model, su.data, su.T_target)))
    n = min(len(p[1]) for p in panels)
    out_frames = []
    for i in range(n):
        sheet = Image.new("RGB", (2 * W + 10, Hh + 44), "white")
        d = ImageDraw.Draw(sheet)
        for k, (title, frames, res) in enumerate(panels):
            x = k * (W + 10)
            wide_img, close_img = (Image.fromarray(a) for a in frames[i])
            close_img = close_img.crop(((W - IW * Hh // IH) // 2, 0, (W + IW * Hh // IH) // 2, Hh)).resize((IW, IH))
            sheet.paste(wide_img, (x, 44))
            ix, iy = x + W - IW - 8, 44 + Hh - IH - 8
            sheet.paste(close_img, (ix, iy))
            d.rectangle([ix - 1, iy - 1, ix + IW, iy + IH], outline=(30, 41, 59), width=1)
            d.text((x + 8, 10), title, fill=(30, 41, 59), font=_font(19))
            if i > n * 0.82:
                ok = res["held"] and res["placed"]
                msg = (f"held: {res['add_mm']:.1f} mm / {res['rot_deg']:.0f}\u00b0 from the real pose" if ok
                       else "dropped")
                tw = d.textlength(msg, font=_font(20))
                d.rectangle([x + 4, 44 + 4, x + 16 + tw, 44 + 36], fill=(255, 255, 255))
                d.text((x + 10, 44 + 8), msg, fill=(21, 128, 61) if ok else (185, 28, 28), font=_font(20))
        out_frames.append(sheet)
    out.mkdir(parents=True, exist_ok=True)
    out_frames[0].save(out / "hand_place.webp", save_all=True, append_images=out_frames[3::3], duration=100, loop=0,
                       quality=45, method=6)
    out_frames[int(n * 0.86)].save(out / "hand_place.jpg", quality=88)
    print(f"wrote {out / 'hand_place.webp'} ({len(out_frames[::3])} frames at 10 fps)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    if not HAND.exists():
        raise SystemExit(f"{HAND} not found; fetch mujoco_menagerie/shadow_hand first (see README)")
    poses = span_poses()
    ctrl, holder, R_rel, t_rel = load()
    results = {}
    for collision, correct in VARIANTS:
        key = f"{collision}_{'reaimed' if correct else 'open_loop'}"
        rows = []
        for seed in range(args.trials):
            r = trial(seed, ctrl, holder, R_rel, t_rel, poses, collision, correct)
            rows.append(r)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        results[key] = rows
    summary = {k: {"trials": len(v), "held": sum(r["held"] for r in v), "placed": sum(r["placed"] for r in v),
                   "add_mm_median": float(np.median([r["add_mm"] for r in v])),
                   "rot_deg_median": float(np.median([r["rot_deg"] for r in v]))} for k, v in results.items()}
    print(json.dumps(summary, indent=1))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "hand_place.json").write_text(json.dumps({"summary": summary, "trials": results,
                                                          "release_gap_m": RELEASE_GAP}, indent=2))
    if not args.no_video:
        video(ctrl, holder, R_rel, t_rel, poses, args.out)


if __name__ == "__main__":
    main()
