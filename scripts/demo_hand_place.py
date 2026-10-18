"""A Shadow Hand places the scanned controller into the scanned holder (MuJoCo).

    python scripts/demo_hand_place.py [--trials 12] [--out docs/figures] [--no-video]
    python scripts/demo_hand_place.py --export-plans output/hand_place_plans.json   # for isaac_hand_place.py

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hand_motion  # noqa: E402
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


def hand_q(x) -> dict[str, float]:
    """Joint targets for a span pose. The index and little fingers follow their neighbours (middle,
    ring) a little less flexed, so they hover over the face instead of pressing on it; wrist straight.
    Each J1 equals its J2 (MuJoCo drives the pair through one tendon)."""
    q = {n: float(v) for n, v in zip(SPAN_JOINTS, x)}
    q.update({"rh_MFJ1": q["rh_MFJ2"], "rh_RFJ1": q["rh_RFJ2"],
              "rh_FFJ4": q["rh_MFJ4"] - 0.1, "rh_FFJ3": 0.7 * q["rh_MFJ3"], "rh_FFJ2": 0.3, "rh_FFJ1": 0.3,
              "rh_LFJ4": q["rh_RFJ4"] + 0.1, "rh_LFJ3": 0.7 * q["rh_RFJ3"], "rh_LFJ2": 0.3, "rh_LFJ1": 0.3})
    return q


def hand_ctrl(model, x) -> np.ndarray:
    """Actuator targets for a span pose (hand_q mapped onto the actuators; tendons get J1 + J2)."""
    q = hand_q(x)
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
    # Studio palette: off-white hand shells, graphite forearm and joints.
    xml = xml.replace('<material name="black" specular="0.5" shininess="0.25" rgba="0.16355 0.16355 0.16355 1"/>',
                      '<material name="black" specular="0.35" shininess="0.5" rgba="0.90 0.90 0.91 1"/>')
    xml = xml.replace('<material name="gray" specular="0.0" shininess="0.25" rgba="0.80848 0.80848 0.80848 1"/>',
                      '<material name="gray" specular="0.3" shininess="0.6" rgba="0.20 0.21 0.23 1"/>')
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
                     f'<geom type="mesh" mesh="ctrl_visual" contype="0" conaffinity="0" rgba="0.1 0.8 0.35 0.3"/></body>')
    sp = T_stand
    scene = f"""<mujoco model="hand_place">
  <include file="right_hand.xml"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="8"/>
  <visual><global offwidth="1920" offheight="1080" fovy="{hand_motion.FOVY_DEG}"/><map zfar="200"/><quality shadowsize="8192" offsamples="8"/>
    <headlight ambient="0.32 0.32 0.33" diffuse="0.30 0.30 0.30" specular="0.05 0.05 0.05"/></visual>
  <asset>{hold.assets_xml}{c.assets_xml}
    <texture type="skybox" builtin="gradient" rgb1="0.97 0.97 0.98" rgb2="0.78 0.80 0.84" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".88 .88 .87" rgb2=".84 .84 .83" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="144 144"/>
  </asset>
  <worldbody>
    <light pos="0.7 -0.5 1.3" dir="-0.45 0.35 -1" diffuse="0.62 0.61 0.60" castshadow="true"/>
    <light pos="-0.6 0.4 0.9" dir="0.5 -0.3 -0.8" diffuse="0.30 0.32 0.36" castshadow="false"/>
    <light pos="-0.2 1.0 0.7" dir="0.1 -0.9 -0.4" diffuse="0.25 0.25 0.25" specular="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="12 12 0.05" material="grid" solref="0.005 1"/>
    <body name="stand" pos="{sp[0, 3]:.5f} {sp[1, 3]:.5f} {STAND_H / 2:.5f}" quat="{" ".join(f"{q:.6f}" for q in quat_wxyz(sp[:3, :3]))}">
      <geom type="box" size="{stand_half[0]} {stand_half[1]} {STAND_H / 2}" rgba="0.62 0.60 0.57 1" friction="0.8 0.005 0.0001"/>
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
    return hand_motion.place_waypoints(su.C_goal, su.al, H_rel, su.g_open, su.transport)


def path(su):
    """Waypoints (hand_motion.wp dicts) for the whole placement (hand_motion.plan_path)."""
    return hand_motion.plan_path(su.T_start, su.C_goal, su.al, su.H, span_frame(su.poses[su.g_contact]),
                                 su.ctrl.face, su.g_open, su.transport)


def execute(su, wps, render=None, fps=30, settle=1.5, on_waypoint=None):
    """Run the waypoints: the mocap target follows hand_motion's blended moves, the finger targets
    blend leg by leg. render(d) is called at `fps` (frames are returned); on_waypoint(k, d) after each
    move, k = the index of the waypoint it ended at."""
    m, d = su.model, su.data
    frames = []

    def step():
        mujoco.mj_step(m, d)
        if render is not None and len(frames) <= d.time * fps:
            frames.append(render(d))

    # Start: put the hand at the first waypoint in the open pose, settled (the weld pulls the hand to the
    # mocap target; starting it there avoids a jolt across the scene).
    wps = list(wps)
    T0 = wps[0]["T"]
    fadr = m.joint("forearm_free").qposadr[0]
    vadr = m.joint("forearm_free").dofadr[0]
    d.qpos[fadr:fadr + 3] = T0[:3, 3]
    d.qpos[fadr + 3:fadr + 7] = quat_wxyz(T0[:3, :3])
    d.qvel[vadr:vadr + 6] = 0
    d.mocap_pos[0], d.mocap_quat[0] = T0[:3, 3], quat_wxyz(T0[:3, :3])
    d.ctrl[:] = hand_ctrl(m, su.poses[wps[0]["g"]])
    mujoco.mj_forward(m, d)
    T_prev, g_prev, k = T0, wps[0]["g"], 1
    while k < len(wps):
        a, b = hand_motion.moves(wps, k)[0]
        mv = hand_motion.Move(T_prev, g_prev, wps[a:b + 1])
        c = [hand_ctrl(m, su.poses[g]) for g in mv.g]
        n = max(1, int(round(mv.duration / m.opt.timestep)))
        for i in range(n):
            T, leg, u = mv.at((i + 1) * m.opt.timestep)
            d.mocap_pos[0] = T[:3, 3]
            d.mocap_quat[0] = quat_wxyz(T[:3, :3])
            d.ctrl[:] = c[leg] + (c[leg + 1] - c[leg]) * u
            step()
        T_prev, g_prev = wps[b]["T"], wps[b]["g"]
        if on_waypoint is not None:
            on_waypoint(b, d)
        if su.correct and wps[b]["tag"] == "preplace":
            # In front of the holder: re-measure the controller in the hand (a wrist camera running the
            # perception pick's registration would; here it is read from the simulation) and re-aim the
            # rest of the placement at it.
            H_rel = np.linalg.inv(body_pose(m, d, "ctrl")) @ body_pose(m, d, "rh_forearm")
            wps[b + 1:] = place_waypoints(su, H_rel)[1:]
        k = b + 1
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


def servo_table(m) -> dict:
    """The finger servos as simulated here, per joint (a tendon actuator's gain on each of its joints)."""
    servo = {}
    for a in range(m.nu):
        kp, frc = float(m.actuator_gainprm[a, 0]), [float(v) for v in m.actuator_forcerange[a]]
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            joints = [m.joint(m.actuator_trnid[a][0]).name]
        else:
            t = m.actuator_trnid[a][0]
            joints = [m.joint(m.wrap_objid[i]).name for i in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t])]
        for j in joints:
            jid = m.joint(j).id
            servo[j] = {"kp": kp, "force_range": frc, "damping": float(m.dof_damping[m.jnt_dofadr[jid]]),
                        "armature": float(m.dof_armature[m.jnt_dofadr[jid]]),
                        "frictionloss": float(m.dof_frictionloss[m.jnt_dofadr[jid]]), "tendon": len(joints) > 1}
    return servo


def export_plans(ctrl, holder, R_rel, t_rel, poses, seeds, path_out: Path) -> None:
    """Everything another engine needs to run the same placement (scripts/isaac_hand_place.py): per
    seed the holder, tray and settled controller poses, the goal, the open-loop waypoints and the
    re-aim inputs; the hand pose targets per key; the finger servos as simulated here."""
    su = Setup(0, ctrl, holder, R_rel, t_rel, "hulls", poses)
    m = su.model
    servo = servo_table(m)
    trials = []
    for seed in seeds:
        su = Setup(seed, ctrl, holder, R_rel, t_rel, "hulls", poses)
        m, d = su.model, su.data
        stand = m.body("stand").id
        geom = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == stand)
        trials.append({
            "seed": seed, "holder": body_pose(m, d, "holder").tolist(), "target": su.T_target.tolist(),
            "controller_start": su.T_start.tolist(), "c_goal": su.C_goal.tolist(), "approach": su.al.tolist(),
            "tray": pose(d.xmat[stand].reshape(3, 3), d.xpos[stand]).tolist(),
            "tray_half_extents": m.geom_size[geom].tolist(),
            "waypoints": [{**w, "T": w["T"].tolist()} for w in path(su)],
        })
    keys = {str(k): hand_q(v) for k, v in su.poses.items()}
    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_out.write_text(json.dumps({"hand_q": keys, "open_key": str(su.g_open), "transport": su.transport,
                                    "servo": servo, "timestep": float(m.opt.timestep),
                                    "release_gap_m": RELEASE_GAP, "trials": trials}, indent=1))
    print(f"wrote {path_out} ({len(trials)} plans)")


def export_task(ctrl, holder, R_rel, t_rel, poses, path_out: Path) -> None:
    """The scene layout and the scripted plan's inputs for a scene-randomizing engine (the Isaac Lab
    task, scripts/rl/): unlike export_plans, no per-seed plans; the engine settles the controller,
    then calls hand_motion.plan_path itself. Poses in the world frame of Setup (z up, table z = 0)."""
    su = Setup(0, ctrl, holder, R_rel, t_rel, "hulls", poses)
    t_lo, t_hi = ctrl.up.min() + 0.005, ctrl.up.max() - 0.005
    yaw0 = float(np.degrees(np.arctan2(EDGE_DIR[1], EDGE_DIR[0]) - np.arctan2(ctrl.top[1], ctrl.top[0])))
    spec = {
        "controller_in_holder": pose(R_rel, t_rel).tolist(), "release_gap_m": RELEASE_GAP,
        "holder_xy": HOLDER_XY.tolist(), "holder_yaw_deg": 0.0,
        "tray_xy": STAND_XY.tolist(), "tray_height_m": STAND_H, "controller_yaw_deg": yaw0,
        # tray box in the controller frame (lying face up on it): center along "top", half extents
        "tray_center_top_m": 0.5 * (t_lo + t_hi), "tray_half_extents": [0.03, 0.5 * (t_hi - t_lo), STAND_H / 2],
        "ctrl_axes": {"right": ctrl.right.tolist(), "top": ctrl.top.tolist(), "face": ctrl.face.tolist()},
        "H": su.H.tolist(), "F": span_frame(su.poses[su.g_contact]).tolist(),
        "open_key": str(su.g_open), "transport": su.transport,
        "hand_q": {str(k): hand_q(v) for k, v in su.poses.items()},
        "servo": servo_table(su.model),
    }
    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_out.write_text(json.dumps(spec, indent=1))
    print(f"wrote {path_out}")


VARIANTS = (("hulls", False), ("hulls", True), ("single", True))


def trial(seed, ctrl, holder, R_rel, t_rel, poses, collision, correct):
    su = Setup(seed, ctrl, holder, R_rel, t_rel, collision, poses, correct=correct)
    execute(su, path(su))
    return {"seed": seed, "collision": collision, "reaimed": correct, "holder_yaw_deg": float(su.hold_yaw),
            **score(ctrl, su.model, su.data, su.T_target)}


CAPTIONS = [  # (from event, text)
    ("start", "Approach: palm over the face, fingers open around its sides"),
    ("pick", "Span grasp level with the center of mass: thumb vs. middle + ring"),
    ("lift", "Lift, yaw palm-down, then tilt upright about the controller's center"),
    ("preplace", "Re-aim: measure the controller in the hand, re-plan the insertion"),
    ("insert", "Slide in, set down 5 mm above the pose measured in the real pair"),
    ("release", "Release and back away"),
    ("away", "Seated where the real controller rests in a scan of the real pair"),
]


def video(ctrl, holder, R_rel, t_rel, poses, out: Path, summary=None, fps: int = 30, size=(1280, 720)):
    """The hero shot (re-aimed run, seed 0) as MP4 (30 fps), a lighter animated WebP for the README, a
    poster frame, and a still comparing where open loop and re-aimed leave the controller."""
    import imageio.v2 as iio
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font

    W, Hh = size
    # No ghost of the real pose: a few mm from the placed scan it pokes through it in patches and reads
    # as broken geometry. The end card gives the distance instead.
    su = Setup(0, ctrl, holder, R_rel, t_rel, "hulls", poses, correct=True, work_tag="video")
    wps = path(su)
    shots = [(t0, hand_motion.camera_track(k))
             for t0, k in hand_motion.shot_keys(su.T_start[:3, 3], su.T_target[:3, 3], wps)]
    ev = hand_motion.event_times(wps)
    caps = sorted((ev[k], txt) for k, txt in CAPTIONS)
    rend = mujoco.Renderer(su.model, Hh, W)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()


    def render(d):
        track = [tr for t0, tr in shots if d.time >= t0][-1]
        x = track(d.time)
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = x[:3], x[3], x[4], x[5]
        rend.update_scene(d, camera=cam, scene_option=opt)
        return d.time, rend.render().copy()

    tail = 3.5
    raw = execute(su, wps, render=render, fps=fps, settle=tail)
    res = score(ctrl, su.model, su.data, su.T_target)
    rend.close()
    f_title, f_cap, f_small = _font(17), _font(25), _font(19)
    t_end = raw[-1][0]
    frames = []
    for t, img in raw:
        im = Image.fromarray(img).convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        d.text((28, 22), "Shadow Hand places the scanned controller into the scanned holder  \u2014  MuJoCo",
               fill=(30, 41, 59, 200), font=f_title)
        # Caption: the latest phase, faded in over 0.3 s and out before the end card.
        k = max([i for i, (t0, _) in enumerate(caps) if t >= t0 - 1e-6], default=0)
        a = min(1.0, (t - caps[k][0]) / 0.3) if k > 0 else min(1.0, t / 0.3 + 0.01)
        a = min(a, max(0.0, (t_end - 1.6 - t) / 0.3))
        if a > 0:
            txt = caps[k][1]
            tw = d.textlength(txt, font=f_cap)
            x0, y0 = 28, Hh - 74
            d.rounded_rectangle([x0, y0, x0 + tw + 36, y0 + 46], 10, fill=(15, 23, 42, int(170 * a)))
            d.text((x0 + 18, y0 + 8), txt, fill=(255, 255, 255, int(255 * a)), font=f_cap)
        if t > t_end - 1.6:  # end card
            b = min(1.0, (t - (t_end - 1.6)) / 0.4)
            lines = [f"Placed: {res['add_mm']:.1f} mm / {res['rot_deg']:.0f}\u00b0 from the real pose"]
            if summary:
                o, r, h = summary["hulls_open_loop"], summary["hulls_reaimed"], summary["single_reaimed"]
                lines += [f"12 jittered trials \u2014 re-aimed {r['held']}/{r['trials']} held, median "
                          f"{r['add_mm_median']:.1f} mm / {r['rot_deg_median']:.0f}\u00b0",
                          f"open loop {o['held']}/{o['trials']}, {o['add_mm_median']:.1f} mm / "
                          f"{o['rot_deg_median']:.0f}\u00b0  \u00b7  one convex hull for the holder: "
                          f"{h['held']}/{h['trials']}"]
            fonts = [f_cap] + [f_small] * (len(lines) - 1)
            wmax = max(d.textlength(L, font=f) for L, f in zip(lines, fonts))
            x0, y0 = 28, Hh - 40 - 36 * len(lines)
            d.rounded_rectangle([x0, y0, x0 + wmax + 36, Hh - 24], 10, fill=(255, 255, 255, int(225 * b)))
            for i, (L, f) in enumerate(zip(lines, fonts)):
                d.text((x0 + 18, y0 + 10 + 36 * i), L, fill=(21, 128, 61, int(255 * b)) if i == 0 else
                       (30, 41, 59, int(255 * b)), font=f)
        frames.append(Image.alpha_composite(im, ov).convert("RGB"))
    out.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(out / "hand_place.mp4", fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
                        macro_block_size=8, output_params=["-crf", "23", "-preset", "slow"]) as w:
        for f in frames:
            w.append_data(np.asarray(f))
    small = [f.resize((800, 450), Image.LANCZOS) for f in frames[::3]]
    small[0].save(out / "hand_place.webp", save_all=True, append_images=small[1:], duration=100, loop=0,
                  quality=50, method=6)
    poster = frames[min(len(frames) - 1, int((ev["away"] + 0.8) * fps))]
    poster.save(out / "hand_place.jpg", quality=90)
    print(f"wrote {out / 'hand_place.mp4'} ({len(frames)} frames, {len(frames) / fps:.1f} s), webp, poster")
    compare_still(ctrl, holder, R_rel, t_rel, poses, out / "hand_place_compare.jpg")


def compare_still(ctrl, holder, R_rel, t_rel, poses, path_out: Path, size=(640, 480)):
    """Where the controller ends up, open loop vs re-aimed (seed 0), with the distance to the real pose."""
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font

    W, Hh = size
    tiles = []
    for correct, title in ((False, "Open loop"), (True, "Re-aimed from the in-hand pose")):
        su = Setup(0, ctrl, holder, R_rel, t_rel, "hulls", poses, correct=correct,
                   work_tag=f"still_{int(correct)}")
        execute(su, path(su), settle=2.0)
        res = score(ctrl, su.model, su.data, su.T_target)
        rend = mujoco.Renderer(su.model, Hh, W)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = su.T_target[:3, 3] + [0, 0, 0.015]
        cam.distance, cam.azimuth, cam.elevation = 0.36 * hand_motion.LENS_PUSH, 128, -10
        rend.update_scene(su.data, camera=cam)
        im = Image.fromarray(rend.render().copy())
        rend.close()
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([12, 12, 12 + d.textlength(title, font=_font(20)) + 24, 48], 8, fill="white")
        d.text((24, 17), title, fill=(30, 41, 59), font=_font(20))
        msg = f"{res['add_mm']:.1f} mm / {res['rot_deg']:.0f}\u00b0 from the real pose"
        d.rounded_rectangle([12, Hh - 48, 12 + d.textlength(msg, font=_font(18)) + 24, Hh - 12], 8, fill="white")
        d.text((24, Hh - 43), msg, fill=(21, 128, 61), font=_font(18))
        tiles.append(im)
    sheet = Image.new("RGB", (2 * W + 8, Hh), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * (W + 8), 0))
    sheet.save(path_out, quality=90)
    print(f"wrote {path_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--export-plans", type=Path, help="write the plans + hand targets (JSON) for Isaac Sim and exit")
    ap.add_argument("--export-task", type=Path, help="write the scene layout + plan inputs (JSON) for the Isaac Lab task")
    args = ap.parse_args()
    if not HAND.exists():
        raise SystemExit(f"{HAND} not found; fetch mujoco_menagerie/shadow_hand first (see README)")
    poses = span_poses()
    ctrl, holder, R_rel, t_rel = load()
    if args.export_task:
        export_task(ctrl, holder, R_rel, t_rel, poses, args.export_task)
        return
    if args.export_plans:
        export_plans(ctrl, holder, R_rel, t_rel, poses, range(args.trials), args.export_plans)
        return
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
        video(ctrl, holder, R_rel, t_rel, poses, args.out, summary=summary)


if __name__ == "__main__":
    main()
