"""A Shadow Hand picks the scanned controller with a top-bottom pinch and places it in the scanned holder.

    python scripts/demo_hand_pinch.py [--trials 12] [--probe]

The second grasp for the placement of demo_hand_place.py (same scene, answer key, motion blending and
re-aim). There the hand spans the controller's width, thumb on one grip, fingers on the other. Here it
pinches the controller's height on its center line, the way a person picks one up to stand it in a
holder: index and middle on the top rim, straddling the center of mass, the thumb under the bottom edge,
all slightly under the edges' widest so that face up the edges rest on the pads. Upright, the thumb is
under the controller and carries its weight.

The motion: come down over the face with the fingers spread past the edges, settle the fingers on the rim,
close the thumb, lift, turn upright, carry, re-aim in front of the holder, set down in its hands (the
holder stands on a pedestal as tall as the tray: upright, this grip's forearm comes from below the
controller), then let go: the fingers lift off the rim and the thumb drops and slides out from under the
bottom edge as the hand backs off, the fingers open, and the open hand nudges the top back onto the
holder before leaving. The holder holds the controller on the same center line, so the thumb meets its
hands for the last couple of centimetres of the set-down.

Grasp design (solve_grasp): the contact points are found on the scan (ray casts), then the forearm pose
and the hand joints are solved together (least squares) to put the three pads on them, facing in, with
no other part of the hand in the controller or (from RELEASE_CHECK up) the holder; then, with the forearm
fixed, the grip (pads squeezed in), hook (fingers at the rim, thumb open), let-go and open poses, each
checked along the way it is reached. Cached in output/.

Robot: mujoco_menagerie/shadow_hand (Apache-2.0); each finger's distal joint coupled to its middle one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_hand_place as HP  # noqa: E402
import hand_motion  # noqa: E402
from demo_holder_place import body_pose, load, pose, yaw_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HAND = HP.HAND
GRASP = ROOT / "output" / "hand_pinch_grasp.json"
JOINTS = ["rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1", "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_MFJ4", "rh_MFJ3",
          "rh_MFJ2", "rh_WRJ1", "rh_WRJ2", "rh_RFJ3", "rh_RFJ2", "rh_LFJ3", "rh_LFJ2"]
PADS = (("rh_thdistal", 0.008), ("rh_ffdistal", 0.0065), ("rh_mfdistal", 0.0065))
KEYS = ("open", "hook", "contact", "grip", "letgo")  # the hand poses (joint targets) of the motion
CLEAR = 0.012  # ring, little and the palm at least this far from the controller (they are
# in the squeeze's way; a curled finger touching the face takes the thumb's push instead of the pinch)

# Contacts (controller frame, m): the pinch along the controller's height on its center line, slightly under
# the edges' widest (so face up the edges rest on the pads); index and middle straddle the center of mass
# (lateral +0.4 cm). The holder holds the controller on the same line (its hands round the bottom edge, its
# head behind the top rim): the thumb slides out from under the bottom edge as the controller is set down.
THUMB_AT = (0.005, 0.036)  # (lateral, height above the back)
INDEX_AT, MIDDLE_AT = (-0.011, 0.028), (0.021, 0.028)
RELEASE_CHECK = 0.020  # the grasp is solved clear of the holder from this far above the seat; lower down, only the
# thumb (under the bottom edge, where the holder's hands are) meets them, and gives
RELEASE_LEAN = 0.005  # set down this far further in than the seat, the controller's back toward the holder's chest
LETGO_THUMB, LETGO_LIFT, LETGO_BACK = 0.6, 0.012, 0.03  # let go: the fingers lift off the rim (they sit under its
# widest: backing off with them on it would pull the top forward) and the thumb drops away (this fraction of the
# way to open) as the hand backs off toward the robot, sliding it out from under the bottom edge
NUDGE = (0.02, -0.015)  # then the open hand pushes the top back onto the holder: this far in (past the seat), lower
SQUEEZE, OPENING = 0.008, 0.020  # grip: pads this far into the surfaces; open: this far out
HOOK_GAP = 0.003  # first step of the close: the finger pads this close to the rim, the thumb still open
FOREARM_MIN_UP = -0.20  # forearm origin vs the grasp along "top": at most this far below it, upright on the holder
PEDESTAL_H, PEDESTAL_HALF = HP.STAND_H, 0.06  # the holder's pedestal: as tall as the tray, 12 cm square
RING_CURL = (1.2, 1.5)  # ring and little J3, J2 (= J1) to start from: curled toward the palm
FINGER_GAIN = 10.0  # finger servo stiffness multiplier (demo_hand_place uses 5); torque limits unchanged


# --- grasp design ------------------------------------------------------------------------------------------

def contact_targets(ctrl) -> list[tuple[np.ndarray, np.ndarray]]:
    """(point, outward surface normal) for the thumb, index and middle pads, in the controller frame: rays
    along the controller's height at the given lateral position and height, up from below the grips for
    the thumb, down from above the top edge for the fingers."""
    mesh = ctrl.asset.render_mesh

    def hit(at, sign):
        o = at[0] * ctrl.right - sign * 0.12 * ctrl.top + np.array([0, 0, at[1]])
        loc, _, _ = mesh.ray.intersects_location([o], [sign * ctrl.top], multiple_hits=False)
        near = np.linalg.norm(mesh.triangles_center - loc[0], axis=1) < 0.005  # the scan is noisy: average
        n = (mesh.face_normals[near] * mesh.area_faces[near, None]).sum(0)
        return loc[0], n / np.linalg.norm(n)

    return [hit(THUMB_AT, 1.0), hit(INDEX_AT, -1.0), hit(MIDDLE_AT, -1.0)]


def grasp_q(x) -> dict[str, float]:
    """Joint targets for a solved pose (JOINTS): each finger's J1 equals its J2."""
    q = {n: float(v) for n, v in zip(JOINTS, x)}
    for f in ("FF", "MF", "RF", "LF"):
        q[f"rh_{f}J1"] = q[f"rh_{f}J2"]
    q["rh_LFJ5"] = 0.1
    return q


class _Solver:
    """The hand next to the controller (identity pose) for the grasp optimization, with the holder where it
    is when the controller is seated in it: a grasp the holder's hands or head would hit is no good either."""

    def __init__(self, ctrl, holder, R_rel, t_rel):
        self.ctrl = ctrl
        self.T_c = pose(np.eye(3), [0, 0, 0.5])
        self.T_h = self.T_c @ np.linalg.inv(pose(R_rel, t_rel))  # the holder, the controller seated in it ...
        self.T_h[:3, 3] -= RELEASE_CHECK * (R_rel.T @ [0, 0, 1.0])  # ... and RELEASE_CHECK (straight up) above that
        m = self.m = HP.build(ctrl, holder, self.T_h, self.T_c, pose(np.eye(3), [5, -5, 0]), (0.01, 0.01), "hulls",
                              ROOT / "data" / "work" / "hand_pinch_solve")
        self.d = mujoco.MjData(m)
        self.cadr = m.joint(m.body("ctrl").jntadr[0]).qposadr[0]
        self.fadr = m.joint("forearm_free").qposadr[0]
        self.J = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
        self.lo = np.array([m.jnt_range[m.joint(n).id][0] for n in JOINTS])
        self.hi = np.array([m.jnt_range[m.joint(n).id][1] for n in JOINTS])
        self.cid, self.hid = m.body("ctrl").id, m.body("holder").id
        self.hand = sorted({int(m.geom_bodyid[g]) for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("rh_")})
        self.pads = {m.body(b).id for b, _ in PADS}
        # ring, little and the palm keep CLEAR from the controller (margin: MuJoCo reports them that close)
        self.clear = {b for b in self.hand if m.body(b).name.startswith(("rh_rf", "rh_lf", "rh_palm"))}
        for gi in range(m.ngeom):
            if m.geom_bodyid[gi] in self.clear:
                m.geom_margin[gi] = CLEAR
        self.targets = contact_targets(ctrl)
        self.center = (self.targets[0][0] + 0.5 * (self.targets[1][0] + self.targets[2][0])) / 2
        # open: every pad OPENING out along its surface normal (the fingers above the top edge, the thumb below
        # the grip): the hand comes down over the face and backs off the seated controller between them
        self.open_offsets = [OPENING * n for _, n in self.targets]

    def set(self, H, x, shift=0.0):
        """Hand at forearm pose H (controller frame), joints x; controller and holder moved by -shift along
        the face normal (= the hand that far above the face / backed off from the seated controller)."""
        m, d = self.m, self.d
        T = self.T_c @ H
        off = -shift * self.ctrl.face
        d.qpos[:] = 0
        d.qpos[self.cadr:self.cadr + 3] = self.T_c[:3, 3] + off
        d.qpos[self.cadr + 3:self.cadr + 7] = [1, 0, 0, 0]
        m.body_pos[self.hid] = self.T_h[:3, 3] + off
        d.qpos[self.fadr:self.fadr + 3] = T[:3, 3]
        d.qpos[self.fadr + 3:self.fadr + 7] = HP.quat_wxyz(T[:3, :3])
        for n, v in grasp_q(x).items():
            d.qpos[self.J[n]] = v
        mujoco.mj_fwdPosition(m, d)

    def pad_poses(self):
        return [(p - self.T_c[:3, 3], n) for p, n in (HP._tip(self.m, self.d, b, r) for b, r in PADS)]

    def penetration(self, allow_pads, holder=True, ctrl=True):
        """Deepest penetration per hand body (m) into the controller (pads allowed `allow_pads`) and the holder."""
        m, d = self.m, self.d
        per = np.zeros(len(self.hand))
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            other = b2 if b1 in self.hand else b1
            hb = int(b1 if b1 in self.hand else b2)
            if (hb not in self.hand or other not in (self.cid, self.hid) or (other == self.hid and not holder)
                    or (other == self.cid and not ctrl)):
                continue
            k = self.hand.index(hb)
            allow = (allow_pads if hb in self.pads and other == self.cid
                     else -CLEAR if hb in self.clear and other == self.cid else 0.0)
            per[k] = max(per[k], -c.dist - allow)
        return np.maximum(per, 0)

    def forearm(self):
        return self.d.xpos[self.m.body("rh_forearm").id] - self.T_c[:3, 3] - self.center

    def residual(self, v, H0, x_ref, mode, x_next):
        H = H0 @ pose(Rotation.from_rotvec(v[3:6]).as_matrix(), v[:3] * 0.1)
        x = v[6:]
        self.set(H, x)
        r = []
        for k, ((p, n), (T, nT)) in enumerate(zip(self.pad_poses(), self.targets)):
            goal = self.goal(mode, k)
            r.extend(1000 * (p - goal))  # mm
            r.append(30 * (1 + n @ nT))  # pad facing into the surface
        if mode == "grip":  # the pads are meant to be inside; only the holder must stay clear
            r.extend(3000 * self.penetration(1.0))
        else:
            r.extend(3000 * self.penetration(0.0015 if mode == "contact" else 0.0))
        if mode == "open":  # clear while coming down onto the face / backing off the seated controller
            for s in (0.01, 0.02, 0.04):
                self.set(H, x, s)
                r.extend(3000 * self.penetration(0.0))
        if mode in ("open", "hook"):  # ... and while the fingers move on to the next pose (open > hook > contact)
            for a in (0.25, 0.5, 0.75):
                self.set(H, x + a * (x_next - x))
                r.extend(3000 * self.penetration(0.0015 if mode == "hook" else 0.0))
            self.set(H, x)
        if mode == "letgo":  # backing off with the thumb out from under the bottom edge: clear of the holder
            for s in (0.01, 0.02, LETGO_BACK):
                self.set(H, x, s)
                r.extend(3000 * self.penetration(0.0, ctrl=False))
            self.set(H, x)
        fo = self.forearm()
        r.append(100 * max(0.0, FOREARM_MIN_UP - fo @ self.ctrl.top))  # upright on the holder: the forearm's drop
        r.append(100 * max(0.0, 0.06 - fo @ self.ctrl.face))  # ... in front of the face (the robot's side)
        r.extend(0.3 * (x - x_ref))
        return np.array(r)

    def goal(self, mode, k):
        T, n = self.targets[k]
        if mode == "grip":
            return T - SQUEEZE * n
        if mode == "open" or (mode == "hook" and k == 0):
            return T + self.open_offsets[k]
        if mode == "letgo" and k == 0:
            return T  # the thumb stays (PinchSetup drops it LETGO_THUMB of the way to open)
        if mode == "letgo":
            return T + LETGO_LIFT * n
        if mode == "hook":
            return T + HOOK_GAP * n
        return T

    def solve(self, H0, x0, mode="contact", x_next=None):
        lb, ub = np.r_[[-np.inf] * 6, self.lo], np.r_[[np.inf] * 6, self.hi]
        if mode != "contact":
            lb[:6], ub[:6] = -1e-9, 1e-9
        res = least_squares(self.residual, np.r_[np.zeros(6), x0], bounds=(lb, ub),
                            args=(H0, x0, mode, x_next if x_next is not None else x0), diff_step=1e-3,
                            max_nfev=3000)
        return H0 @ pose(Rotation.from_rotvec(res.x[3:6]).as_matrix(), res.x[:3] * 0.1), res.x[6:], res.cost

    def report(self, tag, H, x, mode=None):
        self.set(H, x)
        err = [np.linalg.norm(p - self.goal(mode or tag, k)) * 1000 for k, (p, _) in enumerate(self.pad_poses())]
        facing = [-(n @ nT) for (_, n), (_, nT) in zip(self.pad_poses(), self.targets)]
        pen = self.penetration(0.0)
        worst = {self.m.body(self.hand[i]).name: round(pen[i] * 1000, 1) for i in np.argsort(-pen)[:3] if pen[i] > 0}
        print(f"{tag}: pads {np.round(err, 1)} mm from target, facing {np.round(facing, 2)}, penetration (mm) {worst}",
              flush=True)


def _seed_pose(ctrl):
    """A first guess: palm over the top right of the face, fingers toward the top edge, thumb toward the bottom."""
    y = ctrl.top  # thumb -> fingers
    z = -ctrl.face  # palm -> controller
    G = pose(np.c_[np.cross(y, z), y, z], 0.045 * ctrl.right + 0.022 * ctrl.top + np.array([0, 0, 0.03]))
    x0 = np.r_[0.35, 0.8, 0.12, 0.1, -0.2, 0.3, 0.2, 0.7, 0.3, 0.5, 0.6, -0.6, -0.4, RING_CURL, RING_CURL]
    # the seed hand's grasp frame in its forearm frame: midway between the thumb pad and the finger pads
    m = mujoco.MjModel.from_xml_path(str(HAND / "right_hand.xml"))
    d = mujoco.MjData(m)
    J = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
    for n, v in grasp_q(x0).items():
        d.qpos[J[n]] = v
    mujoco.mj_kinematics(m, d)
    pt, _ = HP._tip(m, d, "rh_thdistal", 0.008)
    pf = 0.5 * (HP._tip(m, d, "rh_ffdistal", 0.0065)[0] + HP._tip(m, d, "rh_mfdistal", 0.0065)[0])
    ax = (pf - pt) / np.linalg.norm(pf - pt)
    palm = d.xpos[m.body("rh_palm").id] + d.xmat[m.body("rh_palm").id].reshape(3, 3) @ [0, -0.01, 0.05]
    zz = (pt + pf) / 2 - palm
    zz -= ax * (zz @ ax)
    zz /= np.linalg.norm(zz)
    f = m.body("rh_forearm").id
    F = np.linalg.inv(pose(d.xmat[f].reshape(3, 3), d.xpos[f])) @ pose(np.c_[np.cross(ax, zz), ax, zz], (pt + pf) / 2)
    return G @ np.linalg.inv(F), x0


def solve_grasp(ctrl, holder, R_rel, t_rel, starts: int = 6) -> dict:
    """The grasp: forearm pose H in the controller frame and joint vectors (JOINTS) for open, contact and grip."""
    cfg = {"thumb": THUMB_AT, "index": INDEX_AT, "middle": MIDDLE_AT, "squeeze": SQUEEZE, "opening": OPENING,
           "hook": HOOK_GAP, "ring": RING_CURL, "forearm": FOREARM_MIN_UP, "clear": CLEAR, "check": RELEASE_CHECK,
           "letgo": LETGO_LIFT, "v": 12}
    if GRASP.exists():
        g = json.loads(GRASP.read_text())
        if g.get("cfg") == json.loads(json.dumps(cfg)):
            return {k: np.array(v) if k in KEYS + ("H",) else v for k, v in g.items()}
    s = _Solver(ctrl, holder, R_rel, t_rel)
    H_init, x_init = _seed_pose(ctrl)
    rng = np.random.default_rng(0)
    best = None
    for k in range(starts):
        H0 = H_init if k == 0 else H_init @ pose(Rotation.from_rotvec(rng.normal(0, 0.25, 3)).as_matrix(),
                                                 rng.normal(0, 0.015, 3))
        x0 = np.clip(x_init + (0 if k == 0 else rng.normal(0, 0.2, len(JOINTS))), s.lo, s.hi)
        H, x, cost = s.solve(H0, x0)
        print(f"grasp start {k}: cost {cost:.1f}", flush=True)
        if best is None or cost < best[2]:
            best = (H, x, cost)
    H, x, _ = best
    s.report("contact", H, x)
    _, grip, _ = s.solve(H, x, mode="grip")
    s.report("grip", H, grip)
    _, hook, _ = s.solve(H, x, mode="hook", x_next=x)
    s.report("hook", H, hook)
    _, letgo, _ = s.solve(H, x, mode="letgo")
    s.report("letgo", H, letgo)
    _, opened, _ = s.solve(H, hook, mode="open", x_next=hook)
    s.report("open", H, opened)
    s.set(H, x)
    fo = s.forearm()
    print(f"forearm from the grasp: right {fo @ ctrl.right:+.2f}, top {fo @ ctrl.top:+.2f}, face {fo @ ctrl.face:+.2f} m")
    g = {"joints": JOINTS, "H": H.tolist(), "open": opened.tolist(), "hook": hook.tolist(), "letgo": letgo.tolist(),
         "contact": x.tolist(),
         "grip": grip.tolist(), "center": s.center.tolist(), "cfg": cfg}
    GRASP.parent.mkdir(parents=True, exist_ok=True)
    GRASP.write_text(json.dumps(g, indent=1))
    return solve_grasp(ctrl, holder, R_rel, t_rel)


# --- scene and plan ---------------------------------------------------------------------------------------

class PinchSetup:
    """One trial's scene (demo_hand_place.Setup's layout and jitter) with the photographed grip."""

    def __init__(self, seed, ctrl, holder, R_rel, t_rel, collision, grasp, ghost=False, work_tag=None,
                 transport=2.0, correct=False, finger_gain=FINGER_GAIN):
        rng = np.random.default_rng(seed)
        jit = seed > 0
        hold_xy = HP.HOLDER_XY + (rng.uniform(-0.02, 0.02, 2) if jit else 0)
        self.hold_yaw = 0.0 + (rng.uniform(-10, 10) if jit else 0)
        # The holder on a pedestal as tall as the controller's tray: set down on it upright, the controller is
        # held from below and in front (the thumb under it), and the forearm runs down toward the robot.
        T_hold = pose(yaw_matrix(self.hold_yaw), [*hold_xy, PEDESTAL_H])
        ped = (f'<body name="pedestal" pos="{hold_xy[0]:.5f} {hold_xy[1]:.5f} {PEDESTAL_H / 2:.5f}" '
               f'quat="{" ".join(f"{q:.6f}" for q in HP.quat_wxyz(yaw_matrix(self.hold_yaw)))}">'
               f'<geom type="box" size="{PEDESTAL_HALF} {PEDESTAL_HALF} {PEDESTAL_H / 2}" rgba="0.62 0.60 0.57 1"/></body>')
        self.T_target = T_hold @ pose(R_rel, t_rel)
        self.ctrl, self.correct, self.transport = ctrl, correct, transport
        self.C_goal = self.T_target.copy()
        self.al = -(self.C_goal[:3, :3] @ ctrl.face)  # the approach at the holder: into the face
        self.C_goal[:3, 3] += RELEASE_LEAN * self.al  # set down leaning toward the holder's chest
        stand_xy = HP.STAND_XY + (rng.uniform(-0.02, 0.02, 2) if jit else 0)
        yaw0 = np.degrees(np.arctan2(HP.EDGE_DIR[1], HP.EDGE_DIR[0]) - np.arctan2(ctrl.top[1], ctrl.top[0]))
        yaw0 += rng.uniform(-10, 10) if jit else 0
        R0 = yaw_matrix(yaw0)
        # The tray under the controller, from the grips' ends to 2.5 cm short of the top edge: the index and
        # middle tips go round the back of the top rim (the controller rests on its grips and shoulders).
        l_lo, l_hi = ctrl.lat.min() + 0.004, ctrl.lat.max() - 0.004
        t_lo, t_hi = ctrl.up.min() + 0.003, ctrl.up.max() - 0.025
        self.stand_half = (0.5 * (l_hi - l_lo), 0.5 * (t_hi - t_lo))
        c_loc = 0.5 * (l_lo + l_hi) * ctrl.right[:2] + 0.5 * (t_lo + t_hi) * ctrl.top[:2]
        stand_center = stand_xy + R0[:2, :2] @ c_loc
        T_stand = pose(R0 @ np.c_[ctrl.right, ctrl.top, ctrl.face], [*stand_center, 0.0])
        T_ctrl0 = pose(R0, [*stand_xy, HP.STAND_H + 0.003])
        work = ROOT / "data" / "work" / f"hand_pinch_{work_tag or collision}"
        self.model = HP.build(ctrl, holder, T_hold, T_ctrl0, T_stand, self.stand_half,
                              "hulls" if collision == "hulls" else "single", work,
                              ghost=self.T_target if ghost else None, finger_gain=finger_gain, couple_distal=True,
                              extra_xml=ped)
        m = self.model
        d = self.data = mujoco.MjData(m)
        cadr = m.joint(m.body("ctrl").jntadr[0]).qposadr[0]
        d.qpos[cadr:cadr + 3] = T_ctrl0[:3, 3]
        d.qpos[cadr + 3:cadr + 7] = HP.quat_wxyz(R0)
        fadr = m.joint("forearm_free").qposadr[0]
        d.qpos[fadr:fadr + 3] = [0.0, -0.6, 0.6]
        d.mocap_pos[0] = [0.0, -0.6, 0.6]
        for _ in range(750):
            mujoco.mj_step(m, d)
        d.time = 0.0
        self.T_start = body_pose(m, d, "ctrl")
        self.H = np.asarray(grasp["H"])  # forearm pose in the controller frame
        # The pick approach (hand_motion.plan_path reads it as F's z): straight down onto the face.
        center = np.asarray(grasp["center"])
        G = pose(np.c_[ctrl.right, -ctrl.top, -ctrl.face], center)
        self.F = np.linalg.inv(self.H) @ G
        # hook: the fingers just under the rim, the thumb still off the face (closing goes open > hook > grip,
        # fingers first; on release the thumb lets go first)
        self.q = {k: grasp_q(grasp[k]) for k in KEYS}
        x = np.array(grasp["letgo"], float)
        x[:5] = grasp["contact"][:5] + LETGO_THUMB * (np.asarray(grasp["open"][:5]) - grasp["contact"][:5])
        self.q["letgo"] = grasp_q(x)

    def ctrl_of(self, g):
        return HP.hand_ctrl_q(self.model, self.q[g])

    def place_waypoints(self, H_rel):
        return place_waypoints(self.C_goal, self.al, H_rel, self.transport)


def place_waypoints(C_goal, approach, H_rel, transport):
    """From the pre-place pose on (hand_motion.place_waypoints' moves) with this grasp's release: set down in
    the holder's hands, the fingers lift off the rim and the thumb drops away as the hand backs off LETGO_BACK
    (sliding it out from under the bottom edge), the fingers open, and the open hand nudges the top back onto
    the holder before leaving."""
    wps = hand_motion.place_waypoints(C_goal, approach, H_rel, "open", transport)
    k = next(i for i, w in enumerate(wps) if w["tag"] == "release")
    a = np.asarray(approach, float)
    T = wps[k]["T"].copy()
    T[:3, 3] -= LETGO_BACK * a
    Tn = T.copy()
    Tn[:3, 3] += (LETGO_BACK + NUDGE[0]) * a + [0, 0, NUDGE[1]]
    letgo = dict(wps[k], T=T, g="letgo", dur=0.5, tag="release")
    uncurl = dict(wps[k], T=T, g="open", dur=0.5, tag="uncurl")
    nudge = dict(wps[k], T=Tn, g="open", dur=0.6, tag="nudge")
    off = dict(wps[k], T=T, g="open", dur=0.5, tag="unnudge")
    return wps[:k] + [letgo, uncurl] + ([nudge, off] if NUDGE[0] else []) + wps[k + 1:]


def pick_waypoints(A0, approach, C_mid_H):
    """Come down over the face with the fingers spread past the top edge, curl them round the rim, press
    the thumb on, lift and swing to the mid pose."""
    a = np.asarray(approach, float)

    def shift(T, v):
        return pose(T[:3, :3], T[:3, 3] + v)

    wp = hand_motion.wp
    return [
        wp(shift(A0, -0.10 * a + [0, 0, 0.10]), "open", 0.0, tag="start"),
        wp(shift(A0, -0.05 * a), "open", 1.4, via=True, tag="approach"),
        wp(A0, "open", 0.9, tag="pick"),
        wp(A0, "hook", 0.6, tag="hook"),
        wp(A0, "grip", 0.7, tag="grip"),
        wp(A0, "grip", 0.3, tag="settle"),
        wp(shift(A0, [0, 0, 0.08]), "grip", 1.1, via=True, tag="lift"),
        wp(C_mid_H, "grip", 1.8, via=True, tag="turn"),
    ]


def path(su):
    """Waypoints for the whole placement (the moves of hand_motion.plan_path with this grasp's closing and
    release steps)."""
    A0 = su.T_start @ su.H
    ap = (A0 @ su.F)[:3, 2]
    C = su.C_goal
    f = C[:3, :3] @ su.ctrl.face
    tip = Rotation.align_vectors([[0, 0, 1.0]], [f])[0].as_matrix()
    C_mid = pose(tip @ C[:3, :3], C[:3, 3] - 0.12 * np.asarray(su.al) + [0, 0, 0.10])
    return pick_waypoints(A0, ap, C_mid @ su.H) + place_waypoints(C, su.al, su.H, su.transport)


def pose_hand(su, T_forearm, q: dict):
    """Put the hand (statically) at a forearm pose with joint angles q."""
    m, d = su.model, su.data
    fadr = m.joint("forearm_free").qposadr[0]
    d.qpos[fadr:fadr + 3] = T_forearm[:3, 3]
    d.qpos[fadr + 3:fadr + 7] = HP.quat_wxyz(T_forearm[:3, :3])
    for j in range(m.njnt):
        n = m.joint(j).name
        if n.startswith("rh_"):
            d.qpos[m.jnt_qposadr[j]] = q.get(n, 0.0)
    mujoco.mj_forward(m, d)


def probe_contacts(su, key="grip"):
    """Hand bodies touching the controller / tray with the hand posed (no dynamics) at the grasp."""
    pose_hand(su, su.T_start @ su.H, su.q[key])
    m, d = su.model, su.data
    cid, sid = m.body("ctrl").id, m.body("stand").id
    hits = {}
    for i in range(d.ncon):
        b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
        for a, b in ((b1, b2), (b2, b1)):
            if b in (cid, sid) and m.body(a).name.startswith("rh_"):
                k = f"{m.body(a).name}/{'ctrl' if b == cid else 'tray'}"
                hits[k] = min(hits.get(k, 1.0), round(float(d.contact[i].dist) * 1000, 1))
    return hits


def trial(seed, ctrl, holder, R_rel, t_rel, grasp, correct):
    su = PinchSetup(seed, ctrl, holder, R_rel, t_rel, "hulls", grasp, correct=correct, work_tag=f"trial_{int(correct)}")
    HP.execute(su, path(su))
    return {"seed": seed, "reaimed": correct, **HP.score(ctrl, su.model, su.data, su.T_target)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="solve the grasp and list the contacts at the pick")
    ap.add_argument("--trials", type=int, default=12)
    args = ap.parse_args()
    if not HAND.exists():
        raise SystemExit(f"{HAND} not found; fetch mujoco_menagerie/shadow_hand first (see README)")
    ctrl, holder, R_rel, t_rel = load()
    grasp = solve_grasp(ctrl, holder, R_rel, t_rel)
    if args.probe:
        su = PinchSetup(0, ctrl, holder, R_rel, t_rel, "hulls", grasp, work_tag="probe")
        for k in KEYS:
            print(k, probe_contacts(su, k))
        return
    summary = {}
    for correct in (False, True):
        rows = [trial(s, ctrl, holder, R_rel, t_rel, grasp, correct) for s in range(args.trials)]
        for r in rows:
            print(f"seed {r['seed']:2d} {'re-aimed' if correct else 'open loop'}: placed {r['placed']} held {r['held']} "
                  f"{r['add_mm']:.1f} mm {r['rot_deg']:.1f} deg", flush=True)
        ok = [r for r in rows if r["held"]]
        key = "reaimed" if correct else "open_loop"
        summary[key] = {"trials": len(rows), "held": len(ok), "placed": sum(r["placed"] for r in rows),
                        "add_mm_median": float(np.median([r["add_mm"] for r in ok])) if ok else None,
                        "rot_deg_median": float(np.median([r["rot_deg"] for r in ok])) if ok else None}
        print(key, summary[key], flush=True)
    out = ROOT / "output" / "hand_pinch_trials.json"
    out.write_text(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
