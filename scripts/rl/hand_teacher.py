"""The scripted teacher for the learned hand-place task: scene layout and the scripted plan, per env.

numpy + scipy only (no Isaac imports), so the Isaac Lab task (hand_place_env.py), the rollout and
video script (scripted_rollout.py) and, later, demonstration collection share one implementation.
Inputs come from output/hand_task.json (python scripts/demo_hand_place.py --export-task ...): the same
layout, grasp and motion as the MuJoCo demo (demo_hand_place.py) and its plan (hand_motion.plan_path).

Frames: world z up, table at z = 0, poses as 4x4 matrices; quaternions (x, y, z, w) as Isaac Lab.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hand_motion  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK = os.path.join(ROOT, "output", "hand_task.json")


def yaw(deg: float) -> np.ndarray:
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


def pose7(T) -> np.ndarray:
    """4x4 -> (x, y, z, qx, qy, qz, qw)."""
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()]


def from7(p) -> np.ndarray:
    return hand_motion.pose(Rotation.from_quat(np.asarray(p[3:7], float)).as_matrix(), np.asarray(p[:3], float))


@dataclass
class Randomization:
    """Scene jitter (uniform, +-). The MuJoCo/Isaac benchmark used 0.02 m and 10 deg everywhere."""

    holder_xy: float = 0.02
    holder_yaw: float = 10.0
    tray_xy: float = 0.02
    controller_yaw: float = 10.0


class Spec:
    def __init__(self, path: str = TASK):
        s = json.load(open(path))
        self.raw = s
        self.rel = np.array(s["controller_in_holder"])
        self.gap = float(s["release_gap_m"])
        self.holder_xy, self.holder_yaw = np.array(s["holder_xy"]), float(s["holder_yaw_deg"])
        self.tray_xy, self.tray_h = np.array(s["tray_xy"]), float(s["tray_height_m"])
        self.ctrl_yaw = float(s["controller_yaw_deg"])
        self.tray_top_c, self.tray_half = float(s["tray_center_top_m"]), np.array(s["tray_half_extents"])
        ax = s["ctrl_axes"]
        self.right, self.top, self.face = (np.array(ax[k]) for k in ("right", "top", "face"))
        self.H, self.F = np.array(s["H"]), np.array(s["F"])
        self.open_key, self.transport = s["open_key"], float(s["transport"])
        self.hand_q = s["hand_q"]
        self.servo = s.get("servo", {})
        # the top-bottom pinch (demo_hand_pinch.py --export-task): its goal, plan and release settings
        self.grasp = s.get("grasp", "span")
        self.pinch = s.get("pinch", hand_motion.PINCH)
        self.center = np.array(s.get("grasp_center", [0.0, 0.0, 0.0]))
        self.tray_right_c = float(s.get("tray_center_right_m", 0.0))

    def scene(self, rng: np.random.Generator | None, rz: Randomization = Randomization(),
              layout_yaw: float = 0.0) -> dict:
        """One scene (as demo_hand_place.Setup): holder pose, target and goal, tray, controller start
        (before settling). rng None: the nominal scene (seed 0 of the benchmark). layout_yaw (deg) turns
        the whole cell about the vertical through its origin (for filming: the cells' layout against
        the env grid)."""
        if rng is None:
            def j(a, n=1):
                return np.zeros(n) if n > 1 else 0.0
        else:
            def j(a, n=1):
                return rng.uniform(-a, a, n) if n > 1 else float(rng.uniform(-a, a))
        T_hold = hand_motion.pose(yaw(self.holder_yaw + j(rz.holder_yaw)), [*(self.holder_xy + j(rz.holder_xy, 2)), 0.0])
        T_target = T_hold @ self.rel
        if self.grasp == "pinch":
            C_goal, approach = hand_motion.pinch_goal(T_target, self.face, self.right, self.center, self.pinch)
        else:
            C_goal = T_target.copy()
            C_goal[2, 3] += self.gap
            approach = -(C_goal[:3, :3] @ self.face)
        stand_xy = self.tray_xy + j(rz.tray_xy, 2)
        R0 = yaw(self.ctrl_yaw + j(rz.controller_yaw))
        center = stand_xy + R0[:2, :2] @ (self.tray_top_c * self.top[:2] + self.tray_right_c * self.right[:2])
        T_tray = hand_motion.pose(R0 @ np.c_[self.right, self.top, self.face], [*center, self.tray_h / 2])
        T_ctrl0 = hand_motion.pose(R0, [*stand_xy, self.tray_h + 0.003])
        G = hand_motion.pose(yaw(layout_yaw), np.zeros(3))
        return {"holder": G @ T_hold, "target": G @ T_target, "c_goal": G @ C_goal, "approach": G[:3, :3] @ approach,
                "tray": G @ T_tray, "tray_half_extents": self.tray_half.copy(), "controller_start": G @ T_ctrl0}

    def plan(self, T_start: np.ndarray, sc: dict) -> list[dict]:
        """Waypoints from the settled controller pose (hand_motion.plan_path or pinch_plan)."""
        if self.grasp == "pinch":
            return hand_motion.pinch_plan(T_start, sc["c_goal"], sc["approach"], self.H, self.F, self.face,
                                          self.transport, self.pinch)
        return hand_motion.plan_path(T_start, sc["c_goal"], sc["approach"], self.H, self.F, self.face,
                                     self.open_key, self.transport)

    def replan_place(self, sc: dict, H_rel: np.ndarray) -> list[dict]:
        """The re-aim: place waypoints for the forearm pose measured in the controller frame."""
        if self.grasp == "pinch":
            return hand_motion.pinch_place_waypoints(sc["c_goal"], sc["approach"], H_rel, self.transport, self.pinch)
        return hand_motion.place_waypoints(sc["c_goal"], sc["approach"], H_rel, self.open_key, self.transport)

    def joint_targets(self, names: list[str]) -> dict[str, np.ndarray]:
        """Hand pose key -> joint target vector in the articulation's joint order."""
        return {k: np.array([q.get(n, 0.0) for n in names], np.float32) for k, q in self.hand_q.items()}


class Track:
    """A plan sampled every dt, keeping its moves so that the placement can be re-aimed mid-run.

    From wps[0] (or from `start` = (4x4 pose, key), the current state, through all of wps). P: forearm poses
    (n, 7), Q: joint targets (n, J), ev: {tag: sample index at which the waypoint is reached} (stops and,
    for eased-end moves, passed-through waypoints)."""

    def __init__(self, wps: list[dict], dt: float, keys: dict[str, np.ndarray], start=None):
        self.wps, self.dt, self.keys = list(wps), dt, keys
        if start is None:
            T_prev, g_prev, k = self.wps[0]["T"], self.wps[0]["g"], 1
            P, Q = [pose7(T_prev)[None]], [np.asarray(keys[str(g_prev)])[None]]
        else:
            (T_prev, g_prev), k = start, 0
            P, Q = [], []
        self.moves, ev, count = [], {}, sum(len(x) for x in P)
        while k < len(self.wps):
            a, b = hand_motion.moves(self.wps, k)[0]
            mv = hand_motion.Move(T_prev, g_prev, self.wps[a:b + 1])
            qs = [keys[str(g)] for g in mv.g]
            n = max(1, int(round(mv.duration / dt)))
            i0 = count
            p, q = _samples(mv, qs, np.arange(n))
            P.append(p)
            Q.append(q)
            count += n
            D = float(mv.cum[-1])
            for j, w in enumerate(mv.legs[:-1]):
                if mv.ramp:  # a passed-through waypoint of an eased-end move: when it is reached
                    t_j = hand_motion.time_of(float(mv.cum[j + 1]), D, mv.ramp)
                    ev.setdefault(w["tag"], i0 + max(0, int(round(t_j * n / mv.duration)) - 1))
            ev.setdefault(self.wps[b]["tag"], count - 1)
            self.moves.append((a, b, mv, qs, n, i0))
            T_prev, g_prev = self.wps[b]["T"], self.wps[b]["g"]
            k = b + 1
        self.P = np.concatenate(P).astype(np.float32) if P else np.zeros((0, 7), np.float32)
        self.Q = np.concatenate(Q).astype(np.float32) if Q else np.zeros((0, len(next(iter(keys.values())))), np.float32)
        self.ev = ev

    def reaim(self, k_now: int, place: list[dict]):
        """Re-plan from the pre-place waypoint on (place: its re-aimed replacement, pre-place first), the run
        being at sample k_now. If the pre-place waypoint is passed through, the rest of its move is retargeted
        (hand_motion.Move.retarget: no velocity jump). Returns (first sample index, P tail, Q tail)."""
        tags = [w["tag"] for w in self.wps]
        p = tags.index("preplace")
        a, b, mv, qs, n, i0 = next(m for m in self.moves if m[0] <= p <= m[1])
        if p == b:  # a stop: everything after it re-planned from it
            tail = Track(place[1:], self.dt, self.keys, start=(self.wps[p]["T"], self.wps[p]["g"]))
            return i0 + n, tail.P, tail.Q
        nrem = b - p
        mv.retarget(p - a + 1, place[1:1 + nrem])
        i_from = max(k_now - i0 + 1, 0)
        P, Q = _samples(mv, qs, np.arange(i_from, n), n)
        rest = Track(place[1 + nrem:], self.dt, self.keys, start=(place[nrem]["T"], place[nrem]["g"]))
        return i0 + i_from, np.r_[P.astype(np.float32), rest.P], np.r_[Q.astype(np.float32), rest.Q]


def _samples(mv, qs, idx, n=None):
    """Samples idx (of n, default len(idx)) of a move: forearm poses (m, 7) and blended joint targets (m, J)."""
    n = len(idx) if n is None else n
    pos, quat, leg, u = mv.at_many((np.asarray(idx) + 1) * mv.duration / n)
    quat = Rotation.from_matrix(Rotation.from_quat(quat).as_matrix()).as_quat()  # as pose7 of a 4x4 (same signs)
    qa = np.asarray(qs, float)
    return np.c_[pos, quat], qa[leg] + (qa[leg + 1] - qa[leg]) * u[:, None]


def sample(wps: list[dict], dt: float, keys: dict[str, np.ndarray], start=None):
    """The waypoints' blended moves sampled every dt (Track): forearm poses (n, 7), joint targets (n, J) and
    {tag: sample index}."""
    tr = Track(wps, dt, keys, start)
    return tr.P, tr.Q, tr.ev
