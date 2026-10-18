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
        C_goal = T_target.copy()
        C_goal[2, 3] += self.gap
        approach = -(C_goal[:3, :3] @ self.face)
        stand_xy = self.tray_xy + j(rz.tray_xy, 2)
        R0 = yaw(self.ctrl_yaw + j(rz.controller_yaw))
        center = stand_xy + R0[:2, :2] @ (self.tray_top_c * self.top[:2])
        T_tray = hand_motion.pose(R0 @ np.c_[self.right, self.top, self.face], [*center, self.tray_h / 2])
        T_ctrl0 = hand_motion.pose(R0, [*stand_xy, self.tray_h + 0.003])
        G = hand_motion.pose(yaw(layout_yaw), np.zeros(3))
        return {"holder": G @ T_hold, "target": G @ T_target, "c_goal": G @ C_goal, "approach": G[:3, :3] @ approach,
                "tray": G @ T_tray, "tray_half_extents": self.tray_half.copy(), "controller_start": G @ T_ctrl0}

    def plan(self, T_start: np.ndarray, sc: dict) -> list[dict]:
        """Waypoints from the settled controller pose (hand_motion.plan_path)."""
        return hand_motion.plan_path(T_start, sc["c_goal"], sc["approach"], self.H, self.F, self.face,
                                     self.open_key, self.transport)

    def replan_place(self, sc: dict, H_rel: np.ndarray) -> list[dict]:
        """The re-aim: place waypoints for the forearm pose measured in the controller frame."""
        return hand_motion.place_waypoints(sc["c_goal"], sc["approach"], H_rel, self.open_key, self.transport)

    def joint_targets(self, names: list[str]) -> dict[str, np.ndarray]:
        """Hand pose key -> joint target vector in the articulation's joint order."""
        return {k: np.array([q.get(n, 0.0) for n in names], np.float32) for k, q in self.hand_q.items()}


def sample(wps: list[dict], dt: float, keys: dict[str, np.ndarray], start=None):
    """The waypoints' blended moves sampled every dt, from wps[0] (or from `start` = (4x4 pose, key),
    the current state, through all of wps). Returns forearm poses (n, 7), joint targets (n, J) and
    {tag: sample index at which its move ends}."""
    if start is None:
        T_prev, g_prev, k = wps[0]["T"], wps[0]["g"], 1
        P, Q = [pose7(T_prev)], [keys[str(g_prev)]]
    else:
        (T_prev, g_prev), k = start, 0
        P, Q = [], []
    ev = {}
    while k < len(wps):
        a, b = hand_motion.moves(wps, k)[0]
        mv = hand_motion.Move(T_prev, g_prev, wps[a:b + 1])
        qs = [keys[str(g)] for g in mv.g]
        n = max(1, int(round(mv.duration / dt)))
        for i in range(n):
            T, leg, u = mv.at((i + 1) * mv.duration / n)
            P.append(pose7(T))
            Q.append(qs[leg] + (qs[leg + 1] - qs[leg]) * u)
        T_prev, g_prev = wps[b]["T"], wps[b]["g"]
        ev[wps[b]["tag"]] = len(P) - 1
        k = b + 1
    return np.array(P, np.float32), np.array(Q, np.float32), ev
