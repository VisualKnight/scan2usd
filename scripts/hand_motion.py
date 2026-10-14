"""Hand placement motion: waypoints, blending and the re-aim step, shared by every engine.

numpy + scipy only, so MuJoCo (demo_hand_place.py), Isaac Sim (isaac_hand_place.py) and the Unreal
export run the same plan with the same timing.

A waypoint is a dict: T (4x4 forearm pose, world), g (hand pose key), dur (s), via (pass through
without stopping) and tag (a name for events: cameras, captions, the re-aim). Consecutive via
waypoints and the stop that ends them form one move: a minimum-jerk time scaling over the whole move,
a Hermite spline through the positions and a slerp per leg, so the hand blends through the via points
instead of stopping at each.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def pose(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def wp(T, g, dur, via=False, tag=""):
    return {"T": np.asarray(T, float), "g": g, "dur": float(dur), "via": bool(via), "tag": tag}


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def moves(wps: list[dict], start: int = 0) -> list[tuple[int, int]]:
    """[first, last] index pairs of the moves from waypoint `start` on (each ends at a stop)."""
    out, k = [], start
    while k < len(wps):
        j = k
        while j < len(wps) - 1 and wps[j]["via"]:
            j += 1
        out.append((k, j))
        k = j + 1
    return out


class Move:
    """One move from pose T_prev (hand pose key g_prev) through wps[a..b]."""

    def __init__(self, T_prev, g_prev, legs: list[dict]):
        self.legs = legs
        self.P = np.array([T_prev[:3, 3]] + [w["T"][:3, 3] for w in legs])
        self.R = [T_prev[:3, :3]] + [w["T"][:3, :3] for w in legs]
        self.g = [g_prev] + [w["g"] for w in legs]
        self.d = np.array([w["dur"] for w in legs])
        self.cum = np.r_[0.0, np.cumsum(self.d)]
        self.duration = float(self.cum[-1])
        n = len(legs)
        # Tangents (per second of path parameter): centered differences inside, one-sided at the ends.
        self.m = np.zeros_like(self.P)
        for i in range(n + 1):
            lo, hi = max(i - 1, 0), min(i + 1, n)
            self.m[i] = (self.P[hi] - self.P[lo]) / max(self.cum[hi] - self.cum[lo], 1e-9)
        self.slerps = [Slerp([0, 1], Rotation.from_matrix([self.R[i], self.R[i + 1]])) for i in range(n)]

    def at(self, t: float):
        """(4x4 pose, leg index, leg fraction) at time t in [0, duration]."""
        p = min_jerk(t / self.duration) * self.duration if self.duration > 0 else 0.0
        i = int(np.clip(np.searchsorted(self.cum, p, side="right") - 1, 0, len(self.legs) - 1))
        h = self.d[i]
        u = float(np.clip((p - self.cum[i]) / h, 0, 1)) if h > 0 else 1.0
        h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
        pos = h00 * self.P[i] + h10 * h * self.m[i] + h01 * self.P[i + 1] + h11 * h * self.m[i + 1]
        return pose(self.slerps[i]([u]).as_matrix()[0], pos), i, u


def place_waypoints(C_goal, approach, H_rel, open_key, transport, grip_key="grip"):
    """From the pre-place pose on, for a forearm pose H_rel in the controller frame: in front of the
    holder, slide in along the approach (into the face), set down, open, back out, rise."""

    def at(v):  # forearm pose with the controller at its goal pose shifted by v (world)
        return pose(C_goal[:3, :3], C_goal[:3, 3] + v) @ H_rel

    a = np.asarray(approach, float)
    return [
        wp(at(-0.12 * a + [0, 0, 0.04]), grip_key, transport, tag="preplace"),
        wp(at(-0.05 * a), grip_key, 1.1, via=True, tag="insert"),
        wp(at(np.zeros(3)), grip_key, 1.0, tag="seat"),
        wp(at(np.zeros(3)), open_key, 0.8, tag="release"),
        wp(at(-0.08 * a), open_key, 0.9, via=True, tag="back"),
        wp(at(-0.22 * a + [0, 0, 0.30]), open_key, 1.6, tag="away"),
    ]


def pick_waypoints(A0, approach, C_mid_H, open_key, grip_key="grip"):
    """Start above the controller, come down over its face with the fingers open, close, lift and
    swing (palm still down) to the mid pose; the pre-place pose follows (place_waypoints)."""
    a = np.asarray(approach, float)

    def shift(T, v):
        return pose(T[:3, :3], T[:3, 3] + v)

    return [
        wp(shift(A0, -0.10 * a + [0, 0, 0.10]), open_key, 0.0, tag="start"),
        wp(shift(A0, -0.05 * a), open_key, 1.4, via=True, tag="approach"),
        wp(A0, open_key, 0.9, tag="pick"),
        wp(A0, grip_key, 1.0, tag="grip"),
        wp(A0, grip_key, 0.3, tag="settle"),
        wp(shift(A0, [0, 0, 0.08]), grip_key, 1.1, via=True, tag="lift"),
        wp(C_mid_H, grip_key, 1.8, via=True, tag="turn"),
    ]


def event_times(wps: list[dict]) -> dict[str, float]:
    """Time at which each tagged waypoint is reached (the first waypoint is the start, t = 0)."""
    t, out = 0.0, {}
    for w in wps:
        t += w["dur"]
        out.setdefault(w["tag"], t)
    return out


# The camera's lens: 20 deg vertical field of view, a 57 mm lens on a full-frame 16:9 back. The shots
# are designed as distances for a 45 deg view (MuJoCo's default, a 24 mm lens); they are pushed back by
# LENS_PUSH to keep the framing. A wide lens enlarges whatever is nearest the camera, which here is
# always the hand: at 24 mm the true-scale Shadow Hand looked oversized next to the holder.
FOVY_DEG = 20.0
LENS_PUSH = float(np.tan(np.radians(22.5)) / np.tan(np.radians(FOVY_DEG / 2)))


def focal_mm(sensor_height_mm: float = 20.25) -> float:
    """Focal length giving FOVY_DEG on a sensor of this height (36 x 20.25 mm: full-frame 16:9)."""
    return sensor_height_mm / (2 * float(np.tan(np.radians(FOVY_DEG / 2))))


def camera_track(keys):
    """Smooth camera from keys [(t, lookat xyz, distance, azimuth, elevation)]: a cubic Hermite through
    them (finite-difference tangents, zero at the ends), so the camera eases in and out of the shot."""
    t = np.array([k[0] for k in keys])
    v = np.array([[*k[1], k[2], k[3], k[4]] for k in keys], float)
    m = np.zeros_like(v)
    for i in range(1, len(keys) - 1):
        m[i] = (v[i + 1] - v[i - 1]) / (t[i + 1] - t[i - 1])

    def at(x):
        i = int(np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 2))
        h = t[i + 1] - t[i]
        u = float(np.clip((x - t[i]) / h, 0, 1))
        h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
        return h00 * v[i] + h10 * h * m[i] + h01 * v[i + 1] + h11 * h * m[i + 1]

    return at


def shot_keys(c_start, c_goal, wps):
    """Two shots, cut at the pause in front of the holder. A, from the tray side: establish the cell,
    push in on the grasp, pull back and pan with the turn. B, from the holder's front-right and high
    enough to see over the tray, with the forearm leaving frame left: the insertion, the release, then
    a slow push in and round to the holder's front (the tray drops out of the frame) on the seated
    controller next to the green ghost (the real resting pose). Returns [(start time, keys)]."""
    ev = event_times(wps)
    c0 = np.asarray(c_start, float) + [0, 0, 0.01]
    g = np.asarray(c_goal, float)
    mid = 0.5 * (c0 + g) + [0, 0, 0.08]
    cut = ev["preplace"] - 0.6
    a = [(0.0, 0.6 * c0 + 0.4 * g + [0, 0, 0.10], 1.25, 150, -20),
         (ev["pick"], c0 + [0, 0, 0.02], 0.62, 162, -28),
         (ev["settle"], c0 + [0, 0, 0.03], 0.60, 170, -25),
         (ev["turn"], mid, 0.82, 176, -19),
         (cut, mid + [0, 0.02, 0], 0.84, 178, -18)]
    b = [(cut, g + [0.0, -0.02, 0.05], 0.80, 152, -32),
         (ev["seat"], g + [0, 0, 0.03], 0.62, 146, -28),
         (ev["away"], g + [0, 0, 0.02], 0.46, 112, -18),
         (ev["away"] + 3.5, g + [0, 0, 0.012], 0.33, 82, -12)]
    push = lambda keys: [(t, la, d * LENS_PUSH, az, el) for t, la, d, az, el in keys]  # noqa: E731
    return [(0.0, push(a)), (cut, push(b))]


def orbit_eye(x) -> tuple[np.ndarray, np.ndarray]:
    """(eye, lookat) for a camera track sample (lookat xyz, distance, azimuth, elevation in degrees),
    MuJoCo's free-camera convention (azimuth/elevation of the viewing direction). Vertical FOV FOVY_DEG."""
    look = np.asarray(x[:3], float)
    az, el = np.radians(x[4]), np.radians(x[5])
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return look - x[3] * fwd, look


def camera_at(shots, t):
    """Sample a cut list [(start time, track)] at time t."""
    return ([tr for t0, tr in shots if t >= t0] or [shots[0][1]])[-1](t)
