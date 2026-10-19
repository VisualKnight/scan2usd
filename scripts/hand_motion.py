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


def _ramp_area(u: float) -> float:
    """Integral of min_jerk from 0 to u (0.5 at u = 1)."""
    return u**6 - 3 * u**5 + 2.5 * u**4


def move_duration(D: float, ramp=None) -> float:
    """Real duration of a move of nominal (summed leg) duration D: D under min-jerk, D + ramp with eased ends."""
    return D + min(ramp, D) if ramp else D


def time_scale(t: float, D: float, ramp=None) -> float:
    """Path parameter (nominal seconds, 0..D) at time t into a move. Default: one minimum-jerk profile over the
    whole move. With `ramp` (s): eased ends, the speed rising smoothly (min-jerk shaped) to the nominal over
    `ramp`, constant, then falling the same way, so the legs inside keep their own durations."""
    if not ramp:
        return min_jerk(t / D) * D if D > 0 else 0.0
    r = min(ramp, D)
    T = D + r
    t = min(max(t, 0.0), T)
    if t < r:
        return r * _ramp_area(t / r)
    if t > T - r:
        return D - r * _ramp_area((T - t) / r)
    return 0.5 * r + (t - r)


def time_of(p: float, D: float, ramp=None) -> float:
    """Inverse of time_scale for eased-end moves (the time path parameter p is reached); without a ramp,
    p itself (event times have always been the summed durations)."""
    if not ramp:
        return p
    lo, hi = 0.0, move_duration(D, ramp)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if time_scale(mid, D, ramp) < p else (lo, mid)
    return 0.5 * (lo + hi)


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
        self.ramp = next((w["ramp"] for w in legs if w.get("ramp")), None)  # eased ends (time_scale)
        self.duration = move_duration(float(self.cum[-1]), self.ramp)
        n = len(legs)
        # Tangents (per second of path parameter): centered differences inside, one-sided at the ends.
        self.m = np.zeros_like(self.P)
        for i in range(n + 1):
            lo, hi = max(i - 1, 0), min(i + 1, n)
            self.m[i] = (self.P[hi] - self.P[lo]) / max(self.cum[hi] - self.cum[lo], 1e-9)
        self.slerps = [Slerp([0, 1], Rotation.from_matrix([self.R[i], self.R[i + 1]])) for i in range(n)]

    def retarget(self, k: int, new_legs: list[dict]):
        """Replace legs k.. (same durations) mid-move, e.g. re-aimed after passing waypoint k-1. The tangent at
        the start of leg k is kept, so position and velocity stay continuous."""
        self.legs = self.legs[:k] + list(new_legs)
        for j, w in enumerate(new_legs, start=k):
            self.P[j + 1], self.R[j + 1], self.g[j + 1] = w["T"][:3, 3], w["T"][:3, :3], w["g"]
        n = len(self.legs)
        for i in range(k + 1, n + 1):
            lo, hi = max(i - 1, 0), min(i + 1, n)
            self.m[i] = (self.P[hi] - self.P[lo]) / max(self.cum[hi] - self.cum[lo], 1e-9)
        for i in range(k, n):
            self.slerps[i] = Slerp([0, 1], Rotation.from_matrix([self.R[i], self.R[i + 1]]))

    def at(self, t: float):
        """(4x4 pose, leg index, leg fraction) at time t in [0, duration]."""
        p = time_scale(t, float(self.cum[-1]), self.ramp)
        i = int(np.clip(np.searchsorted(self.cum, p, side="right") - 1, 0, len(self.legs) - 1))
        h = self.d[i]
        u = float(np.clip((p - self.cum[i]) / h, 0, 1)) if h > 0 else 1.0
        h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
        pos = h00 * self.P[i] + h10 * h * self.m[i] + h01 * self.P[i + 1] + h11 * h * self.m[i + 1]
        return pose(self.slerps[i]([u]).as_matrix()[0], pos), i, u

    def at_many(self, ts):
        """Vectorized at() for an array of times: positions (m, 3), quaternions (m, 4; x, y, z, w), leg
        indices (m,) and leg fractions (m,). The same math as at(), one scipy call per leg."""
        ts = np.asarray(ts, float)
        D = float(self.cum[-1])
        p = np.array([time_scale(t, D, self.ramp) for t in ts])
        i = np.clip(np.searchsorted(self.cum, p, side="right") - 1, 0, len(self.legs) - 1)
        h = self.d[i]
        u = np.where(h > 0, np.clip((p - self.cum[i]) / np.where(h > 0, h, 1.0), 0, 1), 1.0)
        h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
        pos = (h00[:, None] * self.P[i] + (h10 * h)[:, None] * self.m[i] + h01[:, None] * self.P[i + 1]
               + (h11 * h)[:, None] * self.m[i + 1])
        quat = np.zeros((len(ts), 4))
        for j in np.unique(i):
            sel = i == j
            quat[sel] = self.slerps[j](u[sel]).as_quat()
        return pos, quat, i, u


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


def plan_path(T_start, C_goal, approach_place, H, F, face, open_key, transport, grip_key="grip"):
    """Waypoints for the whole placement from the settled controller pose T_start (world).

    H: the forearm pose in the controller frame at the grasp; F: the grasp frame in the forearm frame
    (its z is the approach, palm toward the face); face: the controller's face normal (controller
    frame); C_goal: the controller pose to release at; approach_place: the approach at the holder.

    Come down over the face with the fingers open around the sides, close, lift, then turn in two
    stages blended into one move: a yaw with the palm still down (gravity stays along the approach,
    which the grip holds best), then the 90-degree tilt upright, a turn about the grasp axis itself
    (from face up on the tray to face forward on the holder there is no way around that) and the one
    move the grip resists only by friction. The tilt is about the controller's center, so it doesn't
    swing. The hand stops in front of the holder (where the re-aimed variant re-measures the
    controller), slides in, sets it down, opens and backs away.
    """
    A0 = T_start @ H
    ap = (A0 @ F)[:3, 2]  # approach direction in the world (palm toward the face), at the pick
    C = C_goal
    # The goal pose tipped face-up about a horizontal axis: the same yaw as the goal, palm down.
    f = C[:3, :3] @ face
    tip = Rotation.align_vectors([[0, 0, 1.0]], [f])[0].as_matrix()
    C_mid = pose(tip @ C[:3, :3], C[:3, 3] - 0.12 * np.asarray(approach_place) + [0, 0, 0.10])
    return (pick_waypoints(A0, ap, C_mid @ H, open_key, grip_key)
            + place_waypoints(C_goal, approach_place, H, open_key, transport, grip_key))


# --- the top-bottom pinch (demo_hand_pinch.py): goal, pick and place waypoints ------------------------------------
# Release and timing settings; demo_hand_pinch.py documents each (RELEASE_LEAN, PLACE_RIGHT, ...).
PINCH = {"lean": 0.005, "place_right": 0.003, "place_roll_deg": -3.0, "letgo_back": 0.03, "nudge": (0.0, 0.0),
         "ramp": 0.8, "insert_dur": 1.5, "seat_dur": 1.5}


def pinch_goal(T_target, face, right, center, p=PINCH):
    """The pose to set the controller down at (T_target: its pose seated in the holder) and the approach
    (into the face): leaning toward the holder's chest, a little to its right and pre-rolled about the face
    normal (about the grasp center, controller frame) against the drag of the thumb's slide-out."""
    C = np.array(T_target, float).copy()
    al = -(C[:3, :3] @ np.asarray(face, float))
    C[:3, 3] += p["lean"] * al + p["place_right"] * (C[:3, :3] @ np.asarray(right, float))
    if p["place_roll_deg"]:
        c = C[:3, :3] @ np.asarray(center, float) + C[:3, 3]
        R = Rotation.from_rotvec(np.radians(p["place_roll_deg"]) * (C[:3, :3] @ np.asarray(face, float))).as_matrix()
        C[:3, :3] = R @ C[:3, :3]
        C[:3, 3] = c + R @ (C[:3, 3] - c)
    return C, al


def pinch_place_waypoints(C_goal, approach, H_rel, transport, p=PINCH):
    """From the pre-place pose on: no stop in front of the holder (the pre-place pose is passed through, the
    move from the lift to the seat eased in and out over p["ramp"]; a re-aim happens as the hand passes it),
    a slow insertion, then the release: the fingers lift off the rim and the thumb drops away as the hand
    backs off p["letgo_back"] (sliding the thumb out from under the bottom edge), the fingers open,
    optionally an open-hand nudge, and away."""
    wps = place_waypoints(C_goal, approach, H_rel, "open", transport)
    durs = {"insert": p["insert_dur"], "seat": p["seat_dur"]}
    wps = [dict(w, via=True, ramp=p["ramp"]) if w["tag"] == "preplace" else
           dict(w, dur=durs[w["tag"]]) if w["tag"] in durs else w for w in wps]
    k = next(i for i, w in enumerate(wps) if w["tag"] == "release")
    a = np.asarray(approach, float)
    T = wps[k]["T"].copy()
    T[:3, 3] -= p["letgo_back"] * a
    nudge = p["nudge"]
    Tn = T.copy()
    Tn[:3, 3] += (p["letgo_back"] + nudge[0]) * a + [0, 0, nudge[1]]
    letgo = dict(wps[k], T=T, g="letgo", dur=0.5, tag="release")
    uncurl = dict(wps[k], T=T, g="open", dur=0.5, tag="uncurl")
    extra = [dict(wps[k], T=Tn, g="open", dur=0.6, tag="nudge"), dict(wps[k], T=T, g="open", dur=0.5, tag="unnudge")]
    return wps[:k] + [letgo, uncurl] + (extra if nudge[0] else []) + wps[k + 1:]


def pinch_pick_waypoints(A0, approach, C_mid_H):
    """Come down over the face with the fingers spread past the top edge, settle them on the rim (hook),
    close the thumb (grip), lift and swing to the mid pose."""
    a = np.asarray(approach, float)

    def shift(T, v):
        return pose(T[:3, :3], T[:3, 3] + v)

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


def pinch_plan(T_start, C_goal, approach, H, F, face, transport, p=PINCH):
    """The whole pinch placement from the settled controller pose (as plan_path for the span grasp)."""
    A0 = T_start @ H
    ap = (A0 @ F)[:3, 2]
    f = C_goal[:3, :3] @ np.asarray(face, float)
    tip = Rotation.align_vectors([[0, 0, 1.0]], [f])[0].as_matrix()
    C_mid = pose(tip @ C_goal[:3, :3], C_goal[:3, 3] - 0.12 * np.asarray(approach) + [0, 0, 0.10])
    return pinch_pick_waypoints(A0, ap, C_mid @ H) + pinch_place_waypoints(C_goal, approach, H, transport, p)


def event_times(wps: list[dict]) -> dict[str, float]:
    """Time at which each tagged waypoint is reached (the first waypoint is the start, t = 0)."""
    t, out = wps[0]["dur"], {wps[0]["tag"]: wps[0]["dur"]}
    for a, b in moves(wps, 1):
        legs = wps[a:b + 1]
        D = float(sum(w["dur"] for w in legs))
        ramp = next((w["ramp"] for w in legs if w.get("ramp")), None)
        cum = 0.0
        for w in legs:
            cum += w["dur"]
            out.setdefault(w["tag"], t + time_of(cum, D, ramp))
        t += move_duration(D, ramp)
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
    controller. Returns [(start time, keys)]."""
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
