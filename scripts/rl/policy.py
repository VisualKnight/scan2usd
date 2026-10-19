"""The learned hand-place policy: observations, actions, network, and the episode runner shared by training
(train_policy.py) and filming (scripted_rollout.py --policy).

Imitation of the scripted teacher (hand_teacher.py, the pinch of demo_hand_pinch.py), then DAgger: the policy
drives, the teacher labels every state it visits. The policy sees exact simulator state, in its cell's own
frame (so a cell turned for filming looks the same as in training). The default observation set, "scene":

  obs (52)  the plan clock (also as Fourier features inside the network: the targets are sharp functions of
            time); the controller's seated pose in the holder (the target) and where it started (the settled
            pose on the tray, from which the teacher planned); and, from the pre-place time on (when the
            teacher re-aims), the controller's pose and velocity, also relative to both of those poses. Not the
            hand's own state: with it, behavior cloning learns to copy the hand's current pose and drifts off
            ("copycat"); and not the controller before the re-aim: a policy reacting to the nudge of its own
            fingers at the grasp, never seen in the demonstrations, flies off.
  act (42)  the wrist target pose (position, 6D rotation) for the end of the next control step (30 Hz), twice:
            in the controller's start frame and in its goal frame (the teacher's pick waypoints are fixed in the
            first, its place waypoints in the second, so each is nearly scene-independent); blended by the clock
            from the first to the second between the grip and the pre-place pose; then 24 joint targets. The env
            interpolates them over the physics substeps (wrist speed clamped).

Torch only (no Isaac imports): importable once the Isaac app is running.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np
import torch
from torch import nn

import hand_teacher as T

OBS_DIM, ACT_DIM = 91, 42
# Observation sets. "full": everything above. "scene": without the hand's own state (forearm pose, joint
# angles and velocities): with them, behavior cloning learns to copy the hand's current pose (the next target
# is always next to it), and the copy's own errors feed back until the hand flies off ("copycat"). It adds the
# controller's pose relative to where it started and to its seated pose.
OBS_SETS = {"full": 91, "scene": 52}


def rel(pos: torch.Tensor, R: torch.Tensor, fp: torch.Tensor, fR: torch.Tensor):
    """A pose (position, rotation) expressed in the frame (fp, fR)."""
    Rt = fR.transpose(-1, -2)
    return (Rt @ (pos - fp)[..., None])[..., 0], Rt @ R
PLAN_S = 20.0  # the clock's scale (s)
N_FREQ = 24  # Fourier features of the clock (Policy.features)
MAX_STEP_M, MAX_STEP_RAD = 0.03, 0.15  # wrist target change per control step: 0.9 m/s, 4.5 rad/s at 30 Hz
MIN_WRIST_Z = 0.06  # the forearm's origin (its elbow end) never below this (the teacher's lowest: ~0.09 m)
DART_DECAY = float(os.environ.get("DART_DECAY", "0.99"))  # per control step: a slow drift (~3 s), not a jitter


# --- rotations (quaternions x, y, z, w as Isaac Lab) ------------------------------------------------------

def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    m = R
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cand = torch.stack([
        torch.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1], 1 + tr], -1),
        torch.stack([1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0],
                     m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] - m[..., 1, 2]], -1),
        torch.stack([m[..., 0, 1] + m[..., 1, 0], 1 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2],
                     m[..., 1, 2] + m[..., 2, 1], m[..., 0, 2] - m[..., 2, 0]], -1),
        torch.stack([m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1],
                     1 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1], m[..., 1, 0] - m[..., 0, 1]], -1)], -2)
    k = torch.stack([tr, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1).argmax(-1)
    q = torch.gather(cand, -2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True)


def mat_to_6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def sixd_to_mat(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., :3], x[..., 3:6]
    e1 = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    b = b - (e1 * b).sum(-1, keepdim=True) * e1
    e2 = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], -1)


def slerp_clamped(q0: torch.Tensor, q1: torch.Tensor, max_rad: float) -> torch.Tensor:
    """q1, or the rotation max_rad from q0 toward it."""
    d = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(d < 0, -q1, q1)
    ang = 2 * torch.acos(d.abs().clamp(max=1.0))
    t = (max_rad / ang.clamp(min=1e-9)).clamp(max=1.0)
    q = q0 + t * (q1 - q0)
    return q / q.norm(dim=-1, keepdim=True)


# --- network ------------------------------------------------------------------------------------------------

class Policy(nn.Module):
    """MLP on normalized observations to normalized actions (fixed statistics from the first demos). The plan
    clock (input 0) also enters as Fourier features (N_FREQ frequencies over PLAN_S): the targets are sharp
    functions of time, which an MLP fits poorly from one scalar."""

    def __init__(self, obs_set: str = "scene", hidden=(512, 512, 512)):
        super().__init__()
        self.obs_set = obs_set
        obs_dim = OBS_SETS[obs_set]
        layers, d = [], obs_dim + 2 * N_FREQ
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers.append(nn.Linear(d, ACT_DIM))
        self.net = nn.Sequential(*layers)
        self.register_buffer("freqs", torch.arange(1, N_FREQ + 1, dtype=torch.float32) * torch.pi)
        for k in ("obs_mean", "act_mean"):
            self.register_buffer(k, torch.zeros(obs_dim if k.startswith("obs") else ACT_DIM))
        for k in ("obs_std", "act_std"):
            self.register_buffer(k, torch.ones(obs_dim if k.startswith("obs") else ACT_DIM))

    def set_stats(self, obs: torch.Tensor, act: torch.Tensor):
        self.obs_mean.copy_(obs.mean(0))
        self.obs_std.copy_(obs.std(0).clamp(min=1e-3))
        self.act_mean.copy_(act.mean(0))
        self.act_std.copy_(act.std(0).clamp(min=1e-3))

    def features(self, obs: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        x = (obs - self.obs_mean) / self.obs_std
        if noise is not None:
            x = x + noise
        ph = obs[:, :1] * self.freqs  # clock in [0, ~1.2]: up to N_FREQ half-periods over PLAN_S
        return torch.cat([x, torch.sin(ph), torch.cos(ph)], -1)

    def forward_norm(self, obs: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(self.features(obs, noise))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward_norm(obs) * self.act_std + self.act_mean


def save(policy: Policy, path: str, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "meta": {**meta, "obs_set": policy.obs_set}}, path)


def load(path: str, device) -> Policy:
    ck = torch.load(path, map_location=device)
    p = Policy(ck["meta"].get("obs_set", "full")).to(device)
    p.load_state_dict(ck["state_dict"])
    p.eval()
    return p


# --- GPU power governor -------------------------------------------------------------------------------------

class PowerGovernor:
    """Keeps the GPU's average board power under a budget without admin rights (nvidia-smi -pl needs them):
    reads the draw every `every` s and pauses the loop between steps for as long as it is over."""

    def __init__(self, watts: float = 400.0, every: float = 2.0):
        self.watts, self.every, self.pause, self.t, self.log = watts, every, 0.0, 0.0, []

    def draw(self) -> float:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            return float(out.strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            return 0.0

    def tick(self):
        now = time.time()
        if now - self.t >= self.every:
            self.t = now
            w = self.draw()
            self.log.append((now, w, self.pause))
            if w > self.watts:
                self.pause = min(0.5, self.pause + 0.01 + 0.0005 * (w - self.watts))
            elif w < self.watts - 25:
                self.pause = max(0.0, self.pause - 0.005)
        if self.pause > 0:
            time.sleep(self.pause)

    def summary(self) -> dict:
        w = np.array([x[1] for x in self.log]) if self.log else np.zeros(1)
        return {"budget_w": self.watts, "samples": len(self.log), "mean_w": float(w.mean()), "p95_w": float(np.percentile(w, 95)),
                "max_w": float(w.max()), "max_pause_s": float(max([x[2] for x in self.log], default=0.0))}


# --- episodes -------------------------------------------------------------------------------------------------

def controller_points(n=3000) -> np.ndarray:
    from pxr import Usd, UsdGeom

    from hand_place_env import CTRL_USD

    stage = Usd.Stage.Open(CTRL_USD)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/{stage.GetDefaultPrim().GetName()}/render"))
    pts = np.asarray(mesh.GetPointsAttr().Get(), float)
    return pts[np.random.default_rng(1).choice(len(pts), min(n, len(pts)), replace=False)]


class Runner:
    """Runs synchronized episodes on an env: every cell gets a scene, its controller settles, the teacher plans
    from where it settled; each cell is then driven by the teacher or by the policy (use_policy), and the
    teacher's target for the end of each control step is recorded as the label (the re-aim included: the
    teacher measures the controller in the hand when the plan passes its pre-place pose)."""

    def __init__(self, env, spec, layout_yaw: float = 0.0, governor: PowerGovernor | None = None,
                 obs_set: str = "scene"):
        self.obs_set = obs_set
        self.env, self.spec, self.gov = env, spec, governor
        self.dev, self.dt, self.dec = env.device, env.physics_dt, env.cfg.decimation
        g = T.yaw(layout_yaw)
        self.G = torch.tensor(g, dtype=torch.float32, device=self.dev)  # cell frame -> env frame
        self.layout_yaw = layout_yaw
        self.cp = torch.tensor(controller_points(), dtype=torch.float32, device=self.dev)
        self.zero = torch.zeros((env.num_envs, env.cfg.action_space), device=self.dev)

    # frames: poses relative to the env origin (x, y, z, qx, qy, qz, qw) <-> the cell frame
    def to_cell(self, p7: torch.Tensor):
        Gt = self.G.T
        return p7[:, :3] @ Gt.T, Gt @ quat_to_mat(p7[:, 3:7])

    def from_cell(self, pos: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        return torch.cat([pos @ self.G.T, mat_to_quat(self.G @ R)], -1)

    def observe(self, t: float, start7: torch.Tensor, target7: torch.Tensor) -> torch.Tensor:
        e = self.env
        n = e.num_envs
        fp, fR = self.to_cell(e.forearm_pose())
        cp, cR = self.to_cell(e.controller_pose())
        cv = e.controller.data.root_com_vel_w.torch
        lin, ang = cv[:, :3] @ self.G, cv[:, 3:6] @ self.G
        sp, sR = self.to_cell(start7)
        tp, tR = self.to_cell(target7)
        q = e.hand.data.joint_pos.torch
        qd = e.hand.data.joint_vel.torch
        clock = torch.full((n, 1), t / PLAN_S, device=self.dev)
        if self.obs_set == "scene":
            # + the controller relative to where it started and to where it goes (the plan's two frames). The
            # controller's state is shown only from the pre-place time on, as the teacher uses it (its re-aim):
            # before that, the plan runs open loop, and a policy reacting to a nudged controller at the grasp
            # (never seen in the demonstrations) flies off.
            ca, cRa = rel(cp, cR, sp, sR)
            cb, cRb = rel(cp, cR, tp, tR)
            live = float(t >= getattr(self, "t_b", 0.0) - 0.2)
            ctrl = torch.cat([cp, mat_to_6d(cR), lin, 0.1 * ang, ca, mat_to_6d(cRa), cb, mat_to_6d(cRb)], -1) * live
            return torch.cat([clock, tp, mat_to_6d(tR), sp, mat_to_6d(sR), ctrl], -1)
        return torch.cat([clock, fp, mat_to_6d(fR), q, 0.05 * qd, cp, mat_to_6d(cR), lin, 0.1 * ang,
                          tp, mat_to_6d(tR), sp, mat_to_6d(sR)], -1)

    def set_frames(self, start7: torch.Tensor, target7: torch.Tensor, t_a: float, t_b: float):
        """The plan's two frames for this episode (the controller's start pose and its seated pose) and the
        clock interval [t_a, t_b] over which actions hand over from the first to the second."""
        self.fr = (*self.to_cell(start7), *self.to_cell(target7))
        self.t_a, self.t_b = t_a, t_b

    def blend(self, t: float) -> float:
        """Weight of the start frame at plan time t (1 before the lift, 0 from the pre-place pose on)."""
        if t <= self.t_a:
            return 1.0
        if t >= self.t_b:
            return 0.0
        s = (t - self.t_a) / (self.t_b - self.t_a)
        return 1.0 - s * s * (3 - 2 * s)

    def label(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """Teacher targets (env-relative wrist poses, joint targets) as actions: the wrist pose in the start
        frame and in the goal frame (the teacher's pick waypoints are fixed in the first, its place waypoints in
        the second), then the joint targets."""
        pos, R = self.to_cell(P)
        sp, sR, tp, tR = self.fr
        a, Ra = rel(pos, R, sp, sR)
        b, Rb = rel(pos, R, tp, tR)
        return torch.cat([a, mat_to_6d(Ra), b, mat_to_6d(Rb), Q], -1)

    def decode(self, act: torch.Tensor, t: float):
        sp, sR, tp, tR = self.fr
        pa, Ra = sp + (sR @ act[:, :3, None])[..., 0], sR @ sixd_to_mat(act[:, 3:9])
        pb, Rb = tp + (tR @ act[:, 9:12, None])[..., 0], tR @ sixd_to_mat(act[:, 12:18])
        w = self.blend(t)
        pos = w * pa + (1 - w) * pb
        R = sixd_to_mat(w * mat_to_6d(Ra) + (1 - w) * mat_to_6d(Rb))
        return self.from_cell(pos, R), act[:, 18:]

    def step(self, k=1, cb=None):
        for _ in range(k):
            self.env.step(self.zero)
            if self.gov is not None:
                self.gov.tick()
            if cb is not None:
                cb()

    def episode(self, scenes: list[dict], policy: Policy | None = None, use_policy: torch.Tensor | None = None,
                collect: bool = True, frame_cb=None, settle_s: float = 1.5, diag: list | None = None,
                dart: tuple[float, float, float] | None = None):
        """dart = (m, rad, joint rad): the teacher's cells execute its plan with a smooth random offset of about
        this size (an Ornstein-Uhlenbeck drift per control step) while the labels stay the clean plan: the
        demonstrations then show how to get back onto it (DART: Laskey et al. 2017)."""
        e, dev, dt, dec = self.env, self.dev, self.dt, self.dec
        n = e.num_envs
        org = e.scene.env_origins
        use_policy = torch.zeros(n, dtype=torch.bool, device=dev) if use_policy is None else use_policy.to(dev)
        # park 25 cm above the plan's start, settle the controller on its tray
        parks = []
        for sc in scenes:
            T0 = self.spec.plan(sc["controller_start"], sc)[0]["T"].copy()
            T0[2, 3] += 0.25
            parks.append(T0)
        e.on_substep = None
        e.place_scene(scenes, hand_poses=parks)
        e.episode_length_buf[:] = 0
        self.step(int(0.6 * 30), frame_cb)
        starts = np.array([T.from7(x) if np.isfinite(x).all() and np.linalg.norm(x[3:7]) > 0.5 else sc["controller_start"]
                           for x, sc in zip(e.controller_pose().cpu().numpy(), scenes)])
        plans = [self.spec.plan(starts[i], scenes[i]) for i in range(n)]
        for w in plans:
            w[0] = {**w[0], "dur": 1.2}  # glide from the park pose
        tracks = [T.Track(w, dt, e.keys, start=(parks[i], self.spec.open_key)) for i, w in enumerate(plans)]
        L = max(len(tr.P) for tr in tracks) + int(1.0 / dt)
        P = torch.zeros((n, L, 7), device=dev)
        Q = torch.zeros((n, L, len(e.joint_names)), device=dev)
        lengths = torch.zeros(n, dtype=torch.long, device=dev)
        preplace = torch.zeros(n, dtype=torch.long, device=dev)
        for i, tr in enumerate(tracks):
            m = len(tr.P)
            P[i, :m], P[i, m:] = torch.from_numpy(tr.P), torch.from_numpy(tr.P[-1])
            Q[i, :m], Q[i, m:] = torch.from_numpy(tr.Q), torch.from_numpy(tr.Q[-1])
            lengths[i], preplace[i] = m, tr.ev["preplace"]
        start7 = torch.tensor(np.array([T.pose7(s) for s in starts]), dtype=torch.float32, device=dev)
        target7 = torch.tensor(np.array([T.pose7(s["target"]) for s in scenes]), dtype=torch.float32, device=dev)
        ev0 = tracks[0].ev
        self.set_frames(start7, target7, ev0.get("settle", 0) * dt, ev0.get("preplace", ev0.get("seat", 1)) * dt)
        cursor = torch.zeros(n, dtype=torch.long, device=dev)
        reaimed = torch.zeros(n, dtype=torch.bool, device=dev)
        rows = torch.arange(n, device=dev)
        pol = {"prev": None, "next": None, "qprev": None, "qnext": None, "s": 0}
        noise = {"p": torch.zeros((n, 3), device=dev), "r": torch.zeros((n, 3), device=dev),
                 "q": torch.zeros((n, len(e.joint_names)), device=dev)}

        def on_substep(env):
            due = (cursor == preplace) & ~reaimed
            if due.any():  # the teacher's re-aim (measured on whatever drives the cell)
                ids = torch.nonzero(due).squeeze(-1).tolist()
                Cs, Fs = env.controller_pose()[ids].cpu().numpy(), env.forearm_pose()[ids].cpu().numpy()
                for j, i in enumerate(ids):
                    if not (np.isfinite(Cs[j]).all() and np.isfinite(Fs[j]).all()
                            and np.linalg.norm(Cs[j][3:7]) > 0.5 and np.linalg.norm(Fs[j][3:7]) > 0.5):
                        continue  # a blown-up cell: keep the plan as it is
                    H_rel = np.linalg.inv(T.from7(Cs[j])) @ T.from7(Fs[j])
                    if np.linalg.norm(H_rel[:3, 3]) > 0.6:
                        continue  # the controller isn't in the hand: nothing to re-aim
                    k0, p, q = tracks[i].reaim(int(cursor[i]), self.spec.replan_place(scenes[i], H_rel))
                    m = min(len(p), L - k0)
                    if m <= 0:
                        continue
                    pt = torch.from_numpy(p[:m]).to(dev)
                    P[i, k0:k0 + m], Q[i, k0:k0 + m] = pt, torch.from_numpy(q[:m]).to(dev)
                    P[i, k0 + m:], Q[i, k0 + m:] = pt[-1], Q[i, k0 + m - 1]
                    lengths[i] = k0 + m
                reaimed[due] = True
            c = cursor.clamp(max=L - 1)
            wt, jt = P[rows, c].clone(), Q[rows, c].clone()
            if dart is not None:  # the teacher's executed targets drift off its plan (labels stay clean)
                R = quat_to_mat(wt[:, 3:7])
                ax = noise["r"]
                th = ax.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                k_ = ax / th
                Kx = torch.zeros((n, 3, 3), device=dev)
                Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 0] = -k_[:, 2], k_[:, 1], k_[:, 2]
                Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = -k_[:, 0], -k_[:, 1], k_[:, 0]
                s_, c_ = torch.sin(th)[..., None], torch.cos(th)[..., None]
                Rn = torch.eye(3, device=dev) + s_ * Kx + (1 - c_) * (Kx @ Kx)
                qn_ = mat_to_quat(Rn @ R)
                qn_ = torch.where((qn_ * wt[:, 3:7]).sum(-1, keepdim=True) < 0, -qn_, qn_)  # the plan's sign
                wt = torch.cat([wt[:, :3] + noise["p"], qn_], -1)
                jt = jt + noise["q"]
            wt[:, :3] += org
            if pol["next"] is not None and use_policy.any():
                pol["s"] += 1
                a = pol["s"] / dec
                pp = pol["prev"][:, :3] + a * (pol["next"][:, :3] - pol["prev"][:, :3])
                qq = pol["prev"][:, 3:] + a * (pol["next"][:, 3:] - pol["prev"][:, 3:])
                qn = qq.norm(dim=-1, keepdim=True)
                qq = qq / qn.clamp(min=1e-6)
                pw = torch.cat([pp + org, qq], -1)
                jw = pol["qprev"] + a * (pol["qnext"] - pol["qprev"])
                ok = use_policy & (qn.squeeze(-1) > 0.5) & torch.isfinite(pw).all(-1) & torch.isfinite(jw).all(-1)
                wt = torch.where(ok[:, None], pw, wt)
                jt = torch.where(ok[:, None], jw, jt)
            env.wrist_target[:] = wt
            env.joint_target[:] = jt
            cursor.add_(1)

        e.on_substep = on_substep
        obs_buf, act_buf = [], []
        steps = int(np.ceil(int(lengths.max()) / dec))
        with torch.no_grad():
            for k in range(steps):
                t = float(cursor[0]) * dt
                if dart is not None:
                    a_ = DART_DECAY
                    for key, sig in (("p", dart[0]), ("r", dart[1]), ("q", dart[2])):
                        noise[key] = a_ * noise[key] + sig * (1 - a_ * a_) ** 0.5 * torch.randn_like(noise[key])
                need = collect or (policy is not None and use_policy.any())
                if need:
                    obs = self.observe(t, start7, target7)
                    c1 = (cursor + dec - 1).clamp(max=L - 1)
                    if collect:
                        obs_buf.append(obs.cpu())
                        act_buf.append(self.label(P[rows, c1], Q[rows, c1]).cpu())
                    if policy is not None and use_policy.any():
                        if policy == "oracle":  # a check of the control path: the teacher's own label, decoded
                            target, jq = self.decode(self.label(P[rows, c1], Q[rows, c1]), t + dec * dt)
                        else:
                            target, jq = self.decode(policy(torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)), t + dec * dt)
                        target, jq = torch.nan_to_num(target), torch.nan_to_num(jq)
                        if diag is not None:  # the policy's command vs the teacher's label, and the controller
                            lp, lq = self.decode(self.label(P[rows, c1], Q[rows, c1]), t + dec * dt)
                            u = use_policy
                            cpose = e.controller_pose()
                            diag.append({"t": t, "pos_err_mm": float((target[u, :3] - lp[u, :3]).norm(dim=-1).median() * 1000),
                                         "rot_err_deg": float(torch.rad2deg(2 * torch.acos((target[u, 3:] * lp[u, 3:]).sum(-1).abs().clamp(max=1))).median()),
                                         "joint_err_rad": float((jq[u] - lq[u]).abs().mean()),
                                         "ctrl_z_cm": float(cpose[u, 2].median() * 100),
                                         "hand_ctrl_cm": float((e.forearm_pose()[u, :3] - cpose[u, :3]).norm(dim=-1).median() * 100)})
                        target[:, 2] = target[:, 2].clamp(min=MIN_WRIST_Z)  # the forearm stays above the floor
                        prev = torch.nan_to_num(e.wrist_target.clone())
                        prev[:, :3] -= org
                        dp = target[:, :3] - prev[:, :3]
                        dn = dp.norm(dim=-1, keepdim=True)
                        target[:, :3] = prev[:, :3] + dp * (MAX_STEP_M / dn.clamp(min=1e-9)).clamp(max=1.0)
                        target[:, 3:] = slerp_clamped(prev[:, 3:], target[:, 3:], MAX_STEP_RAD)
                        pol.update(prev=prev, next=target, qprev=e.joint_target.clone(), qnext=jq, s=0)
                self.step(1, frame_cb)
        e.on_substep = None
        self.step(int(settle_s * 30), frame_cb)
        res = self.score(target7)
        data = None
        if collect:
            o, a = torch.stack(obs_buf, 1), torch.stack(act_buf, 1)  # (n, steps, dim)
            ok = (torch.isfinite(o).all(-1).all(-1) & (o.abs().amax((-1, -2)) < 100) & torch.isfinite(a).all(-1).all(-1)
                  & (a.abs().amax((-1, -2)) < 5))
            data = (o, a, ok)
        # a cell whose simulation blew up (non-finite, or the controller far off) counts as failed
        bad = ~torch.isfinite(res["add_mm"]) | (res["add_mm"] > 2000)
        res["held"] &= ~bad
        res["placed"] &= ~bad
        res["broken"] = bad
        return res, data

    def score(self, target7: torch.Tensor) -> dict:
        from isaaclab.utils.math import quat_apply

        C = self.env.controller_pose()

        def apply(p, pts):
            return quat_apply(p[:, None, 3:7].expand(-1, len(pts), -1), pts[None].expand(len(p), -1, -1)) + p[:, None, :3]

        end, true = apply(C, self.cp), apply(target7, self.cp)
        add = (end - true).norm(dim=-1).mean(-1) * 1000
        d = (C[:, 3:7] * target7[:, 3:7]).sum(-1).abs().clamp(max=1.0)
        rot = torch.rad2deg(2 * torch.acos(d))
        held = end[..., 2].min(-1).values > 0.02
        placed = held & (add < 20)
        return {"add_mm": add.cpu(), "rot_deg": rot.cpu(), "held": held.cpu(), "placed": placed.cpu()}


def summarize(res: dict, mask: torch.Tensor | None = None) -> dict:
    m = torch.ones_like(res["placed"]) if mask is None else mask.cpu()
    if m.sum() == 0:
        return {"envs": 0}
    add, rot = res["add_mm"][m], res["rot_deg"][m]
    out = {"envs": int(m.sum()), "held": int(res["held"][m].sum()), "placed": int(res["placed"][m].sum()),
           "placed_frac": float(res["placed"][m].float().mean()), "add_mm_median": float(add.nanmedian()),
           "rot_deg_median": float(rot.nanmedian())}
    if "broken" in res:
        out["broken"] = int(res["broken"][m].sum())
    return out


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
