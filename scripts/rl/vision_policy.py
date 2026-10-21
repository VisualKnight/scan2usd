"""The vision-based hand-place policy: pose estimates from the cameras (vision_env.py) feed the learned policy
(policy.py) in place of the simulator's exact state.

"See, then act": the policy's inputs are poses (where the controller settled, where it belongs, and, from the
re-aim on, where it is in the hand), so two networks estimate them from the two RGB-D views and nothing else:

  scene net   once per episode, after the scene settles and before the hand comes in: the holder's pose on
              the table (x, y, yaw: it stands flat) and the controller's settled pose (position, 6D rotation).
              The controller's seated (target) pose follows from the holder's: its seat in the holder is fixed.
  hand net    every control step from just before the re-aim: the controller's pose relative to the forearm.
              The forearm's pose is an input (a robot knows its own pose); velocities are differences of
              successive estimates.

Both: a ResNet-18 trunk (ImageNet weights, a fourth input channel for depth, no max-pool: 1/8 resolution),
shared by both views; per view, spatial-softmax keypoints (Levine et al. 2016: expected image coordinates, for
precise positions) and pooled features; an MLP on both views (and the forearm pose). Torch only, like
policy.py.
"""

from __future__ import annotations

import os

import torch
import torchvision
from torch import nn

import policy as PL

DEPTH_MID, DEPTH_SCALE = 0.8, 0.3  # m: depth normalized around the cameras' working distance
RGB_MID, RGB_SCALE = 0.35, 0.25


# --- storage: uint8 colour, uint16 depth (mm) -------------------------------------------------------------

def to_storage(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, V, 4, H, W) float (vision_env.VisionHandPlaceEnv.images) -> uint8 (n, V, 3, H, W), uint16 mm (n, V, H, W)."""
    rgb = (img[:, :, :3].clamp(0, 1) * 255 + 0.5).to(torch.uint8)
    dep = (img[:, :, 3].clamp(0, 60.0) * 1000 + 0.5).to(torch.int32).to(torch.uint16)
    return rgb.cpu(), dep.cpu()


def from_storage(rgb: torch.Tensor, dep: torch.Tensor) -> torch.Tensor:
    return torch.cat([rgb.float() / 255.0, dep.to(torch.int32).float()[:, :, None] / 1000.0], 2)


def normalize(img: torch.Tensor) -> torch.Tensor:
    """(n, V, 4, H, W) colour in [0, 1] and depth in m -> the network's input."""
    rgb = (img[:, :, :3] - RGB_MID) / RGB_SCALE
    d = img[:, :, 3:4]
    dep = torch.where(d > 0, (d - DEPTH_MID) / DEPTH_SCALE, torch.zeros_like(d))
    return torch.cat([rgb, dep], 2)


def corrupt(img: torch.Tensor, depth_sigma: float = 0.006, drop: float = 0.05, holes: int = 4,
            rgb_sigma: float = 0.04) -> torch.Tensor:
    """A worse sensor than training saw: depth noise (m), dropped depth pixels (share), missing depth patches
    (per view) and colour noise; the levels a phone LiDAR or a RealSense shows on small objects at ~0.7 m."""
    n, v, _, h, w = img.shape
    rgb = (img[:, :, :3] + rgb_sigma * torch.randn_like(img[:, :, :3])).clamp(0, 1)
    dep = img[:, :, 3:4]
    valid = (dep > 0) & (torch.rand_like(dep) >= drop)
    dep = dep + depth_sigma * torch.randn_like(dep)
    yy = torch.arange(h, device=img.device)[None, None, :, None]
    xx = torch.arange(w, device=img.device)[None, None, None, :]
    for _ in range(holes):
        cy = torch.randint(0, h, (n, v, 1, 1), device=img.device)
        cx = torch.randint(0, w, (n, v, 1, 1), device=img.device)
        m = ((yy - cy).abs() <= 6) & ((xx - cx).abs() <= 9)
        valid &= ~m[:, :, None]
    return torch.cat([rgb, torch.where(valid, dep, torch.zeros_like(dep))], 2)


# --- poses as network targets -----------------------------------------------------------------------------

def pose9(p7: torch.Tensor) -> torch.Tensor:
    """(x y z qx qy qz qw) -> position and 6D rotation."""
    return torch.cat([p7[:, :3], PL.mat_to_6d(PL.quat_to_mat(p7[:, 3:7]))], -1)


def from9(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[:, :3], PL.mat_to_quat(PL.sixd_to_mat(x[:, 3:9]))], -1)


def holder4(p7: torch.Tensor) -> torch.Tensor:
    """A holder pose (flat on the table) -> x, y, cos yaw, sin yaw."""
    R = PL.quat_to_mat(p7[:, 3:7])
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    return torch.stack([p7[:, 0], p7[:, 1], torch.cos(yaw), torch.sin(yaw)], -1)


def holder_target7(h4: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    """The controller's seated pose from an estimated holder pose (x, y, cos, sin) and the fixed seat `rel`
    (4x4, controller in holder)."""
    n = len(h4)
    yaw = torch.atan2(h4[:, 3], h4[:, 2])
    c, s = torch.cos(yaw), torch.sin(yaw)
    H = torch.zeros((n, 4, 4), device=h4.device)
    H[:, 0, 0], H[:, 0, 1], H[:, 1, 0], H[:, 1, 1] = c, -s, s, c
    H[:, 2, 2] = H[:, 3, 3] = 1.0
    H[:, 0, 3], H[:, 1, 3] = h4[:, 0], h4[:, 1]
    Tt = H @ rel
    return torch.cat([Tt[:, :3, 3], PL.mat_to_quat(Tt[:, :3, :3])], -1)


def relative7(child7: torch.Tensor, parent7: torch.Tensor) -> torch.Tensor:
    """child's pose in parent's frame."""
    p, R = PL.rel(child7[:, :3], PL.quat_to_mat(child7[:, 3:7]), parent7[:, :3], PL.quat_to_mat(parent7[:, 3:7]))
    return torch.cat([p, PL.mat_to_quat(R)], -1)


def compose7(parent7: torch.Tensor, local7: torch.Tensor) -> torch.Tensor:
    Rp = PL.quat_to_mat(parent7[:, 3:7])
    p = parent7[:, :3] + (Rp @ local7[:, :3, None])[..., 0]
    return torch.cat([p, PL.mat_to_quat(Rp @ PL.quat_to_mat(local7[:, 3:7]))], -1)


def pose_errors(est7: torch.Tensor, true7: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Position error (mm) and rotation error (deg) per row."""
    pos = (est7[:, :3] - true7[:, :3]).norm(dim=-1) * 1000
    d = (est7[:, 3:7] * true7[:, 3:7]).sum(-1).abs().clamp(max=1.0)
    return pos, torch.rad2deg(2 * torch.acos(d))


# --- network ------------------------------------------------------------------------------------------------

class SpatialSoftmax(nn.Module):
    """Feature maps -> the expected (x, y) image coordinates of each map's activation, in [-1, 1]."""

    def __init__(self, channels: int, keypoints: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, keypoints, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        k = self.proj(f)
        n, c, h, w = k.shape
        a = torch.softmax(k.flatten(2) / self.log_temp.exp(), -1).view(n, c, h, w)
        ys = torch.linspace(-1, 1, h, device=f.device)
        xs = torch.linspace(-1, 1, w, device=f.device)
        return torch.cat([(a.sum(2) * xs).sum(-1), (a.sum(3) * ys).sum(-1)], -1)


class PoseNet(nn.Module):
    def __init__(self, out_dim: int, extra_dim: int = 0, views: int = 2, keypoints: int = 64, pretrained: bool = True):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        conv1 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv1.weight[:, :3] = r.conv1.weight
            conv1.weight[:, 3:] = r.conv1.weight.mean(1, keepdim=True)
        self.trunk = nn.Sequential(conv1, r.bn1, r.relu, r.layer1, r.layer2, r.layer3)  # no max-pool: 1/8
        self.ssm = SpatialSoftmax(256, keypoints)
        self.views, self.extra_dim = views, extra_dim
        d = views * (2 * keypoints + 256) + extra_dim
        self.head = nn.Sequential(nn.Linear(d, 512), nn.ELU(), nn.Linear(512, 512), nn.ELU(), nn.Linear(512, out_dim))
        for k, dim in (("out_mean", out_dim), ("extra_mean", max(extra_dim, 1))):
            self.register_buffer(k, torch.zeros(dim))
        for k, dim in (("out_std", out_dim), ("extra_std", max(extra_dim, 1))):
            self.register_buffer(k, torch.ones(dim))

    def set_stats(self, out: torch.Tensor, extra: torch.Tensor | None = None):
        self.out_mean.copy_(out.mean(0))
        self.out_std.copy_(out.std(0).clamp(min=1e-4))
        if extra is not None:
            self.extra_mean.copy_(extra.mean(0))
            self.extra_std.copy_(extra.std(0).clamp(min=1e-4))

    def forward_norm(self, x: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """x: normalized images (n, V, 4, H, W) -> normalized outputs."""
        n, v = x.shape[:2]
        f = self.trunk(x.flatten(0, 1))
        feats = torch.cat([self.ssm(f), f.mean((2, 3))], -1).view(n, -1)
        if self.extra_dim:
            feats = torch.cat([feats, (extra - self.extra_mean) / self.extra_std], -1)
        return self.head(feats)

    def forward(self, img: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """img: (n, V, 4, H, W) colour in [0, 1], depth in m -> outputs in their units."""
        return self.forward_norm(normalize(img), extra) * self.out_std + self.out_mean


SCENE_OUT, HAND_OUT, HAND_EXTRA = 4 + 9, 9, 9
_CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
SCENE_NET = os.path.join(_CK, "vision_scene_net.pt")  # the published networks (fp16 weights)
HAND_NET = os.path.join(_CK, "vision_hand_net.pt")


def save_net(net: PoseNet, path: str, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "meta": {**meta, "out_dim": net.head[-1].out_features,
                                                         "extra_dim": net.extra_dim, "views": net.views}}, path)


def load_net(path: str, device) -> PoseNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ck["meta"]
    net = PoseNet(m["out_dim"], m["extra_dim"], m["views"], pretrained=False).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


# --- the runner -----------------------------------------------------------------------------------------------

class VisionRunner(PL.Runner):
    """A Runner whose policy sees the cameras' estimates instead of the simulator's state.

    scene:   "vision" (the scene net) or "exact"
    in_hand: "vision" (the hand net), "mean" (the average grasp: no live estimate, as open loop) or "exact"
    Errors of every estimate against the truth are kept in self.errors (per episode)."""

    WARMUP_S = 0.5  # in-hand estimates start this long before the policy sees them (the velocity filter)

    def __init__(self, env, spec, scene_net: PoseNet | None, hand_net: PoseNet | None, scene: str = "vision",
                 in_hand: str = "vision", vel_alpha: float = 0.5, noise: bool = False, **kw):
        super().__init__(env, spec, **kw)
        assert self.layout_yaw == 0.0, "the cameras are placed for the unturned cell"
        self.scene_net, self.hand_net = scene_net, hand_net
        self.scene_mode, self.hand_mode, self.vel_alpha, self.noise = scene, in_hand, vel_alpha, noise
        self.rel = torch.tensor(spec.rel, dtype=torch.float32, device=self.dev)
        self.errors: dict = {}

    def images(self) -> torch.Tensor:
        img = self.env.images()
        return corrupt(img) if self.noise else img

    @torch.no_grad()
    def scene_estimate(self, start7, target7):
        self._prev = None
        self.live = None
        self.errors = {"start_mm": None, "start_deg": None, "target_mm": None, "target_deg": None,
                       "hand_mm": [], "hand_deg": []}
        if self.scene_mode == "exact":
            self.seen = {"start": start7.cpu().numpy(), "target": target7.cpu().numpy()}
            z = torch.zeros(len(start7))
            self.errors.update(start_mm=z, start_deg=z, target_mm=z, target_deg=z)
            return start7, target7
        out = self.scene_net(self.images())
        est_target = holder_target7(out[:, :4], self.rel)
        est_start = from9(out[:, 4:13])
        for k, (e, t) in (("start", (est_start, start7)), ("target", (est_target, target7))):
            mm, deg = pose_errors(e, t)
            self.errors[f"{k}_mm"], self.errors[f"{k}_deg"] = mm.cpu(), deg.cpu()
        self.seen = {"start": est_start.cpu().numpy(), "target": est_target.cpu().numpy()}
        return est_start, est_target

    @torch.no_grad()
    def controller_state(self, t):
        if self.hand_mode == "exact":
            return super().controller_state(t)
        e = self.env
        n = e.num_envs
        if t < getattr(self, "t_b", 0.0) - self.WARMUP_S:
            z = torch.zeros((n, 3), device=self.dev)
            return torch.cat([z, torch.tensor([[0.0, 0, 0, 1]], device=self.dev).expand(n, 4)], -1), z, z
        f7 = e.forearm_pose()
        if self.hand_mode == "mean":
            local = from9(self.hand_net.out_mean[None].expand(n, -1))
        else:
            local = from9(self.hand_net(self.images(), pose9(f7)))
        est7 = compose7(f7, local)
        mm, deg = pose_errors(est7, e.controller_pose())
        self.errors["hand_mm"].append(mm.cpu())
        self.errors["hand_deg"].append(deg.cpu())
        dt = self.dt * self.dec
        if self._prev is None:
            lin = ang = torch.zeros((n, 3), device=self.dev)
        else:
            p7, plin, pang = self._prev
            raw_lin = (est7[:, :3] - p7[:, :3]) / dt
            dR = PL.quat_to_mat(est7[:, 3:7]) @ PL.quat_to_mat(p7[:, 3:7]).transpose(-1, -2)
            raw_ang = rotvec(dR) / dt
            a = self.vel_alpha
            lin, ang = a * raw_lin + (1 - a) * plin, a * raw_ang + (1 - a) * pang
        self._prev = (est7, lin, ang)
        self.live = est7.cpu().numpy()
        return est7, lin, ang

    def error_summary(self) -> dict:
        out = {}
        for k in ("start", "target"):
            if self.errors.get(f"{k}_mm") is not None:
                out[f"{k}_mm_median"] = float(self.errors[f"{k}_mm"].median())
                out[f"{k}_deg_median"] = float(self.errors[f"{k}_deg"].median())
                out[f"{k}_mm_p95"] = float(self.errors[f"{k}_mm"].quantile(0.95))
        if self.errors.get("hand_mm"):
            mm, deg = torch.cat(self.errors["hand_mm"]), torch.cat(self.errors["hand_deg"])
            out.update(hand_mm_median=float(mm.median()), hand_deg_median=float(deg.median()),
                       hand_mm_p95=float(mm.quantile(0.95)))
        return out


def rotvec(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrices -> rotation vectors (axis * angle)."""
    cos = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2).clamp(-1, 1)
    ang = torch.acos(cos)
    v = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    s = torch.sin(ang).clamp(min=1e-6)
    return v * (ang / (2 * s))[:, None]


__all__ = ["PoseNet", "VisionRunner", "to_storage", "from_storage", "normalize", "pose9", "from9", "holder4",
           "holder_target7", "relative7", "compose7", "pose_errors", "save_net", "load_net",
           "SCENE_OUT", "HAND_OUT", "HAND_EXTRA"]
