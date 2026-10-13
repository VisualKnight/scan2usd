"""Align scans of the same object (or of an object inside a bigger scene) up to scale.

Photogrammetry scans come in arbitrary units and frames, so two scans of one object, or an
object and a scene containing it, differ by a similarity transform: scale s, rotation R,
translation t (x' = s R x + t). This module estimates it without external libraries:

* `similarity_icp`: point-to-point ICP with a closed-form similarity update (Umeyama) and
  trimmed correspondences, so partial overlap (occlusion, clutter in the target) is tolerated.
* `global_register`: tries many initial rotations (all of SO(3), or yaw only for upright objects
  standing on the same table), refines the best few with ICP, and keeps the one that explains
  the most source points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class Similarity:
    s: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.s * x @ self.R.T + self.t

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = self.s * self.R, self.t
        return m

    def inverse(self) -> "Similarity":
        ri = self.R.T
        return Similarity(1.0 / self.s, ri, -ri @ self.t / self.s)

    def then(self, other: "Similarity") -> "Similarity":
        """`other` applied after `self`."""
        return Similarity(other.s * self.s, other.R @ self.R, other.s * other.R @ self.t + other.t)


@dataclass
class Fit:
    transform: Similarity
    rmse: float  # over the trimmed (kept) correspondences, target units
    inlier_fraction: float  # source points within `tau` of the target
    tau: float


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True, fixed_scale: float = 1.0) -> Similarity:
    """Least-squares similarity mapping src onto dst (corresponding rows); scale fixed if not with_scale."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    sgn = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sgn[2] = -1
    r = u @ np.diag(sgn) @ vt
    s = float((d * sgn).sum() / a.var(0).sum()) if with_scale else fixed_scale
    return Similarity(s, r, mu_d - s * r @ mu_s)


def voxel_down(x: np.ndarray, voxel: float, max_points: int | None = None, seed: int = 0) -> np.ndarray:
    """One point per occupied voxel (the first), optionally capped by random subsampling."""
    keys = np.floor((x - x.min(0)) / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    out = x[np.sort(first)]
    if max_points and len(out) > max_points:
        out = out[np.random.default_rng(seed).choice(len(out), max_points, replace=False)]
    return out


def principal_extent(x: np.ndarray, q: float = 1.0) -> float:
    """Length along the first principal axis between the q-th and (100-q)-th percentiles."""
    c = x - x.mean(0)
    _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 20000)], full_matrices=False)
    p = c @ vt[0]
    return float(np.percentile(p, 100 - q) - np.percentile(p, q))


def similarity_icp(src: np.ndarray, dst: np.ndarray, init: Similarity, iters: int = 40, trim: float = 0.8,
                   tau: float | None = None, with_scale: bool = True, tree: cKDTree | None = None) -> Fit:
    """Trimmed point-to-point ICP with a similarity update (source -> target).

    Only the `trim` fraction of closest source->target pairs counts, so parts of the source missing
    from the target (occlusion) don't pull the fit. One-way ICP can't constrain scale against a
    cluttered target well, so callers should fix the scale (`with_scale=False`) whenever it is
    known independently (e.g. metric assets), or treat the estimated scale with care.
    """
    tree = tree or cKDTree(dst)
    tf = init
    for _ in range(iters):
        moved = tf.apply(src)
        d, j = tree.query(moved)
        keep = d <= np.quantile(d, trim)
        new = umeyama(src[keep], dst[j[keep]], with_scale, tf.s)
        step = np.linalg.norm(new.apply(src[:50]) - moved[:50], axis=1).max()
        tf = new
        if step < 1e-7 * (1 + np.abs(dst).max()):
            break
    d, _ = tree.query(tf.apply(src))
    keep = d <= np.quantile(d, trim)
    tau = tau if tau is not None else 2.0 * float(np.median(d))
    return Fit(tf, float(np.sqrt(np.mean(d[keep] ** 2))), float(np.mean(d <= tau)), tau)


def candidate_rotations(n: int = 240, yaw_only: bool = False, seed: int = 0) -> list[np.ndarray]:
    if yaw_only:
        return [Rotation.from_euler("z", a, degrees=True).as_matrix() for a in np.linspace(0, 360, n, endpoint=False)]
    return list(Rotation.random(n, random_state=seed).as_matrix())


def global_register(src: np.ndarray, dst: np.ndarray, scale: float, rotations: list[np.ndarray] | None = None,
                    tau: float = None, coarse_points: int = 3000, keep: int = 6, trim: float = 0.7,
                    anchor_src: np.ndarray | None = None, anchor_dst: np.ndarray | None = None,
                    with_scale: bool = True) -> Fit:
    """Best similarity src -> dst over many initial rotations.

    `scale` is the initial scale guess; `anchor_src`/`anchor_dst` the points brought together
    initially (default: centroids). `tau` (target units) defines an inlier for scoring; the winner
    has the most source points within `tau` of the target (ties: lower trimmed RMSE).
    """
    rotations = rotations if rotations is not None else candidate_rotations()
    rng = np.random.default_rng(0)
    s_small = src[rng.choice(len(src), min(coarse_points, len(src)), replace=False)]
    d_small = dst[rng.choice(len(dst), min(4 * coarse_points, len(dst)), replace=False)]
    tree_small, tree = cKDTree(d_small), cKDTree(dst)
    a_src = src.mean(0) if anchor_src is None else anchor_src
    a_dst = dst.mean(0) if anchor_dst is None else anchor_dst
    tau = tau if tau is not None else 0.01 * principal_extent(dst)
    coarse = []
    for r in rotations:
        init = Similarity(scale, r, a_dst - scale * r @ a_src)
        f = similarity_icp(s_small, d_small, init, iters=12, trim=trim, tau=tau, with_scale=with_scale,
                           tree=tree_small)
        coarse.append((-f.inlier_fraction, f.rmse, f))
    coarse.sort(key=lambda c: (c[0], c[1]))
    best = None
    for _, _, f in coarse[:keep]:
        g = similarity_icp(src, dst, f.transform, iters=60, trim=trim, tau=tau, with_scale=with_scale, tree=tree)
        if best is None or (g.inlier_fraction, -g.rmse) > (best.inlier_fraction, -best.rmse):
            best = g
    return best


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the rotation taking a to b."""
    c = (np.trace(a.T @ b) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
