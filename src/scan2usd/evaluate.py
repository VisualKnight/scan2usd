"""Compare a reconstructed asset with ground truth (synthetic captures, or lab scans)."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _rot_z(deg: float) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(np.radians(deg), [0, 0, 1])


def align(source: trimesh.Trimesh, reference: trimesh.Trimesh, yaw_step: float = 3.0, samples: int = 4000,
          seed: int = 0) -> np.ndarray:
    """Rigid transform taking `source` onto `reference`.

    Both meshes are expected Z-up, resting on z = 0 (as the pipeline outputs them), so the
    unknown is mostly a rotation about Z: brute-force yaw, then refine with ICP.
    """
    src = source.sample(samples, seed=seed)
    ref = reference.sample(samples * 2, seed=seed + 1)
    tree = cKDTree(ref)
    best, best_err = np.eye(4), np.inf
    for yaw in np.arange(0.0, 360.0, yaw_step):
        m = _rot_z(yaw)
        p = src @ m[:3, :3].T
        m[:3, 3] = [*(ref[:, :2].mean(0) - p[:, :2].mean(0)), 0.0]
        err = tree.query(p + m[:3, 3])[0].mean()
        if err < best_err:
            best, best_err = m, err
    refined, _, _ = trimesh.registration.icp(src, ref, initial=best, threshold=1e-7, max_iterations=60,
                                             reflection=False, translation=True, scale=False)
    return refined


def compare(recon: trimesh.Trimesh, gt: trimesh.Trimesh, samples: int = 30000, seed: int = 0,
            base_exclude: float = 0.01) -> dict:
    """Point-to-surface distances in mm after alignment, plus volume error.

    accuracy:     reconstruction -> ground-truth surface (is what we built correct?)
    completeness: ground truth -> reconstructed surface (did we capture everything?)
    The "*_above_base" variants ignore the bottom `base_exclude` meters, which a tabletop
    capture cannot see (the object rests on it).
    """
    m = align(recon, gt, seed=seed)
    aligned = recon.copy()
    aligned.apply_transform(m)
    rp = aligned.sample(samples, seed=seed)
    gp = gt.sample(samples, seed=seed + 1)
    acc = np.abs(trimesh.proximity.closest_point(gt, rp)[1]) * 1000.0
    comp = np.abs(trimesh.proximity.closest_point(aligned, gp)[1]) * 1000.0
    z0 = gt.bounds[0][2] + base_exclude
    acc_up, comp_up = acc[rp[:, 2] > z0], comp[gp[:, 2] > z0]

    def fscore(tau_mm: float, a=acc, c=comp) -> float:
        p, r = np.mean(a < tau_mm), np.mean(c < tau_mm)
        return float(2 * p * r / (p + r)) if p + r > 0 else 0.0

    out = {
        "accuracy_mm": {"mean": float(acc.mean()), "median": float(np.median(acc)), "p95": float(np.percentile(acc, 95))},
        "completeness_mm": {"mean": float(comp.mean()), "median": float(np.median(comp)), "p95": float(np.percentile(comp, 95))},
        "accuracy_above_base_mm": {"mean": float(acc_up.mean()), "p95": float(np.percentile(acc_up, 95))},
        "completeness_above_base_mm": {"mean": float(comp_up.mean()), "p95": float(np.percentile(comp_up, 95))},
        "fscore_2mm_above_base": fscore(2.0, acc_up, comp_up),
        "chamfer_mm": float((acc.mean() + comp.mean()) / 2),
        "fscore_1mm": fscore(1.0),
        "fscore_2mm": fscore(2.0),
        "extents_mm": {"reconstruction": [float(v * 1000) for v in aligned.extents],
                       "ground_truth": [float(v * 1000) for v in gt.extents]},
    }
    if recon.is_watertight and gt.is_watertight:
        out["volume_cm3"] = {"reconstruction": float(recon.volume * 1e6), "ground_truth": float(gt.volume * 1e6),
                             "error_pct": float((recon.volume / gt.volume - 1) * 100)}
    return out
