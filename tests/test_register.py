import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from scan2usd.register import (Similarity, candidate_rotations, global_register, rotation_angle_deg,
                               similarity_icp, umeyama)


def _shape(seed=0):
    """An asymmetric test object: a box with a cylinder knob off-center (no rotational symmetry)."""
    box = trimesh.creation.box((0.15, 0.10, 0.05))
    knob = trimesh.creation.cylinder(radius=0.015, height=0.04)
    knob.apply_translation((0.04, 0.02, 0.04))
    return np.vstack([box.sample(4000, seed=seed), knob.sample(1500, seed=seed)])


def test_umeyama_recovers_similarity():
    x = np.random.default_rng(0).normal(size=(200, 3))
    tf = Similarity(2.5, Rotation.from_euler("xyz", (10, -40, 70), degrees=True).as_matrix(), np.array([1, -2, 3.0]))
    est = umeyama(x, tf.apply(x))
    assert np.isclose(est.s, 2.5) and rotation_angle_deg(est.R, tf.R) < 1e-4 and np.allclose(est.t, tf.t)


def test_similarity_composition_and_inverse():
    x = np.random.default_rng(1).normal(size=(10, 3))
    a = Similarity(2.0, Rotation.from_euler("z", 30, degrees=True).as_matrix(), np.array([1.0, 0, 0]))
    b = Similarity(0.5, Rotation.from_euler("x", -60, degrees=True).as_matrix(), np.array([0, 2.0, 1]))
    assert np.allclose(a.then(b).apply(x), b.apply(a.apply(x)))
    assert np.allclose(a.inverse().apply(a.apply(x)), x)


def test_global_register_partial_cluttered_target_known_scale():
    src = _shape()
    true = Similarity(7.0, Rotation.from_euler("xyz", (120, 35, -80), degrees=True).as_matrix(), np.array([3.0, 1, 2]))
    tgt = true.apply(_shape(seed=1))
    tgt = tgt[tgt[:, 0] > np.percentile(tgt[:, 0], 25)]  # occlude a quarter of the target
    clutter = np.random.default_rng(2).uniform(tgt.min(0) - 0.5, tgt.max(0) + 0.5, (1500, 3))
    fit = global_register(src, np.vstack([tgt, clutter]), scale=7.0, rotations=candidate_rotations(300),
                          tau=0.02, with_scale=False)  # tau ~2% of the object's size
    assert fit.transform.s == 7.0
    assert rotation_angle_deg(fit.transform.R, true.R) < 3.0
    assert np.linalg.norm(fit.transform.apply(src.mean(0)) - true.apply(src.mean(0))) < 0.03 * 7.0


def test_icp_refines_small_offset():
    src = _shape()
    true = Similarity(1.3, Rotation.from_euler("z", 20, degrees=True).as_matrix(), np.array([0.1, 0.0, 0.05]))
    dst = true.apply(src)
    init = Similarity(1.25, Rotation.from_euler("z", 26, degrees=True).as_matrix(), np.array([0.11, 0.01, 0.05]))
    fit = similarity_icp(src, dst, init, iters=80, trim=1.0)
    assert fit.rmse < 1e-4 and abs(fit.transform.s - 1.3) < 1e-3
