import numpy as np
import pytest
import trimesh

from scan2usd.demo import make_mug
from scan2usd.evaluate import compare
from scan2usd.reconstruct import _rotation_to_z, fit_plane_ransac, isolate_object, read_ply_points, write_ply_points


def _scene_cloud(seed=0):
    """A tilted, offset 'table' with a box standing on it and a flat card lying on it."""
    rng = np.random.default_rng(seed)
    table = np.c_[rng.uniform(-0.3, 0.3, (40000, 2)), rng.normal(0, 0.0003, 40000)]
    box = trimesh.creation.box((0.06, 0.04, 0.08))
    box.apply_translation([0, 0, 0.04])
    obj = box.sample(15000, seed=seed)
    card = np.c_[rng.uniform(0.1, 0.16, (3000, 1)), rng.uniform(-0.1, -0.06, (3000, 1)), np.full((3000, 1), 0.0006)]
    pts = np.vstack([table, obj, card])
    rot = trimesh.transformations.rotation_matrix(0.7, [1, 0.3, 0])[:3, :3]
    return pts @ rot.T * 3.7 + [1.0, -2.0, 0.5], len(table), len(obj)  # arbitrary COLMAP-like frame


def test_ply_roundtrip(tmp_path):
    xyz = np.random.default_rng(0).normal(size=(50, 3))
    nrm = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    rgb = np.random.default_rng(1).integers(0, 255, (50, 3)).astype(np.uint8)
    path = tmp_path / "p.ply"
    write_ply_points(path, xyz, nrm, rgb)
    d = read_ply_points(path)
    assert np.allclose(np.c_[d["x"], d["y"], d["z"]], xyz, atol=1e-6)
    assert np.array_equal(np.c_[d["red"], d["green"], d["blue"]], rgb)


def test_rotation_to_z():
    for n in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [0.3, -0.5, 0.8]):
        n = np.asarray(n, float) / np.linalg.norm(n)
        r = _rotation_to_z(n)
        assert np.allclose(r @ n, [0, 0, 1])
        assert np.linalg.det(r) == pytest.approx(1.0)


def test_plane_ransac_finds_table():
    pts, n_table, _ = _scene_cloud()
    _, _, inliers = fit_plane_ransac(pts, threshold=0.004)
    assert inliers[:n_table].mean() > 0.95


def test_isolate_object_removes_table_and_card():
    pts, n_table, n_obj = _scene_cloud()
    aligned, keep, _, _, _ = isolate_object(pts, spacing=0.004, plane_threshold=0.004)
    kept = set(keep.tolist())
    obj_idx = range(n_table, n_table + n_obj)
    assert sum(i in kept for i in obj_idx) / n_obj > 0.8  # all but the bottom face (on the table)
    assert not any(i in kept for i in range(n_table))  # no table
    assert not any(i in kept for i in range(n_table + n_obj, len(pts)))  # no card
    obj = aligned[keep]
    assert obj[:, 2].min() > -1e-3  # Z-up, above the table
    ext = obj.max(0) - obj.min(0)
    assert ext[2] == pytest.approx(0.08 * 3.7, rel=0.05)  # box height is vertical


def test_compare_identical_after_yaw():
    gt = make_mug(sections=48)
    rec = gt.copy()
    rec.apply_transform(trimesh.transformations.rotation_matrix(np.radians(73), [0, 0, 1]))
    m = compare(rec, gt, samples=20000)
    assert m["accuracy_mm"]["mean"] < 0.6
    assert m["fscore_2mm"] > 0.97
    assert abs(m["volume_cm3"]["error_pct"]) < 1e-6


def test_compare_reports_above_base_metrics():
    gt = make_mug(sections=48)
    m = compare(gt.copy(), gt, samples=8000)
    assert m["fscore_2mm_above_base"] > 0.97
    assert m["accuracy_above_base_mm"]["mean"] < 0.5


def test_main_body_prefers_point_support_over_size():
    """A big open Poisson sheet (large area) vs the small closed body the points sit on."""
    from scan2usd.reconstruct import _main_body

    obj = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    sheet = trimesh.creation.box((12.0, 12.0, 0.01))
    sheet.apply_translation((0, 0, 0.2))
    sheet = trimesh.Trimesh(sheet.vertices, sheet.faces[:-2])  # open: area 288 vs volume 4.2
    points = obj.sample(5000, seed=0)
    assert sheet.area > obj.volume and not sheet.is_watertight
    assert _main_body([sheet, obj], points) == 1
