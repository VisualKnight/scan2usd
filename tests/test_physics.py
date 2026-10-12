import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from scan2usd.demo import make_mug
from scan2usd.physics import convex_decompose, estimate_mass_properties, principal_frame, single_hull


def test_box_mass_and_inertia_analytic():
    a, b, c = 0.1, 0.2, 0.3
    box = trimesh.creation.box((a, b, c))
    mp = estimate_mass_properties(box, density=1000.0)
    m = 1000.0 * a * b * c
    assert mp.mass == pytest.approx(m)
    expected = sorted([m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12])
    assert np.allclose(mp.principal_moments, expected)
    assert np.allclose(mp.center_of_mass, 0.0, atol=1e-12)
    assert mp.volume_source == "render_mesh"


def test_measured_mass_sets_density():
    box = trimesh.creation.box((0.1, 0.1, 0.1))
    mp = estimate_mass_properties(box, mass=0.25)
    assert mp.mass == pytest.approx(0.25)
    assert mp.density == pytest.approx(250.0)


def test_density_or_mass_required():
    box = trimesh.creation.box((0.1, 0.1, 0.1))
    with pytest.raises(ValueError):
        estimate_mass_properties(box)
    with pytest.raises(ValueError):
        estimate_mass_properties(box, density=1.0, mass=1.0)


def test_principal_frame_reconstructs_rotated_tensor():
    box = trimesh.creation.box((0.1, 0.2, 0.3))
    rot = Rotation.from_euler("xyz", [20, 35, -50], degrees=True).as_matrix()
    box.apply_transform(np.r_[np.c_[rot, np.zeros(3)], [[0, 0, 0, 1]]])
    mp = estimate_mass_properties(box, density=500.0)
    w, x, y, z = mp.principal_axes_quat_wxyz
    r = Rotation.from_quat([x, y, z, w]).as_matrix()
    assert np.allclose(r @ np.diag(mp.principal_moments) @ r.T, mp.inertia_tensor, atol=1e-12)
    moments, axes = principal_frame(np.asarray(mp.inertia_tensor))
    assert np.linalg.det(axes) == pytest.approx(1.0)


def test_open_mesh_uses_hull_union():
    box = trimesh.creation.box((0.1, 0.1, 0.1))
    box.update_faces(np.arange(2, len(box.faces)))
    mp = estimate_mass_properties(box, single_hull(box), density=1000.0)
    assert mp.volume_source == "collision_hull_union"
    assert mp.mass == pytest.approx(1.0, rel=1e-6)


def test_convex_decomposition_of_mug():
    mug = make_mug(sections=48)
    hulls = convex_decompose(mug, threshold=0.08, max_hulls=16, proxy_faces=2000)
    assert 2 <= len(hulls) <= 16
    assert all(h.is_convex for h in hulls)
    assert all(len(h.vertices) <= 64 for h in hulls)
    # Hulls hug the mug far tighter than its single convex hull would.
    union = trimesh.boolean.union(hulls, engine="manifold")
    assert union.volume < 0.6 * mug.convex_hull.volume
    assert union.volume > 0.9 * mug.volume
