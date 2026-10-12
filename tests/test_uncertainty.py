import numpy as np
import pytest
import trimesh

from scan2usd.schemas import Uncertainty
from scan2usd.uncertainty import (
    BASE,
    OBSERVED,
    SEALED,
    cavity_hypothesis,
    observation_field,
    property_intervals,
    sample_physical,
)
from scan2usd.usd_io import AssetData


def _sealed_cup():
    """What Poisson gives for a cup whose interior was never seen: a closed cylinder.

    Truth: an open cup (outer radius 4 cm, wall 4 mm, bottom 6 mm, height 10 cm). Observed points
    cover the outside wall and the rim ring only, as from a camera that never looked in.
    """
    r, h, wall = 0.04, 0.10, 0.004
    solid = trimesh.creation.cylinder(radius=r, height=h, sections=96)
    solid.apply_translation([0, 0, h / 2])
    solid = solid.subdivide().subdivide()
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * np.pi, 20000)
    z = rng.uniform(0.002, h, 20000)
    outside = np.c_[r * np.cos(a), r * np.sin(a), z]
    ra = rng.uniform(r - wall, r, 4000)
    aa = rng.uniform(0, 2 * np.pi, 4000)
    rim = np.c_[ra * np.cos(aa), ra * np.sin(aa), np.full(4000, h)]
    inner = trimesh.creation.cylinder(radius=r - wall, height=h)
    inner.apply_translation([0, 0, h / 2 + 0.006])
    true_volume = solid.volume - inner.volume * (h - 0.006) / h
    return solid, np.vstack([outside, rim]), true_volume


def test_observation_labels_sealed_top():
    mesh, pts, _ = _sealed_cup()
    f = observation_field(mesh, pts, near=0.002)
    top_center = (mesh.vertices[:, 2] > 0.099) & (np.hypot(*mesh.vertices[:, :2].T) < 0.02)
    side = (np.hypot(*mesh.vertices[:, :2].T) > 0.039) & (mesh.vertices[:, 2] > 0.01) & (mesh.vertices[:, 2] < 0.09)
    bottom = mesh.vertices[:, 2] < 1e-4
    assert np.all(f.labels[top_center] == SEALED)
    assert np.mean(f.labels[side] == OBSERVED) > 0.95
    assert np.all(f.labels[bottom & (np.hypot(*mesh.vertices[:, :2].T) < 0.03)] == BASE)
    assert 0.5 < f.observed_fraction < 0.95


def test_cavity_hypothesis_opens_the_cup():
    pytest.importorskip("skimage")
    mesh, pts, true_volume = _sealed_cup()
    f = observation_field(mesh, pts, near=0.002)
    carved, frac = cavity_hypothesis(mesh, pts, f, wall=0.004, divisions=96)
    assert carved is not None and 0.5 < frac < 0.95
    # The carved shape is much closer to the open cup than the sealed one.
    assert abs(carved.volume / true_volume - 1) < 0.5 < abs(mesh.volume / true_volume - 1)
    # Its outside is unchanged: same footprint and height.
    assert np.allclose(carved.extents, mesh.extents, atol=0.004)


def _asset(mesh, mass=0.5):
    return AssetData(name="cup", render_mesh=mesh, hulls=[mesh.convex_hull], mass=mass,
                     center_of_mass=np.asarray(mesh.center_mass), diagonal_inertia=np.array([1e-3, 1e-3, 1e-3]),
                     principal_axes_wxyz=np.array([1.0, 0, 0, 0]),
                     physics_material={"static_friction": 0.6, "dynamic_friction": 0.5, "restitution": 0.1})


def test_intervals_span_hypotheses_and_sources():
    mesh, _, true_volume = _sealed_cup()
    a = _asset(mesh, mass=0.5)
    rho = a.mass / mesh.volume
    u = property_intervals(a, "ceramic", a.physics_material, scale_source="ruler", scale_precision=0.001,
                           carved_volume=true_volume, carved_com=np.array([0, 0, 0.03]))
    assert u.mass[0] < rho * true_volume < a.mass < u.mass[1]  # truth and nominal both inside
    assert u.scale == pytest.approx((0.99, 1.01), abs=1e-6)  # 1 mm on a 10 cm object
    assert u.com_min[2] < 0.03 < u.com_max[2]
    measured = property_intervals(a, "ceramic", a.physics_material, measured_mass=0.21)
    assert measured.mass_source == "measured" and measured.mass[1] - measured.mass[0] < 0.01
    assert u.static_friction[0] < 0.6 < u.static_friction[1]


def test_samples_stay_inside_intervals():
    mesh, _, _ = _sealed_cup()
    a = _asset(mesh)
    u = Uncertainty(scale=(0.98, 1.02), mass=(0.3, 0.7), com_min=(-0.01, -0.01, 0.0), com_max=(0.01, 0.01, 0.08),
                    static_friction=(0.4, 0.8), dynamic_friction=(0.3, 0.7), restitution=(0.05, 0.2))
    rng = np.random.default_rng(1)
    for _ in range(20):
        s, info = sample_physical(a, u, rng)
        assert 0.3 <= s.mass <= 0.7 and 0.98 <= info["scale"] <= 1.02
        assert np.all(s.center_of_mass >= np.array(u.com_min) - 1e-9) and np.all(s.center_of_mass <= np.array(u.com_max) + 1e-9)
        assert 0.3 <= s.physics_material["dynamic_friction"] <= 0.7
    assert a.mass == 0.5  # the nominal asset is untouched
