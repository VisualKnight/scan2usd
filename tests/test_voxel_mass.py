import numpy as np
import pytest
import trimesh

from scan2usd.demo import make_mug
from scan2usd.physics import estimate_mass_properties, single_hull


def _punch_holes(mesh, n=6, seed=0):
    rng = np.random.default_rng(seed)
    keep = np.ones(len(mesh.faces), bool)
    keep[rng.choice(len(mesh.faces), n, replace=False)] = False
    out = mesh.copy()
    out.update_faces(keep)
    return out


def test_voxel_fill_handles_scan_holes_box():
    box = trimesh.creation.box((0.1, 0.06, 0.04)).subdivide_to_size(0.01)
    holey = _punch_holes(box)
    assert not holey.is_watertight
    mp = estimate_mass_properties(holey, single_hull(box), density=1000.0)
    assert mp.volume_source == "voxel_fill"
    assert mp.mass == pytest.approx(1000.0 * box.volume, rel=0.06)
    exact = estimate_mass_properties(box, density=1000.0)
    assert np.allclose(mp.principal_moments, exact.principal_moments, rtol=0.12)


def test_voxel_fill_keeps_mug_cavity_open():
    mug = make_mug(sections=48).subdivide_to_size(0.004)
    holey = _punch_holes(mug, n=10)
    mp = estimate_mass_properties(holey, single_hull(mug), density=2400.0)
    assert mp.volume_source == "voxel_fill"
    # Close to the real wall volume, nowhere near the filled convex hull.
    assert mp.volume == pytest.approx(mug.volume, rel=0.15)
    assert mp.volume < 0.5 * mug.convex_hull.volume
