import numpy as np
import pytest
import trimesh

from scan2usd.demo import make_mug
from scan2usd.mesh_ops import clean_mesh, decimate, orient_and_scale


def test_clean_drops_floaters_and_keeps_object():
    body = trimesh.creation.box((0.1, 0.1, 0.1))
    speck = trimesh.creation.icosphere(radius=0.003)
    speck.apply_translation([0.5, 0, 0])
    mesh, rep = clean_mesh(trimesh.util.concatenate([body, speck]))
    assert rep.components_in == 2 and rep.components_kept == 1
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(0.001, rel=1e-6)


def test_clean_fills_small_hole():
    box = trimesh.creation.box((0.1, 0.1, 0.1))
    box.update_faces(np.arange(1, len(box.faces)))
    assert not box.is_watertight
    mesh, rep = clean_mesh(box)
    assert rep.holes_filled and mesh.is_watertight


def test_orient_y_up_centimeters():
    # 10 x 30 x 20 cm box, Y-up: the 30 cm side is "up".
    box = trimesh.creation.box((10.0, 30.0, 20.0))
    box.apply_translation([5, 7, 9])
    mesh, scale = orient_and_scale(box, up_axis="y", units="cm")
    assert scale == pytest.approx(0.01)
    assert np.allclose(sorted(mesh.extents), [0.1, 0.2, 0.3])
    assert mesh.extents[2] == pytest.approx(0.3)
    lo, hi = mesh.bounds
    assert lo[2] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose((lo[:2] + hi[:2]) / 2, 0.0)


def test_scale_to_measurement():
    box = trimesh.creation.box((2.0, 1.0, 4.0))
    mesh, scale = orient_and_scale(box, scale_to=0.12, scale_axis="z")
    assert mesh.extents[2] == pytest.approx(0.12)
    assert scale == pytest.approx(0.03)


def test_decimate_keeps_watertight_and_colors():
    mug = make_mug(sections=64).subdivide_to_size(0.004)
    mug.visual = trimesh.visual.ColorVisuals(mug, vertex_colors=np.tile([200, 50, 50, 255], (len(mug.vertices), 1)))
    low, notes = decimate(mug, 5000)
    assert len(low.faces) <= 5000
    assert low.is_watertight
    assert low.visual.kind == "vertex"
    assert np.allclose(low.visual.vertex_colors[:, :3].mean(0), [200, 50, 50], atol=2)
    assert low.volume == pytest.approx(mug.volume, rel=0.02)
