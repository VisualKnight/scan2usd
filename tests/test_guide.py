import numpy as np
import trimesh

from scan2usd.guide import View, hemisphere_views, suggest_views, view_quality
from scan2usd.uncertainty import observation_field


def _sealed_cup():
    r, h = 0.04, 0.10
    solid = trimesh.creation.cylinder(radius=r, height=h, sections=96)
    solid.apply_translation([0, 0, h / 2])
    solid = solid.subdivide().subdivide()
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * np.pi, 20000)
    side = np.c_[r * np.cos(a), r * np.sin(a), rng.uniform(0.002, h, 20000)]
    return solid, side  # top never observed


def test_view_quality_occlusion_and_angle():
    mesh = trimesh.creation.box((0.1, 0.1, 0.1))
    surface = np.asarray(mesh.sample(20000, seed=0))
    targets = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, -0.05]])  # top and bottom face centers
    normals = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    top_view = View(np.array([0.0, 0.0, 0.4]), np.array([0.0, 0.0, -1.0]))
    q = view_quality(top_view, surface, targets, normals)
    assert q[0] > 0.9 and q[1] == 0.0  # the bottom is hidden behind the box
    grazing = View(np.array([0.4, 0.0, 0.05]), np.array([-1.0, 0.0, 0.0]))
    assert view_quality(grazing, surface, targets[:1], normals[:1])[0] == 0.0  # edge-on: no use


def test_guidance_looks_into_the_unseen_top():
    mesh, pts = _sealed_cup()
    field = observation_field(mesh, pts, near=0.002)
    center = np.array([0.0, 0.0, 0.04])
    cands = hemisphere_views(center, 0.3)
    existing = [v for v in cands if v.name.endswith(("el10", "el20"))]  # low rings only
    sugg, stats = suggest_views(mesh, field, cands, existing, k=4)
    assert len(sugg) >= 2
    assert sugg[0].elevation_deg > 45  # the first suggestion looks down at the sealed top
    assert np.mean([s.elevation_deg for s in sugg]) > 40
    assert stats["covered_fraction"] > 0.5
    assert sugg[0].gain >= sugg[-1].gain  # greedy: diminishing returns
