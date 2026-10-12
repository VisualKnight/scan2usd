import numpy as np
import pytest
import trimesh

pytest.importorskip("xatlas")

from scan2usd.bake import bake_color, unwrap  # noqa: E402
from scan2usd.mesh_ops import has_texture  # noqa: E402


def _two_tone_sphere(subdiv=5):
    """Dense sphere: red on top (z > 0), blue below."""
    s = trimesh.creation.icosphere(subdivisions=subdiv, radius=0.05)
    colors = np.where(s.vertices[:, 2:3] > 0, [[220, 40, 40, 255]], [[40, 60, 220, 255]]).astype(np.uint8)
    s.visual = trimesh.visual.ColorVisuals(s, vertex_colors=colors)
    return s


def test_unwrap_gives_uvs_in_unit_square():
    low = unwrap(trimesh.creation.icosphere(subdivisions=2))
    uv = np.asarray(low.visual.uv)
    assert uv.min() >= 0.0 and uv.max() <= 1.0
    assert len(uv) == len(low.vertices)


def test_bake_transfers_vertex_colors():
    dense = _two_tone_sphere()
    low = dense.simplify_quadric_decimation(face_count=800)
    baked, tex, stats = bake_color(dense, low, size=256)
    assert has_texture(baked)
    assert tex.size == (256, 256)
    assert 0.2 < stats["atlas_coverage"] <= 1.0
    # Sample the baked texture at points on each hemisphere.
    img = np.asarray(tex, np.float64)
    uv = np.asarray(baked.visual.uv)
    v = baked.vertices
    for mask, expect in ((v[:, 2] > 0.02, (220, 40, 40)), (v[:, 2] < -0.02, (40, 60, 220))):
        px = np.clip((uv[mask, 0] * 255).round().astype(int), 0, 255)
        py = np.clip(((1 - uv[mask, 1]) * 255).round().astype(int), 0, 255)
        assert np.allclose(np.median(img[py, px], axis=0), expect, atol=25)
