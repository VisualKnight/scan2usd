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


def _bumpy_plane(n=200, size=0.1, amp=0.004, waves=3):
    """Dense heightfield z = amp * sin(kx) * sin(ky) and its analytic normal function."""
    k = 2 * np.pi * waves / size
    xs = np.linspace(-size / 2, size / 2, n)
    gx, gy = np.meshgrid(xs, xs)
    z = amp * np.sin(k * gx) * np.sin(k * gy)
    verts = np.c_[gx.ravel(), gy.ravel(), z.ravel()]
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    faces = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
    dense = trimesh.Trimesh(verts, faces, process=False)

    def normal(x, y):
        dzdx = amp * k * np.cos(k * x) * np.sin(k * y)
        dzdy = amp * k * np.sin(k * x) * np.cos(k * y)
        nrm = np.c_[-dzdx, -dzdy, np.ones_like(x)]
        return nrm / np.linalg.norm(nrm, axis=1, keepdims=True)

    return dense, normal


def _decode_world_normals(img, low, size):
    """Texel positions and the world-space normals a renderer gets from the baked map."""
    from scan2usd.bake import _rasterize, vertex_tangents
    from scan2usd.mesh_ops import corner_normals

    face_id, bary = _rasterize(np.asarray(low.visual.uv), np.asarray(low.faces), size)
    inside = face_id >= 0
    faces = np.asarray(low.faces)[face_id[inside]]
    w = bary[inside][:, :, None]
    pos = (low.vertices[faces] * w).sum(1)
    T, hand = vertex_tangents(low, np.asarray(low.visual.uv))
    n = (corner_normals(low)[face_id[inside]] * w).sum(1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t = (T[faces] * w).sum(1)
    t -= n * (t * n).sum(1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t) * np.sign((hand[faces] * bary[inside]).sum(1))[:, None]
    ts = np.asarray(img, np.float64)[inside] / 255.0 * 2 - 1
    world = ts[:, :1] * t + ts[:, 1:2] * b + ts[:, 2:3] * n
    return pos, world / np.linalg.norm(world, axis=1, keepdims=True)


def test_normal_map_round_trip():
    """Baked tangent-space normals, decoded with the same tangent frame, match the true surface."""
    from scan2usd.bake import bake_normal_map

    dense, true_normal = _bumpy_plane()
    flat = trimesh.Trimesh([[-0.05, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0]],
                           [[0, 1, 2], [0, 2, 3]], process=False)
    low = unwrap(flat.subdivide().subdivide())
    size = 256
    img, stats = bake_normal_map(dense, low, size=size)
    assert stats["mean_deviation_deg"] > 5  # the bumps are captured
    pos, world = _decode_world_normals(img, low, size)
    err = np.degrees(np.arccos(np.clip((world * true_normal(pos[:, 0], pos[:, 1])).sum(1), -1, 1)))
    assert np.median(err) < 3.0


def test_normal_map_is_flat_for_matching_surfaces():
    from scan2usd.bake import bake_normal_map

    dense = trimesh.creation.icosphere(subdivisions=5, radius=0.05)
    low = unwrap(trimesh.creation.icosphere(subdivisions=3, radius=0.05))
    img, stats = bake_normal_map(dense, low, size=128)
    assert stats["mean_deviation_deg"] < 3.0
    rgb = np.asarray(img)
    assert np.median(rgb[..., 2]) > 245  # mostly (0, 0, 1) in tangent space


def test_normal_map_keeps_hard_edges_flat():
    """A low-poly cylinder baked against itself: what a renderer decodes on the long side faces must be
    the true radial normal, not the rim's averaged vertex normals smeared down the wall."""
    from scan2usd.bake import bake_normal_map

    cyl = trimesh.creation.cylinder(radius=0.05, height=0.02, sections=64)
    low = unwrap(cyl)
    size = 256
    img, _ = bake_normal_map(cyl, low, size=size, samples=400_000)
    pos, world = _decode_world_normals(img, low, size)
    r = np.linalg.norm(pos[:, :2], axis=1)
    side = (r > 0.0495) & (np.abs(pos[:, 2]) < 0.008)  # the wall, away from the rims
    radial = np.c_[pos[side, :2] / r[side, None], np.zeros(side.sum())]
    err = np.degrees(np.arccos(np.clip((world[side] * radial).sum(1), -1, 1)))
    assert np.percentile(err, 95) < 5.0
