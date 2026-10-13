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


def test_load_obj_finds_texture_in_other_folder(tmp_path):
    """Archives often keep textures in materials/textures while the .mtl expects them beside the OBJ."""
    from PIL import Image

    from scan2usd.mesh_ops import has_texture, load_mesh

    (tmp_path / "meshes").mkdir()
    (tmp_path / "materials" / "textures").mkdir(parents=True)
    Image.new("RGB", (8, 8), (200, 30, 30)).save(tmp_path / "materials" / "textures" / "texture.png")
    (tmp_path / "meshes" / "model.mtl").write_text("newmtl m0\nmap_Kd texture.png\n")
    (tmp_path / "meshes" / "model.obj").write_text(
        "mtllib model.mtl\nusemtl m0\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
        "vt 0 0\nvt 1 0\nvt 0 1\nvt 1 1\n"
        "f 1/1 3/3 2/2\nf 1/1 2/2 4/4\nf 1/1 4/4 3/3\nf 2/2 3/3 4/4\n")
    mesh = load_mesh(tmp_path / "meshes" / "model.obj")
    assert has_texture(mesh)


def test_fit_disc_base_replaces_pitted_base():
    import trimesh
    from scan2usd.mesh_ops import fit_disc_base

    base = trimesh.creation.cylinder(radius=0.05, height=0.018, sections=64)
    base.apply_translation((0.01, -0.02, 0.009))
    body = trimesh.creation.box((0.03, 0.03, 0.15))
    body.apply_translation((0.01, -0.02, 0.018 + 0.075 - 0.002))
    good = trimesh.boolean.union([base, body], engine="manifold")
    pts, fi = trimesh.sample.sample_surface(good, 30000, seed=0)
    nrm = good.face_normals[fi]
    # A pitted scan: dents in the base's side.
    bad = good.copy()
    side = (np.abs(bad.vertices[:, 2] - 0.009) < 0.008) & (np.linalg.norm(bad.vertices[:, :2] - (0.01, -0.02), axis=1) > 0.04)
    bad.vertices[side, :2] -= 0.3 * (bad.vertices[side, :2] - (0.01, -0.02))
    new, fit = fit_disc_base(bad, pts, nrm)
    assert abs(fit["z_top"] - 0.018) < 0.001 and abs(fit["radius"] - 0.05) < 0.001
    assert np.allclose(fit["center"], (0.01, -0.02), atol=0.001)
    assert new.is_watertight and abs(new.volume - good.volume) / good.volume < 0.03
