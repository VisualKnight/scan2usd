import numpy as np
import pytest

from scan2usd.usd_io import SurfaceMaterial, read_usd


def test_usd_roundtrip(box_asset):
    path, box, mp = box_asset
    asset = read_usd(path)
    assert asset.name == "box"
    assert asset.stage_info == {"up_axis": "Z", "meters_per_unit": 1.0, "kilograms_per_unit": 1.0, "default_prim": "box"}
    assert {"PhysicsRigidBodyAPI", "PhysicsMassAPI"} <= set(asset.root_schemas)
    assert asset.mass == pytest.approx(mp.mass, rel=1e-6)
    assert np.allclose(asset.center_of_mass, mp.center_of_mass, atol=1e-6)
    assert np.allclose(asset.inertia_tensor(), mp.inertia_tensor, rtol=1e-5, atol=1e-10)
    assert len(asset.render_mesh.faces) == len(box.faces)
    assert len(asset.hulls) == 1
    assert asset.collider_info[0]["approximation"] == "convexHull"
    assert asset.physics_material["static_friction"] == pytest.approx(0.5)
    assert asset.has_render_material


def test_textured_asset_references_texture(tmp_path):
    import trimesh
    from PIL import Image

    from scan2usd.physics import estimate_mass_properties, single_hull
    from scan2usd.usd_io import PhysicsMaterial, write_usd

    tex = tmp_path / "textures" / "albedo.png"
    tex.parent.mkdir()
    Image.new("RGB", (4, 4), (120, 100, 80)).save(tex)
    box = trimesh.creation.box((0.1, 0.1, 0.1))
    box.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(box.vertices), 2)))

    path = tmp_path / "tex.usda"
    write_usd(path, "tex", box, single_hull(box), estimate_mass_properties(box, density=1.0),
              SurfaceMaterial(texture=tex), PhysicsMaterial())
    asset = read_usd(path)
    assert [p.name for p in asset.texture_paths] == ["albedo.png"]
    assert asset.texture_paths[0].exists()
    assert "./textures/albedo.png" in path.read_text()
