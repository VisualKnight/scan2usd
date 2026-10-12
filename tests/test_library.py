import hashlib

import mujoco
import numpy as np
import pytest
import trimesh
from pxr import Usd

from scan2usd.articulation import build_parts, make_keepsake_box, write_articulated_usd
from scan2usd.library import build_library, library_entries
from scan2usd.mjcf import asset_body
from scan2usd.physics import estimate_mass_properties
from scan2usd.usd_io import PhysicsMaterial, SurfaceMaterial, read_usd, write_usd


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("lib")
    # An L-shaped asset with two convex pieces.
    a = trimesh.creation.box((0.1, 0.03, 0.02))
    a.apply_translation([0, 0, 0.01])
    b = trimesh.creation.box((0.03, 0.03, 0.08))
    b.apply_translation([0.035, 0, 0.04])
    mesh = trimesh.util.concatenate([a, b])
    ell = root / "assets" / "ell" / "ell.usda"
    write_usd(ell, "ell", mesh, [a, b], estimate_mass_properties(mesh, [a, b], density=800.0),
              SurfaceMaterial(), PhysicsMaterial(static_friction=0.6, dynamic_friction=0.5))
    box_mesh, cfg = make_keepsake_box()
    parts = build_parts(box_mesh, cfg["plane_origin"], cfg["plane_normal"], material="wood", collision="hull")
    hinged = root / "assets" / "hinged" / "hinged.usda"
    hinged.parent.mkdir(parents=True)
    write_articulated_usd(hinged, "hinged", parts, cfg["hinge"], SurfaceMaterial(), PhysicsMaterial())
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ell, hinged)}
    out = root / "library" / "lib.usda"
    entries = build_library([ell, hinged], out)
    return out, entries, digests, mesh


def test_payloads_are_relative_and_assets_untouched(library):
    out, entries, digests, _ = library
    text = out.read_text()
    assert "payload = @../assets/ell/ell.usda@" in text
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == d for p, d in digests.items())
    assert [e.name for e in entries] == ["ell", "hinged"]


def test_open_unloaded_is_cheap(library):
    out = library[0]
    stage = Usd.Stage.Open(str(out), Usd.Stage.LoadNone)
    prim = stage.GetPrimAtPath("/Library/ell")
    assert prim and not prim.IsLoaded()
    assert not stage.GetPrimAtPath("/Library/ell/render")  # geometry not composed until loaded
    info = {e["name"]: e for e in library_entries(out)}
    assert info["ell"]["colliders"] == {"decomposed": 2, "convex_hull": 1, "bounding_box": 1}
    assert info["hinged"]["colliders"]["convex_hull"] == 2  # one per rigid body
    assert sorted(info["ell"]["variants"]["friction"]) == ["high", "low", "measured"]


def test_collision_variants(library):
    out, _, _, mesh = library
    dec = read_usd(out, "/Library/ell")
    assert len(dec.hulls) == 2
    hull = read_usd(out, "/Library/ell", {"collision": "convex_hull"})
    assert len(hull.hulls) == 1
    assert hull.hulls[0].volume == pytest.approx(mesh.convex_hull.volume, rel=1e-3)
    box = read_usd(out, "/Library/ell", {"collision": "bounding_box"})
    assert np.allclose(box.hulls[0].bounds, mesh.bounds, atol=1e-6)
    # Mass properties come from the asset, whichever collider is selected.
    assert hull.mass == pytest.approx(dec.mass)


def test_friction_variants(library):
    out = library[0]
    assert read_usd(out, "/Library/ell").physics_material["static_friction"] == pytest.approx(0.6)
    high = read_usd(out, "/Library/ell", {"friction": "high", "collision": "convex_hull"})
    assert high.physics_material["static_friction"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        read_usd(out, "/Library/ell", {"friction": "sticky"})


def test_variant_feeds_mujoco(library):
    out = library[0]
    body = asset_body(read_usd(out, "/Library/ell", {"collision": "bounding_box"}), "ell")
    xml = f"""<mujoco><asset>{body.assets_xml}</asset><worldbody>{body.body_xml}</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml, assets=body.files)
    assert model.ngeom >= 1
