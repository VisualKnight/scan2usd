import numpy as np
import pytest
import trimesh
from PIL import Image

from scan2usd.physics import estimate_mass_properties, single_hull
from scan2usd.usd_io import PhysicsMaterial, SurfaceMaterial, write_usd


def make_box_asset(path, extents=(0.1, 0.06, 0.04), density=1000.0, surface=None):
    box = trimesh.creation.box(extents)
    box.apply_translation([0, 0, extents[2] / 2])
    hulls = single_hull(box)
    mp = estimate_mass_properties(box, hulls, density=density)
    write_usd(path, "box", box, hulls, mp, surface or SurfaceMaterial(), PhysicsMaterial())
    return box, mp


def make_textured_box(size=0.08) -> trimesh.Trimesh:
    box = trimesh.creation.box((size, size, size))
    box.unmerge_vertices()
    uv = (box.vertices[:, :2] / size) + 0.5
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(60, 200, (32, 32, 3), dtype=np.uint8))
    box.visual = trimesh.visual.TextureVisuals(uv=uv, image=img)
    return box


@pytest.fixture
def box_asset(tmp_path):
    path = tmp_path / "box.usda"
    box, mp = make_box_asset(path)
    return path, box, mp
