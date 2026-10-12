import numpy as np
import trimesh

from scan2usd.cli import main
from scan2usd.preview import contact_sheet, preview_file, render


def test_render_draws_object_with_its_color():
    box = trimesh.creation.box((0.1, 0.1, 0.1))
    img = np.asarray(render([box], colors=[(200, 40, 40)], size=96))
    assert img.shape == (96, 96, 3)
    obj = np.any(img < 250, axis=2)
    assert 0.15 < obj.mean() < 0.9
    assert img[obj][:, 0].mean() > img[obj][:, 2].mean() + 40  # reddish


def test_render_depth_order():
    near = trimesh.creation.box((0.05, 0.05, 0.05))
    far = trimesh.creation.box((0.2, 0.2, 0.2))
    far.apply_translation([-0.3, 0, 0])  # behind `near` for azimuth 0 (camera on +X)
    img = np.asarray(render([far, near], colors=[(0, 0, 255), (255, 0, 0)], size=64, azimuth=0, elevation=0))
    center = img[32, 32]
    assert center[0] > center[2]


def test_preview_usd_asset(box_asset, tmp_path):
    path, _, _ = box_asset
    sheet = preview_file(path, size=64)
    assert sheet.width == 128  # render + collision panels
    out = tmp_path / "p.png"
    assert main(["preview", str(path), "-o", str(out), "--size", "64"]) == 0
    assert out.exists()


def test_contact_sheet_layout():
    from PIL import Image

    sheet = contact_sheet([("a", Image.new("RGB", (50, 40))), ("b", Image.new("RGB", (30, 40)))], caption_h=10)
    assert sheet.size == (80, 50)
