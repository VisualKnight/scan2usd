import numpy as np
import trimesh
from PIL import Image

from scan2usd.carve import background_votes, subtract_volume


def test_subtract_volume_cuts_a_slot():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    pitch = 0.02
    origin = np.array([-0.6, -0.6, -0.6])
    n = int(round(1.2 / pitch)) + 1
    idx = np.indices((n, n, n)).reshape(3, -1).T
    p = origin + pitch * idx
    # A 0.2-wide slot all the way through y, below z = 0.2.
    mask = ((np.abs(p[:, 0]) < 0.1) & (p[:, 2] < 0.2)).reshape(n, n, n)
    out = subtract_volume(box, mask, origin, pitch, sigma=0.0)
    assert out.is_watertight
    removed = box.volume - out.volume
    assert abs(removed - 0.2 * 1.0 * 0.7) < 0.03  # slot 0.2 wide, 1 deep, from z = -0.5 to 0.2
    assert not out.contains([[0.0, 0.0, 0.0]])[0]
    assert out.contains([[0.3, 0.0, 0.0]])[0]


def test_subtract_empty_mask_is_identity():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    out = subtract_volume(box, np.zeros((4, 4, 4), bool), (0, 0, 0), 0.1)
    assert np.isclose(out.volume, box.volume)


def test_background_votes(tmp_path):
    # One camera at the origin looking down +z; the image is bright except a dark square in the middle.
    w = h = 100
    img = np.full((h, w), 230, np.uint8)
    img[40:60, 40:60] = 20
    Image.fromarray(img).save(tmp_path / "a.png")
    cameras = {1: ("PINHOLE", w, h, np.array([100.0, 100.0, 50.0, 50.0]))}
    images = {"a.png": (1, np.eye(3), np.zeros(3))}
    pts = np.array([[0.0, 0.0, 1.0],      # center: projects onto the dark square
                    [0.3, 0.3, 1.0],      # pixel (80, 80): background
                    [0.0, 0.0, -1.0],     # behind the camera
                    [0.11, 0.0, 1.0]])    # pixel (61, 50): background, but next to the dark edge
    votes, seen = background_votes(pts, cameras, images, tmp_path, erode=3)
    assert seen.tolist() == [1, 1, 0, 1]
    assert votes.tolist() == [0, 1, 0, 0]
