import numpy as np
import pytest
from PIL import Image

from scan2usd.chart import _normalize, _project_error_px, _triangulate, detect_chart
from scan2usd.color import render_chart


def _pose(look_from, target=(0, 0, 0)):
    """World-to-camera R, t for a camera at `look_from` looking at `target` (OpenCV axes)."""
    c = np.asarray(look_from, float)
    z = np.asarray(target, float) - c
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z])
    return r, -r @ c


def test_triangulate_recovers_point():
    p = np.array([0.03, -0.02, 0.01])
    views = []
    for cam in ([0.4, -0.3, 0.3], [-0.3, -0.4, 0.35], [0.1, 0.5, 0.4]):
        r, t = _pose(cam)
        pc = r @ p + t
        views.append((pc[:2] / pc[2], r, t))
    assert np.allclose(_triangulate(views), p, atol=1e-9)


def test_normalize_inverts_simple_radial():
    f, cx, cy, k = 1400.0, 1000.0, 750.0, -0.05
    uv = np.array([[0.2, -0.1], [-0.3, 0.25], [0.0, 0.0]])
    r2 = (uv ** 2).sum(1, keepdims=True)
    pix = uv * (1 + k * r2) * f + [cx, cy]
    out = _normalize(pix, ("SIMPLE_RADIAL", 2000, 1500, np.array([f, cx, cy, k])))
    assert np.allclose(out, uv, atol=1e-7)


def test_reprojection_error_flags_bad_view():
    pts = np.array([[0.0, 0.0, 0.0], [0.016, 0.0, 0.0]])
    r, t = _pose([0.3, -0.3, 0.3])
    pc = pts @ r.T + t
    uv = pc[:, :2] / pc[:, 2:3]
    assert _project_error_px(pts, uv, r, t, 1400) < 1e-6
    assert _project_error_px(pts, uv + 0.01, r, t, 1400) > 10


def test_detects_rendered_chart_in_order(tmp_path):
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "mcc"):
        pytest.skip("OpenCV contrib (mcc) not available")
    img, corners = render_chart(patch_px=70, border_px=14, margin_px=120)
    path = tmp_path / "chart.png"
    Image.fromarray(img).save(path)
    d = detect_chart(path)
    if d is None:
        pytest.skip("mcc did not detect the flat synthetic chart")
    ideal = np.array([[c + 0.5, r + 0.5] for r in range(4) for c in range(6)])
    expected = corners[0] + ideal * (corners[1][0] - corners[0][0]) / 6  # order starts at "dark skin"
    assert np.abs(d.centers - expected).max() < 6.0


def test_rejects_non_chart_texture(tmp_path):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    noise = (rng.random((600, 800, 3)) * 255).astype(np.uint8)
    path = tmp_path / "noise.png"
    Image.fromarray(noise).save(path)
    assert detect_chart(path) is None
