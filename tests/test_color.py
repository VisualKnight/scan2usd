import numpy as np
import pytest

from scan2usd import color


@pytest.mark.parametrize(
    "lab1, lab2, expected",
    [
        # Reference pairs from Sharma, Wu & Dalal (2005), Table 1.
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
        ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
        ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
        ((2.0776, 0.0795, -1.135), (0.9033, -0.0636, -0.5514), 0.9082),
    ],
)
def test_delta_e_2000_reference_pairs(lab1, lab2, expected):
    assert color.delta_e_2000(np.array(lab1), np.array(lab2)) == pytest.approx(expected, abs=1e-4)


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 101)
    assert np.allclose(color.linear_to_srgb(color.srgb_to_linear(x)), x, atol=1e-9)


def test_white_patch_lab_is_neutral():
    lab = color.linear_rgb_to_lab(np.ones(3))
    assert lab[0] == pytest.approx(100.0, abs=1e-3)
    assert abs(lab[1]) < 1e-3 and abs(lab[2]) < 1e-3


def test_patch_centers_axis_aligned():
    corners = np.array([[0, 0], [600, 0], [600, 400], [0, 400]], float)
    centers = color.patch_centers(corners)
    assert centers.shape == (24, 2)
    assert np.allclose(centers[0], [50, 50])
    assert np.allclose(centers[23], [550, 350])


def test_sampling_recovers_reference_without_cast():
    img, corners = color.render_chart(noise=0.0)
    measured = color.sample_chart(img, corners)
    de = color.delta_e_2000(color.linear_rgb_to_lab(measured), color.linear_rgb_to_lab(color.reference_linear()))
    assert de.max() < 0.5  # only 8-bit quantization error


def test_ccm_removes_color_cast():
    cast = 0.85 * np.array([[1.10, 0.06, -0.02], [0.03, 0.95, 0.02], [-0.03, 0.08, 0.70]])
    img, corners = color.render_chart(cast)
    cal = color.fit_ccm(color.sample_chart(img, corners))
    assert cal.delta_e_before["mean"] > 4.0
    assert cal.delta_e_after["mean"] < 0.6
    assert np.allclose(cal.matrix, np.linalg.inv(cast), atol=0.03)


def test_ccm_excludes_clipped_patches():
    measured = np.clip(color.reference_linear() * 1.3, 0, 1)
    cal = color.fit_ccm(measured)
    assert cal.patches_used < 24
    assert np.allclose(cal.matrix, np.eye(3) / 1.3, atol=0.02)


def test_calibration_save_load(tmp_path):
    cal = color.fit_ccm(color.reference_linear() * 0.8)
    p = tmp_path / "cal.json"
    cal.save(p)
    assert np.allclose(color.Calibration.load(p).matrix, cal.matrix)


def test_apply_ccm_preserves_alpha():
    rgba = np.array([[[100, 120, 140, 7]]], dtype=np.uint8)
    out = color.apply_ccm_srgb8(rgba, np.eye(3))
    assert out.shape == rgba.shape
    assert out[0, 0, 3] == 7
    assert np.abs(out[..., :3].astype(int) - rgba[..., :3]).max() <= 1


def test_albedo_range_stats():
    stats = color.albedo_range_stats(np.array([[0, 0, 0], [255, 255, 255], [128, 128, 128]], dtype=np.uint8))
    assert stats["too_dark"] == pytest.approx(1 / 3)
    assert stats["too_bright"] == pytest.approx(1 / 3)


def test_gray_card_recovers_white_balance_and_exposure():
    cast = np.diag([1.3, 1.0, 0.6]) * 0.8
    measured = np.array([0.18, 0.18, 0.18]) @ cast.T
    cal = color.fit_neutral(measured, reflectance=0.18)
    assert cal.method == "neutral"
    assert np.allclose(cal.matrix, np.linalg.inv(cast), atol=1e-9)
    assert cal.delta_e_after["max"] < 1e-6
    # Every color is fixed when the cast is a pure per-channel gain.
    ref = color.reference_linear()
    fixed = color.apply_ccm_linear(ref @ cast.T, cal.matrix)
    assert np.allclose(fixed, ref, atol=1e-9)


def test_white_balance_only_keeps_green_exposure():
    measured = np.array([0.5, 0.4, 0.3])
    cal = color.fit_neutral(measured, reflectance=None)
    assert cal.method == "white_balance"
    out = measured @ cal.matrix.T
    assert np.allclose(out, [0.4, 0.4, 0.4])


def test_neutral_rejects_clipped_target():
    with pytest.raises(ValueError):
        color.fit_neutral(np.array([1.0, 0.9, 0.8]))


def test_gray_card_vs_full_chart_on_demo_cast():
    """A gray card fixes most of the demo cast; the 24-patch chart also fixes channel cross-talk."""
    from scan2usd.demo import CAMERA_CAST

    ref = color.reference_linear()
    measured = np.clip(ref @ CAMERA_CAST.T, 0, 1)
    gray = color.fit_neutral(measured[21], reflectance=float(ref[21].mean()))  # "neutral 5" patch
    gray_de = color.delta_e_2000(color.linear_rgb_to_lab(color.apply_ccm_linear(measured, gray.matrix)),
                                 color.linear_rgb_to_lab(ref)).mean()
    full = color.fit_ccm(measured)
    assert full.delta_e_before["mean"] > gray_de > full.delta_e_after["mean"]


def test_sample_region():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:10, 5:10] = 118
    assert np.allclose(color.sample_region(img, (5, 5, 10, 10)), color.srgb_to_linear(118 / 255))
