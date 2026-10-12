"""Color calibration against an X-Rite / Calibrite ColorChecker Classic (24 patches).

All fitting happens in linear sRGB (D65). The fitted 3x3 color-correction matrix (CCM)
maps camera-measured linear RGB to reference linear RGB, so applying it to a scan's
albedo texture (or vertex colors) removes the capture's lighting / camera color cast.
Accuracy is reported as CIEDE2000 against the reference patches.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

PATCH_NAMES = [
    "dark skin", "light skin", "blue sky", "foliage", "blue flower", "bluish green",
    "orange", "purplish blue", "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow", "magenta", "cyan",
    "white 9.5", "neutral 8", "neutral 6.5", "neutral 5", "neutral 3.5", "black 2",
]

# Commonly published 8-bit sRGB (D65) values for the ColorChecker Classic, row-major
# with "dark skin" top-left. Replace with your own spectrophotometer measurements for
# best accuracy (see `load_reference_csv`).
COLORCHECKER_SRGB8 = np.array([
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
    [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
    [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161],
    [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
], dtype=np.float64)

CHART_ROWS, CHART_COLS = 4, 6

# Linear sRGB (D65) -> CIE XYZ
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_D65_WHITE = _RGB_TO_XYZ @ np.ones(3)

# Plausible PBR albedo range in linear space (~sRGB 30..240): charcoal to fresh snow.
ALBEDO_MIN_LINEAR = 0.0126
ALBEDO_MAX_LINEAR = 0.8714


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def linear_rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    xyz = np.asarray(rgb, dtype=np.float64) @ _RGB_TO_XYZ.T
    t = xyz / _D65_WHITE
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(t > eps, np.cbrt(t), (kappa * t + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 color difference (Sharma, Wu & Dalal 2005)."""
    lab1, lab2 = np.asarray(lab1, dtype=np.float64), np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2.0
    g = 0.5 * (1.0 - np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dh = np.where(c1p * c2p == 0.0, 0.0, dh)
    dHp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2.0))

    Lp_bar = (L1 + L2) / 2.0
    Cp_bar = (c1p + c2p) / 2.0
    h_sum = h1p + h2p
    hp_bar = np.where(
        c1p * c2p == 0.0,
        h_sum,
        np.where(
            np.abs(h1p - h2p) <= 180.0,
            h_sum / 2.0,
            np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0),
        ),
    )
    t = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    r_c = 2.0 * np.sqrt(Cp_bar**7 / (Cp_bar**7 + 25.0**7))
    s_l = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * Cp_bar
    s_h = 1.0 + 0.015 * Cp_bar * t
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c
    return np.sqrt(
        (dLp / s_l) ** 2 + (dCp / s_c) ** 2 + (dHp / s_h) ** 2 + r_t * (dCp / s_c) * (dHp / s_h)
    )


def reference_linear() -> np.ndarray:
    return srgb_to_linear(COLORCHECKER_SRGB8 / 255.0)


def load_reference_csv(path: str | Path) -> np.ndarray:
    """Load 24 reference patches as linear RGB from a CSV with columns r,g,b (0..1 sRGB)."""
    rows = [
        [float(v) for v in line.split(",")[:3]]
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "r"))
    ]
    data = np.array(rows, dtype=np.float64)
    if data.shape != (24, 3):
        raise ValueError(f"expected 24 rows of r,g,b in {path}, got {data.shape}")
    return srgb_to_linear(data)


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct linear transform: 3x3 H such that dst ~ H @ src (4+ point pairs)."""
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        a.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(a, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def patch_centers(corners: np.ndarray) -> np.ndarray:
    """Image-space centers of the 24 patches.

    `corners` are the outer chart corners in pixels, ordered top-left ("dark skin"
    corner), top-right, bottom-right, bottom-left.
    """
    chart = np.array([[0, 0], [CHART_COLS, 0], [CHART_COLS, CHART_ROWS], [0, CHART_ROWS]], float)
    h = _homography(chart, np.asarray(corners, dtype=np.float64))
    grid = np.array([[c + 0.5, r + 0.5, 1.0] for r in range(CHART_ROWS) for c in range(CHART_COLS)])
    p = grid @ h.T
    return p[:, :2] / p[:, 2:3]


def sample_chart(image_srgb: np.ndarray, corners: np.ndarray, window_fraction: float = 0.35) -> np.ndarray:
    """Median linear RGB of each patch, sampled in a square window around its center."""
    img = np.asarray(image_srgb, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    lin = srgb_to_linear(img[..., :3])
    corners = np.asarray(corners, dtype=np.float64)
    patch_w = np.linalg.norm(corners[1] - corners[0]) / CHART_COLS
    patch_h = np.linalg.norm(corners[3] - corners[0]) / CHART_ROWS
    half = max(1, int(round(min(patch_w, patch_h) * window_fraction / 2.0)))
    out = []
    for cx, cy in patch_centers(corners):
        x, y = int(round(cx)), int(round(cy))
        win = lin[max(0, y - half): y + half + 1, max(0, x - half): x + half + 1].reshape(-1, 3)
        if win.size == 0:
            raise ValueError("patch window falls outside the image; check chart corners")
        out.append(np.median(win, axis=0))
    return np.array(out)


@dataclass
class Calibration:
    ccm: list[list[float]]
    delta_e_before: dict = field(default_factory=dict)
    delta_e_after: dict = field(default_factory=dict)
    patches_used: int = 24
    method: str = "colorchecker24"

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.ccm, dtype=np.float64)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls(**json.loads(Path(path).read_text()))


def _de_stats(measured: np.ndarray, reference: np.ndarray) -> dict:
    de = delta_e_2000(linear_rgb_to_lab(measured), linear_rgb_to_lab(reference))
    return {"mean": float(de.mean()), "median": float(np.median(de)), "max": float(de.max())}


def fit_ccm(measured: np.ndarray, reference: np.ndarray | None = None, clip_level: float = 0.98) -> Calibration:
    """Least-squares 3x3 CCM mapping measured linear RGB to reference linear RGB.

    Patches with any channel at or above `clip_level` are treated as clipped and excluded.
    """
    measured = np.asarray(measured, dtype=np.float64)
    reference = reference_linear() if reference is None else np.asarray(reference, dtype=np.float64)
    usable = np.all(measured < clip_level, axis=1)
    if usable.sum() < 6:
        raise ValueError("too few unclipped patches to fit a color correction; reduce exposure")
    m, *_ = np.linalg.lstsq(measured[usable], reference[usable], rcond=None)
    ccm = m.T
    corrected = np.clip(measured @ ccm.T, 0.0, None)
    return Calibration(
        ccm=ccm.tolist(),
        delta_e_before=_de_stats(measured, reference),
        delta_e_after=_de_stats(corrected, reference),
        patches_used=int(usable.sum()),
    )


def apply_ccm_linear(rgb_linear: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(rgb_linear, dtype=np.float64) @ np.asarray(ccm).T, 0.0, 1.0)


def sample_region(image_srgb: np.ndarray, box) -> np.ndarray:
    """Median linear RGB inside a pixel box (x0, y0, x1, y1)."""
    img = np.asarray(image_srgb, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    win = srgb_to_linear(img[y0:y1, x0:x1, :3]).reshape(-1, 3)
    if win.size == 0:
        raise ValueError(f"empty region {box}")
    return np.median(win, axis=0)


def fit_neutral(measured: np.ndarray, reflectance: float | None = 0.18) -> Calibration:
    """Per-channel gains from a neutral target (gray card or white paper).

    With a known `reflectance` (18% for a standard gray card) this corrects white balance
    *and* exposure. With `reflectance=None` (e.g. printer paper of unknown brightness)
    it corrects white balance only and keeps the green channel's exposure. Unlike the
    24-patch fit, it cannot correct cross-channel (hue) errors.
    """
    m = np.atleast_2d(np.asarray(measured, dtype=np.float64))
    if np.any(m <= 0) or np.any(m >= 0.98):
        raise ValueError("neutral target is black or clipped; adjust exposure")
    level = m[:, 1] if reflectance is None else np.full(len(m), float(reflectance))
    target = np.repeat(level[:, None], 3, axis=1)
    gains = (m * target).sum(0) / (m * m).sum(0)
    ccm = np.diag(gains)
    corrected = np.clip(m @ ccm.T, 0.0, None)
    return Calibration(
        ccm=ccm.tolist(),
        delta_e_before=_de_stats(m, target),
        delta_e_after=_de_stats(corrected, target),
        patches_used=len(m),
        method="neutral" if reflectance is not None else "white_balance",
    )


def apply_ccm_srgb8(rgb8: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    """Apply a CCM to 8-bit sRGB data (image or vertex colors); alpha is preserved."""
    rgb8 = np.asarray(rgb8)
    lin = srgb_to_linear(rgb8[..., :3] / 255.0)
    out = np.round(linear_to_srgb(apply_ccm_linear(lin, ccm)) * 255.0).astype(np.uint8)
    if rgb8.shape[-1] == 4:
        out = np.concatenate([out, rgb8[..., 3:4].astype(np.uint8)], axis=-1)
    return out


def albedo_range_stats(rgb8: np.ndarray) -> dict:
    """Fraction of texels outside the plausible PBR albedo range (by linear luminance)."""
    lin = srgb_to_linear(np.asarray(rgb8)[..., :3].reshape(-1, 3) / 255.0)
    lum = lin @ _RGB_TO_XYZ[1]
    return {
        "too_dark": float(np.mean(lum < ALBEDO_MIN_LINEAR)),
        "too_bright": float(np.mean(lum > ALBEDO_MAX_LINEAR)),
        "mean_luminance": float(lum.mean()),
    }


def render_chart(
    camera_matrix: np.ndarray | None = None,
    patch_px: int = 60,
    border_px: int = 10,
    margin_px: int = 40,
    noise: float = 0.004,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize an 8-bit sRGB photo of the chart, optionally with a camera color cast.

    Returns (image, corners). Used by the demo and tests.
    """
    rng = np.random.default_rng(seed)
    cast = np.eye(3) if camera_matrix is None else np.asarray(camera_matrix, dtype=np.float64)
    patches = np.clip(reference_linear() @ cast.T, 0.0, 1.0)
    cell = patch_px + border_px
    w = margin_px * 2 + CHART_COLS * cell + border_px
    h = margin_px * 2 + CHART_ROWS * cell + border_px
    img = np.full((h, w, 3), 0.18, dtype=np.float64)
    x0, y0 = margin_px, margin_px
    img[y0: y0 + CHART_ROWS * cell + border_px, x0: x0 + CHART_COLS * cell + border_px] = 0.01
    for i, rgb in enumerate(patches):
        r, c = divmod(i, CHART_COLS)
        px = x0 + border_px + c * cell
        py = y0 + border_px + r * cell
        img[py: py + patch_px, px: px + patch_px] = rgb
    img = np.clip(img + rng.normal(0.0, noise, img.shape), 0.0, 1.0)
    img8 = np.round(linear_to_srgb(img) * 255.0).astype(np.uint8)
    # Corners of the 6x4 patch grid (outer edge of the cell lattice, border-centered).
    half_b = border_px / 2.0
    gx0, gy0 = x0 + half_b, y0 + half_b
    gx1, gy1 = gx0 + CHART_COLS * cell, gy0 + CHART_ROWS * cell
    corners = np.array([[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1]])
    return img8, corners
