"""ColorChecker detection in capture photos: automatic color calibration and metric scale.

With a ColorChecker Classic in some of the photos, Scan2USD can (1) calibrate color
without hand-picked chart corners and (2) recover true size without a ruler: the chart's
patch centers are triangulated with COLMAP's camera poses, and their spacing in the
reconstruction is compared with the chart's physical patch pitch.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .color import CHART_COLS, CHART_ROWS, _homography, reference_linear, srgb_to_linear

# COLMAP camera models: id -> (name, number of params)
_CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
                  4: ("OPENCV", 8)}


@dataclass
class ChartDetection:
    image: Path
    centers: np.ndarray  # (24, 2) pixel centers, row-major from "dark skin"
    pitch_px: float      # mean distance between neighboring patch centers

    def lattice_corners(self) -> np.ndarray:
        """Outer corners of the 6x4 patch lattice (TL, TR, BR, BL), the convention of color.sample_chart."""
        ideal = np.array([[c + 0.5, r + 0.5] for r in range(CHART_ROWS) for c in range(CHART_COLS)], float)
        h = _homography(ideal, self.centers)
        corners = np.array([[0, 0, 1], [CHART_COLS, 0, 1], [CHART_COLS, CHART_ROWS, 1], [0, CHART_ROWS, 1]], float)
        p = corners @ h.T
        return p[:, :2] / p[:, 2:3]


def detect_chart(image_path: str | Path, max_color_error: float = 0.03) -> ChartDetection | None:
    """Find a ColorChecker Classic (24 patches) with OpenCV's mcc module."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    det = cv2.mcc.CCheckerDetector_create()
    det.setColorChartType(cv2.mcc.MCC24)
    if not det.process(img, 1):
        return None
    quads = np.asarray(det.getBestColorChecker().getColorCharts(), dtype=np.float64).reshape(24, 4, 2)
    centers = quads.mean(axis=1)
    h, w = img.shape[:2]
    if np.any(centers < 2) or np.any(centers[:, 0] > w - 3) or np.any(centers[:, 1] > h - 3):
        return None  # chart partly out of frame: patch positions are extrapolated
    # Resolve the patch order with color: each patch has a known reference color, so the right
    # ordering is the one a single 3x3 color matrix explains best. (mcc's order isn't consistent
    # across viewpoints; a wrong order collapses the triangulated grid.)
    rgb = img[..., ::-1]
    xs = np.clip(centers[:, 0].round().astype(int), 0, w - 1)
    ys = np.clip(centers[:, 1].round().astype(int), 0, h - 1)
    measured = np.array([srgb_to_linear(rgb[max(0, y - 2): y + 3, max(0, x - 2): x + 3].reshape(-1, 3) / 255.0).mean(0)
                         for x, y in zip(xs, ys)])
    ref = reference_linear()
    idx = np.arange(24).reshape(CHART_ROWS, CHART_COLS)
    best_order, best_err = None, np.inf
    for order in (idx, idx[::-1, ::-1], idx[::-1, :], idx[:, ::-1]):
        m = measured[order.ravel()]
        ccm, *_ = np.linalg.lstsq(m, ref, rcond=None)
        err = float(np.abs(m @ ccm - ref).mean())
        if err < best_err:
            best_order, best_err = order.ravel(), err
    if best_err > max_color_error:
        return None  # colors don't match a ColorChecker: likely a false detection on texture
    centers = centers[best_order]
    grid = centers.reshape(CHART_ROWS, CHART_COLS, 2)
    pitch = float(np.mean(np.r_[np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
                                np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()]))
    return ChartDetection(Path(image_path), centers, pitch)


def best_chart_image(images: list[Path]) -> ChartDetection | None:
    """The detection where the chart appears largest (most pixels per patch)."""
    best = None
    for path in images:
        d = detect_chart(path)
        if d is not None and (best is None or d.pitch_px > best.pitch_px):
            best = d
    return best


# --- COLMAP sparse model ------------------------------------------------------------------

def read_colmap_model(sparse_dir: str | Path) -> tuple[dict, dict]:
    """cameras {id: (model, w, h, params)} and images {name: (camera_id, R, t)} from a binary model."""
    sparse_dir = Path(sparse_dir)
    cameras = {}
    with open(sparse_dir / "cameras.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cam_id, model_id = struct.unpack("<ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            name, n = _CAMERA_MODELS[model_id]
            cameras[cam_id] = (name, w, h, np.array(struct.unpack(f"<{n}d", f.read(8 * n))))
    images = {}
    with open(sparse_dir / "images.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            _img_id = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            t = np.array(struct.unpack("<3d", f.read(24)))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while (ch := f.read(1)) != b"\x00":
                name += ch
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * n_pts, 1)
            from scipy.spatial.transform import Rotation

            images[name.decode()] = (cam_id, Rotation.from_quat([qx, qy, qz, qw]).as_matrix(), t)
    return cameras, images


def _normalize(pixels: np.ndarray, camera) -> np.ndarray:
    """Pixel -> undistorted normalized image coordinates."""
    model, _, _, p = camera
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = p[0], p[1], p[2]
        uv = (pixels - [cx, cy]) / f
        k1 = p[3] if model != "SIMPLE_PINHOLE" else 0.0
        k2 = p[4] if model == "RADIAL" else 0.0
    elif model in ("PINHOLE", "OPENCV"):
        uv = (pixels - [p[2], p[3]]) / [p[0], p[1]]
        k1, k2 = (p[4], p[5]) if model == "OPENCV" else (0.0, 0.0)
    else:
        raise ValueError(f"unsupported camera model {model}")
    und = uv.copy()
    for _ in range(10):  # invert the radial model by fixed-point iteration
        r2 = (und ** 2).sum(1, keepdims=True)
        und = uv / (1 + k1 * r2 + k2 * r2 ** 2)
    return und


def _triangulate(rays: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Linear (DLT) triangulation from normalized points and [R|t] poses."""
    a = []
    for uv, r, t in rays:
        p = np.c_[r, t]
        a.append(uv[0] * p[2] - p[0])
        a.append(uv[1] * p[2] - p[1])
    _, _, vt = np.linalg.svd(np.asarray(a))
    x = vt[-1]
    return x[:3] / x[3]


def _project_error_px(pts: np.ndarray, uv: np.ndarray, r: np.ndarray, t: np.ndarray, f: float) -> float:
    pc = pts @ r.T + t
    if np.any(pc[:, 2] <= 0):
        return np.inf
    return float(np.median(np.linalg.norm(pc[:, :2] / pc[:, 2:3] - uv, axis=1)) * f)


def chart_metric_scale(images_dir: str | Path, sparse_dir: str | Path, pitch_m: float, max_views: int = 40,
                       inlier_px: float = 4.0, iterations: int = 200, seed: int = 0) -> dict:
    """Meters per reconstruction unit, from the ColorChecker seen in several registered photos.

    Views are combined with RANSAC: false or misordered detections disagree with the others
    under reprojection and are dropped before the final triangulation.
    """
    cameras, images = read_colmap_model(sparse_dir)
    names = sorted(images)
    step = max(1, len(names) // max_views)
    views = []
    for name in names[::step]:
        d = detect_chart(Path(images_dir) / name)
        if d is None:
            continue
        cam_id, r, t = images[name]
        views.append((_normalize(d.centers, cameras[cam_id]), r, t, float(cameras[cam_id][3][0])))
    if len(views) < 2:
        raise ValueError("the chart was found in fewer than 2 registered photos")

    def triangulate(subset):
        return np.array([_triangulate([(views[k][0][i], views[k][1], views[k][2]) for k in subset]) for i in range(24)])

    rng = np.random.default_rng(seed)
    best = list(range(len(views)))
    best_count = -1
    for _ in range(iterations):
        pair = rng.choice(len(views), 2, replace=False)
        pts = triangulate(pair)
        inliers = [k for k, (uv, r, t, f) in enumerate(views) if _project_error_px(pts, uv, r, t, f) < inlier_px]
        if len(inliers) > best_count:
            best, best_count = inliers, len(inliers)
    if best_count < 2:
        raise ValueError("chart detections don't agree across photos")
    pts = triangulate(best)
    grid = pts.reshape(CHART_ROWS, CHART_COLS, 3)
    gaps = np.r_[np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(), np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()]
    pitch_units = float(np.median(gaps))
    return {"meters_per_unit": pitch_m / pitch_units, "pitch_units": pitch_units, "views": len(views),
            "inlier_views": len(best), "pitch_spread": float(np.std(gaps) / pitch_units)}
