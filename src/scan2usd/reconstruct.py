"""Photos -> object mesh: COLMAP reconstruction, table-plane alignment, object isolation.

The output mesh is Z-up with the table at z = 0, but in COLMAP's arbitrary units.
Metric scale comes later from a ruler measurement (`build --scale-to`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
              "int": "<i4", "int32": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}


@dataclass
class ReconReport:
    images: int = 0
    registered_images: int = 0
    dense_points: int = 0
    plane_inlier_fraction: float = 0.0
    object_points: int = 0
    poisson_depth: int = 0
    mesh_faces: int = 0
    watertight: bool = False
    timings_s: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def find_colmap() -> Path:
    env = os.environ.get("SCAN2USD_COLMAP")
    candidates = [Path(env)] if env else []
    candidates.append(Path(__file__).resolve().parents[2] / "tools" / "colmap" / "bin" / "colmap.exe")
    on_path = shutil.which("colmap")
    if on_path:
        candidates.append(Path(on_path))
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("COLMAP not found; set SCAN2USD_COLMAP or unpack it to tools/colmap/")


def _run(cmd: list[str], log: Path) -> None:
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"COLMAP failed ({proc.returncode}); see {log}")


def read_ply_points(path: str | Path) -> dict[str, np.ndarray]:
    """Read a binary little-endian vertex PLY (as written by COLMAP) into named arrays."""
    with open(path, "rb") as f:
        header, props, count = [], [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property") and "list" not in line:
                _, typ, name = line.split()
                props.append((name, _PLY_TYPES[typ]))
            elif line == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"{path}: only binary little-endian PLY is supported")
        data = np.frombuffer(f.read(np.dtype(props).itemsize * count), dtype=np.dtype(props), count=count)
    return {name: np.asarray(data[name]) for name, _ in props}


def write_ply_points(path: str | Path, xyz: np.ndarray, normals: np.ndarray, rgb: np.ndarray) -> None:
    """Write points in COLMAP's fused.ply layout so its Poisson mesher can read them."""
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(len(xyz), dtype=dt)
    for i, k in enumerate("xyz"):
        arr[k] = xyz[:, i]
        arr["n" + k] = normals[:, i]
    for i, k in enumerate(("red", "green", "blue")):
        arr[k] = rgb[:, i]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def fit_plane_ransac(pts: np.ndarray, threshold: float, iterations: int = 800, seed: int = 0):
    """Dominant plane (n, d) with n.p + d = 0, refined by least squares on inliers."""
    rng = np.random.default_rng(seed)
    best, best_count = None, -1
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        count = int(np.sum(np.abs(pts @ n - n @ a) < threshold))
        if count > best_count:
            best, best_count = (n, -n @ a), count
    n, d = best
    inliers = np.abs(pts @ n + d) < threshold
    centroid = pts[inliers].mean(0)
    _, _, vt = np.linalg.svd(pts[inliers] - centroid, full_matrices=False)
    n = vt[2]
    return n, -n @ centroid, inliers


def _rotation_to_z(n: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(n, z), float(n @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def isolate_object(xyz: np.ndarray, spacing: float, plane_threshold: float):
    """Align the table plane to z = 0 (Z-up), drop it, and keep the object standing on it.

    The object is found in 2D: points above the table are binned onto the table plane
    and the occupied blob holding the most points wins. Dark or low-texture objects give
    sparse, patchy 3D points, but their walls still stack into one solid footprint.
    """
    from scipy import ndimage

    n, d, inliers = fit_plane_ransac(xyz, plane_threshold)
    signed = xyz @ n + d
    off = np.abs(signed) > 3 * plane_threshold
    if np.mean(signed[off] > 0) < 0.5:  # object should be on the positive side
        n, d, signed = -n, -d, -signed
    rot = _rotation_to_z(n)
    origin = -d * n
    aligned = (xyz - origin) @ rot.T
    idx = np.nonzero(aligned[:, 2] > 2.5 * plane_threshold)[0]
    if len(idx) == 0:
        raise ValueError("nothing found above the table plane")

    cell = 8.0 * spacing
    xy = aligned[idx, :2]
    lo = xy.min(0)
    ij = np.floor((xy - lo) / cell).astype(int)
    counts = np.zeros(ij.max(0) + 1, dtype=np.int32)
    np.add.at(counts, (ij[:, 0], ij[:, 1]), 1)
    pad = 3
    occupied = ndimage.binary_closing(np.pad(counts >= 3, pad), iterations=2)[pad:-pad, pad:-pad]
    labels, num = ndimage.label(occupied)
    if num == 0:
        raise ValueError("no object footprint found above the table")
    point_labels = labels[ij[:, 0], ij[:, 1]]
    weights = np.bincount(point_labels, minlength=num + 1)
    weights[0] = 0
    keep = idx[point_labels == weights.argmax()]

    # Statistical outlier removal: drop points far from their neighbors.
    tree = cKDTree(aligned[keep])
    dist, _ = tree.query(aligned[keep], k=9)
    mean_d = dist[:, 1:].mean(1)
    keep = keep[mean_d < mean_d.mean() + 2.5 * mean_d.std()]
    return aligned, keep, rot, origin, float(inliers.mean())


def _median_spacing(xyz: np.ndarray, sample: int = 20000, seed: int = 0) -> float:
    """Median nearest-neighbor distance (the cloud's sampling density)."""
    rng = np.random.default_rng(seed)
    probe = xyz[rng.choice(len(xyz), min(sample, len(xyz)), replace=False)]
    dist, _ = cKDTree(xyz).query(probe, k=2)
    return float(np.median(dist[:, 1]))


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _image_dir(images: Path, workspace: Path) -> tuple[Path, int]:
    """COLMAP wants a folder of only images; link them into the workspace if needed."""
    files = sorted(p for p in images.iterdir() if p.is_file())
    pics = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    if len(pics) == len(files):
        return images, len(pics)
    target = workspace / "images"
    target.mkdir(parents=True, exist_ok=True)
    for p in pics:
        dst = target / p.name
        if not dst.exists():
            try:
                os.link(p, dst)
            except OSError:
                shutil.copy2(p, dst)
    return target, len(pics)


def reconstruct(images: str | Path, workspace: str | Path, quality: str = "medium",
                single_camera: bool = True, poisson_depth: int | None = None) -> tuple[Path, ReconReport]:
    images, workspace = Path(images).resolve(), Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    images, count = _image_dir(images, workspace)
    rep = ReconReport(images=count)
    colmap = str(find_colmap())
    log = workspace / "colmap.log"

    t = time.perf_counter()
    dense = workspace / "dense" / "0"
    fused = dense / "fused.ply"
    if not fused.exists():
        # Sparse via the automatic pipeline, then dense steps explicitly: the automatic
        # dense mode also Poisson-meshes the whole scene (table included), which is slow and unused.
        sparse = workspace / "sparse" / "0"
        if not (sparse / "images.bin").exists():
            _run([colmap, "automatic_reconstructor", "--workspace_path", str(workspace), "--image_path", str(images),
                  "--data_type", "individual", "--quality", quality, "--single_camera", str(int(single_camera)),
                  "--dense", "0", "--use_gpu", "1"], log)
        if not sparse.exists():
            raise RuntimeError(f"COLMAP could not register the photos; see {log}")
        max_size = {"low": 1000, "medium": 1600, "high": 2400, "extreme": -1}[quality]
        dense.mkdir(parents=True, exist_ok=True)  # image_undistorter crashes if it's missing
        _run([colmap, "image_undistorter", "--image_path", str(images), "--input_path", str(sparse),
              "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", str(max_size)], log)
        # Mirrors COLMAP's automatic quality presets (coarser = faster), keeping geometric filtering.
        speed = {"low": ["--PatchMatchStereo.window_step", "2", "--PatchMatchStereo.window_radius", "4",
                         "--PatchMatchStereo.num_samples", "7", "--PatchMatchStereo.num_iterations", "3"],
                 "medium": ["--PatchMatchStereo.window_step", "2", "--PatchMatchStereo.window_radius", "4",
                            "--PatchMatchStereo.num_samples", "10", "--PatchMatchStereo.num_iterations", "4"]}
        _run([colmap, "patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
              "--PatchMatchStereo.geom_consistency", "true", *speed.get(quality, [])], log)
        _run([colmap, "stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
              "--input_type", "geometric", "--output_path", str(fused)], log)
    else:
        rep.notes.append("reused existing dense reconstruction")
    rep.timings_s["colmap"] = round(time.perf_counter() - t, 1)
    if not fused.exists():
        raise RuntimeError(f"no dense point cloud produced; see {log}")
    sparse_images = workspace / "sparse" / "0" / "images.bin"
    if sparse_images.exists():
        with open(sparse_images, "rb") as f:
            rep.registered_images = int(np.frombuffer(f.read(8), dtype="<u8")[0])

    t = time.perf_counter()
    ply = read_ply_points(fused)
    xyz = np.c_[ply["x"], ply["y"], ply["z"]].astype(np.float64)
    nrm = np.c_[ply["nx"], ply["ny"], ply["nz"]].astype(np.float64)
    rgb = np.c_[ply["red"], ply["green"], ply["blue"]].astype(np.uint8)
    rep.dense_points = len(xyz)
    spacing = _median_spacing(xyz)
    aligned, keep, rot, origin, plane_frac = isolate_object(xyz, spacing, plane_threshold=2.0 * spacing)
    rep.plane_inlier_fraction = round(plane_frac, 3)
    rep.object_points = int(len(keep))
    obj_xyz, obj_nrm, obj_rgb = aligned[keep], nrm[keep] @ rot.T, rgb[keep]
    cropped = workspace / "object_points.ply"
    write_ply_points(cropped, obj_xyz, obj_nrm, obj_rgb)
    rep.timings_s["isolate"] = round(time.perf_counter() - t, 1)

    t = time.perf_counter()
    if poisson_depth is None:
        # One octree cell per ~1.5 point spacings: finer only fits noise and invents sheets.
        obj_spacing = _median_spacing(obj_xyz)
        extent = float((obj_xyz.max(0) - obj_xyz.min(0)).max())
        poisson_depth = int(np.clip(np.ceil(np.log2(extent / (1.5 * obj_spacing))), 6, 11))
    rep.poisson_depth = poisson_depth
    raw_mesh = workspace / "object_poisson.ply"
    _run([colmap, "poisson_mesher", "--input_path", str(cropped), "--output_path", str(raw_mesh),
          "--PoissonMeshing.depth", str(poisson_depth), "--PoissonMeshing.point_weight", "4",
          "--PoissonMeshing.trim", "0", "--PoissonMeshing.color", "1"], log)
    mesh = trimesh.load(str(raw_mesh), force="mesh", process=True)
    # Poisson closes the unseen base with a blob; cut it flat where the object met the table.
    mesh = mesh.slice_plane([0, 0, 0], [0, 0, 1], cap=True)
    bodies = mesh.split(only_watertight=False)
    if len(bodies) > 1:
        sizes = [b.volume if b.is_watertight else b.area for b in bodies]
        mesh = bodies[int(np.argmax(sizes))]
        rep.notes.append(f"kept the main body, dropped {len(bodies) - 1} Poisson fragment(s)")
    _, nearest = cKDTree(obj_xyz).query(mesh.vertices)
    colors = obj_rgb[nearest]
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.c_[colors, np.full(len(colors), 255, np.uint8)])
    out = workspace / "object_mesh.ply"
    mesh.export(str(out))
    rep.mesh_faces = int(len(mesh.faces))
    rep.watertight = bool(mesh.is_watertight)
    rep.timings_s["mesh"] = round(time.perf_counter() - t, 1)
    (workspace / "recon_report.json").write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")
    return out, rep
