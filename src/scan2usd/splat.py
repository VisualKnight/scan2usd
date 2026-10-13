"""Hybrid assets: a Gaussian splat for appearance, the mesh and convex hulls for physics.

A splat (trained on the same photos, e.g. with Brush) reproduces view-dependent appearance far
better than a textured mesh, but a robot simulator can't collide with it. Here the splat becomes
one more prim inside the physics asset: a `UsdVol.ParticleField3DGaussianSplat` (OpenUSD 26.03+)
under the rigid body, so it moves with physics, while the hulls keep doing the contacts.

The splat data stays in the reconstruction's (COLMAP) frame, untouched; the prim's xform carries
the same similarity transform the build applied to the mesh (align to the table, metric scale,
recenter). That keeps provenance, and avoids rotating spherical-harmonics coefficients by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


@dataclass
class Splat:
    positions: np.ndarray  # (N, 3)
    scales: np.ndarray  # (N, 3) standard deviations (linear, not log)
    rotations: np.ndarray  # (N, 4) unit quaternions, (w, x, y, z)
    opacities: np.ndarray  # (N,) in [0, 1] (not logit)
    sh: np.ndarray  # (N, K, 3) radiance SH coefficients, K = (degree + 1)^2

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def degree(self) -> int:
        return int(round(np.sqrt(self.sh.shape[1]))) - 1

    def subset(self, keep: np.ndarray) -> "Splat":
        return Splat(self.positions[keep], self.scales[keep], self.rotations[keep], self.opacities[keep],
                     self.sh[keep])

    def base_colors(self) -> np.ndarray:
        """View-independent (DC) color in [0, 1]."""
        return np.clip(0.5 + SH_C0 * self.sh[:, 0], 0.0, 1.0)


def read_splat_ply(path) -> Splat:
    """Read a 3D Gaussian Splatting .ply (the INRIA layout written by Brush, gsplat, Nerfstudio...)."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path}: expected a binary little-endian PLY")
    count = next(int(h.split()[-1]) for h in header if h.startswith("element vertex"))
    types = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
    props = [h.split() for h in header if h.startswith("property")]
    v = np.frombuffer(raw, np.dtype([(p[2], types[p[1]]) for p in props]), count=count, offset=end)
    col = lambda *names: np.stack([v[n].astype(np.float64) for n in names], 1)  # noqa: E731

    n_rest = sum(1 for p in props if p[2].startswith("f_rest_"))
    k = n_rest // 3  # coefficients per channel beyond DC
    sh = np.zeros((count, k + 1, 3))
    sh[:, 0] = col("f_dc_0", "f_dc_1", "f_dc_2")
    if k:
        rest = col(*[f"f_rest_{i}" for i in range(n_rest)])  # channel-major: R block, G block, B block
        sh[:, 1:] = rest.reshape(count, 3, k).transpose(0, 2, 1)
    q = col("rot_0", "rot_1", "rot_2", "rot_3")
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return Splat(
        positions=col("x", "y", "z"),
        scales=np.exp(col("scale_0", "scale_1", "scale_2")),
        rotations=q,
        opacities=1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64))),
        sh=sh,
    )


def asset_transform(workspace, usd_path) -> np.ndarray:
    """4x4 matrix (column vectors) taking reconstruction (COLMAP) coordinates into the asset frame.

    The build's chain: isolate_object's table alignment (rot, origin), the metric scale s, and the
    recentering offset t:  P = s * rot @ (X - origin) + t.
    """
    import json

    import trimesh

    from .reconstruct import _median_spacing, isolate_object, read_ply_points
    from .usd_io import read_usd

    workspace, usd_path = Path(workspace), Path(usd_path)
    ply = read_ply_points(workspace / "dense" / "0" / "fused.ply")
    xyz = np.c_[ply["x"], ply["y"], ply["z"]].astype(np.float64)
    sp = _median_spacing(xyz)
    _, _, rot, origin, _ = isolate_object(xyz, sp, 2.0 * sp)
    s = float(json.loads((usd_path.parent / "build_report.json").read_text())["scale_applied"])
    recon = trimesh.load(workspace / "object_mesh.ply", force="mesh")
    t = read_usd(usd_path).render_mesh.bounds[0] - s * recon.bounds[0]
    m = np.eye(4)
    m[:3, :3] = s * rot
    m[:3, 3] = t - s * rot @ origin
    return m


def transform_points(m: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ m[:3, :3].T + m[:3, 3]


def silhouette(vertices_cam: np.ndarray, faces: np.ndarray, k: np.ndarray, w: int, h: int,
               supersample: int = 3) -> np.ndarray:
    """Anti-aliased silhouette (uint8 alpha) of a mesh given in camera coordinates (+Z forward)."""
    from PIL import Image, ImageDraw

    z = vertices_cam[:, 2:3]
    uv = (vertices_cam[:, :2] / np.maximum(z, 1e-9)) * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    uv *= supersample
    tri = uv[faces]
    front = (z[faces][..., 0] > 0).all(1)
    img = Image.new("L", (w * supersample, h * supersample), 0)
    draw = ImageDraw.Draw(img)
    for t in tri[front]:
        draw.polygon([tuple(p) for p in t], fill=255)
    return np.asarray(img.resize((w, h), Image.BOX))


def _rename_images_bin(src: Path, dst: Path, rename) -> None:
    """Copy a COLMAP images.bin, renaming every image (layout unchanged otherwise)."""
    import struct

    data = src.read_bytes()
    out = bytearray(data[:8])
    pos = 8
    for _ in range(struct.unpack_from("<Q", data, 0)[0]):
        head = data[pos:pos + 64]  # id (4) + qvec (32) + tvec (24) + camera id (4)
        pos += 64
        end = data.index(b"\x00", pos)
        name = rename(data[pos:end].decode())
        pos = end + 1
        n_pts = struct.unpack_from("<Q", data, pos)[0]
        body = data[pos:pos + 8 + 24 * n_pts]
        pos += 8 + 24 * n_pts
        out += head + name.encode() + b"\x00" + body
    dst.write_bytes(bytes(out))


def masked_dataset(model_dir, image_dir, usd_path, m: np.ndarray, out_dir, exclude=()) -> dict:
    """A COLMAP dataset whose photos carry the asset's silhouette as alpha (RGBA PNG).

    The mesh tells the splat where the object is: projected through each solved camera, it gives an
    exact object mask, so a splat trainer that honors transparency (Brush) learns the object alone,
    with no background Gaussians to crop away and no see-through holes where they used to be.
    `m` maps reconstruction coordinates to the asset frame (asset_transform).
    """
    import shutil

    from PIL import Image

    from .chart import read_colmap_model
    from .usd_io import read_usd

    model_dir, image_dir, out_dir = Path(model_dir), Path(image_dir), Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    mesh = read_usd(usd_path).render_mesh
    verts = transform_points(np.linalg.inv(m), np.asarray(mesh.vertices))
    faces = np.asarray(mesh.faces)
    cameras, images = read_colmap_model(model_dir)
    png = lambda n: str(Path(n).with_suffix(".png"))  # noqa: E731
    coverage = []
    for name, (cam_id, r, t) in images.items():
        if name in exclude:
            continue
        model, w, h, p = cameras[cam_id]
        single = ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE")
        fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if model in single else p[:4]
        n_intr = 3 if model in single else 4
        if len(p) > n_intr and np.abs(p[n_intr:]).max() > 1e-3:
            raise ValueError(f"{model} camera with lens distortion {p[n_intr:]}: use the undistorted model "
                             "(<workspace>/dense/0/sparse) and its images")
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        alpha = silhouette(verts @ np.asarray(r).T + np.asarray(t), faces, k, int(w), int(h))
        rgb = Image.open(image_dir / name).convert("RGB")
        if rgb.size != (w, h):
            rgb = rgb.resize((int(w), int(h)), Image.LANCZOS)
        Image.fromarray(np.dstack([np.asarray(rgb), alpha])).save(out_dir / "images" / png(name))
        coverage.append(float((alpha > 127).mean()))
    for f in model_dir.iterdir():
        if f.name != "images.bin" and f.is_file():
            shutil.copy(f, out_dir / "sparse" / "0" / f.name)
    _rename_images_bin(model_dir / "images.bin", out_dir / "sparse" / "0" / "images.bin", png)
    return {"images": len(coverage), "object_fraction_median": float(np.median(coverage)), "excluded": len(exclude)}


@dataclass
class CropReport:
    total: int
    kept: int
    margin_m: float
    removed_far: int
    removed_below: int
    removed_faint: int
    removed_large: int


def crop_to_asset(splat: Splat, m: np.ndarray, mesh, margin: float | None = None,
                  min_opacity: float = 0.05, max_size: float | None = None) -> tuple[Splat, CropReport]:
    """Keep the Gaussians that belong to the object: near its surface, above the table, not faint,
    not huge. `mesh` is the asset's render mesh (asset frame, meters)."""
    from scipy.spatial import cKDTree

    extent = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    margin = margin if margin is not None else max(0.003, 0.02 * extent)
    max_size = max_size if max_size is not None else 0.05 * extent
    s = float(np.cbrt(abs(np.linalg.det(m[:3, :3]))))
    p = transform_points(m, splat.positions)
    surface = np.asarray(mesh.sample(200_000)) if len(mesh.faces) else np.asarray(mesh.vertices)
    dist, _ = cKDTree(surface).query(p, distance_upper_bound=4 * margin)
    far = dist > margin
    below = p[:, 2] < 0.5 * margin
    faint = splat.opacities < min_opacity
    large = s * splat.scales.max(1) > max_size
    keep = ~(far | below | faint | large)
    rep = CropReport(len(splat), int(keep.sum()), margin, int(far.sum()), int((below & ~far).sum()),
                     int((faint & ~far & ~below).sum()), int((large & ~far & ~below & ~faint).sum()))
    return splat.subset(keep), rep


def define_splat(stage, path: str, splat: Splat, m: np.ndarray | None = None):
    """Author a UsdVol.ParticleField3DGaussianSplat prim; `m` (column-vector 4x4) becomes its xform."""
    from pxr import Gf, UsdGeom, UsdVol, Vt

    gs = UsdVol.ParticleField3DGaussianSplat.Define(stage, path)
    f32 = lambda a: np.ascontiguousarray(a, np.float32)  # noqa: E731
    gs.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(f32(splat.positions)))
    gs.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(f32(splat.scales)))
    gs.CreateOrientationsAttr(Vt.QuatfArray.FromNumpy(f32(splat.rotations[:, [1, 2, 3, 0]])))  # Vt: x,y,z,w
    gs.CreateOpacitiesAttr(Vt.FloatArray.FromNumpy(f32(splat.opacities)))
    gs.CreateRadianceSphericalHarmonicsDegreeAttr(splat.degree)
    sh_attr = gs.CreateRadianceSphericalHarmonicsCoefficientsAttr(
        Vt.Vec3fArray.FromNumpy(f32(splat.sh.reshape(-1, 3))))
    pv = UsdGeom.Primvar(sh_attr)
    pv.SetElementSize(splat.sh.shape[1])
    pv.SetInterpolation(UsdGeom.Tokens.vertex)
    pad = 3.0 * splat.scales.max(1)[:, None]
    lo, hi = (splat.positions - pad).min(0), (splat.positions + pad).max(0)
    gs.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    gs.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(f32(splat.base_colors())))
    if m is not None:
        UsdGeom.Xformable(gs.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))
    return gs


def write_hybrid(asset_usd, splat: Splat, m: np.ndarray, out_dir, default: str = "splat",
                 metadata: dict | None = None, proxy_inset: float | None = None) -> Path:
    """A hybrid asset: a thin layer referencing the physics asset, plus the splat (binary .usdc) and an
    `appearance` variant set: mesh | splat | splat_proxy. Physics (rigid body, mass, hulls) is unchanged."""
    import json
    import os

    from pxr import Sdf, Usd, UsdGeom, Vt

    asset_usd, out_dir = Path(asset_usd), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Usd.Stage.Open(str(asset_usd))
    root = src.GetDefaultPrim()
    name = root.GetName()
    render = next((p for p in root.GetChildren() if p.GetName() == "render"), None)
    if render is None:
        raise ValueError(f"{asset_usd}: no render mesh under /{name}")

    splat_file = out_dir / f"{name}_splat.usdc"
    s_stage = Usd.Stage.CreateNew(str(splat_file))
    UsdGeom.SetStageUpAxis(s_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(s_stage, 1.0)
    define_splat(s_stage, "/splat", splat, m)
    s_stage.SetDefaultPrim(s_stage.GetPrimAtPath("/splat"))
    # Shadow proxy: the render mesh pulled inward along its normals. The Gaussians receive shadows in
    # RTX; at the surface itself the proxy would shadow the very Gaussians it stands in for.
    from .usd_io import read_usd

    rm = read_usd(asset_usd).render_mesh
    inset = proxy_inset if proxy_inset is not None else 0.05 * float(np.max(rm.extents))
    # Thin walls (a mug): never move a vertex past the middle of the wall, or the opposite surface
    # ends up outside the object, in front of the Gaussians it should sit behind.
    origin = rm.vertices - 1e-4 * rm.vertex_normals
    hit, ray, _ = rm.ray.intersects_location(origin, -rm.vertex_normals, multiple_hits=False)
    thickness = np.full(len(origin), np.inf)
    thickness[ray] = np.linalg.norm(hit - origin[ray], axis=1)
    per_vertex = np.minimum(inset, 0.5 * thickness)[:, None]
    pm = UsdGeom.Mesh.Define(s_stage, "/shadow_proxy")
    pm.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(
        rm.vertices - per_vertex * rm.vertex_normals, np.float32)))
    pm.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(rm.faces), 3, np.int32)))
    pm.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(rm.faces.reshape(-1), np.int32)))
    pm.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(pm).CreatePrimvar("hideForCamera", Sdf.ValueTypeNames.Bool).Set(True)
    s_stage.GetRootLayer().Save()

    out = out_dir / f"{name}.usda"
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = stage.DefinePrim(f"/{name}")
    prim.GetReferences().AddReference(Path(os.path.relpath(asset_usd, out_dir)).as_posix())
    stage.SetDefaultPrim(prim)
    prim.SetKind("component")
    sp = stage.DefinePrim(f"/{name}/splat")
    sp.GetPayloads().AddPayload(splat_file.name)
    stage.DefinePrim(f"/{name}/shadow_proxy").GetReferences().AddReference(splat_file.name, "/shadow_proxy")
    info = {"splats": len(splat), "sh_degree": splat.degree, "source_frame": "reconstruction (COLMAP)",
            "shadow_proxy_inset_m": inset, **(metadata or {})}
    prim.SetCustomDataByKey("scan2usd:splat", json.dumps(info))

    vs = prim.GetVariantSets().AddVariantSet("appearance")
    for v in ("mesh", "splat", "splat_proxy"):
        vs.AddVariant(v)
        vs.SetVariantSelection(v)
        with vs.GetVariantEditContext():
            UsdGeom.Imageable(stage.GetPrimAtPath(f"/{name}/render")).CreateVisibilityAttr(
                "inherited" if v == "mesh" else "invisible")
            UsdGeom.Imageable(stage.GetPrimAtPath(f"/{name}/splat")).CreateVisibilityAttr(
                "invisible" if v == "mesh" else "inherited")
            # Omniverse RTX: primvars:hideForCamera keeps the proxy out of camera rays, while it still casts
            # shadows (the splat's own lighting is baked, and it casts none).
            UsdGeom.Imageable(stage.GetPrimAtPath(f"/{name}/shadow_proxy")).CreateVisibilityAttr(
                "inherited" if v == "splat_proxy" else "invisible")
    vs.SetVariantSelection(default)
    stage.GetRootLayer().documentation = (
        f"Scan2USD hybrid asset: {asset_usd.name} (mesh, hulls, mass) + Gaussian splat ({len(splat)} splats). "
        "Variant set 'appearance' = mesh | splat | splat_proxy (splat seen by the camera, plus an inset copy "
        "of the mesh hidden from camera rays that casts the shadows: Omniverse RTX).")
    stage.GetRootLayer().Save()
    return out
