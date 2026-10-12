import numpy as np
import pytest
import trimesh

from scan2usd.splat import Splat, crop_to_asset, define_splat, read_splat_ply, transform_points


def _write_ply(path, n, rng, degree=3):
    k = (degree + 1) ** 2 - 1
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(3 * k)] + [
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = rng.normal(size=(n, len(names))).astype("<f4")
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {p}" for p in names] + ["end_header"]
    path.write_bytes(("\n".join(header) + "\n").encode() + data.tobytes())
    return dict(zip(names, data.T))


def test_read_ply_layout(tmp_path):
    rng = np.random.default_rng(0)
    cols = _write_ply(tmp_path / "s.ply", 50, rng)
    s = read_splat_ply(tmp_path / "s.ply")
    assert len(s) == 50 and s.degree == 3 and s.sh.shape == (50, 16, 3)
    # f_rest is channel-major (15 R, 15 G, 15 B): coefficient j of channel c is f_rest_{c*15+j}.
    assert np.allclose(s.sh[:, 1, 1], cols["f_rest_15"])
    assert np.allclose(s.sh[:, 15, 2], cols["f_rest_44"])
    assert np.allclose(s.scales[:, 0], np.exp(cols["scale_0"]), rtol=1e-5)
    assert np.allclose(np.linalg.norm(s.rotations, axis=1), 1.0)
    assert ((s.opacities > 0) & (s.opacities < 1)).all()


def _splat(points, scale=0.002):
    n = len(points)
    q = np.tile([1.0, 0, 0, 0], (n, 1))
    return Splat(points, np.full((n, 3), scale), q, np.full(n, 0.9), np.zeros((n, 16, 3)))


def test_crop_keeps_surface_gaussians():
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    mesh.apply_translation((0, 0, 0.05))
    on = np.asarray(mesh.sample(300, return_index=False))
    on = on[on[:, 2] > 0.01]
    far = np.array([[0.3, 0, 0.05], [0, 0.4, 0.2]])
    table = np.array([[0.051, 0.0, 0.0005]])  # touching the object, but on the table plane
    m = np.diag([2.0, 2.0, 2.0, 1.0])  # reconstruction units are half of the asset's
    s = _splat(np.vstack([on, far, table]) / 2.0)
    kept, rep = crop_to_asset(s, m, mesh)
    assert rep.kept == len(on) and rep.removed_far == 2 and rep.removed_below == 1
    assert np.allclose(transform_points(m, kept.positions), on)


def test_hybrid_asset_variants(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdVol

    from scan2usd.splat import write_hybrid

    stage = Usd.Stage.CreateNew(str(tmp_path / "a.usda"))
    root = UsdGeom.Xform.Define(stage, "/obj")
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    mesh = UsdGeom.Mesh.Define(stage, "/obj/render")
    mesh.CreatePointsAttr([tuple(map(float, v)) for v in box.vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(box.faces))
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in box.faces.reshape(-1)])
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    m = np.eye(4)
    m[:3, 3] = (0.1, 0, 0)
    out = write_hybrid(tmp_path / "a.usda", _splat(np.zeros((4, 3))), m, tmp_path / "hy")
    st = Usd.Stage.Open(str(out))
    splat = UsdVol.ParticleField3DGaussianSplat(st.GetPrimAtPath("/obj/splat"))
    assert splat and len(splat.GetPositionsAttr().Get()) == 4
    assert splat.GetRadianceSphericalHarmonicsDegreeAttr().Get() == 3
    assert len(splat.GetRadianceSphericalHarmonicsCoefficientsAttr().Get()) == 4 * 16
    world = UsdGeom.Xformable(splat.GetPrim()).ComputeLocalToWorldTransform(0)
    assert np.allclose(world.ExtractTranslation(), (0.1, 0, 0))
    render, prim = UsdGeom.Imageable(st.GetPrimAtPath("/obj/render")), st.GetPrimAtPath("/obj")
    proxy = UsdGeom.Mesh(st.GetPrimAtPath("/obj/shadow_proxy"))
    assert render.ComputeVisibility() == "invisible" and proxy.ComputeVisibility() == "invisible"
    prim.GetVariantSet("appearance").SetVariantSelection("mesh")
    assert render.ComputeVisibility() == "inherited"
    assert UsdGeom.Imageable(splat.GetPrim()).ComputeVisibility() == "invisible"
    prim.GetVariantSet("appearance").SetVariantSelection("splat_proxy")
    assert render.ComputeVisibility() == "invisible" and proxy.ComputeVisibility() == "inherited"
    assert UsdGeom.PrimvarsAPI(proxy.GetPrim()).GetPrimvar("hideForCamera").Get() is True
    # Inset by 5% of the size (5 mm) along vertex normals (box corners: diagonal, 2.9 mm per axis).
    assert np.abs(np.asarray(proxy.GetPointsAttr().Get())).max() < 0.05 - 0.0025


def test_define_splat_quaternion_order(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd

    st = Usd.Stage.CreateInMemory()
    s = _splat(np.zeros((1, 3)))
    s.rotations[0] = [np.cos(0.3), np.sin(0.3), 0, 0]
    gs = define_splat(st, "/s", s)
    q = gs.GetOrientationsAttr().Get()[0]
    assert np.isclose(q.GetReal(), np.cos(0.3)) and np.isclose(q.GetImaginary()[0], np.sin(0.3))

def test_silhouette_of_a_square():
    from scan2usd.splat import silhouette

    # A 0.2 m square, 1 m in front of a 100 px focal camera: 20 x 20 px, centered.
    v = np.array([[-0.1, -0.1, 1.0], [0.1, -0.1, 1.0], [0.1, 0.1, 1.0], [-0.1, 0.1, 1.0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    k = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    a = silhouette(v, f, k, 100, 80)
    assert a.shape == (80, 100)
    assert abs((a / 255.0).sum() - 400) < 25
    assert a[40, 50] == 255 and a[5, 5] == 0


def test_rename_images_bin(tmp_path):
    import struct

    from scan2usd.chart import read_colmap_model
    from scan2usd.splat import _rename_images_bin

    def image(i, name, pts):
        return (struct.pack("<i4d3di", i, 1, 0, 0, 0, 0.1 * i, 0, 0, 1) + name.encode() + b"\x00"
                + struct.pack("<Q", len(pts)) + b"".join(struct.pack("<ddq", *p) for p in pts))

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "images.bin").write_bytes(
        struct.pack("<Q", 2) + image(1, "a.jpg", [(1.0, 2.0, 7)]) + image(2, "b.jpg", []))
    (tmp_path / "src" / "cameras.bin").write_bytes(struct.pack("<Q", 0))
    (tmp_path / "dst").mkdir()
    (tmp_path / "dst" / "cameras.bin").write_bytes(struct.pack("<Q", 0))
    _rename_images_bin(tmp_path / "src" / "images.bin", tmp_path / "dst" / "images.bin", lambda n: n[:-4] + ".png")
    _, images = read_colmap_model(tmp_path / "dst")
    assert sorted(images) == ["a.png", "b.png"]
    assert np.allclose(images["b.png"][2], (0.2, 0, 0))
