import os
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdShade

from scan2usd.physics import estimate_mass_properties, single_hull
from scan2usd.simready import INTERMEDIATE, SimReadyInfo, bad_path_parts, face_normals, package, validate
from scan2usd.usd_io import PhysicsMaterial, SurfaceMaterial, write_usd

INFO = SimReadyInfo(category="box", qcode="Q639460", material="wood", license="Apache-2.0")


def _textured_asset(tmp_path):
    tex = tmp_path / "src" / "textures" / "albedo.png"
    tex.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (200, 40, 40)).save(tex)
    box = trimesh.creation.box((0.1, 0.06, 0.04))
    box.apply_translation([0, 0, 0.02])
    box.visual = trimesh.visual.TextureVisuals(uv=np.full((len(box.vertices), 2), 0.5))
    path = tmp_path / "src" / "red_box.usda"
    write_usd(path, "red_box", box, single_hull(box), estimate_mass_properties(box, density=500.0),
              SurfaceMaterial(texture=tex), PhysicsMaterial())
    return path


def test_build_writes_hull_normals_and_float3_texture_output(tmp_path):
    stage = Usd.Stage.Open(str(_textured_asset(tmp_path)))
    hull = UsdGeom.Mesh(stage.GetPrimAtPath("/red_box/collision/hull_000"))
    n = np.asarray(hull.GetNormalsAttr().Get())
    assert hull.GetNormalsInterpolation() == UsdGeom.Tokens.uniform
    assert len(n) == len(hull.GetFaceVertexCountsAttr().Get())
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)
    tex = UsdShade.Shader(stage.GetPrimAtPath("/red_box/Looks/M_surface/albedoTexture"))
    assert tex.GetOutput("rgb").GetTypeName() == Sdf.ValueTypeNames.Float3
    surf = UsdShade.Shader(stage.GetPrimAtPath("/red_box/Looks/M_surface/PreviewSurface"))
    assert surf.GetInput("diffuseColor").GetConnectedSources()[0][0].source.GetPath() == tex.GetPath()


def test_face_normals_point_outward():
    box = trimesh.creation.box((1, 1, 1))
    n = face_normals(np.asarray(box.vertices), np.full(len(box.faces), 3), np.asarray(box.faces).ravel())
    assert np.allclose(n, box.face_normals, atol=1e-9)


def test_package_layout_and_content(tmp_path):
    src = _textured_asset(tmp_path)
    main = package(src, tmp_path / "pkg", INFO)
    folder = tmp_path / "pkg" / "red_box" / INTERMEDIATE
    assert main == folder / "sm_red_box_01.usda"
    assert [p.name for p in folder.glob("*.usd*")] == [main.name]  # NP.005: the only USD file there
    assert (folder / "textures" / "albedo.png").exists()
    thumb = folder / ".thumbs" / "256x256" / f"{main.name}.png"
    assert Image.open(thumb).size == (256, 256)

    stage = Usd.Stage.Open(str(main))
    root = stage.GetDefaultPrim()
    assert root.GetAttribute("semantics:labels:class").Get() == ["box"]
    assert root.GetAttribute("semantics:labels:wikidata_qcode").Get() == ["Q639460"]
    for path in ("/red_box/Looks/M_surface", "/red_box/Looks/PM_contact"):
        assert stage.GetPrimAtPath(path).GetAttribute("semantics:labels:material").Get() == ["wood"]
    tex = stage.GetPrimAtPath("/red_box/Looks/M_surface/albedoTexture").GetAttribute("inputs:file").Get()
    assert tex.path == "./textures/albedo.png" and tex.resolvedPath

    data = stage.GetRootLayer().customLayerData
    md = data["SimReady_Metadata"]
    assert {data[k] for k in ("asset_name", "asset_type")} == {"red_box", "prop"}
    assert data["source_file"] and data["usd_date_generated"]
    assert md["qcode"] == "Q639460" and md["rigid_body_count"] == 1
    assert md["mass"] == pytest.approx(500.0 * 0.1 * 0.06 * 0.04, rel=1e-3)
    assert np.allclose(md["asset_extents"], (0.1, 0.06, 0.04), atol=1e-5)
    for k in ("asset_name", "asset_type", "source_file", "usd_date_generated"):  # each field authored once
        assert k not in md

    render = UsdGeom.Mesh(stage.GetPrimAtPath("/red_box/render"))
    color = UsdGeom.PrimvarsAPI(render).GetPrimvar("displayColor").Get()[0]
    assert np.allclose(color, (0.578, 0.021, 0.021), atol=0.01)  # the sRGB texel (200, 40, 40), linear
    assert src.read_text().count("SemanticsLabelsAPI") == 0  # the source asset is untouched


def test_package_keeps_vertex_colors(tmp_path):
    box = trimesh.creation.box((0.05, 0.05, 0.05))
    box.visual = trimesh.visual.ColorVisuals(box, vertex_colors=np.tile([30, 160, 60, 255], (len(box.vertices), 1)))
    src = tmp_path / "src" / "green_box.usda"
    write_usd(src, "green_box", box, single_hull(box), estimate_mass_properties(box, density=500.0),
              SurfaceMaterial(use_vertex_colors=True), PhysicsMaterial())
    stage = Usd.Stage.Open(str(package(src, tmp_path / "pkg", INFO)))
    pv = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/green_box/render")).GetPrimvar("displayColor")
    assert pv.GetInterpolation() == UsdGeom.Tokens.vertex and len(pv.Get()) == len(box.vertices)
    reader = UsdShade.Shader(stage.GetPrimAtPath("/green_box/Looks/M_surface/colorReader"))
    assert reader.GetOutput("result").GetTypeName() == Sdf.ValueTypeNames.Float3
    thumb = np.asarray(Image.open(tmp_path / "pkg" / "green_box" / INTERMEDIATE / ".thumbs" / "256x256" /
                                  "sm_green_box_01.usda.png").convert("RGB"), int)
    lit = thumb[(thumb != 255).any(-1)]
    assert lit[:, 1].mean() > lit[:, 0].mean() + 30  # rendered green, not the grey fallback


def test_package_twice_and_repackage_a_package(tmp_path):
    src = _textured_asset(tmp_path)
    first = package(src, tmp_path / "pkg", INFO)
    again = package(src, tmp_path / "pkg", INFO)  # replaces the earlier package
    assert again == first and again.exists()
    redo = package(first, tmp_path / "pkg", INFO)  # the package itself as the source
    assert redo == first
    stage = Usd.Stage.Open(str(redo))
    assert stage.GetDefaultPrim().GetAttribute("semantics:labels:class").Get() == ["box"]
    assert (redo.parent / "textures" / "albedo.png").exists()
    assert not list((tmp_path / "pkg" / "red_box").glob(".*building*"))


def test_package_keeps_same_named_textures_apart(tmp_path):
    a, b = tmp_path / "src" / "a" / "Texture.png", tmp_path / "src" / "b" / "texture.png"  # case differs only
    for f, c in ((a, (200, 40, 40)), (b, (128, 128, 255))):
        f.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), c).save(f)
    box = trimesh.creation.box((0.05, 0.05, 0.05))
    box.visual = trimesh.visual.TextureVisuals(uv=np.full((len(box.vertices), 2), 0.5))
    src = tmp_path / "src" / "two_tex.usda"
    write_usd(src, "two_tex", box, single_hull(box), estimate_mass_properties(box, density=500.0),
              SurfaceMaterial(texture=a, normal_texture=b), PhysicsMaterial())
    stage = Usd.Stage.Open(str(package(src, tmp_path / "pkg", INFO, thumbnail=False)))
    files = {p.GetName(): p.GetAttribute("inputs:file").Get().path for p in stage.Traverse()
             if p.GetAttribute("inputs:file")}
    assert files == {"albedoTexture": "./textures/Texture.png", "normalTexture": "./textures/texture_1.png"}
    tex = tmp_path / "pkg" / "two_tex" / INTERMEDIATE / "textures"
    assert Image.open(tex / "Texture.png").getpixel((0, 0)) == (200, 40, 40)
    assert Image.open(tex / "texture_1.png").getpixel((0, 0)) == (128, 128, 255)


@pytest.mark.skipif(os.name != "nt", reason="open files block directory renames on Windows only")
def test_package_keeps_the_old_one_when_a_file_is_locked(tmp_path):
    src = _textured_asset(tmp_path)
    main = package(src, tmp_path / "pkg", INFO)
    with open(main.parent / "textures" / "albedo.png", "rb"):  # e.g. a viewer holding a texture
        with pytest.raises(PermissionError):
            package(src, tmp_path / "pkg", INFO)
    assert main.exists() and (main.parent / ".thumbs" / "256x256" / f"{main.name}.png").exists()
    assert not (main.parent.parent / f".{INTERMEDIATE}_building").exists()
    assert package(src, tmp_path / "pkg", INFO) == main  # and works once the file is released


def test_package_grasp_lines_cross_the_object(tmp_path):
    from scan2usd.schemas import Grasp, write_grasps

    src = _textured_asset(tmp_path)
    pose = np.eye(4)
    pose[:3, 3] = (0.0, 0.0, 0.12)
    write_grasps(src, [Grasp(pose=pose, gripper="panda", type="pinch", success_rate=1.0, trials=4)])
    stage = Usd.Stage.Open(str(package(src, tmp_path / "pkg", INFO, thumbnail=False)))
    lines = [p for p in stage.Traverse() if p.GetName().startswith("grasp_identifier")]
    assert len(lines) == 1
    curve = UsdGeom.BasisCurves(lines[0])
    a, b = np.asarray(curve.GetPointsAttr().Get(), float)
    assert np.allclose(a, (0, 0, 0.12), atol=1e-6)
    assert b[2] < 0.02 < a[2]  # through the box's center (z 0.02) and out the far side
    assert curve.GetExtentAttr().Get() is not None and curve.ComputePurpose() == UsdGeom.Tokens.guide


def test_bad_path_parts_and_qcode():
    root = Path(Path.cwd().anchor)  # D:\ on Windows, / elsewhere
    assert bad_path_parts(root / "Local Projects" / "x" / "a.usda") == ["Local Projects"]
    assert bad_path_parts(root / "scan2usd_simready" / "a" / "b.usda") == []
    with pytest.raises(ValueError):
        SimReadyInfo(category="mug", qcode="mug", material="ceramic", license="x")


@pytest.mark.skipif(not os.environ.get("SIMREADY_VALIDATE"), reason="set SIMREADY_VALIDATE to the simready-validate "
                    "executable (pip install simready-validate simready-foundation-tier-core, Python 3.12)")
def test_package_passes_robotics_prop(tmp_path):
    if bad_path_parts(tmp_path):
        pytest.skip(f"{tmp_path} has directory names NP.003 rejects")
    report = validate(package(_textured_asset(tmp_path), tmp_path / "pkg", INFO))
    assert report["passed"], {k: f["failing"] for k, f in report["features"].items() if f["blocking"]}
