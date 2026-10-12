import json

import numpy as np
from PIL import Image

from scan2usd.cli import main
from scan2usd.color import render_chart
from scan2usd.demo import CAMERA_CAST, run_demo
from scan2usd.pipeline import BuildOptions, build_asset
from scan2usd.usd_io import read_usd


def test_demo_end_to_end(tmp_path):
    res, glaze = run_demo(tmp_path, target_faces=8000)
    assert (tmp_path / "preview.png").exists()
    assert res.validation.status == "pass", res.validation.to_markdown()
    s = res.summary
    assert s["clean"]["components_in"] == 3 and s["clean"]["components_kept"] == 1
    assert s["mass_properties"]["volume_source"] == "render_mesh"
    # ~155 cm^3 of ceramic at 2400 kg/m^3.
    assert 0.33 < s["mass_properties"]["mass"] < 0.41
    assert glaze["captured"] > 4.0 and glaze["calibrated"] < 1.5
    assert (tmp_path / "asset" / "ceramic_mug.usda").exists()
    assert (tmp_path / "asset" / "ceramic_mug.glb").exists()
    assert (tmp_path / "asset" / "validation.md").exists()
    assert (tmp_path / "asset" / "preview.png").exists()
    assert json.loads((tmp_path / "asset" / "build_report.json").read_text())["demo_glaze_delta_e2000"]


def test_textured_build_calibrates_texture(tmp_path):
    from conftest import make_textured_box

    src = tmp_path / "in" / "crate.glb"
    src.parent.mkdir()
    make_textured_box().export(str(src))
    img, corners = render_chart(CAMERA_CAST)
    chart = tmp_path / "in" / "chart.png"
    Image.fromarray(img).save(chart)

    opts = BuildOptions(name="crate", material="cardboard", collision="hull", chart_image=chart,
                        chart_corners=corners.tolist(), run_sim=False)
    res = build_asset(src, tmp_path / "out", opts)
    assert res.validation.status == "pass", res.validation.to_markdown()
    tex = tmp_path / "out" / "textures" / "crate_albedo.png"
    assert tex.exists()
    assert res.summary["albedo"]["source"] == "texture"
    assert (tmp_path / "out" / "color_calibration.json").exists()
    asset = read_usd(res.usd_path)
    assert asset.texture_paths and asset.texture_paths[0].resolve() == tex.resolve()
    # The warm camera cast is removed: the inverse CCM lifts blue relative to red.
    before = np.asarray(make_textured_box().visual.material.image.convert("RGB"), dtype=float)
    after = np.asarray(Image.open(tex).convert("RGB"), dtype=float)
    assert (after[..., 2] - after[..., 0]).mean() > (before[..., 2] - before[..., 0]).mean() + 10


def test_baked_build_writes_pbr_material_and_normal_map(tmp_path):
    import trimesh
    from pxr import Usd, UsdShade

    dense = trimesh.creation.icosphere(subdivisions=5, radius=0.04)
    dense.vertices *= (1 + 0.03 * np.sin(40 * dense.vertices[:, :1]))  # surface detail to bake
    dense.visual = trimesh.visual.ColorVisuals(dense, vertex_colors=np.tile([180, 60, 40, 255], (len(dense.vertices), 1)))
    src = tmp_path / "bumpy.ply"
    dense.export(str(src))
    opts = BuildOptions(name="bumpy", material="ceramic", collision="hull", target_faces=2000, bake_texture=256,
                        run_sim=False, usda=True, preview=False)
    res = build_asset(src, tmp_path / "out", opts)
    normal = tmp_path / "out" / "textures" / "bumpy_normal.png"
    assert normal.exists()
    stage = Usd.Stage.Open(str(res.usd_path))
    shaders = {p.GetName(): UsdShade.Shader(p) for p in stage.Traverse() if p.IsA(UsdShade.Shader)}
    assert shaders["normalTexture"].GetInput("sourceColorSpace").Get() == "raw"
    surf = shaders["PreviewSurface"]
    assert surf.GetInput("normal").HasConnectedSource()
    assert surf.GetInput("clearcoat").Get() == 1.0  # ceramic glaze look preset
    asset = read_usd(res.usd_path)
    assert len(asset.texture_paths) == 1 and asset.texture_paths[0].name == "bumpy_albedo.png"
    assert res.summary["albedo"]["look"]["preset"] == "ceramic"


def test_cli_build_and_validate(tmp_path, capsys):
    import trimesh

    src = tmp_path / "block.obj"
    block = trimesh.creation.box((20.0, 10.0, 5.0))  # centimeters
    block.export(str(src))
    out = tmp_path / "out"
    code = main(["build", str(src), "-o", str(out), "--units", "cm", "--material", "wood",
                 "--collision", "hull", "--no-sim", "--usda"])
    assert code == 0
    usd = out / "block.usda"
    asset = read_usd(usd)
    assert np.allclose(sorted(asset.render_mesh.extents), [0.05, 0.1, 0.2])
    assert abs(asset.mass - 700.0 * 0.001) < 1e-6
    assert main(["validate", str(usd), "--no-sim", "--json", str(tmp_path / "v.json")]) == 0
    assert json.loads((tmp_path / "v.json").read_text())["status"] == "pass"
    assert main(["materials"]) == 0
    assert "ceramic" in capsys.readouterr().out


def test_cli_build_with_gray_card(tmp_path):
    import trimesh

    from scan2usd.color import linear_to_srgb

    cast = np.diag([1.25, 1.0, 0.7])
    gray = np.round(linear_to_srgb(np.array([0.18, 0.18, 0.18]) @ cast.T) * 255).astype(np.uint8)
    photo = np.full((60, 80, 3), 30, np.uint8)
    photo[20:40, 30:50] = gray
    Image.fromarray(photo).save(tmp_path / "gray.png")

    src = tmp_path / "ball.ply"
    ball = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    captured = np.round(linear_to_srgb(np.array([0.3, 0.3, 0.3]) @ cast.T) * 255).astype(np.uint8)
    ball.visual = trimesh.visual.ColorVisuals(ball, vertex_colors=np.tile([*captured, 255], (len(ball.vertices), 1)))
    ball.export(str(src))

    out = tmp_path / "out"
    code = main(["build", str(src), "-o", str(out), "--collision", "hull", "--no-sim", "--no-gltf",
                 "--gray-card-image", str(tmp_path / "gray.png"), "--gray-card-box", "30,20,50,40"])
    assert code == 0
    cal = json.loads((out / "color_calibration.json").read_text())
    assert cal["method"] == "neutral"
    assert np.allclose(np.diag(cal["ccm"]), 1 / np.diag(cast), atol=0.03)
    asset = read_usd(out / "ball.usd")
    rgb = asset.render_mesh.visual.vertex_colors[:, :3].astype(float).mean(0)
    assert np.ptp(rgb) <= 3  # the cast is gone: neutral gray again
