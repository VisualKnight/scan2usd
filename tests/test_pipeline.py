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
