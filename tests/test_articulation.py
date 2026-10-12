import mujoco
import numpy as np
import pytest

from scan2usd.articulation import (
    build_parts,
    make_keepsake_box,
    mujoco_xml,
    read_articulation,
    split_mesh,
    write_articulated_usd,
)
from scan2usd.usd_io import PhysicsMaterial, SurfaceMaterial


@pytest.fixture(scope="module")
def box_usd(tmp_path_factory):
    mesh, cfg = make_keepsake_box()
    parts = build_parts(mesh, cfg["plane_origin"], cfg["plane_normal"], material="wood", collision="hull")
    path = tmp_path_factory.mktemp("art") / "box.usda"
    write_articulated_usd(path, "keepsake_box", parts, cfg["hinge"], SurfaceMaterial(base_color=(0.6, 0.4, 0.25)),
                          PhysicsMaterial())
    return path, mesh, parts


def test_split_preserves_volume():
    mesh, cfg = make_keepsake_box()
    below, above = split_mesh(mesh, cfg["plane_origin"], cfg["plane_normal"])
    assert below.is_watertight and above.is_watertight
    assert below.volume + above.volume == pytest.approx(mesh.volume, rel=1e-6)
    assert above.bounds[0][2] == pytest.approx(cfg["plane_origin"][2], abs=1e-9)


def test_articulated_usd_structure(box_usd):
    path, _, parts = box_usd
    art = read_articulation(path)
    assert art["articulation_root"]
    assert len(art["bodies"]) == 2
    j = art["joints"][0]
    assert j["body0"].endswith("/base") and j["body1"].endswith("/lid")
    assert np.allclose(j["axis"], [-1, 0, 0], atol=1e-6)
    assert (j["lower_deg"], j["upper_deg"]) == (0.0, 105.0)
    total = sum(p.mass.mass for p in parts)
    assert total == pytest.approx(700.0 * 0.14 * 0.10 * 0.07, rel=1e-3)


def test_hinge_respects_limits_in_mujoco(box_usd):
    path, _, _ = box_usd
    body, extra = mujoco_xml(path, fixed_base=True)
    xml = (f'<mujoco><option timestep="0.001"/><asset>{extra["assets"]}</asset><worldbody>{body}</worldbody>'
           f'{extra["contact"]}<actuator><motor joint="hinge" gear="1" ctrlrange="-2 2"/></actuator></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml, extra["files"])
    data = mujoco.MjData(model)
    qadr = model.joint("hinge").qposadr[0]
    peak = 0.0
    data.ctrl[0] = 0.25  # N*m, ~3x what gravity needs: opens the lid
    for _ in range(1500):
        mujoco.mj_step(model, data)
        peak = max(peak, data.qpos[qadr])
    assert np.degrees(data.qpos[qadr]) == pytest.approx(105.0, abs=4.0)
    assert np.degrees(peak) < 109.0
    data.ctrl[0] = 0.0  # past 90 degrees the lid's weight keeps it open against the limit
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert np.degrees(data.qpos[qadr]) > 100.0
    data.ctrl[0] = -0.25  # push it closed: stops at the lower limit
    low = np.inf
    for _ in range(3000):
        mujoco.mj_step(model, data)
        low = min(low, data.qpos[qadr])
    assert np.degrees(data.qpos[qadr]) < 2.0
    assert np.degrees(low) > -4.0
    assert np.isfinite(data.qpos).all()


def test_articulate_cli(tmp_path):
    from scan2usd.cli import main

    mesh, _ = make_keepsake_box()
    src = tmp_path / "box.ply"
    mesh.export(str(src))
    code = main(["articulate", str(src), "-o", str(tmp_path / "out"), "--name", "box",
                 "--split", "0,0,0.052,0,0,1", "--hinge", "0,0.05,0.052,-1,0,0", "--limits", "0,90",
                 "--collision", "hull"])
    assert code == 0
    art = read_articulation(tmp_path / "out" / "box.usda")
    assert art["joints"][0]["upper_deg"] == 90.0
