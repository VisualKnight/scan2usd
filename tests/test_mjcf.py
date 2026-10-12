import mujoco
import numpy as np
import pytest

from scan2usd.mjcf import asset_body


@pytest.mark.parametrize("collision", ["hulls", "single"])
def test_asset_body_loads_and_falls(box_asset, collision):
    path, _, mp = box_asset
    body = asset_body(str(path), "obj", pos=(0, 0, 0.2), collision=collision)
    xml = (f'<mujoco><asset>{body.assets_xml}</asset><worldbody>'
           f'<geom type="plane" size="1 1 .1"/>{body.body_xml}</worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml, body.files)
    data = mujoco.MjData(model)
    bid = model.body("obj").id
    assert model.body_mass[bid] == pytest.approx(mp.mass, rel=1e-5)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert data.xpos[bid][2] == pytest.approx(0.0, abs=2e-3)  # resting on the floor


def test_asset_body_rejects_unknown_collision(box_asset):
    path, _, _ = box_asset
    with pytest.raises(ValueError):
        asset_body(str(path), "obj", collision="voxels")
