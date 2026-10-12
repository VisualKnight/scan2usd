import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from scan2usd.simulate import drop_test
from scan2usd.validate import FAIL, PASS, validate_usd


def _status(rep, name):
    return next(c.status for c in rep.checks if c.name == name)


def test_good_box_passes(box_asset):
    path, _, _ = box_asset
    rep = validate_usd(path)
    assert rep.status == PASS, rep.to_markdown()
    assert len(rep.drop_tests) == 3
    assert all(d["stable"] and d["settled"] for d in rep.drop_tests)


def test_invalid_inertia_fails(box_asset):
    path, _, _ = box_asset
    stage = Usd.Stage.Open(str(path))
    api = UsdPhysics.MassAPI(stage.GetDefaultPrim())
    api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(1e-5, 1e-5, 1.0))  # violates triangle inequality
    api.GetMassAttr().Set(0.0)
    stage.Save()
    rep = validate_usd(path, run_sim=False)
    assert rep.status == FAIL
    assert _status(rep, "inertia triangle inequality") == FAIL
    assert _status(rep, "mass > 0") == FAIL


def test_wrong_units_and_up_axis_fail(box_asset):
    path, _, _ = box_asset
    stage = Usd.Stage.Open(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.Save()
    rep = validate_usd(path, run_sim=False)
    assert _status(rep, "up axis is Z") == FAIL
    assert _status(rep, "meters per unit = 1") == FAIL


def test_unreadable_file_reports_failure(tmp_path):
    bad = tmp_path / "missing.usda"
    rep = validate_usd(bad)
    assert rep.status == FAIL


def test_drop_test_box_settles_flat():
    box = trimesh.creation.box((0.1, 0.1, 0.05))
    box.apply_translation([0, 0, 0.025])
    box.density = 500.0
    res = drop_test([box], float(box.mass), box.center_mass, box.moment_inertia, pose="upright")
    assert res.stable and res.settled
    assert res.final_height_m < 0.002  # body origin returns to the floor
    assert res.resting_penetration_m < 0.001
