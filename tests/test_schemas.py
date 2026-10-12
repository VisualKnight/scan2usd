import subprocess
import sys

import numpy as np
import pytest
from pxr import Usd

from scan2usd.schemas import (
    GRASP_API,
    UNCERTAINTY_API,
    Grasp,
    Uncertainty,
    read_grasps,
    read_uncertainty,
    register,
    write_grasps,
    write_uncertainty,
)


def test_schema_registers_with_fallbacks():
    assert register()
    d = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(GRASP_API)
    assert "scan2usd:grasp:successRate" in d.GetPropertyNames()
    assert Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(UNCERTAINTY_API) is not None
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/g", "Xform")
    prim.ApplyAPI(GRASP_API)
    assert prim.HasAPI(GRASP_API)
    assert prim.GetAttribute("scan2usd:grasp:gripper").Get() == "panda"  # schema fallback


def _rot_z(deg):
    a = np.radians(deg)
    m = np.eye(4)
    m[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return m


def test_grasps_round_trip_sorted_and_scoped(box_asset):
    path, _, _ = box_asset
    p1 = _rot_z(30)
    p1[:3, 3] = [0.01, 0.02, 0.03]
    grasps = [Grasp(pose=p1, gripper="panda", type="rim", width=0.044, success_rate=0.5, trials=4,
                    evaluated_with="mujoco:decomposed", by_variant={"decomposed": 0.5, "convex_hull": 0.0}),
              Grasp(pose=_rot_z(-90), gripper="panda", success_rate=0.9, trials=10, expected_success=0.8)]
    assert write_grasps(path, grasps) == 2
    write_grasps(path, [Grasp(pose=np.eye(4), gripper="shadow_hand", type="power", success_rate=1.0)],
                 replace=True, gripper="shadow_hand")
    panda = read_grasps(path, gripper="panda")
    assert [g.success_rate for g in panda] == pytest.approx([0.9, 0.5])  # ranked by expected/success
    rim = panda[1]
    assert np.allclose(rim.pose, p1, atol=1e-6)
    assert rim.by_variant == {"decomposed": 0.5, "convex_hull": 0.0} and rim.width == pytest.approx(0.044)
    assert len(read_grasps(path)) == 3
    # Replacing panda grasps keeps the hand's.
    write_grasps(path, grasps[:1], gripper="panda")
    assert len(read_grasps(path, gripper="panda")) == 1 and len(read_grasps(path, gripper="shadow_hand")) == 1
    stage = Usd.Stage.Open(str(path))
    assert all(p.HasAPI(GRASP_API) for p in stage.GetPrimAtPath("/box/grasps").GetChildren())


def test_uncertainty_round_trip(box_asset):
    path, _, _ = box_asset
    u = Uncertainty(scale=(0.98, 1.02), scale_source="chart", mass=(0.2, 0.3), mass_source="density+sealed-volume",
                    com_min=(-0.01, -0.01, 0.01), com_max=(0.01, 0.01, 0.03), static_friction=(0.4, 0.7),
                    dynamic_friction=(0.3, 0.6), restitution=(0.05, 0.2), observed_fraction=0.8,
                    sealed_volume_fraction=0.3)
    write_uncertainty(path, u)
    back = read_uncertainty(path)
    assert back.scale_source == "chart" and back.mass_source == "density+sealed-volume"
    for f in ("scale", "mass", "com_min", "com_max", "static_friction", "restitution"):
        assert np.allclose(getattr(back, f), getattr(u, f), atol=1e-6)
    assert back.observed_fraction == pytest.approx(0.8)
    assert Usd.Stage.Open(str(path)).GetDefaultPrim().HasAPI(UNCERTAINTY_API)


def test_files_readable_without_plugin(box_asset, tmp_path):
    """A plain USD process (no scan2usd import) still reads the data."""
    path, _, _ = box_asset
    write_grasps(path, [Grasp(pose=np.eye(4), success_rate=0.75)])
    code = ("from pxr import Usd; s = Usd.Stage.Open(r'%s'); p = s.GetPrimAtPath('/box/grasps/g_000');"
            "print(p.GetAttribute('scan2usd:grasp:successRate').Get(), 'Scan2UsdGraspAPI' in str(p.GetMetadata('apiSchemas')))") % path
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out[:2] == ["0.75", "True"]
