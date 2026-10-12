"""Simulation stability checks in MuJoCo: drop the asset onto a plane and watch it settle."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

DROP_POSES = {
    "upright": (0.0, 0.0, 0.0),
    "tilted_25deg": (25.0, 10.0, 0.0),
    "on_side": (90.0, 0.0, 30.0),
}


@dataclass
class DropResult:
    pose: str
    stable: bool
    settled: bool
    settle_time_s: float | None
    max_penetration_m: float
    resting_penetration_m: float
    energy_gain_j: float
    final_speed_mps: float
    final_angular_speed_rps: float
    final_height_m: float
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _mjcf(hull_count: int, mass: float, com, inertia: np.ndarray, friction: float, quat_wxyz, height: float) -> str:
    i = inertia
    full = f"{i[0,0]:.9g} {i[1,1]:.9g} {i[2,2]:.9g} {i[0,1]:.9g} {i[0,2]:.9g} {i[1,2]:.9g}"
    assets = "\n".join(f'    <mesh name="hull_{k}" file="hull_{k}.stl"/>' for k in range(hull_count))
    geoms = "\n".join(
        f'      <geom type="mesh" mesh="hull_{k}" friction="{friction:.4g} 0.005 0.0001"/>' for k in range(hull_count)
    )
    q = " ".join(f"{v:.9g}" for v in quat_wxyz)
    c = " ".join(f"{v:.9g}" for v in com)
    return f"""<mujoco model="drop_test">
  <compiler angle="radian"/>
  <option timestep="0.001" integrator="implicitfast">
    <flag energy="enable"/>
  </option>
  <default>
    <geom solref="0.005 1"/>
  </default>
  <asset>
{assets}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" friction="{friction:.4g} 0.005 0.0001"/>
    <body name="asset" pos="0 0 {height:.6g}" quat="{q}">
      <freejoint/>
      <inertial pos="{c}" mass="{mass:.9g}" fullinertia="{full}"/>
{geoms}
    </body>
  </worldbody>
</mujoco>"""


def drop_test(
    hulls: list[trimesh.Trimesh],
    mass: float,
    center_of_mass: np.ndarray,
    inertia: np.ndarray,
    friction: float = 0.5,
    pose: str = "upright",
    drop_height: float = 0.05,
    duration: float = 3.0,
) -> DropResult:
    """Drop the rigid body from `drop_height` above the floor and evaluate the outcome.

    stable:  no NaNs, no energy gain beyond tolerance, no deep penetration, no tunneling.
    settled: came to rest before `duration` (round objects may legitimately keep rolling).
    """
    import mujoco

    rot = Rotation.from_euler("xyz", DROP_POSES[pose], degrees=True)
    combined = trimesh.util.concatenate(hulls)
    rotated = combined.vertices @ rot.as_matrix().T
    lowest = rotated[:, 2].min()
    size = float(combined.extents.max())
    height = drop_height - lowest
    x, y, z, w = rot.as_quat()

    assets = {f"hull_{k}.stl": h.export(file_type="stl") for k, h in enumerate(hulls)}
    xml = _mjcf(len(hulls), mass, center_of_mass, np.asarray(inertia), friction, (w, x, y, z), height)
    model = mujoco.MjModel.from_xml_string(xml, assets)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    e0 = float(data.energy[0] + data.energy[1])

    pen_tol = max(0.001, 0.01 * size)
    impact_tol = max(0.005, 0.05 * size)
    steps = int(duration / model.opt.timestep)
    rest_window = int(0.25 / model.opt.timestep)
    max_pen = 0.0
    rest_pen = 0.0
    max_gain = 0.0
    rest_steps_needed = rest_window
    rest_count = 0
    settle_time = None
    for step in range(steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            return DropResult(pose, False, False, None, float("nan"), float("nan"), float("nan"), float("nan"),
                              float("nan"), float("nan"), "simulation diverged (NaN)")
        pen = max((-float(c.dist) for c in data.contact[: data.ncon]), default=0.0)
        max_pen = max(max_pen, pen)
        if step >= steps - rest_window:
            rest_pen = max(rest_pen, pen)
        max_gain = max(max_gain, float(data.energy[0] + data.energy[1]) - e0)
        v = float(np.linalg.norm(data.qvel[:3]))
        wv = float(np.linalg.norm(data.qvel[3:6]))
        if v < 0.005 and wv < 0.05:
            rest_count += 1
            if rest_count >= rest_steps_needed and settle_time is None:
                settle_time = (step + 1 - rest_steps_needed) * model.opt.timestep
        else:
            rest_count = 0

    final_v = float(np.linalg.norm(data.qvel[:3]))
    final_w = float(np.linalg.norm(data.qvel[3:6]))
    final_h = float(data.qpos[2])
    # Allow energy "gain" of 1% of the drop's potential energy for integrator noise.
    gain_tol = 0.01 * mass * 9.81 * max(drop_height, 0.01) + 1e-6
    msgs = []
    if max_gain > gain_tol:
        msgs.append(f"energy increased by {max_gain:.3g} J")
    if max_pen > impact_tol:
        msgs.append(f"impact penetration {max_pen * 1000:.1f} mm exceeds {impact_tol * 1000:.1f} mm")
    if rest_pen > pen_tol:
        msgs.append(f"resting penetration {rest_pen * 1000:.2f} mm exceeds {pen_tol * 1000:.1f} mm")
    if final_h < -size:
        msgs.append("fell through the floor")
    if abs(data.qpos[0]) > 1.0 or abs(data.qpos[1]) > 1.0:
        msgs.append("travelled more than 1 m")
    stable = not msgs
    settled = settle_time is not None and final_v < 0.005 and final_w < 0.05
    if not settled:
        msgs.append("did not come to rest")
    return DropResult(
        pose=pose,
        stable=stable,
        settled=settled,
        settle_time_s=settle_time,
        max_penetration_m=max_pen,
        resting_penetration_m=rest_pen,
        energy_gain_j=max_gain,
        final_speed_mps=final_v,
        final_angular_speed_rps=final_w,
        final_height_m=final_h,
        message="; ".join(msgs),
    )
