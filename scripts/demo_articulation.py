"""Robotics demo: an articulated Scan2USD asset (box + hinged lid) in MuJoCo.

    python scripts/demo_articulation.py [--out docs/figures]

Builds a keepsake box from one closed 'scan' (split plane + hinge axis), writes it as an
articulated OpenUSD asset (UsdPhysics.RevoluteJoint with limits under an articulation
root), then opens the lid, drops a scanned toy inside and closes the lid again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from scan2usd.articulation import build_parts, make_keepsake_box, mujoco_xml, write_articulated_usd
from scan2usd.mjcf import asset_body
from scan2usd.preview import _font
from scan2usd.usd_io import PhysicsMaterial, SurfaceMaterial, read_usd

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "output" / "gso_assets" / "great_dinos_triceratops_toy" / "great_dinos_triceratops_toy.usda"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = p.parse_args()

    mesh, cfg = make_keepsake_box(size=(0.18, 0.13, 0.10), lid_height=0.02, wall=0.008, hollow=True)
    parts = build_parts(mesh, cfg["plane_origin"], cfg["plane_normal"], material="wood")
    usd = ROOT / "output" / "articulated" / "keepsake_box.usda"
    write_articulated_usd(usd, "keepsake_box", parts, cfg["hinge"],
                          SurfaceMaterial(base_color=(0.55, 0.36, 0.2), roughness=0.6), PhysicsMaterial(0.5, 0.4, 0.2))
    box_xml, extra = mujoco_xml(usd, fixed_base=True)
    box_xml = box_xml.replace('rgba="0.62 0.42 0.25 1"', 'rgba="0.55 0.36 0.2 1"')

    toy_asset = read_usd(TOY)
    toy = asset_body(toy_asset, "toy", pos=(0.0, -0.005, 0.24), quat=(0.7071, 0, 0, 0.7071))
    xml = f"""<mujoco model="articulated_box">
  <option timestep="0.001" integrator="implicitfast" cone="elliptic"/>
  <visual><global offwidth="960" offheight="720"/><quality shadowsize="4096"/></visual>
  <asset>{extra['assets']}{toy.assets_xml}
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.80 0.84 0.90" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".82 .82 .80" rgb2=".74 .74 .72" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.6 1.2" dir="-0.2 0.5 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light pos="-0.5 0.4 1" dir="0.5 -0.4 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom type="plane" size="1 1 0.05" material="grid"/>
    {box_xml}
    <body name="holder" mocap="true" pos="0 -0.005 0.24"/>
    {toy.body_xml}
  </worldbody>
  {extra['contact']}
  <equality><weld name="hold" body1="toy" body2="holder" solref="0.02 1"/></equality>
  <actuator><motor name="lid_motor" joint="hinge" gear="1" ctrlrange="-1 1"/></actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml, {**extra["files"], **toy.files})
    data = mujoco.MjData(model)
    qadr = model.joint("hinge").qposadr[0]
    toy_id = model.body("toy").id
    renderer = mujoco.Renderer(model, 540, 720)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0, 0, 0.07)
    cam.distance, cam.elevation, cam.azimuth = 0.6, -38, 62

    # (end time, lid torque N*m, holding the toy?)
    schedule = [(0.3, 0.0, True), (1.2, 0.35, True), (2.6, 0.0, False), (3.8, -0.35, False), (4.6, -0.35, False)]
    frames, trace, t_prev = [], [], 0.0
    for t_end, torque, hold in schedule:
        data.eq_active[0] = hold
        while data.time < t_end:
            data.ctrl[0] = torque
            mujoco.mj_step(model, data)
            if len(frames) <= data.time * 30:
                renderer.update_scene(data, camera=cam)
                img = Image.fromarray(renderer.render().copy())
                d = ImageDraw.Draw(img)
                d.rectangle([0, 0, img.width, 36], fill=(255, 255, 255))
                angle = float(np.degrees(data.qpos[qadr]))
                d.text((12, 7), f"Articulated USD: revolute joint, limit 0-105°   lid {angle:5.1f}°",
                       fill=(30, 41, 59), font=_font(18))
                frames.append(img)
                trace.append(round(angle, 2))
    toy_pos = data.xpos[toy_id]
    inside = bool(abs(toy_pos[0]) < 0.082 and abs(toy_pos[1]) < 0.057 and toy_pos[2] < 0.09)
    result = {"usd": str(usd.relative_to(ROOT)), "max_lid_deg": max(trace), "final_lid_deg": trace[-1],
              "toy_inside_box": inside, "toy_final_pos_m": [float(v) for v in toy_pos],
              "base_mass_g": parts[0].mass.mass * 1000, "lid_mass_g": parts[1].mass.mass * 1000,
              "finite": bool(np.isfinite(data.qpos).all())}
    print(json.dumps(result, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out / "articulation_demo.webp", save_all=True, append_images=frames[1:], duration=33,
                   loop=0, quality=70, method=4)
    frames[int(len(frames) * 0.45)].save(args.out / "articulation_demo.jpg", quality=88)
    (args.out / "articulation_demo.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
