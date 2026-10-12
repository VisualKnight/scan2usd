"""Dexterous grasping of scanned assets: a Shadow Hand (MuJoCo Menagerie) power-grasps each object.

    python scripts/demo_hand_grasp.py [--workers 16]

For every asset in the USD library (scan2usd library) and two of its collision variants
(convex decomposition vs. one convex hull), 90 power grasps are sampled: 54 from the side
(6 directions x 3 heights x 3 standoffs) and 36 from above (6 yaws x 2 palm gaps x 3 offsets).
Each trial: approach, close the hand (Menagerie's "close hand"
pose), lift 15 cm, shake +-2 cm at 3 Hz, hold. Success = still lifted and in the hand.
This is grasp evaluation by sampling in simulation; the success rate per asset measures how
graspable it is, and how much that depends on collision fidelity.

Robot: mujoco_menagerie/shadow_hand (Apache-2.0); fetch with
    git -C data/external/menagerie sparse-checkout add shadow_hand
Writes docs/figures/hand_grasp.{json,png,webp,jpg}.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
HAND = ROOT / "data" / "external" / "menagerie" / "shadow_hand"
LIB = ROOT / "output" / "library" / "scan2usd_library.usda"
VARIANTS = ["decomposed", "convex_hull"]
YAWS = [0, 60, 120, 180, 240, 300]
SIDE_HEIGHTS = [0.3, 0.5, 0.7]  # side grasp: height as a fraction of the object's height
SIDE_STANDOFF = [-0.01, 0.01, 0.03]  # palm gap from the object's surface (m)
TOP_GAP = [0.005, 0.02]  # top grasp: palm height above the object's top (m)
TOP_OFFSET = [0.0, 0.025, 0.05]  # object center behind the grasp site, toward the palm (m)
SKIP = {"keepsake_box"}  # articulated: no single rigid body to lift
SITE_IN_FOREARM = np.array([0.045, 0.0, 0.337])  # grasp_site at zero wrist angles
R_FP = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # forearm -> palm rotation
# Phases (s): approach, close, lift, shake, hold
T_DESCEND, T_CLOSE, T_LIFT, T_SHAKE, T_HOLD = 1.0, 1.0, 1.2, 1.5, 0.6
LIFT, SHAKE_AMP, SHAKE_HZ = 0.15, 0.02, 3.0

TITLES = {"ace_coffee_mug_kristen_16_oz_cup": "Printed mug", "cole_hardware_bowl_scirocco_yellowblue": "Bowl",
          "great_dinos_triceratops_toy": "Android figure", "schleich_lion_action_figure": "Lion",
          "womens_bluefish_2eye_boat_shoe_in_tan": "Boat shoe", "ceramic_mug": "Ceramic mug",
          "speckled_mug": "Speckled mug", "buddha_head": "Buddha (4.6 kg)"}


def forearm_pose(site: np.ndarray, yaw_deg: float, mode: str = "side"):
    """Forearm (mocap) pose for a grasp at `site` (the palm's grasp site).

    side: palm facing the object along `yaw`, thumb up, fingers wrap horizontally (holding a can).
    top:  palm facing down, fingers pointing along `yaw`, curling down around the object.
    """
    from scipy.spatial.transform import Rotation

    a = np.radians(yaw_deg)
    toward = np.array([np.cos(a), np.sin(a), 0.0])
    if mode == "side":
        x = np.array([0.0, 0.0, 1.0])  # palm x: little finger -> thumb side, up
        y = -toward  # the palm faces its -y
        z = np.cross(x, y)  # finger direction (horizontal)
    else:
        y = np.array([0.0, 0.0, 1.0])  # palm faces down
        z = toward  # fingers point along yaw
        x = np.cross(y, z)
    r_palm = np.c_[x, y, z]
    r_f = r_palm @ R_FP.T
    pos = site - r_f @ SITE_IN_FOREARM
    q = Rotation.from_matrix(r_f).as_quat()  # x y z w
    return pos, np.array([q[3], q[0], q[1], q[2]]), toward


def build(work: Path, body, finger_gain: float | None = None) -> "mujoco.MjModel":  # noqa: F821
    """Hand + object scene. finger_gain scales the finger/thumb position-servo stiffness: Menagerie's
    servos (kp 0.4-1.5 N·m/rad) reach their target around an object and then barely squeeze. The
    torque limits (±1-3 N·m, i.e. up to ~20 N at a fingertip) are unchanged."""
    import mujoco

    if finger_gain is None:
        finger_gain = float(os.environ.get("SCAN2USD_FINGER_GAIN", "5"))
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(HAND, work)
    xml = (work / "right_hand.xml").read_text()
    # The forearm is a free body welded to a mocap target. Moving a mocap body directly teleports it
    # (zero velocity to the solver), so friction never drags the object along; a weld gives the hand
    # real velocity.
    xml = xml.replace('<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">',
                      '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1"><freejoint name="forearm_free"/>')
    (work / "right_hand.xml").write_text(xml)
    for name, data in body.files.items():
        (work / "assets" / name).write_bytes(data)
    scene = f"""<mujoco model="hand_grasp">
  <include file="right_hand.xml"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="960" offheight="720"/><quality shadowsize="4096"/></visual>
  <asset>{body.assets_xml}
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.80 0.84 0.90" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".82 .82 .80" rgb2=".74 .74 .72" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.6 1.2" dir="-0.2 0.5 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light pos="-0.5 0.5 1.0" dir="0.4 -0.4 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="grid" solref="0.005 1"/>
    <body name="target" mocap="true" quat="0 1 0 1"/>
    {body.body_xml}
  </worldbody>
  <equality><weld body1="target" body2="rh_forearm" solref="0.005 1"/></equality>
</mujoco>"""
    (work / "grasp_scene.xml").write_text(scene)
    model = mujoco.MjModel.from_xml_path(str(work / "grasp_scene.xml"))
    for a in range(model.nu):
        if "WRJ" not in model.actuator(a).name:  # position servo: gain kp, bias -kp * q
            model.actuator_gainprm[a, 0] *= finger_gain
            model.actuator_biasprm[a, 1] *= finger_gain
    return model


def pose_ctrl(model, key: str) -> np.ndarray:
    """Actuator targets for a Menagerie keyframe pose (wrist kept straight)."""
    import mujoco

    kf = {"open hand": np.zeros(24),
          "close hand": np.array([0, 0, 0, 1.571, 1.571, 1.571, 0, 1.571, 1.571, 1.571, 0, 1.571, 1.571, 1.571,
                                  0, 0, 1.571, 1.571, 1.571, 0.17, 1.2, 0, 0.61, 0.52])}[key]
    jnames = ["rh_WRJ2", "rh_WRJ1", "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1", "rh_MFJ4", "rh_MFJ3", "rh_MFJ2",
              "rh_MFJ1", "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1", "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2",
              "rh_LFJ1", "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1"]
    q = dict(zip(jnames, kf))
    ctrl = np.zeros(model.nu)
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            ctrl[a] = q[model.joint(model.actuator_trnid[a][0]).name]
        else:  # tendon: sum of its joints
            t = model.actuator_trnid[a][0]
            adr, num = model.tendon_adr[t], model.tendon_num[t]
            ctrl[a] = sum(model.wrap_prm[i] * q[model.joint(model.wrap_objid[i]).name] for i in range(adr, adr + num))
    return ctrl


_cache: dict = {}


def asset(prim: str, variant: str):
    from scan2usd.usd_io import read_usd

    key = (prim, variant)
    if key not in _cache:
        _cache[key] = read_usd(LIB, prim, {"collision": variant})
    return _cache[key]


def trial(p: dict, render: bool = False):
    import mujoco

    from scan2usd.mjcf import asset_body

    t_start = time.time()
    a = asset(p["prim"], p["variant"])
    body = asset_body(a, "obj", pos=(0, 0, 0.001), collision="hulls")
    body = replace(body, body_xml=body.body_xml)
    model = build(ROOT / "data" / "work" / "hand" / p["id"], body)
    data = mujoco.MjData(model)
    obj = model.body("obj").id
    mujoco.mj_forward(model, data)
    lo, hi = a.render_mesh.bounds
    center = (lo + hi) / 2
    a_rad = np.radians(p["yaw"])
    toward = np.array([np.cos(a_rad), np.sin(a_rad), 0.0])
    if p["mode"] == "side":
        # Palm on the object's surface (+ standoff b): half the footprint along the approach
        # direction, minus the 3.5 cm between the palm and its grasp site.
        v = a.render_mesh.vertices
        radius = float(np.max(-(v[:, :2] - center[:2]) @ toward[:2]))
        height = max(0.045, lo[2] + p["a"] * (hi[2] - lo[2]))
        site = np.r_[center[:2] - toward[:2] * max(0.0, radius - 0.035 + p["b"]), height]
        pos, quat, _ = forearm_pose(site, p["yaw"], "side")
        above = pos - toward * 0.12  # approach horizontally
    else:
        # Palm a gap `a` above the top; object center `b` behind the grasp site (toward the palm).
        site = np.r_[center[:2] + toward[:2] * p["b"], hi[2] + p["a"] - 0.035]
        pos, quat, _ = forearm_pose(site, p["yaw"], "top")
        above = pos + np.array([0.0, 0.0, 0.12])  # approach from above
    data.mocap_pos[0], data.mocap_quat[0] = above, quat
    adr = model.joint("forearm_free").qposadr[0]
    data.qpos[adr:adr + 3], data.qpos[adr + 3:adr + 7] = above, quat  # start the hand at its target
    open_c, close_c = pose_ctrl(model, "open hand"), pose_ctrl(model, "close hand")
    data.ctrl[:] = open_c
    for _ in range(100):  # settle the object
        mujoco.mj_step(model, data)
    z0 = float(data.xipos[obj][2])
    dt = model.opt.timestep
    frames, cam, renderer = [], None, None
    if render:
        renderer = mujoco.Renderer(model, 480, 640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (site[0], site[1], 0.1)
        cam.distance, cam.elevation, cam.azimuth = 0.75, -18, 130

    def run(seconds, fn):
        n = int(seconds / dt)
        for i in range(n):
            fn((i + 1) / n)
            mujoco.mj_step(model, data)
            if render and len(frames) <= data.time * 30:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())

    smooth = lambda s: 0.5 - 0.5 * np.cos(np.pi * s)  # noqa: E731
    run(T_DESCEND, lambda s: data.mocap_pos.__setitem__(0, above + (pos - above) * smooth(s)))
    run(T_CLOSE, lambda s: data.ctrl.__setitem__(slice(None), open_c + (close_c - open_c) * smooth(s)))
    lifted = pos + np.array([0, 0, LIFT])
    run(T_LIFT, lambda s: data.mocap_pos.__setitem__(0, pos + (lifted - pos) * smooth(s)))
    z_lift = float(data.xipos[obj][2])

    def shake(s):
        t = s * T_SHAKE
        data.mocap_pos[0] = lifted + SHAKE_AMP * np.array([np.sin(2 * np.pi * SHAKE_HZ * t),
                                                          np.sin(2 * np.pi * SHAKE_HZ * t + np.pi / 2), 0.0])
    run(T_SHAKE, shake)
    run(T_HOLD, lambda s: data.mocap_pos.__setitem__(0, lifted))
    z_end = float(data.xipos[obj][2])
    palm = data.xpos[model.body("rh_palm").id]
    dist = float(np.linalg.norm(data.xipos[obj] - palm))
    res = {**p, "lifted_m": round(z_lift - z0, 4), "held_m": round(z_end - z0, 4), "palm_dist_m": round(dist, 3),
           "lift_ok": bool(z_lift - z0 > 0.10), "success": bool(z_end - z0 > 0.10 and dist < 0.2),
           "mass_kg": round(float(a.mass), 3), "colliders": len(a.hulls), "seconds": round(time.time() - t_start, 1)}
    if render:
        renderer.close()
    shutil.rmtree(ROOT / "data" / "work" / "hand" / p["id"], ignore_errors=True)
    return (res, frames) if render else res


def grid(prims):
    out = []
    for prim in prims:
        for v in VARIANTS:
            for yaw in YAWS:
                for a in SIDE_HEIGHTS:
                    for b in SIDE_STANDOFF:
                        out.append({"prim": prim, "variant": v, "mode": "side", "yaw": yaw, "a": a, "b": b})
                for a in TOP_GAP:
                    for b in TOP_OFFSET:
                        out.append({"prim": prim, "variant": v, "mode": "top", "yaw": yaw, "a": a, "b": b})
    for k, p in enumerate(out):
        p["id"] = f"h{k:04d}"
    return out


def figure(results, out: Path) -> None:
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font

    prims = list(dict.fromkeys(r["prim"] for r in results))
    rate = {(r["prim"], v): np.mean([x["success"] for x in results if x["prim"] == r["prim"] and x["variant"] == v])
            for r in results for v in VARIANTS}
    w, row, left, bar = 980, 40, 220, 300
    img = Image.new("RGB", (w, 110 + row * len(prims)), "white")
    d = ImageDraw.Draw(img)
    d.text((16, 12), "Shadow Hand power grasps per scanned asset (90 sampled grasps each)", fill=(30, 41, 59),
           font=_font(21))
    d.text((16, 44), "success = lifted 15 cm, survives a ±2 cm / 3 Hz shake, still in the hand", fill=(100, 116, 139),
           font=_font(15))
    colors = {"decomposed": (37, 99, 235), "convex_hull": (234, 88, 12)}
    for k, v in enumerate(VARIANTS):
        x = left + k * (bar + 60)
        d.rectangle([x, 74, x + 14, 88], fill=colors[v])
        d.text((x + 20, 71), {"decomposed": "convex decomposition", "convex_hull": "one convex hull"}[v],
               fill=(51, 65, 85), font=_font(15))
    for i, prim in enumerate(prims):
        y = 100 + i * row
        name = prim.split("/")[-1]
        d.text((16, y + 8), TITLES.get(name, name), fill=(51, 65, 85), font=_font(16))
        for k, v in enumerate(VARIANTS):
            x = left + k * (bar + 60)
            r = rate[(prim, v)]
            d.rectangle([x, y + 8, x + bar, y + 28], fill=(241, 245, 249))
            d.rectangle([x, y + 8, x + int(bar * r), y + 28], fill=colors[v])
            d.text((x + bar + 8, y + 8), f"{r * 100:.0f}%", fill=(51, 65, 85), font=_font(15))
    img.save(out)


def video(picks, out_dir: Path) -> None:
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font

    clips = []
    for p in picks:
        res, frames = trial(p, render=True)
        name = p["prim"].split("/")[-1]
        clips.append((frames, f"{TITLES.get(name, name)} · {'decomposition' if p['variant'] == 'decomposed' else 'one hull'}",
                      res["success"]))
    n = min(len(c[0]) for c in clips)
    cols = 2 if len(clips) == 4 else len(clips)
    sheets = []
    for i in range(0, n, 2):
        tiles = []
        for frames, title, ok in clips:
            t = Image.fromarray(frames[i])
            d = ImageDraw.Draw(t)
            d.rectangle([0, 0, t.width, 32], fill=(255, 255, 255))
            d.text((10, 6), title, fill=(30, 41, 59), font=_font(17))
            if i > n * 0.85:
                d.text((10, t.height - 34), "held" if ok else "dropped", fill=(21, 128, 61) if ok else (185, 28, 28),
                       font=_font(22))
            tiles.append(t)
        rows = (len(tiles) + cols - 1) // cols
        s = Image.new("RGB", (cols * tiles[0].width, rows * tiles[0].height), "white")
        for k, t in enumerate(tiles):
            s.paste(t, ((k % cols) * t.width, (k // cols) * t.height))
        sheets.append(s.resize((s.width * 3 // 4, s.height * 3 // 4)))
    sheets[0].save(out_dir / "hand_grasp.webp", save_all=True, append_images=sheets[1:], duration=66, loop=0,
                   quality=60, method=4)
    sheets[int(len(sheets) * 0.62)].save(out_dir / "hand_grasp.jpg", quality=86)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--video-only", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    out = ROOT / "docs" / "figures"
    if not HAND.exists():
        raise SystemExit("fetch mujoco_menagerie/shadow_hand first (see the docstring)")
    from scan2usd.library import library_entries

    prims = [e["path"] for e in library_entries(LIB) if e["name"] not in SKIP]
    if not args.video_only:
        params = grid(prims)
        t0 = time.time()
        with Pool(args.workers) as pool:
            results = []
            for r in pool.imap_unordered(trial, params):
                results.append(r)
                if len(results) % 24 == 0:
                    print(f"[{len(results)}/{len(params)}] {time.time() - t0:.0f} s", flush=True)
        results.sort(key=lambda r: r["id"])
        summary = {"trials": results, "minutes": round((time.time() - t0) / 60, 1),
                   "rates": {f"{p}|{v}": float(np.mean([r["success"] for r in results
                                                         if r["prim"] == p and r["variant"] == v]))
                             for p in prims for v in VARIANTS}}
        (out / "hand_grasp.json").write_text(json.dumps(summary, indent=2))
        figure(results, out / "hand_grasp.png")
        for k, v in summary["rates"].items():
            print(f"  {k}: {v * 100:.0f}%")
    results = json.loads((out / "hand_grasp.json").read_text())["trials"]
    picks = []
    for name in ("speckled_mug", "schleich_lion_action_figure", "great_dinos_triceratops_toy",
                 "cole_hardware_bowl_scirocco_yellowblue"):
        ok = [r for r in results if r["prim"].endswith(name) and r["variant"] == "decomposed" and r["success"]]
        if ok:
            picks.append({k: ok[0][k] for k in ("prim", "variant", "mode", "yaw", "a", "b", "id")})
    if picks and not args.no_video:
        video(picks[:4], out)


if __name__ == "__main__":
    main()
