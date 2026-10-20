"""Generated vs scanned: are image-to-3D assets accurate enough for physics (and for looks)?

    python scripts/study_generative.py --stage inputs                       # main venv
    <genai venv>\\python.exe scripts\\genai_generate.py --model all          # shapes (separate venv, see there)
    <genai venv>\\python.exe scripts\\genai_generate.py --model paint,paint2mv   # textures
    python scripts/study_generative.py --stage score,physics,look,figure    # figure also writes the summary

Image-to-3D models (Hunyuan3D 2.1, TripoSG: one photo; Hunyuan3D-2mv: four) are fed photos of the
same objects the photogrammetry benchmark uses, and their meshes enter the pipeline exactly where
a photogrammetry mesh does (scan2usd build, scaled to the same one measured height, same convex
decomposition and density). Then the same yardsticks:

  inputs   benchmark objects: views rendered in the capture's own Blender scene without the color
           chart (which a generator would take for part of the object): one three-quarter view at
           30 degrees, and front/left/back/right at 8 degrees for the multi-view model. Real controller
           and holder: hand-picked phone photos, plus four views chosen from the COLMAP camera poses.
           Writes output/genai/inputs.json.
  score    build each generated mesh into an asset; surface error vs the ground-truth mesh (as the
           benchmark), plus shape-only error (best free scale and rotation), and what physics reads:
           volume (mass at a known density), center of mass and inertia vs ground truth.
  physics  the real-world check with generated assets: generated controller dropped into the
           generated holder (MuJoCo), scored against the pose measured in the scan of the real pair;
           also for two more generation seeds.
  look     each textured asset re-photographed in the capture's Blender scene from 8 held-out
           cameras, compared with the photos (compare_splat.py's protocol).
  figure   docs/figures/genai_{geometry,drop,look}.jpg and the summary docs/figures/genai.json.

Model weights and the generation venv live outside the repository (D:\\local_projects\\genai by default).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "output" / "genai"
FIG = ROOT / "docs" / "figures"

# benchmark key -> capture folder, material (as run_benchmark.py)
from run_benchmark import OBJECTS  # noqa: E402

REAL = {  # real objects: hand-picked single photo (front three-quarter of the holder, the face of the controller)
    "cod_holder": "IMG_0020.jpg",
    "controller": "IMG_0060.jpg",
}


def _lookat_center(centers: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """The point closest (least squares) to all optical axes: what the photos are aimed at."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centers, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def _views(names, centers, forward, up, single_name=None, single_elev=30.0, ok=None):
    """Single view (nearest `single_elev` degrees, or the given photo) and four views around it,
    from the photos marked `ok` (all by default)."""
    ok = np.ones(len(names), bool) if ok is None else ok
    target = _lookat_center(centers, forward)
    up = up / np.linalg.norm(up)
    e1 = np.cross(up, [1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.cross(up, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)  # (e1, e2, up) right-handed: azimuth counterclockwise seen from above
    rel = centers - target
    elev = np.degrees(np.arctan2(rel @ up, np.linalg.norm(rel - np.outer(rel @ up, up), axis=1)))
    azim = np.degrees(np.arctan2(rel @ e2, rel @ e1))
    i0 = names.index(single_name) if single_name else int(np.argmin(np.where(ok, np.abs(elev - single_elev), 1e9)))
    low = elev < elev.min() + 10  # the lowest ring: closest to the side views the model expects
    usable = low & ok

    def nearest(az):
        d = np.abs((azim - az + 180) % 360 - 180)
        d[~usable] = 1e9
        k = int(np.argmin(d))
        return k, float(d[k])

    # The four views are anchored where all four have a usable photo closest to 90 degrees apart
    # (the real captures: at the single photo's side, as the model's "front").
    anchors = [azim[i0]] if single_name else np.arange(0.0, 360.0, 2.0)
    a0 = min(anchors, key=lambda a: max(nearest(a + off)[1] for off in (0, 90, 180, 270)))
    mv = {}
    for role, off in (("front", 0), ("left", 90), ("back", 180), ("right", 270)):
        k, err = nearest(a0 + off)
        mv[role] = {"photo": names[k], "azimuth_err_deg": err, "elevation_deg": float(elev[k])}
    return {"single": {"photo": names[i0], "elevation_deg": float(elev[i0])}, "multiview": mv}


BLENDER = os.environ.get("BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
SINGLE = (35.0, 30.0)  # azimuth, elevation (deg): a three-quarter phone shot
MV_ELEV = 8.0  # the multi-view model is trained on side views (0 deg): as low as a phone on the table


def _render_inputs(key, folder):
    """The input photos, rendered in the benchmark capture's own Blender scene (same table, light,
    phone camera and distance) but without the color chart, which a generator would take for part
    of the object. One three-quarter view, and front/left/back/right views for the multi-view model
    (left = camera on the object's left, azimuth +90 deg, as Hunyuan3D-2mv expects)."""
    import trimesh

    cap = ROOT / "data" / "synthetic" / folder
    rel = (cap / "gt_path.txt").read_text().strip() if (cap / "gt_path.txt").exists() else "gt_mesh.ply"
    mesh = cap / rel
    ext = trimesh.load(str(mesh), force="mesh").extents
    dist = max(0.2, 2.0 * float(max(ext)))  # make_synthetic_capture.py's camera distance
    target = [0.0, 0.0, float(ext[2]) * 0.45]  # the capture's aim point (object centered, on z = 0)
    out = OUT / "renders" / key
    views = {"single": SINGLE, **{r: (SINGLE[0] + off, MV_ELEV) for r, off in
                                  (("front", 0), ("left", 90), ("back", 180), ("right", 270))}}
    poses = []
    for name, (az, el) in views.items():
        a, e = np.radians(az), np.radians(el)
        loc = np.array(target) + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        poses.append({"name": f"{name}.jpg", "location": loc.tolist(), "target": target})
    if not all((out / "images" / p["name"]).exists() for p in poses):
        out.mkdir(parents=True, exist_ok=True)
        (out / "poses.json").write_text(json.dumps(poses, indent=1))
        cmd = [BLENDER, "-b", "--factory-startup", "--python", str(ROOT / "scripts" / "blender_render_capture.py"),
               "--", "--mesh", str(mesh), "--out", str(out), "--poses", str(out / "poses.json"), "--no-chart"]
        if rel != "gt_mesh.ply":
            cmd.append("--keep-materials")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"photo_dir": str((out / "images").relative_to(ROOT)),
            "single": {"photo": "single.jpg", "azimuth_deg": SINGLE[0], "elevation_deg": SINGLE[1]},
            "multiview": {r: {"photo": f"{r}.jpg", "azimuth_err_deg": 0.0, "elevation_deg": MV_ELEV}
                          for r in ("front", "left", "back", "right")}}


def stage_inputs(state):
    from scan2usd.chart import read_colmap_model

    items = {}
    for key, (folder, material, display) in OBJECTS.items():
        items[key] = {"kind": "benchmark", "display": display, "material": material, **_render_inputs(key, folder)}
        print(f"rendered inputs for {key}", flush=True)
    for key, photo in REAL.items():
        _, imgs = read_colmap_model(ROOT / "data" / "work" / key / "sparse" / "0")
        names = sorted(imgs)
        R = np.array([imgs[n][1] for n in names])
        t = np.array([imgs[n][2] for n in names])
        centers = -np.einsum("nji,nj->ni", R, t)  # C = -R^T t
        forward = R[:, 2, :]  # camera z axis in world: third row of R
        up = -R[:, 1, :].mean(0)  # phones held upright: the mean image-up direction is the world up
        v = _views(names, centers, forward, up, single_name=photo)
        items[key] = {"kind": "real", "display": key, "material": "plastic",
                      "photo_dir": str((ROOT / "data" / "raw" / key).relative_to(ROOT)), **v}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inputs.json").write_text(json.dumps(items, indent=1))
    for k, v in items.items():
        mv = ", ".join(f"{r} {m['photo']} ({m['azimuth_err_deg']:.0f}\u00b0 off, {m['elevation_deg']:.0f}\u00b0 up)"
                       for r, m in v["multiview"].items())
        print(f"{k}: single {v['single']['photo']} ({v['single']['elevation_deg']:.0f}\u00b0 up); {mv}")
    state["inputs"] = str((OUT / "inputs.json").relative_to(ROOT))



SOURCES = {  # source key -> label (photogrammetry: the benchmark's own asset)
    "photogrammetry": "Photogrammetry (78 photos)",
    "hunyuan2mv": "Hunyuan3D-2mv (4 photos)",
    "hunyuan21": "Hunyuan3D 2.1 (1 photo)",
    "triposg": "TripoSG (1 photo)",
}


def _report(usd: Path) -> dict:
    return json.loads(next(usd.parent.glob("*report*.json")).read_text())


def build_generated(model: str, key: str, it: dict, scale_to: float, rebuild: bool = False) -> Path:
    """A generated mesh through the same build as a photogrammetry mesh: same one measured height
    (scale_to, along z), same cleanup, decimation, convex decomposition and material density."""
    from scan2usd.pipeline import BuildOptions, build_asset

    out = OUT / "assets" / model / key
    usd = out / f"{key}.usda"
    if usd.exists() and not rebuild:
        return usd
    import shutil

    shutil.rmtree(out, ignore_errors=True)
    opts = BuildOptions(name=key, material=it["material"], up_axis="y", units="m", scale_to=scale_to,
                        scale_axis="z", usda=True)
    res = build_asset(OUT / "meshes" / model / f"{key}.ply", out, opts)
    return res.usd_path


def score_asset(usd: Path, gt) -> dict:
    """The benchmark's surface metrics (as scaled), shape-only metrics (free scale and rotation), and
    the mass properties a simulator reads vs the ground-truth solid at the same density."""
    from scan2usd.evaluate import align, compare
    from scan2usd.register import candidate_rotations, global_register
    from scan2usd.usd_io import read_usd

    mesh = read_usd(usd).render_mesh
    rep = _report(usd)
    m = compare(mesh, gt)
    out = {"as_scaled": {k: m[k] for k in ("accuracy_mm", "completeness_mm", "chamfer_mm", "fscore_2mm",
                                             "fscore_2mm_above_base", "extents_mm") if k in m}}
    # Shape only: the best similarity (free scale and rotation): of the rigid alignment, that alignment
    # refined with free scale, and a global rotation search (generators may tilt or squash), the one
    # with the lowest two-way distance. Distances by KD-tree on dense samples (~0.2 mm apart).
    from scipy.spatial import cKDTree

    from scan2usd.register import Similarity, similarity_icp

    M = align(mesh, gt)
    src = np.asarray(mesh.sample(6000, seed=0))
    dst = np.asarray(gt.sample(12000, seed=1))
    rigid = Similarity(1.0, M[:3, :3], M[:3, 3])
    cands = [rigid, similarity_icp(src, dst, rigid, iters=60, trim=0.9, with_scale=True).transform,
             global_register(src, dst, scale=1.0, rotations=candidate_rotations(96, seed=1), tau=0.002,
                             with_scale=True, keep=6, trim=0.9).transform]
    m_pts = np.asarray(mesh.sample(200000, seed=2))
    g_pts = np.asarray(gt.sample(200000, seed=3))
    g_tree = cKDTree(g_pts)
    best = None
    for tf in cands:
        if not 0.6 < tf.s < 1.6:  # a collapsed or exploded fit
            continue
        a_pts = tf.apply(m_pts)
        acc = g_tree.query(a_pts)[0] * 1000
        comp = cKDTree(a_pts).query(g_pts)[0] * 1000
        ch = float((acc.mean() + comp.mean()) / 2)
        if best is None or ch < best[0]:
            pr, rc = np.mean(acc < 2), np.mean(comp < 2)
            best = (ch, tf, float(2 * pr * rc / (pr + rc)) if pr + rc else 0.0, float(acc.mean()))
    out["shape_only"] = {"best_scale": float(best[1].s), "chamfer_mm": best[0], "fscore_2mm": best[2],
                         "accuracy_mm": {"mean": best[3]}}
    # What the physics engine reads: mass properties from the asset's colliders/mesh.
    mp = rep["mass_properties"]
    rho = mp["density"]
    g = gt.copy()
    g.density = rho
    com = M[:3, :3] @ np.asarray(mp["center_of_mass"]) + M[:3, 3]
    gp, ap = np.sort(g.principal_inertia_components), np.sort(np.asarray(mp["principal_moments"]))
    out["physics"] = {
        "density": rho, "volume_cm3": mp["volume"] * 1e6, "gt_volume_cm3": float(g.volume * 1e6),
        "mass_error_pct": float((mp["volume"] / g.volume - 1) * 100),
        "com_offset_mm": float(np.linalg.norm(com - g.center_mass) * 1000),
        "inertia_error_pct": float(np.mean(np.abs(ap / gp - 1)) * 100),
        "volume_source": mp.get("volume_source"),
    }
    out["asset"] = {"hulls": rep["collision_hulls"], "validation": rep["validation"]["status"]
                    if isinstance(rep["validation"], dict) else rep["validation"], "triangles": rep["render_triangles"]}
    return out


def stage_score(state, rebuild: bool = False):
    from evaluate_synthetic import load_ground_truth

    items = json.loads((OUT / "inputs.json").read_text())
    scores = json.loads((OUT / "scores.json").read_text()) if (OUT / "scores.json").exists() else {}
    for key, (folder, material, display) in OBJECTS.items():
        cap = ROOT / "data" / "synthetic" / folder
        gt = load_ground_truth(cap)
        height = round(float(gt.extents[2]), 3)  # the benchmark's one ruler measurement
        scores.setdefault(key, {"display": display})
        for src in SOURCES:
            if src == "photogrammetry":
                usd = next((cap / "work" / "asset").glob("*.usda"))
            else:
                if not (OUT / "meshes" / src / f"{key}.ply").exists():
                    continue
                usd = build_generated(src, key, items[key], height, rebuild)
            r = score_asset(usd, gt)
            meta = OUT / "meshes" / src / f"{key}.json"
            if meta.exists():
                mj = json.loads(meta.read_text())
                r["generation"] = {"seconds": mj["seconds"], "peak_gpu_gb": mj["peak_gpu_gb"]}
            scores[key][src] = r
            a, s, p = r["as_scaled"], r["shape_only"], r["physics"]
            print(f"{key[:22]:22s} {src:15s} F@2mm {a['fscore_2mm']:.3f} chamfer {a['chamfer_mm']:5.2f} mm | "
                  f"shape-only F {s['fscore_2mm']:.3f} (scale x{s['best_scale']:.3f}) | mass {p['mass_error_pct']:+6.1f}% "
                  f"CoM {p['com_offset_mm']:4.1f} mm inertia {p['inertia_error_pct']:5.1f}%", flush=True)
            (OUT / "scores.json").write_text(json.dumps(scores, indent=1))
    state["scores"] = str((OUT / "scores.json").relative_to(ROOT))



REAL_MODELS = ("hunyuan2mv", "hunyuan21", "triposg")


def _rigid_to(src_pts, dst_pts, yaw_only, with_scale=False):
    from scan2usd.register import candidate_rotations, global_register, voxel_down

    s = voxel_down(src_pts, 0.002, 12000)
    d = voxel_down(dst_pts, 0.002, 30000)
    rots = candidate_rotations(72, yaw_only=True) if yaw_only else candidate_rotations(480, seed=3)
    return global_register(s, d, scale=1.0, rotations=rots, tau=0.002, trim=0.8, with_scale=with_scale, keep=8)


def _build_real(model, name, scanned_pts, meas):
    """A generated real object as an asset: upright (holder: as generated; controller: lying face up
    like the scanned one, found by registration since the generator saw it from above at an angle),
    scaled to the same tape measurement as the scan, built like the scanned asset."""
    import shutil

    import trimesh

    from scan2usd.mesh_ops import orient_and_scale
    from scan2usd.pipeline import BuildOptions, build_asset

    out = OUT / "real" / model / name
    usd = out / f"{name}.usda"
    if usd.exists():
        return usd
    raw = trimesh.load(str(OUT / "meshes" / model / f"{name}.ply"), force="mesh")
    if len(raw.faces) > 400_000:
        raw = raw.simplify_quadric_decimation(face_count=400_000)
    m, _ = orient_and_scale(raw, up_axis="y")
    if name == "controller":  # orientation only (free scale), then the measurement sets the size
        f = _rigid_to(np.asarray(m.sample(20000, seed=0)), scanned_pts, yaw_only=False, with_scale=True)
        T = np.eye(4)
        T[:3, :3] = f.transform.R
        m.apply_transform(T)
    height = meas["height_m"] if name == "cod_holder" else meas["height_face_up_m"]
    tmp = out.parent / f"{name}_upright.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.export(str(tmp))
    shutil.rmtree(out, ignore_errors=True)
    opts = BuildOptions(name=name, material="plastic", mass=meas["mass_kg"], up_axis="z", units="m", scale_to=height,
                        scale_axis="z", usda=True, run_sim=True, preview=True, max_hulls=64, hull_threshold=0.03)
    return build_asset(tmp, out, opts).usd_path


def drop_test(h_asset, c_asset, R_true, t_true, trials=12):
    """study_holder_fit.stage_sim's drop: the controller released 3 cm above its real pose (jittered
    +-8 mm, +-8 deg), 3 s of MuJoCo; held = not on the table; error = mean point distance (ADD)."""
    import mujoco
    from scipy.spatial.transform import Rotation

    from scan2usd.mjcf import asset_body
    from scan2usd.register import rotation_angle_deg

    cp = np.asarray(c_asset.render_mesh.sample(4000, seed=0))
    true_pts = cp @ R_true.T + t_true
    rng = np.random.default_rng(0)
    starts = [(np.zeros(3), 0.0)] + [(np.r_[rng.uniform(-0.008, 0.008, 2), 0.0], rng.uniform(-8, 8))
                                     for _ in range(trials - 1)]
    hold = asset_body(h_asset, "holder", collision="hulls", free=False)
    rows = []
    for dxy, dyaw in starts:
        R0 = Rotation.from_euler("z", dyaw, degrees=True).as_matrix() @ R_true
        t0 = t_true + dxy + np.array([0, 0, 0.03])
        q0 = Rotation.from_matrix(R0).as_quat()
        ctrl = asset_body(c_asset, "ctrl", pos=t0, quat=(q0[3], *q0[:3]), collision="hulls")
        xml = f"""<mujoco><option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
<asset>{hold.assets_xml}{ctrl.assets_xml}</asset>
<worldbody><geom type="plane" size="1 1 0.05" friction="0.6 0.005 0.0001"/>{hold.body_xml}{ctrl.body_xml}</worldbody>
</mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml, {**hold.files, **ctrl.files})
        data = mujoco.MjData(model)
        for _ in range(int(3.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        b = model.body("ctrl").id
        R1, t1 = data.xmat[b].reshape(3, 3), data.xpos[b].copy()
        sim = cp @ R1.T + t1
        rows.append({"add_mm": float(np.linalg.norm(sim - true_pts, axis=1).mean() * 1000),
                     "rot_deg": rotation_angle_deg(R1, R_true), "held": bool(sim[:, 2].min() > 0.005)})
    held = [r for r in rows if r["held"]]
    return {"trials": rows, "held": len(held), "of": len(rows),
            "add_mm_median": float(np.median([r["add_mm"] for r in rows])),
            "rot_deg_median": float(np.median([r["rot_deg"] for r in rows]))}


def stage_physics(state):
    """The real-world check with generated assets: generated controller into generated holder."""
    from scan2usd.usd_io import read_usd

    hf = json.loads((ROOT / "output" / "holder_fit" / "state.json").read_text())
    meas = json.loads((ROOT / "data" / "raw" / "measurements.json").read_text())
    rel = hf["truth"]["controller_in_holder"]
    T_rel = np.eye(4)
    T_rel[:3, :3], T_rel[:3, 3] = np.array(rel["R"]), np.array(rel["t"])
    scanned = {n: read_usd(ROOT / f"output/holder_fit/assets/{n}/{n}.usda") for n in ("cod_holder", "controller")}
    pts = {n: np.asarray(a.render_mesh.sample(60000, seed=0)) for n, a in scanned.items()}
    res = {"photogrammetry": {"drop": {k: hf["sim"]["decomposed"][k] for k in ("add_mm_median", "rot_deg_median")}
                              | {"held": hf["sim"]["decomposed"]["held_in_holder"], "of": hf["sim"]["decomposed"]["of"]}}}
    runs = [m for base in REAL_MODELS for m in (base, f"{base}_s1", f"{base}_s2")
            if (OUT / "meshes" / m / "controller.ply").exists()]  # seed 0, and the robustness seeds
    for model in runs:
        r = {}
        T = {}
        assets = {}
        for name in ("cod_holder", "controller"):
            usd = _build_real(model, name, pts[name], meas[name])
            a = read_usd(usd)
            assets[name] = a
            g = np.asarray(a.render_mesh.sample(30000, seed=0))
            # generated asset -> scanned asset (rigid, both metric): how far apart are the shapes?
            src = g[g[:, 2] > 0.025] if name == "cod_holder" else g  # the holder: figure above the base
            dst = pts[name][pts[name][:, 2] > 0.025] if name == "cod_holder" else pts[name]
            f = _rigid_to(src, dst, yaw_only=name == "cod_holder")
            T[name] = f.transform.matrix()
            from scipy.spatial import cKDTree

            d_gs = cKDTree(pts[name]).query(f.transform.apply(g))[0]
            d_sg = cKDTree(f.transform.apply(g)).query(pts[name])[0]
            r[name] = {"vs_scan_chamfer_mm": float((d_gs.mean() + d_sg.mean()) / 2 * 1000),
                       "vs_scan_fscore_2mm": float(2 * np.mean(d_gs < 0.002) * np.mean(d_sg < 0.002)
                                                   / max(np.mean(d_gs < 0.002) + np.mean(d_sg < 0.002), 1e-9)),
                       "extents_mm": [float(v * 1000) for v in a.render_mesh.extents],
                       "hulls": len(a.hulls)}
        # The real resting pose, in the generated holder's frame, for the generated controller.
        Tg = np.linalg.inv(T["cod_holder"]) @ T_rel @ T["controller"]
        r["drop"] = drop_test(assets["cod_holder"], assets["controller"], Tg[:3, :3], Tg[:3, 3])
        d = r["drop"]
        print(f"{model:12s} holder vs scan: chamfer {r['cod_holder']['vs_scan_chamfer_mm']:.1f} mm, F@2mm "
              f"{r['cod_holder']['vs_scan_fscore_2mm']:.2f} | controller: {r['controller']['vs_scan_chamfer_mm']:.1f} mm, "
              f"F {r['controller']['vs_scan_fscore_2mm']:.2f} | drop: held {d['held']}/{d['of']}, median "
              f"{d['add_mm_median']:.1f} mm / {d['rot_deg_median']:.0f} deg", flush=True)
        res[model] = r
    (OUT / "physics.json").write_text(json.dumps(res, indent=1))
    state["physics"] = str((OUT / "physics.json").relative_to(ROOT))



LOOK_SOURCES = {  # source -> (textured glb, label)
    "ground_truth": (None, "Ground truth (re-render)"),
    "photogrammetry": (None, "Photogrammetry (78 photos)"),
    "hunyuan2mv": ("textured/hunyuan2mv", "Hunyuan3D-2mv + paint (4 photos)"),
    "hunyuan21": ("textured/hunyuan21", "Hunyuan3D 2.1 + paint (1 photo)"),
}
LOOK_VIEWS = list(range(0, 78, 10))  # 8 of the capture's photos, all rings; none was a generator input


def LOOK_OBJECTS(looks):  # noqa: N802
    """Objects whose ground truth, re-rendered the same way, reproduces the capture's photos (> 30 dB):
    only there does the comparison measure the asset. (The procedural mug's capture predates a change
    in how the scene places it: 15.9 dB.)"""
    return [k for k in OBJECTS if k in looks and looks[k].get("ground_truth", {}).get("psnr_db", 0) > 30]


def _aligned_glb(src, key, cap, gt, height):
    """The textured asset placed in the capture's world (= the ground-truth frame), as a Y-up glb."""
    import trimesh

    from scan2usd.evaluate import align
    from scan2usd.mesh_ops import orient_and_scale

    out = OUT / "look" / src / f"{key}.glb"
    if out.exists():
        return out
    if src == "ground_truth":
        rel = (cap / "gt_path.txt").read_text().strip() if (cap / "gt_path.txt").exists() else "gt_mesh.ply"
        return cap / rel  # rendered exactly as the capture was
    if src == "photogrammetry":
        m = trimesh.load(str(next((cap / "work" / "asset").glob("*.glb"))), force="mesh")
        m, _ = orient_and_scale(m, up_axis="y")  # the asset's glb: metric, Y-up
        # Its vertex colors are sRGB, but glTF reads COLOR_0 as linear (Blender would wash them out).
        from scan2usd.color import srgb_to_linear

        vc = np.asarray(m.visual.vertex_colors).copy()
        vc[:, :3] = np.round(srgb_to_linear(vc[:, :3] / 255.0) * 255).astype(np.uint8)
        m.visual.vertex_colors = vc
    else:
        m = trimesh.load(str(OUT / LOOK_SOURCES[src][0] / f"{key}.glb"), force="mesh")
        m, _ = orient_and_scale(m, up_axis="y", scale_to=height, scale_axis="z")  # as the geometry was scored
    m.apply_transform(align(m, gt))
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # Z-up -> glTF Y-up
    out.parent.mkdir(parents=True, exist_ok=True)
    m.export(str(out))
    return out


def _gt_mask(gt, c2w, K, scale):
    """Ground-truth silhouette in a photo (pure-PyTorch rasterizer, genai_raster.py)."""
    import torch

    import genai_raster

    W, H = int(K["width"] * scale), int(K["height"] * scale)
    fx, fy, cx, cy = K["fx"] * scale, K["fy"] * scale, K["cx"] * scale, K["cy"] * scale
    pc = (gt.vertices - c2w[:3, 3]) @ c2w[:3, :3]
    u, v = fx * pc[:, 0] / pc[:, 2] + cx, fy * pc[:, 1] / pc[:, 2] + cy
    z = pc[:, 2]
    clip = np.c_[((u - 0.5) / (W - 1) - 0.5) * 2, ((v - 0.5) / (H - 1) - 0.5) * 2,
                 (z - z.min()) / (np.ptp(z) + 1e-9) * 2 - 1, np.ones(len(z))]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fi, _ = genai_raster.rasterize(torch.tensor(clip, dtype=torch.float32, device=dev)[None],
                                   torch.tensor(gt.faces, dtype=torch.int32, device=dev), (H, W))
    return fi.cpu().numpy() > 0


def stage_look(state, scale=0.5):
    """Re-photograph each textured asset in the benchmark's own Blender scene from 8 capture cameras
    and compare with the capture's photos (compare_splat.py's protocol: inside the object, white
    balance matched, CIEDE2000 / PSNR / SSIM)."""
    from scipy import ndimage

    from compare_splat import score
    from evaluate_synthetic import load_ground_truth

    looks = {}
    for key, (folder, material, display) in OBJECTS.items():
        cap = ROOT / "data" / "synthetic" / folder
        gt = load_ground_truth(cap)
        height = round(float(gt.extents[2]), 3)
        cj = json.loads((cap / "cameras.json").read_text())
        cams = [cj["images"][i] for i in LOOK_VIEWS]
        poses = [{"name": c["name"], "camera_to_world": c["camera_to_world"]} for c in cams]
        looks[key] = {"display": display}
        for src in LOOK_SOURCES:
            if LOOK_SOURCES[src][0] and not (OUT / LOOK_SOURCES[src][0] / f"{key}.glb").exists():
                continue
            mesh = _aligned_glb(src, key, cap, gt, height)
            d = OUT / "look" / "renders" / src / key
            if not all((d / "images" / c["name"]).exists() for c in cams):
                d.mkdir(parents=True, exist_ok=True)
                (d / "poses.json").write_text(json.dumps(poses))
                cmd = [BLENDER, "-b", "--factory-startup", "--python",
                       str(ROOT / "scripts" / "blender_render_capture.py"), "--", "--mesh", str(mesh), "--out", str(d),
                       "--poses", str(d / "poses.json"), "--no-chart", "--keep-materials",
                       "--resolution-scale", str(scale)]
                if src != "ground_truth":
                    cmd.append("--keep-position")
                if src == "ground_truth" and mesh.name == "gt_mesh.ply":
                    cmd.remove("--keep-materials")  # the procedural mug: its glaze is a Blender material
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rows = []
            for c in cams:
                from PIL import Image

                photo = Image.open(cap / "images" / c["name"]).convert("RGB")
                photo = np.asarray(photo.resize((int(photo.width * scale), int(photo.height * scale)), Image.LANCZOS))
                rend = np.asarray(Image.open(d / "images" / c["name"]).convert("RGB"))
                mask = _gt_mask(gt, np.array(c["camera_to_world"]), cj["intrinsics"], scale)
                inner = ndimage.binary_erosion(mask, iterations=3)
                s, _ = score(photo, rend, inner)
                rows.append(s)
            med = {k: float(np.median([r[k] for r in rows])) for k in ("delta_e2000_wb_median", "psnr_db", "ssim")}
            looks[key][src] = {"views": [c["name"] for c in cams], "per_view": rows, **med}
            print(f"{key[:22]:22s} {src:15s} dE {med['delta_e2000_wb_median']:5.2f}  PSNR {med['psnr_db']:5.2f} dB  "
                  f"SSIM {med['ssim']:.3f}", flush=True)
            (OUT / "look.json").write_text(json.dumps(looks, indent=1))
    state["look"] = str((OUT / "look.json").relative_to(ROOT))



def _heat(d_mm):
    """Per-vertex error colors: green (0) -> yellow (2.5 mm) -> red (5 mm and more)."""
    t = np.clip(np.asarray(d_mm) / 5.0, 0, 1)[:, None]
    g, y, r = np.array([46, 160, 90]), np.array([240, 200, 40]), np.array([210, 50, 40])
    c = np.where(t < 0.5, g + (y - g) * (t / 0.5), y + (r - y) * ((t - 0.5) / 0.5))
    return np.c_[c, np.full(len(c), 255)].astype(np.uint8)


def _photo_tile(path, size):
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB").resize((size, size), Image.LANCZOS)


def figure_geometry(size=210):
    """Rows: benchmark objects. Columns: the input photo, then each source's asset colored by its
    distance to the ground-truth surface (as scaled to the one measured height, rigidly aligned)."""
    from PIL import Image, ImageDraw
    from scipy.spatial import cKDTree

    from evaluate_synthetic import load_ground_truth
    from scan2usd.evaluate import align
    from scan2usd.preview import _font, render
    from scan2usd.usd_io import read_usd

    scores = json.loads((OUT / "scores.json").read_text())
    cols = ["input", *SOURCES]
    head = {"input": "Input photo", **{k: v.replace(" (", "\n(") for k, v in SOURCES.items()}}
    pad, top, lab = 8, 58, 22
    W = pad + len(cols) * (size + pad)
    H = top + len(OBJECTS) * (size + lab + pad) + 40
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for j, c in enumerate(cols):
        x = pad + j * (size + pad)
        for k, line in enumerate(head[c].split("\n")):
            d.text((x + size / 2, 10 + 20 * k), line, fill=(30, 41, 59), font=_font(15 if k == 0 else 13),
                   anchor="ma")
    for i, (key, (folder, material, display)) in enumerate(OBJECTS.items()):
        cap = ROOT / "data" / "synthetic" / folder
        gt = load_ground_truth(cap)
        tree = cKDTree(np.asarray(gt.sample(300000, seed=0)))
        y = top + i * (size + lab + pad)
        sheet.paste(_photo_tile(OUT / "inputs" / key / "single.png", size), (pad, y))
        d.text((pad + 4, y + size + 3), display.replace("*", ""), fill=(71, 85, 105), font=_font(13))
        for j, src in enumerate(SOURCES, start=1):
            usd = next((cap / "work" / "asset").glob("*.usda")) if src == "photogrammetry" else \
                OUT / "assets" / src / key / f"{key}.usda"
            m = read_usd(usd).render_mesh
            m.apply_transform(align(m, gt))
            import trimesh

            m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=_heat(tree.query(m.vertices)[0] * 1000))
            img = render([m], size=size, azimuth=SINGLE[0] + 180 - 90, elevation=22)
            x = pad + j * (size + pad)
            sheet.paste(img, (x, y))
            f = scores[key][src]["as_scaled"]["fscore_2mm"]
            d.text((x + size / 2, y + size + 3), f"F@2mm {f:.2f}", fill=(30, 41, 59), font=_font(13), anchor="ma")
    # legend
    ly = H - 30
    for k in range(200):
        col = tuple(int(v) for v in _heat([k / 199 * 5.0])[0, :3])
        d.line([(pad + 120 + k, ly), (pad + 120 + k, ly + 12)], fill=col)
    d.text((pad, ly - 1), "distance to truth", fill=(71, 85, 105), font=_font(13))
    d.text((pad + 120, ly + 14), "0", fill=(71, 85, 105), font=_font(12))
    d.text((pad + 320, ly + 14), "\u2265 5 mm", fill=(71, 85, 105), font=_font(12), anchor="ra")
    d.text((W - pad, ly), "Hunyuan3D-2mv got four photos (front, left, back, right); the others one",
           fill=(71, 85, 105), font=_font(13), anchor="ra")
    FIG.mkdir(parents=True, exist_ok=True)
    sheet.save(FIG / "genai_geometry.jpg", quality=88)
    print("wrote", FIG / "genai_geometry.jpg")


def _settle(h_asset, c_asset, Tg):
    import mujoco
    from scipy.spatial.transform import Rotation

    from scan2usd.mjcf import asset_body

    q0 = Rotation.from_matrix(Tg[:3, :3]).as_quat()
    hold = asset_body(h_asset, "holder", collision="hulls", free=False)
    ctrl = asset_body(c_asset, "ctrl", pos=Tg[:3, 3] + [0, 0, 0.03], quat=(q0[3], *q0[:3]), collision="hulls")
    xml = f"""<mujoco><option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
<asset>{hold.assets_xml}{ctrl.assets_xml}</asset>
<worldbody><geom type="plane" size="1 1 0.05" friction="0.6 0.005 0.0001"/>{hold.body_xml}{ctrl.body_xml}</worldbody>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml, {**hold.files, **ctrl.files})
    data = mujoco.MjData(m)
    for _ in range(1500):
        mujoco.mj_step(m, data)
    b = m.body("ctrl").id
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = data.xmat[b].reshape(3, 3), data.xpos[b]
    return T


def figure_drop(size=300):
    """The real pair: what each source was given, and where its controller settles in its holder
    (MuJoCo, release 3 cm above the real pose; green: the real pose)."""
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font, render
    from scan2usd.usd_io import read_usd

    phys = json.loads((OUT / "physics.json").read_text())
    items = json.loads((OUT / "inputs.json").read_text())
    hf = json.loads((ROOT / "output" / "holder_fit" / "state.json").read_text())
    rel = hf["truth"]["controller_in_holder"]
    T_rel = np.eye(4)
    T_rel[:3, :3], T_rel[:3, 3] = np.array(rel["R"]), np.array(rel["t"])
    scanned = {n: read_usd(ROOT / f"output/holder_fit/assets/{n}/{n}.usda") for n in ("cod_holder", "controller")}
    pts = {n: np.asarray(a.render_mesh.sample(60000, seed=0)) for n, a in scanned.items()}
    cols = ["photogrammetry", *REAL_MODELS]
    label = {"photogrammetry": "Photogrammetry", "hunyuan2mv": "Hunyuan3D-2mv", "hunyuan21": "Hunyuan3D 2.1",
             "triposg": "TripoSG"}
    given = {"photogrammetry": "81 + 78 + 38 photos", "hunyuan2mv": "4 photos each", "hunyuan21": "1 photo each",
             "triposg": "1 photo each"}
    pad, top = 10, 64
    W = pad + len(cols) * (size + pad)
    thumb = (size - 3 * 4) // 4
    H = top + thumb + 8 + size + 70
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for j, src in enumerate(cols):
        x = pad + j * (size + pad)
        d.text((x + size / 2, 10), label[src], fill=(30, 41, 59), font=_font(18), anchor="ma")
        d.text((x + size / 2, 34), given[src], fill=(71, 85, 105), font=_font(14), anchor="ma")
        if src == "photogrammetry":
            A, Tg = scanned, T_rel
            names = []
        else:
            A = {n: read_usd(OUT / "real" / src / n / f"{n}.usda") for n in ("cod_holder", "controller")}
            T = {}
            for name in ("cod_holder", "controller"):
                g = np.asarray(A[name].render_mesh.sample(30000, seed=0))
                s_ = g[g[:, 2] > 0.025] if name == "cod_holder" else g
                d_ = pts[name][pts[name][:, 2] > 0.025] if name == "cod_holder" else pts[name]
                T[name] = _rigid_to(s_, d_, yaw_only=name == "cod_holder").transform.matrix()
            Tg = np.linalg.inv(T["cod_holder"]) @ T_rel @ T["controller"]
            roles = ["single"] if src != "hunyuan2mv" else ["front", "left"]
            names = [(n, items[n]["single"]["photo"] if r == "single" else items[n]["multiview"][r]["photo"])
                     for n in ("cod_holder", "controller") for r in roles]
        for k, (obj, photo) in enumerate(names[:4]):
            sheet.paste(_photo_tile(OUT / "inputs" / obj / Path(photo).with_suffix(".png").name, thumb),
                        (x + k * (thumb + 4), top))
        if not names:
            d.text((x + size / 2, top + thumb / 2), "(the full scans)", fill=(148, 163, 184), font=_font(14),
                   anchor="mm")
        T1 = _settle(A["cod_holder"], A["controller"], Tg)
        c_end, ghost = A["controller"].render_mesh.copy(), A["controller"].render_mesh.copy()
        c_end.apply_transform(T1)
        ghost.apply_transform(Tg)
        h = A["cod_holder"].render_mesh.copy()
        for mm in (h, c_end, ghost):
            mm.visual = None
        img = render([h, c_end, ghost], colors=[(150, 150, 156), (232, 128, 64), (64, 190, 100)], size=size,
                     azimuth=-70, elevation=14)
        sheet.paste(img, (x, top + thumb + 8))
        r = phys[src]["drop"]
        ok = r["held"] == r["of"]
        d.text((x + size / 2, top + thumb + 14 + size), f"held {r['held']}/{r['of']}", fill=(21, 128, 61) if ok
               else (185, 28, 28), font=_font(17), anchor="ma")
        d.text((x + size / 2, top + thumb + 38 + size), f"median {r['add_mm_median']:.1f} mm / "
               f"{r['rot_deg_median']:.0f}\u00b0 from the real pose", fill=(71, 85, 105), font=_font(13), anchor="ma")
    sheet.save(FIG / "genai_drop.jpg", quality=88)
    print("wrote", FIG / "genai_drop.jpg")


def figure_look(objects=("ACE_Coffee_Mug_Kristen_16_oz_cup", "Great_Dinos_Triceratops_Toy",
                         "Schleich_Lion_Action_Figure", "Womens_Bluefish_2Eye_Boat_Shoe_in_Tan"), view=2, size=240):
    """One held-out capture photo per object vs each textured asset re-photographed in the same scene."""
    from PIL import Image, ImageDraw
    from scipy import ndimage

    from evaluate_synthetic import load_ground_truth
    from scan2usd.preview import _font

    looks = json.loads((OUT / "look.json").read_text())
    cols = ["photo", "photogrammetry", "hunyuan2mv", "hunyuan21"]
    head = {"photo": "Held-out photo", "photogrammetry": "Photogrammetry", "hunyuan2mv": "Hunyuan3D-2mv + paint",
            "hunyuan21": "Hunyuan3D 2.1 + paint"}
    pad, top, lab = 8, 34, 22
    W = pad + len(cols) * (size + pad)
    H = top + len(objects) * (size + lab + pad)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for j, c in enumerate(cols):
        d.text((pad + j * (size + pad) + size / 2, 10), head[c], fill=(30, 41, 59), font=_font(15), anchor="ma")
    for i, key in enumerate(objects):
        folder = OBJECTS[key][0]
        cap = ROOT / "data" / "synthetic" / folder
        cj = json.loads((cap / "cameras.json").read_text())
        cam = cj["images"][LOOK_VIEWS[view]]
        gt = load_ground_truth(cap)
        mask = _gt_mask(gt, np.array(cam["camera_to_world"]), cj["intrinsics"], 0.5)
        ys, xs = np.nonzero(ndimage.binary_dilation(mask, iterations=12))
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        y = top + i * (size + lab + pad)
        for j, c in enumerate(cols):
            x = pad + j * (size + pad)
            if c == "photo":
                im = Image.open(cap / "images" / cam["name"]).convert("RGB")
                im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
                txt = OBJECTS[key][2].replace("*", "")
            else:
                im = Image.open(OUT / "look" / "renders" / c / key / "images" / cam["name"]).convert("RGB")
                r = looks[key][c]
                txt = f"\u0394E {r['delta_e2000_wb_median']:.1f} \u00b7 PSNR {r['psnr_db']:.1f} dB"
            sheet.paste(im.crop(box).resize((size, size), Image.LANCZOS), (x, y))
            d.text((x + size / 2, y + size + 3), txt, fill=(71, 85, 105), font=_font(13), anchor="ma")
    sheet.save(FIG / "genai_look.jpg", quality=88)
    print("wrote", FIG / "genai_look.jpg")


def stage_figure(state, only=None):
    if only != "look":
        figure_geometry()
        figure_drop()
    if (OUT / "look.json").exists():
        figure_look()
    stage_summary(state)



def stage_summary(state):
    """docs/figures/genai.json: per source, the averages the docs quote (and the per-object rows)."""
    scores = json.loads((OUT / "scores.json").read_text())
    phys = json.loads((OUT / "physics.json").read_text())
    looks = json.loads((OUT / "look.json").read_text()) if (OUT / "look.json").exists() else {}
    out = {"sources": {}, "objects": {}, "real_pair": phys, "look": looks,
           "protocol": {"scale": "one measured height (the benchmark's ruler value) for every source",
                        "inputs": "benchmark: views rendered in the capture's Blender scene without the chart; "
                                  "real pair: phone photos from the scans' captures",
                        "seed": 0}}
    for src, label in SOURCES.items():
        rows = [scores[k][src] for k in OBJECTS if src in scores.get(k, {})]
        gen = [json.loads((OUT / "meshes" / src / f"{k}.json").read_text()) for k in OBJECTS
               if (OUT / "meshes" / src / f"{k}.json").exists()]
        s = {"label": label, "objects": len(rows),
             "fscore_2mm_mean": float(np.mean([r["as_scaled"]["fscore_2mm"] for r in rows])),
             "chamfer_mm_median": float(np.median([r["as_scaled"]["chamfer_mm"] for r in rows])),
             "shape_only_fscore_2mm_mean": float(np.mean([r["shape_only"]["fscore_2mm"] for r in rows])),
             "mass_abs_error_pct_median": float(np.median([abs(r["physics"]["mass_error_pct"]) for r in rows])),
             "com_offset_mm_median": float(np.median([r["physics"]["com_offset_mm"] for r in rows])),
             "inertia_error_pct_median": float(np.median([r["physics"]["inertia_error_pct"] for r in rows])),
             "wins_fscore_vs_photogrammetry": int(sum(scores[k][src]["as_scaled"]["fscore_2mm"]
                                                      > scores[k]["photogrammetry"]["as_scaled"]["fscore_2mm"]
                                                      for k in OBJECTS if src in scores.get(k, {})))}
        if gen:
            s["seconds_mean"] = float(np.mean([g["seconds"] for g in gen]))
            s["peak_gpu_gb"] = float(max(g["peak_gpu_gb"] for g in gen))
        else:  # photogrammetry: COLMAP + build, from the benchmark's own timings
            ev = [json.loads((ROOT / "data" / "synthetic" / OBJECTS[k][0] / "work" / "evaluation.json").read_text())
                  for k in OBJECTS]
            s["seconds_mean"] = float(np.mean([sum(e["reconstruction"]["timings_s"].values()) + e["build_time_s"]
                                               for e in ev]))
        if src in phys:
            s["real_drop"] = phys[src]["drop"]
            seeds = [phys[k]["drop"] for k in (src, f"{src}_s1", f"{src}_s2") if k in phys]
            s["real_drop_seeds"] = [{"held": x["held"], "of": x["of"], "add_mm_median": x["add_mm_median"]}
                                    for x in seeds]
        lk = [looks[k][src] for k in LOOK_OBJECTS(looks) if src in looks[k]]
        if lk:
            s["look"] = {m: float(np.median([v[m] for v in lk])) for m in ("delta_e2000_wb_median", "psnr_db", "ssim")}
        out["sources"][src] = s
    for k in OBJECTS:
        out["objects"][k] = {src: {"fscore_2mm": v["as_scaled"]["fscore_2mm"], "chamfer_mm": v["as_scaled"]["chamfer_mm"],
                                   "mass_error_pct": v["physics"]["mass_error_pct"],
                                   "com_offset_mm": v["physics"]["com_offset_mm"]}
                             for src, v in scores[k].items() if src in SOURCES}
    (FIG / "genai.json").write_text(json.dumps(out, indent=1))
    for src, s in out["sources"].items():
        print(f"{src:15s} F@2mm {s['fscore_2mm_mean']:.3f} (wins {s['wins_fscore_vs_photogrammetry']}/6) "
              f"|mass err| {s['mass_abs_error_pct_median']:.1f}% CoM {s['com_offset_mm_median']:.1f} mm "
              f"t {s['seconds_mean']:.0f} s {('drop ' + str(s['real_drop']['held']) + '/12') if 'real_drop' in s else ''} "
              f"{s.get('look', '')}")
    state["summary"] = str((FIG / "genai.json").relative_to(ROOT))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="inputs")
    args = ap.parse_args()
    sp = OUT / "state.json"
    state = json.loads(sp.read_text()) if sp.exists() else {}
    for stage in args.stage.split(","):
        globals()[f"stage_{stage}"](state)
        OUT.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
