"""Real capture study: a scanned controller holder, a scanned controller, and the two together.

    python scripts/study_holder_fit.py [--stage prep,scale,merge,base,gap,assets,truth,sim,video,figure,lead]

Data: four iPhone 13 Pro captures (data/raw/, git-ignored), each reconstructed with
`scan2usd reconstruct` into data/work/<name>/:
  cod_holder               81 photos, the figure (a controller holder) standing on its base
  controller               78 photos, the controller face up
  controller_standing      38 photos, the controller standing on its grips, leaning on a pen box
  holder_with_controller   58 photos, the controller resting in the figure's hands
Measurements (data/raw/measurements.json): holder height 21 cm, 528 g; controller 5.5 cm tall
lying face up (to the right thumbstick), 243 g.

Stages (outputs in output/holder_fit/):
  prep    remove the pen box the standing controller leaned on: its top is the big horizontal plane,
          its footprint (plus a margin) is cut away, and its red end cap by color.
  scale   metric scale for each scan: heights from the measurements (the combined scene's top is
          the holder's helmet, so it is checked by fitting the holder into it).
  merge   the standing pass is registered onto the face-up pass (similarity ICP over a rotation
          search, after removing the pen box); their union, minus the face-up pass's paper-skirt band,
          is meshed (Poisson) into one controller.
  base    the holder's glossy black base is barely seen from the side, so Poisson pits it; it is
          replaced by a cylinder fitted to its densely measured flat top (mesh_ops.fit_disc_base).
  gap     Poisson bridged the gap between the legs; the photos see through it, so it is carved out
          (carve.background_votes / subtract_volume; docs/figures/holder_gap.jpg).
  assets  scan2usd builds: holder (plastic, 528 g) and controller (plastic, 243 g), convex decomposition.
  truth   both assets registered into the combined scan: the real resting pose of the controller
          relative to the holder.
  sim     MuJoCo: the controller dropped into the holder's hands from above the real pose; where it
          settles vs where it really rests, for the decomposed and single-hull holder.
  video   side-by-side drop animation with a ghost of the real pose (docs/figures/holder_fit.{webp,jpg}).
  figure  per-trial error strip (docs/figures/holder_fit_results.png).
  lead    the docs' lead strip: one source photo (cropped tight) -> each scan -> the drop.

"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scan2usd.reconstruct import _median_spacing, find_colmap, read_ply_points, write_ply_points
from scan2usd.register import candidate_rotations, global_register, rotation_angle_deg, similarity_icp, voxel_down

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "work"
OUT = ROOT / "output" / "holder_fit"
FIG = ROOT / "docs" / "figures"
MEAS = json.loads((ROOT / "data" / "raw" / "measurements.json").read_text())


def load(name, file="object_points.ply"):
    p = read_ply_points(WORK / name / file)
    return (np.c_[p["x"], p["y"], p["z"]].astype(float), np.c_[p["nx"], p["ny"], p["nz"]].astype(float),
            np.c_[p["red"], p["green"], p["blue"]].astype(np.uint8))


def stage_prep(state):
    """Remove the pen box the standing controller leaned on (data/work/controller_standing)."""
    xyz, nrm, rgb = load("controller_standing")
    H = xyz[:, 2].max()
    hz = np.abs(nrm[:, 2]) > 0.9
    hist, edges = np.histogram(xyz[hz, 2] / H, bins=40, range=(0, 0.6))
    z_top = edges[hist.argmax()] + 0.5 * (edges[1] - edges[0])  # the box top: the big horizontal plane
    top = hz & (np.abs(xyz[:, 2] / H - z_top) < 0.03)
    zt = np.median(xyz[top, 2])
    pts = xyz[top & (np.abs(xyz[:, 2] - zt) < 0.01 * H)]
    c = pts[:, :2].mean(0)
    _, _, vt = np.linalg.svd(pts[:, :2] - c)
    uv = (pts[:, :2] - c) @ vt.T
    lo, hi = np.percentile(uv, 0.5, 0), np.percentile(uv, 99.5, 0)
    m = 0.02 * H
    uv_all = (xyz[:, :2] - c) @ vt.T
    box = np.all((uv_all > lo - m) & (uv_all < hi + m), 1) & (xyz[:, 2] < zt + 0.012 * H)
    red = (rgb[:, 0].astype(int) - rgb[:, 1:].max(1).astype(int) > 50) & (xyz[:, 2] < 0.45 * H)  # the end cap
    keep = ~(box | red)
    d, _ = cKDTree(xyz[keep]).query(xyz[keep], k=8)
    md = d[:, 1:].mean(1)
    idx = np.nonzero(keep)[0][md < md.mean() + 3 * md.std()]
    write_ply_points(WORK / "controller_standing" / "controller_points_clean.ply", xyz[idx], nrm[idx], rgb[idx])
    state["prep"] = {"box_top_z_frac": float(zt / H), "box_footprint": (hi - lo).tolist(),
                     "removed": int(len(xyz) - len(idx)), "kept": int(len(idx))}
    print(f"pen box removed: {len(xyz) - len(idx):,} of {len(xyz):,} points (top at {zt / H:.2f} of the height)")


def stage_scale(state):
    s = {"cod_holder": MEAS["cod_holder"]["height_m"] / np.percentile(load("cod_holder")[0][:, 2], 99.9),
         "controller": MEAS["controller"]["height_face_up_m"] / np.percentile(load("controller")[0][:, 2], 99.9),
         "holder_with_controller": MEAS["cod_holder"]["height_m"]
         / np.percentile(load("holder_with_controller")[0][:, 2], 99.9)}
    state["scale"] = {k: float(v) for k, v in s.items()}
    print("metric scale (m per scan unit):", {k: round(v, 5) for k, v in state["scale"].items()})


def stage_merge(state):
    s_up = state["scale"]["controller"]
    up, up_n, up_c = load("controller")
    up = up * s_up
    # The face-up pass's grips fused with the paper below ~6 mm; the standing pass sees them instead.
    band = up[:, 2] > 0.006
    st, st_n, st_c = load("controller_standing", "controller_points_clean.ply")
    sp_up, sp_st = _median_spacing(up), _median_spacing(st)
    guess = 1.0
    from scan2usd.register import principal_extent
    guess = principal_extent(up[band]) / principal_extent(st)
    src, dst = voxel_down(st, 2.0 * sp_st, 15000), voxel_down(up, 2.0 * sp_up, 40000)
    fit = global_register(src, dst, scale=guess, rotations=candidate_rotations(600, seed=1), tau=0.0015, trim=0.6)
    tf = fit.transform
    state["merge"] = {"scale_guess": guess, "scale": tf.s, "rmse_mm": fit.rmse * 1000,
                      "inliers_1p5mm": fit.inlier_fraction}
    print(f"standing -> face-up: scale {tf.s:.5f} (guess {guess:.5f}), trimmed RMSE {fit.rmse * 1000:.2f} mm, "
          f"{fit.inlier_fraction * 100:.0f}% of points within 1.5 mm")
    st_m, st_nm = tf.apply(st), st_n @ tf.R.T
    pts = np.vstack([up[band], st_m])
    nrm = np.vstack([up_n[band], st_nm])
    col = np.vstack([up_c[band], st_c])
    # The two passes were lit differently: equalize the standing pass's colors to the face-up pass
    # on the overlap (per-channel gain).
    d, j = cKDTree(up[band]).query(st_m)
    ov = d < 0.0015
    if ov.sum() > 500:
        gain = np.median(up_c[band][j[ov]].astype(float), 0) / np.maximum(np.median(st_c[ov].astype(float), 0), 1)
        col[band.sum():] = np.clip(st_c * gain, 0, 255).astype(np.uint8)
        state["merge"]["color_gain"] = gain.tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    write_ply_points(OUT / "controller_merged_points.ply", pts, nrm, col)
    mesh_points(OUT / "controller_merged_points.ply", OUT / "controller_merged.ply", pts, col)
    mesh_points(OUT / "controller_merged_points.ply", OUT / "controller_closed.ply", pts, col, trim_far=False)
    state["merge"]["points"] = int(len(pts))


def stage_base(state):
    """The holder's round base: replace the pitted scan with a fitted cylinder (mesh_ops.fit_disc_base)."""
    import trimesh

    from scan2usd.mesh_ops import fit_disc_base

    mesh = trimesh.load(WORK / "cod_holder" / "object_mesh.ply", force="mesh")
    pts, nrm, _ = load("cod_holder")
    fixed, fit = fit_disc_base(mesh, pts, nrm)
    fixed.export(OUT / "cod_holder_base_fixed.ply")
    s = state["scale"]["cod_holder"]
    state["base"] = {"top_mm": fit["z_top"] * s * 1000, "radius_mm": fit["radius"] * s * 1000,
                     "rim_rms_mm": fit["rim_rms"] * s * 1000, "watertight": bool(fixed.is_watertight)}
    print(f"holder base: disc r {fit['radius'] * s * 1000:.1f} mm, top {fit['z_top'] * s * 1000:.1f} mm "
          f"(rim fit RMS {fit['rim_rms'] * s * 1000:.2f} mm), watertight {fixed.is_watertight}")


def stage_gap(state):
    """The gap between the legs: Poisson bridged it; carve out what the photos see straight through.

    Seen from the front, the legs are dark against bright paper, so every photo votes for the 3D
    points through which it sees background (carve.background_votes). Within a window around the
    gap the votes give its outline in the front plane (x-z), which is cut through the legs' depth as
    a slot; behind the legs only the few views looking through the slot can see, so a lower vote
    count is accepted there. Below ~35 mm the photos see the gap only against the black base (no
    contrast): the slot's outline at the lowest resolved level is extended down to the base.
    """
    import trimesh
    from scipy import ndimage

    from scan2usd.carve import background_votes, subtract_volume
    from scan2usd.chart import read_colmap_model
    from scan2usd.reconstruct import isolate_object

    s = state["scale"]["cod_holder"]
    dense = WORK / "cod_holder" / "dense" / "0"
    ply = read_ply_points(dense / "fused.ply")
    xyz = np.c_[ply["x"], ply["y"], ply["z"]].astype(np.float64)
    spacing = _median_spacing(xyz)
    # Recover the scan's object frame (deterministic) and check it reproduces object_points.ply.
    aligned, keep, rot, origin, _ = isolate_object(xyz, spacing, plane_threshold=2.0 * spacing)
    assert np.allclose(aligned[keep][:1000], load("cod_holder")[0][:1000], atol=1e-5), "object frame mismatch"
    cams, imgs = read_colmap_model(dense / "sparse")

    mm = 0.5
    z0 = state["base"]["top_mm"] + 1.0
    X, Y, Z = np.arange(-50, 40, mm), np.arange(-45, 60, mm), np.arange(z0, 100, mm)
    grid = np.stack(np.meshgrid(X, Y, Z, indexing="ij"), -1).reshape(-1, 3) / 1000 / s  # scan units
    votes, _ = background_votes(grid @ rot + origin, cams, imgs, dense / "images")
    votes = votes.reshape(len(X), len(Y), len(Z))

    K, window = 6, (X > -26) & (X < 12)
    legs = (Y > -12) & (Y < 20)  # the legs' front-to-back extent
    outline = ((votes[:, legs, :] >= K).mean(1) >= 0.5) & window[:, None]
    outline = ndimage.binary_closing(ndimage.binary_opening(outline, iterations=2), iterations=2)
    lab, _ = ndimage.label(outline)
    outline = lab == 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    width = outline.sum(0) * mm
    lowest = int(np.argmax(width >= 8.0))
    top = int(np.nonzero(width)[0].max())
    outline[:, :lowest] = outline[:, lowest:lowest + 12].any(1, keepdims=True)
    outline = ndimage.gaussian_filter(outline.astype(np.float32), 1.0) > 0.5
    slot = np.zeros(votes.shape, bool)
    slot[:, (Y > -35) & (Y < 22), :] = outline[:, None, :]
    behind = (votes >= 2) & (Y >= 18)[None, :, None] & window[:, None, None] & (Z < 80)[None, None, :]
    # The outline test is conservative (half the legs' depth must agree); also take free voxels
    # that touch the slot directly, which trims what Poisson added to the legs' inner faces.
    beside = (votes >= 4) & window[:, None, None] & (Z < 80)[None, None, :]
    extra = ndimage.binary_opening(behind | beside, iterations=1)
    lab, _ = ndimage.label(slot | extra)
    slot = np.isin(lab, np.unique(lab[slot]))

    mesh = trimesh.load(OUT / "cod_holder_base_fixed.ply", force="mesh")
    carved = subtract_volume(mesh, slot, np.array([X[0], Y[0], Z[0]]) / 1000 / s, mm / 1000 / s)
    carved.export(OUT / "cod_holder_gap.ply")
    state["gap"] = {"removed_cm3": float((mesh.volume - carved.volume) * (s * 100) ** 3),
                    "crotch_mm": float(Z[top]), "lowest_resolved_mm": float(Z[lowest]),
                    "max_width_mm": float(width.max()), "watertight": bool(carved.is_watertight)}
    print(f"leg gap: carved {state['gap']['removed_cm3']:.1f} cm3, up to {Z[top]:.0f} mm "
          f"(resolved from {Z[lowest]:.0f} mm), widest {width.max():.0f} mm, watertight {carved.is_watertight}")
    _gap_figure(mesh, carved, s, rot, origin, cams, imgs, dense / "images")


def _gap_figure(before, after, s, rot, origin, cams, imgs, image_dir, name="IMG_0001.jpg"):
    """The front photo, with the volume the carve removed tinted: the photos see background there."""
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font

    cam_id, R, t = imgs[name]
    _, w, h, p = cams[cam_id]

    def silhouette(mesh):
        v = mesh.vertices
        low = mesh.faces[v[mesh.faces].mean(1)[:, 2] * s < 0.11]
        pc = (v @ rot + origin) @ R.T + t
        uv = np.c_[p[0] * pc[:, 0] / pc[:, 2] + p[2], p[1] * pc[:, 1] / pc[:, 2] + p[3]]
        im = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(im)
        for f in low:
            d.polygon([tuple(uv[f[0]]), tuple(uv[f[1]]), tuple(uv[f[2]])], fill=255)
        return np.asarray(im) > 0

    sb, sa = silhouette(before), silhouette(after)
    photo = np.asarray(Image.open(image_dir / name).convert("RGB")).astype(float)
    cut = sb & ~sa
    photo[cut] = 0.45 * photo[cut] + 0.55 * np.array([34, 197, 94])
    ys, xs = np.nonzero(sb)
    y0 = max(int(np.nonzero(cut)[0].min()) - 120, 0)  # from just above the carved region down
    crop = Image.fromarray(photo.astype(np.uint8)).crop((xs.min() - 20, y0, xs.max() + 20, ys.max() + 20))
    crop = crop.resize((560, round(crop.height * 560 / crop.width)), Image.LANCZOS)
    out = Image.new("RGB", (crop.width, crop.height + 40), "white")
    out.paste(crop, (0, 40))
    ImageDraw.Draw(out).text((10, 10), "Green: removed from the scan (the photos see through it)",
                             fill=(21, 128, 61), font=_font(18))
    out.save(FIG / "holder_gap.jpg", quality=88)


def stage_assets(state):
    """Physics assets from the scans, scaled to the measured sizes and masses."""
    import shutil

    from scan2usd.pipeline import BuildOptions, build_asset

    jobs = {
        "cod_holder": (OUT / "cod_holder_gap.ply", MEAS["cod_holder"]["height_m"], "z",
                       MEAS["cod_holder"]["mass_kg"]),
        # The merged controller is already metric (face-up scale): no rescale.
        "controller": (OUT / "controller_closed.ply", None, "z", MEAS["controller"]["mass_kg"]),
    }
    state["assets"] = {}
    for name, (mesh, scale_to, axis, mass) in jobs.items():
        out = OUT / "assets" / name
        shutil.rmtree(out, ignore_errors=True)
        opts = BuildOptions(name=name, material="plastic", mass=mass, up_axis="z", units="m", scale_to=scale_to,
                            scale_axis=axis, usda=True, run_sim=True, preview=True, gltf=True,
                            bake_texture=2048, max_hulls=64, hull_threshold=0.03)
        res = build_asset(mesh, out, opts)
        mp = res.summary["mass_properties"]
        state["assets"][name] = {"usd": str(res.usd_path.relative_to(ROOT)).replace("\\", "/"),
                                 "extents_m": res.summary["extents_m"], "hulls": res.summary["collision_hulls"],
                                 "mass_kg": mp.get("mass"), "validation": res.validation.status,
                                 "checks": res.validation.counts}
        print(f"{name}: extents {np.round(res.summary['extents_m'], 4)} m, {res.summary['collision_hulls']} hulls, "
              f"validation {res.validation.status} {res.validation.counts}")


def mesh_points(ply, out, pts, col, depth=None, trim_far=True):
    import trimesh

    if depth is None:
        extent = float(np.ptp(pts, 0).max())
        depth = int(np.clip(np.ceil(np.log2(extent / (1.5 * _median_spacing(pts)))), 6, 11))
    raw = out.with_name(out.stem + "_poisson.ply")
    subprocess.run([str(find_colmap()), "poisson_mesher", "--input_path", str(ply), "--output_path", str(raw),
                    "--PoissonMeshing.depth", str(depth), "--PoissonMeshing.point_weight", "4",
                    "--PoissonMeshing.trim", "0", "--PoissonMeshing.color", "1"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    m = trimesh.load(str(raw), force="mesh", process=True)
    if not trim_far:  # keep Poisson's closed surface (needed for mass and collision)
        bodies = m.split(only_watertight=False)
        m = max(bodies, key=lambda b: len(b.faces)) if len(bodies) > 1 else m
        _, nearest = cKDTree(pts).query(m.vertices)
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.c_[col[nearest], np.full(len(m.vertices), 255)])
        m.export(str(out))
        print(f"meshed {out.name}: {len(m.faces):,} faces, watertight {m.is_watertight}")
        return m
    # Poisson closes a blob over unseen regions: drop surface far from any measured point.
    d, nearest = cKDTree(pts).query(m.vertices)
    far = d > 6 * _median_spacing(pts)
    m.update_faces(~far[m.faces].any(1))
    m.remove_unreferenced_vertices()
    bodies = m.split(only_watertight=False)
    if len(bodies) > 1:
        m = max(bodies, key=lambda b: len(b.faces))
    _, nearest = cKDTree(pts).query(m.vertices)
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.c_[col[nearest], np.full(len(m.vertices), 255)])
    m.export(str(out))
    print(f"meshed {out.name}: {len(m.faces):,} faces, watertight {m.is_watertight}")


def asset_points(name, n=40000):
    """Points sampled on a built asset's render mesh, in asset (metric) coordinates."""
    from scan2usd.usd_io import read_usd

    a = read_usd(ROOT / f"output/holder_fit/assets/{name}/{name}.usda")
    return np.asarray(a.render_mesh.sample(n, return_index=False, seed=0)), a  # seeded: reproducible numbers


def stage_truth(state):
    """Where does the controller really sit? Register both assets into the combined scan (metric)."""
    from scipy.spatial.transform import Rotation

    scene = load("holder_with_controller")[0] * state["scale"]["holder_with_controller"]
    hp, _ = asset_points("cod_holder", 60000)
    cp, _ = asset_points("controller")
    tau = 0.002
    # Holder: both stand on the table (z = 0, Z-up), so search yaw only; scale is metric on both sides.
    # Only the figure above its base is used: the base is a fitted disc in the asset but a pitted,
    # paper-fringed scan in the combined capture, and a disc has no yaw to offer anyway.
    above = 0.025
    h_src = voxel_down(hp[hp[:, 2] > above], 0.002, 12000)
    h_dst = voxel_down(scene[scene[:, 2] > above], 0.0015, 60000)
    h_fit = global_register(h_src, h_dst, scale=1.0, rotations=candidate_rotations(72, yaw_only=True), tau=tau,
                            trim=0.6, with_scale=False, keep=8,
                            anchor_src=np.array([*hp[:, :2].mean(0), 0.0]),
                            anchor_dst=np.array([*scene[:, :2].mean(0), 0.0]))
    # Scene scale check: free scale refinement from the holder fit.
    h_free = similarity_icp(h_src, h_dst, h_fit.transform, iters=60, trim=0.6, tau=tau)
    # Controller: the scene points the holder doesn't explain, then a full rotation search.
    d, _ = cKDTree(h_fit.transform.apply(hp)).query(scene)
    rest = scene[d > 0.004]
    c_fit = global_register(voxel_down(cp, 0.002, 10000), voxel_down(rest, 0.0015, 40000), scale=1.0,
                            rotations=candidate_rotations(720, seed=3), tau=tau, trim=0.6, with_scale=False)
    # Controller pose in the holder's frame: T_hc = T_h^-1 o T_c
    h_inv = h_fit.transform.inverse()
    rel = c_fit.transform.then(h_inv)
    state["truth"] = {
        "holder": {"rmse_mm": h_fit.rmse * 1000, "inliers_2mm": h_fit.inlier_fraction,
                   "free_scale": h_free.transform.s},
        "controller": {"rmse_mm": c_fit.rmse * 1000, "inliers_2mm": c_fit.inlier_fraction},
        "controller_in_holder": {"R": rel.R.tolist(), "t": rel.t.tolist(),
                                 "euler_xyz_deg": Rotation.from_matrix(rel.R).as_euler("xyz", degrees=True).tolist()},
        "scene_points_left_by_holder": int(len(rest)),
    }
    print(f"holder in scene: RMSE {h_fit.rmse * 1000:.2f} mm, {h_fit.inlier_fraction * 100:.0f}% within 2 mm; "
          f"free-scale check s = {h_free.transform.s:.4f}")
    print(f"controller in scene: RMSE {c_fit.rmse * 1000:.2f} mm, {c_fit.inlier_fraction * 100:.0f}% within 2 mm "
          f"(of {len(rest):,} scene points left after the holder)")
    print("controller in holder frame: t (mm)", np.round(rel.t * 1000, 1), "euler xyz (deg)",
          np.round(state["truth"]["controller_in_holder"]["euler_xyz_deg"], 1))
    state["truth"]["note"] = "transforms map asset coordinates into the combined scan (metric, Z-up, table z=0)"
    np.save(OUT / "truth_transforms.npy", np.stack([h_fit.transform.matrix(), c_fit.transform.matrix()]))


def stage_sim(state, trials: int = 12):
    """Drop the scanned controller into the scanned holder (MuJoCo) and compare with the real pose."""
    import mujoco
    from scipy.spatial.transform import Rotation

    from scan2usd.mjcf import asset_body

    rel = state["truth"]["controller_in_holder"]
    R_true, t_true = np.array(rel["R"]), np.array(rel["t"])
    cp, c_asset = asset_points("controller", 4000)
    _, h_asset = asset_points("cod_holder", 10)
    true_pts = cp @ R_true.T + t_true
    rng = np.random.default_rng(0)
    starts = [(np.zeros(3), 0.0)] + [(np.r_[rng.uniform(-0.008, 0.008, 2), 0.0], rng.uniform(-8, 8))
                                     for _ in range(trials - 1)]
    results = {}
    for variant, coll in (("decomposed", "hulls"), ("one hull", "single")):
        hold = asset_body(h_asset, "holder", collision=coll, free=False)
        rows = []
        for k, (dxy, dyaw) in enumerate(starts):
            Rz = Rotation.from_euler("z", dyaw, degrees=True).as_matrix()
            R0 = Rz @ R_true
            t0 = t_true + dxy + np.array([0, 0, 0.03])  # 3 cm above the real resting pose
            q0 = Rotation.from_matrix(R0).as_quat()  # x, y, z, w
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
            R1 = data.xmat[b].reshape(3, 3)
            t1 = data.xpos[b].copy()
            sim_pts = cp @ R1.T + t1
            add = float(np.linalg.norm(sim_pts - true_pts, axis=1).mean())
            start_pts = cp @ R0.T + t0
            rows.append({"start": k, "add_mm": add * 1000, "start_add_mm": float(np.linalg.norm(start_pts - true_pts, axis=1).mean() * 1000),
                         "rot_deg": rotation_angle_deg(R1, R_true), "start_rot_deg": rotation_angle_deg(R0, R_true),
                         "trans_mm": float(np.linalg.norm(t1 - t_true) * 1000),
                         "min_z_mm": float(sim_pts[:, 2].min() * 1000), "speed": float(np.linalg.norm(data.qvel[:3]))})
        add = np.array([r["add_mm"] for r in rows])
        rot = np.array([r["rot_deg"] for r in rows])
        held = np.array([r["min_z_mm"] > 5 for r in rows])  # not fallen to the table
        results[variant] = {"trials": rows, "add_mm_median": float(np.median(add)), "rot_deg_median": float(np.median(rot)),
                            "start_add_mm_median": float(np.median([r["start_add_mm"] for r in rows])),
                            "held_in_holder": int(held.sum()), "of": len(rows)}
        print(f"{variant:10s}: held {held.sum()}/{len(rows)}, ADD median {np.median(add):.1f} mm "
              f"(p90 {np.percentile(add, 90):.1f}; at release {results[variant]['start_add_mm_median']:.1f}), "
              f"rotation median {np.median(rot):.1f} deg")
    state["sim"] = {k: {kk: vv for kk, vv in v.items() if kk != "trials"} for k, v in results.items()}
    (OUT / "sim_trials.json").write_text(json.dumps(results, indent=2))


def stage_video(state):
    """Side-by-side drop animation (decomposed vs one-hull holder) with a ghost of the real pose."""
    import mujoco
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation

    from scan2usd.mjcf import asset_body
    from scan2usd.preview import _font

    rel = state["truth"]["controller_in_holder"]
    R_true, t_true = np.array(rel["R"]), np.array(rel["t"])
    _, c_asset = asset_points("controller", 10)
    _, h_asset = asset_points("cod_holder", 10)
    q = Rotation.from_matrix(R_true).as_quat()
    qt = (q[3], *q[:3])
    normal = R_true @ np.array([0, 0, 1.0])
    az = float(np.degrees(np.arctan2(-normal[1], -normal[0]))) + 35
    clips = []
    for variant, coll in (("convex decomposition (64 hulls)", "hulls"), ("one convex hull", "single")):
        hold = asset_body(h_asset, "holder", collision=coll, free=False, rgba=(0.32, 0.32, 0.36, 1), visual_faces=60000)
        ctrl = asset_body(c_asset, "ctrl", pos=t_true + np.array([0, 0, 0.03]), quat=qt, collision="hulls",
                          rgba=(0.92, 0.92, 0.9, 1), visual_faces=60000)
        ghost = (f'<body name="ghost" pos="{" ".join(map(str, t_true))}" quat="{" ".join(map(str, qt))}">'
                 f'<geom type="mesh" mesh="ctrl_visual" contype="0" conaffinity="0" rgba="0.1 0.8 0.35 0.35"/></body>')
        xml = f"""<mujoco><option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
<visual><global offwidth="960" offheight="720"/><quality shadowsize="4096"/>
<headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/></visual>
<asset>{hold.assets_xml}{ctrl.assets_xml}
<texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.82 0.86 0.92" width="256" height="256"/>
<texture name="grid" type="2d" builtin="checker" rgb1=".86 .86 .84" rgb2=".78 .78 .76" width="512" height="512"/>
<material name="grid" texture="grid" texrepeat="8 8"/></asset>
<worldbody><light pos="0.3 -0.4 0.8" dir="-0.3 0.4 -1" castshadow="true"/>
<geom type="plane" size="1 1 0.05" material="grid" friction="0.6 0.005 0.0001"/>
{hold.body_xml}{ctrl.body_xml}{ghost}</worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml, {**hold.files, **ctrl.files})
        data = mujoco.MjData(model)
        r = mujoco.Renderer(model, 480, 640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (t_true[0], t_true[1], 0.09)
        cam.distance, cam.elevation, cam.azimuth = 0.42, -15, az
        frames = []
        for i in range(int(2.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            if i % 17 == 0:
                r.update_scene(data, camera=cam)
                img = Image.fromarray(r.render())
                d = ImageDraw.Draw(img)
                d.rectangle([0, 0, 640, 30], fill="white")
                d.text((10, 5), f"holder collision: {variant}", fill=(30, 41, 59), font=_font(17))
                frames.append(img)
        r.close()
        clips.append(frames)
    n = min(len(c) for c in clips)
    sheets = []
    for i in range(n):
        s = Image.new("RGB", (1280, 480), "white")
        for k, c in enumerate(clips):
            s.paste(c[i], (k * 640, 0))
        if i > n * 0.75:
            dd = ImageDraw.Draw(s)
            dd.text((12, 446), "green ghost = where it really rests (from the combined scan)", fill=(21, 128, 61), font=_font(16))
        sheets.append(s)
    FIG.mkdir(parents=True, exist_ok=True)
    sheets[0].save(FIG / "holder_fit.webp", save_all=True, append_images=sheets[1:] + [sheets[-1]] * 20, duration=50,
                   loop=0, quality=65, method=4)
    sheets[-1].save(FIG / "holder_fit.jpg", quality=88)
    print(f"wrote holder_fit.webp ({n} frames)")


def stage_figure(state):
    """Results strip: per-trial settle error for both holder collision variants."""
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font

    tr = json.loads((OUT / "sim_trials.json").read_text())
    img = Image.new("RGB", (1280, 330), "white")
    d = ImageDraw.Draw(img)
    d.text((16, 10), "Real capture → simulation → back to reality: does the scanned controller sit where it really sits?",
           fill=(30, 41, 59), font=_font(20))
    d.text((16, 40), "12 drops from 3 cm above the real pose (±8 mm, ±8° jitter). Error: mean distance of the controller's "
           "surface points from its pose in the combined scan (ADD).", fill=(100, 116, 139), font=_font(14))
    x0, y0, bw, add_max = 300, 80, 820, 40
    for row, (name, lab, colr) in enumerate((("decomposed", "holder: convex decomposition (64 hulls)", (37, 99, 235)),
                                             ("one hull", "holder: one convex hull", (234, 88, 12)))):
        y = y0 + row * 90
        d.text((16, y + 10), lab, fill=(51, 65, 85), font=_font(15))
        d.line([x0, y + 60, x0 + bw, y + 60], fill=(203, 213, 225))
        for v in range(0, add_max + 1, 10):
            xx = x0 + bw * v / add_max
            d.line([xx, y + 56, xx, y + 64], fill=(148, 163, 184))
            d.text((xx - 8, y + 66), f"{v} mm", fill=(148, 163, 184), font=_font(11))
        for t in tr[name]["trials"]:
            v = t["add_mm"]
            xx, yy = x0 + bw * min(v, add_max) / add_max, y + 30 + (t["start"] % 4) * 5
            d.ellipse([xx - 5, yy - 5, xx + 5, yy + 5], fill=colr if v <= add_max else None, outline=colr)
        s = state["sim"][name]
        txt = (f"held {s['held_in_holder']}/{s['of']} · median {s['add_mm_median']:.1f} mm, {s['rot_deg_median']:.0f}°"
               if s["held_in_holder"] else f"held 0/{s['of']}: slides off and falls (median {s['add_mm_median']:.0f} mm away) →")
        d.text((x0, y + 4), txt, fill=colr, font=_font(15))
    t = state["truth"]
    d.text((16, 270), f"Scans: holder 81 photos, controller 78 + 38 photos (two passes merged, {state['merge']['rmse_mm']:.1f} mm RMSE), "
           f"combined 58 photos. The holder fits the combined scan at {t['holder']['rmse_mm']:.1f} mm RMSE, the controller at "
           f"{t['controller']['rmse_mm']:.1f} mm;", fill=(71, 85, 105), font=_font(13))
    d.text((16, 290), f"independent scale estimates agree within {abs(t['holder']['free_scale'] - 1) * 100:.1f}%. The reference pose "
           "itself is good to a few mm (dark, glossy figure).", fill=(71, 85, 105), font=_font(13))
    img.save(FIG / "holder_fit_results.png")


def stage_lead(state, photo: str = "IMG_0024.jpg"):
    """Lead strip for the docs: the real pair (one source photo, cropped) -> each scan alone -> the drop."""
    import trimesh
    from PIL import Image, ImageDraw

    from scan2usd.preview import _font, render

    S = 420
    im = Image.open(ROOT / "data" / "raw" / "holder_with_controller" / photo).convert("RGB")
    side, cx, top = 2750, im.width // 2, 360  # tight on the pair; drops most of the floor and flyers
    photo_img = im.crop((cx - side // 2, top, cx + side // 2, top + side)).resize((S, S), Image.LANCZOS)

    def scan(path, az, el):
        full = trimesh.load(path, force="mesh")
        m = full.simplify_quadric_decimation(face_count=250_000) if len(full.faces) > 250_000 else full.copy()
        _, idx = cKDTree(full.vertices).query(m.vertices)
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.asarray(full.visual.vertex_colors)[idx])
        return render([m], size=S, azimuth=az, elevation=el)

    holder = scan(OUT / "cod_holder_gap.ply", -90, 12)
    ctrl = scan(OUT / "controller_merged.ply", -90, 55)
    sim = Image.open(FIG / "holder_fit.jpg").convert("RGB").crop((95, 31, 545, 431)).resize((S, int(S * 400 / 450)))
    simp = Image.new("RGB", (S, S), "white")
    simp.paste(sim, (0, 0))
    ImageDraw.Draw(simp).text((10, S - 36), "green: where it really rests (photo 1)", fill=(21, 128, 61), font=_font(17))
    add = state["sim"]["decomposed"]["add_mm_median"]
    panels = [(photo_img, "1 - The real pair", "iPhone photo (58 taken)"),
              (holder, "2 - Holder, scanned alone", "81 photos to 3D"),
              (ctrl, "3 - Controller, scanned alone", "78 + 38 photos, two passes"),
              (simp, "4 - Simulated drop (MuJoCo)", f"settles {add:.1f} mm from the real pose")]
    gap, top_h = 50, 70
    out = Image.new("RGB", (4 * S + 3 * gap + 40, S + top_h + 20), "white")
    d = ImageDraw.Draw(out)
    for k, (img, t, sub) in enumerate(panels):
        x = 20 + k * (S + gap)
        d.text((x, 8), t, fill=(30, 41, 59), font=_font(24))
        d.text((x, 40), sub, fill=(100, 116, 139), font=_font(17))
        out.paste(img, (x, top_h))
        d.rectangle([x, top_h, x + S - 1, top_h + S - 1], outline=(226, 232, 240))
        if k < 3:
            ax, ay = x + S + 10, top_h + S // 2
            d.polygon([(ax, ay - 12), (ax + 28, ay), (ax, ay + 12)], fill=(37, 99, 235))
    out.save(FIG / "holder_fit_lead.jpg", quality=88)  # re-encoded: no EXIF / GPS
    print("wrote holder_fit_lead.jpg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="prep,scale,merge,base,gap,assets,truth,sim,video,figure,lead")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for st in args.stage.split(","):
        {"prep": stage_prep, "scale": stage_scale, "merge": stage_merge, "base": stage_base, "gap": stage_gap,
         "assets": stage_assets, "truth": stage_truth,
         "sim": stage_sim, "video": stage_video, "figure": stage_figure, "lead": stage_lead}[st](state)
        state_path.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
