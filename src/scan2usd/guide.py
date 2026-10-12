"""Physics-aware capture guidance: which extra photos would fix what matters for simulation?

Generic capture apps guide for visual coverage. Here the target is the asset's *uncertainty*
(scan2usd.uncertainty): surface regions that were sealed or only inferred, weighted by how much they
matter physically (a sealed cup opening changes collision, mass and grasps; the base the object stood
on cannot be photographed anyway and is ignored). Candidate phone viewpoints are scored by how well
they would see those regions (visibility via a point-splat z-buffer, viewing angle), and chosen
greedily so that each region gets the two good views photogrammetry needs to triangulate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .uncertainty import BASE, INFERRED, SEALED, ObservationField


@dataclass
class View:
    center: np.ndarray  # camera position (asset frame, m)
    forward: np.ndarray  # unit viewing direction
    name: str = ""


@dataclass
class Suggestion:
    view: View
    gain: float  # weighted, newly covered uncertain surface (fraction of the total)
    azimuth_deg: float
    elevation_deg: float
    distance_m: float


def target_weights(field: ObservationField, mesh, grasp_points: np.ndarray | None = None, grasp_radius: float = 0.03):
    """Per-vertex importance: how uncertain (1 - confidence), how physically relevant (sealed > inferred,
    base ignored), boosted near grasp locations stored in the asset."""
    w = 1.0 - field.confidence
    w = w * np.where(field.labels == SEALED, 1.0, np.where(field.labels == INFERRED, 0.5, 0.0))
    w[field.labels == BASE] = 0.0
    if grasp_points is not None and len(grasp_points):
        from scipy.spatial import cKDTree

        d, _ = cKDTree(grasp_points).query(mesh.vertices)
        w = w * np.where(d < grasp_radius, 2.0, 1.0)
    return w


def view_quality(view: View, surface: np.ndarray, targets: np.ndarray, normals: np.ndarray, fov_deg: float = 65.0,
                 res: tuple[int, int] = (200, 150), tol: float = 0.004, max_incidence_deg: float = 70.0) -> np.ndarray:
    """0..1 per target: visible from `view` (not occluded by `surface` samples) and facing it."""
    f = view.forward / np.linalg.norm(view.forward)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:  # looking straight down
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    w, h = res
    focal = (w / 2) / np.tan(np.radians(fov_deg) / 2)

    def project(p):
        d = p - view.center
        z = d @ f
        z_safe = np.where(z > 1e-6, z, 1e-6)
        px = np.floor((d @ r) / z_safe * focal + w / 2).astype(int)
        py = np.floor(-(d @ u) / z_safe * focal + h / 2).astype(int)
        ok = (z > 1e-3) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        return z, px, py, ok

    z, px, py, ok = project(surface)
    zbuf = np.full((h, w), np.inf)
    np.minimum.at(zbuf, (py[ok], px[ok]), z[ok])
    zbuf = ndimage.minimum_filter(zbuf, size=3)  # close holes between splatted points
    zt, tx, ty, tok = project(targets)
    vis = np.zeros(len(targets), bool)
    vis[tok] = zt[tok] <= zbuf[ty[tok], tx[tok]] + tol
    to_cam = view.center - targets
    cosang = (normals * to_cam).sum(1) / np.maximum(np.linalg.norm(to_cam, axis=1), 1e-9)
    c0 = np.cos(np.radians(max_incidence_deg))
    return vis * np.clip((cosang - c0) / (1 - c0), 0.0, 1.0)


def hemisphere_views(center: np.ndarray, distance: float, elevations=range(10, 90, 10), azimuth_step: int = 20) -> list[View]:
    """Phone positions a person can reach: around the object at `distance`, aimed at its center."""
    views = []
    for el in elevations:
        for az in range(0, 360, azimuth_step):
            a, e = np.radians(az), np.radians(el)
            d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            views.append(View(center + distance * d, -d, f"az{az:03d}_el{el:02d}"))
    return views


def capture_views(workspace, usd_path) -> list[View]:
    """The photos actually taken (COLMAP's solved cameras), in the asset's frame.

    Same chain as the build: COLMAP world -> table-aligned (rot, origin) -> scaled and recentered.
    """
    import json
    from pathlib import Path

    import trimesh

    from .chart import read_colmap_model
    from .reconstruct import _median_spacing, isolate_object, read_ply_points
    from .usd_io import read_usd

    workspace, usd_path = Path(workspace), Path(usd_path)
    dense = workspace / "dense" / "0"
    ply = read_ply_points(dense / "fused.ply")
    xyz = np.c_[ply["x"], ply["y"], ply["z"]].astype(np.float64)
    sp = _median_spacing(xyz)
    _, _, rot, origin, _ = isolate_object(xyz, sp, 2.0 * sp)
    s = float(json.loads((usd_path.parent / "build_report.json").read_text())["scale_applied"])
    recon = trimesh.load(workspace / "object_mesh.ply", force="mesh")
    t = read_usd(usd_path).render_mesh.bounds[0] - s * recon.bounds[0]
    _, images = read_colmap_model(dense / "sparse")
    views = []
    for name, (_, r, tc) in images.items():
        r = np.asarray(r)
        c_w = -r.T @ np.asarray(tc)
        views.append(View(s * rot @ (c_w - origin) + t, rot @ (r.T @ np.array([0.0, 0.0, 1.0])), name))
    return views


def heatmap_colors(field: ObservationField) -> np.ndarray:
    """RGBA per vertex: observed gray, inferred amber, sealed red, base dark."""
    c = np.tile(np.array([200, 200, 205, 255], np.uint8), (len(field.labels), 1))
    c[field.labels == INFERRED] = (245, 158, 11, 255)
    c[field.labels == SEALED] = (220, 38, 38, 255)
    c[field.labels == BASE] = (120, 120, 120, 255)
    return c


def plan_figure(mesh, existing: list[View], suggestions: list[Suggestion], size: int = 520):
    """Top-down 'where to stand' sketch: object footprint, photos taken (gray), suggested (numbered)."""
    from PIL import Image, ImageDraw

    from .preview import _font

    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    pts = [v.center[:2] for v in existing] + [s.view.center[:2] for s in suggestions] + list(mesh.vertices[:, :2])
    lo, hi = np.min(pts, 0), np.max(pts, 0)
    c, half = (lo + hi) / 2, max(hi - lo) / 2 * 1.12

    def xy(p):
        q = (np.asarray(p[:2]) - c) / half
        return size / 2 + q[0] * size / 2, size / 2 - q[1] * size / 2

    hull = mesh.convex_hull.vertices[:, :2]
    from scipy.spatial import ConvexHull

    ring = hull[ConvexHull(hull).vertices]
    d.polygon([xy(p) for p in ring], fill=(226, 232, 240), outline=(100, 116, 139))
    for v in existing:
        x, y = xy(v.center)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(148, 163, 184))
    for i, s in enumerate(suggestions):
        x, y = xy(s.view.center)
        d.ellipse([x - 13, y - 13, x + 13, y + 13], fill=(220, 38, 38))
        d.text((x - 5, y - 10), str(i + 1), fill="white", font=_font(16))
    d.text((10, 8), "top view: gray = photos taken, red = take next", fill=(51, 65, 85), font=_font(15))
    return img


def run_guide(usd_path, workspace, out_dir, k: int = 8, candidates: list[View] | None = None):
    """Analyze a scan and write a shot list (index.html + images) to `out_dir`. Returns suggestions, stats."""
    import copy
    from pathlib import Path

    import trimesh

    from .preview import render
    from .schemas import read_grasps
    from .uncertainty import observation_field, workspace_points
    from .usd_io import read_usd

    import json

    usd_path, workspace, out_dir = Path(usd_path), Path(workspace), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    asset = read_usd(usd_path)
    mesh = asset.render_mesh
    s = float(json.loads((usd_path.parent / "build_report.json").read_text())["scale_applied"])
    pts, _ = workspace_points(asset, workspace, s)
    field = observation_field(mesh, pts)
    existing = capture_views(workspace, usd_path)
    if candidates is None:
        lo, hi = mesh.bounds
        center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + 0.4 * (hi[2] - lo[2])])
        dist = float(np.median([np.linalg.norm(v.center - center) for v in existing]))
        candidates = hemisphere_views(center, dist)
    grasps = read_grasps(usd_path)
    gp = np.array([g.position for g in grasps]) if grasps else None
    sugg, stats = suggest_views(mesh, field, candidates, existing, k=k, grasp_points=gp)
    heat = copy.deepcopy(mesh)
    heat.visual = trimesh.visual.ColorVisuals(heat, vertex_colors=heatmap_colors(field))
    thumbs = []
    for i, sg in enumerate(sugg):
        name = f"view_{i + 1}.jpg"
        render([heat], size=360, azimuth=sg.azimuth_deg, elevation=sg.elevation_deg).save(out_dir / name, quality=88)
        thumbs.append(name)
    plan_figure(mesh, existing, sugg).save(out_dir / "plan.png")
    first = existing[0].center if existing else None
    first_az = None
    if first is not None:
        lo, hi = mesh.bounds
        rel = first - (lo + hi) / 2
        first_az = float(np.degrees(np.arctan2(rel[1], rel[0])))
    (out_dir / "index.html").write_text(shot_list_html(asset.name, sugg, thumbs, "plan.png", stats, first_az),
                                         encoding="utf-8")
    return sugg, {**stats, "observed_fraction": field.observed_fraction, "existing_photos": len(existing)}


def suggest_views(mesh, field: ObservationField, candidates: list[View], existing: list[View] | None = None, k: int = 8,
                  grasp_points: np.ndarray | None = None, samples: int = 60000, max_targets: int = 5000, seed: int = 0,
                  existing_credit: float = 0.5) -> tuple[list[Suggestion], dict]:
    """Greedy choice of `k` candidate views covering the asset's physically relevant uncertainty."""
    rng = np.random.default_rng(seed)
    surface = np.asarray(mesh.sample(samples, seed=seed))
    w_all = target_weights(field, mesh, grasp_points)
    idx = np.nonzero(w_all > 0)[0]
    if len(idx) == 0:
        return [], {"total_weight": 0.0, "covered_fraction": 1.0}
    if len(idx) > max_targets:
        idx = rng.choice(idx, max_targets, replace=False, p=w_all[idx] / w_all[idx].sum())
    targets, normals, w = mesh.vertices[idx], np.asarray(mesh.vertex_normals)[idx], w_all[idx]
    need = np.full(len(idx), 2.0)
    for v in existing or []:  # partial credit: these views already saw the region and it still failed
        need -= existing_credit * view_quality(v, surface, targets, normals)
    need = np.clip(need, 0.0, 2.0)
    total = float((w * need).sum())
    q = np.array([view_quality(v, surface, targets, normals) for v in candidates])
    chosen, out = set(), []
    lo, hi = mesh.bounds
    obj_center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + 0.4 * (hi[2] - lo[2])])
    for _ in range(min(k, len(candidates))):
        gains = (w[None] * np.minimum(q, need[None])).sum(1)
        gains[list(chosen)] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        chosen.add(best)
        need = np.clip(need - q[best], 0.0, 2.0)
        v = candidates[best]
        rel = v.center - obj_center
        out.append(Suggestion(v, float(gains[best] / max(total, 1e-12)), float(np.degrees(np.arctan2(rel[1], rel[0]))),
                              float(np.degrees(np.arcsin(rel[2] / np.linalg.norm(rel)))), float(np.linalg.norm(rel))))
    stats = {"total_weight": total, "covered_fraction": float(1 - (w * need).sum() / max(total, 1e-12)),
             "targets": int(len(idx))}
    return out, stats


def shot_list_html(asset_name: str, suggestions: list[Suggestion], thumbs: list[str], plan_png: str, stats: dict,
                   first_azimuth_deg: float | None = None) -> str:
    """A phone-friendly page: one card per suggested photo."""
    cards = []
    for i, (s, t) in enumerate(zip(suggestions, thumbs)):
        if first_azimuth_deg is not None:
            turn = (s.azimuth_deg - first_azimuth_deg + 540) % 360 - 180
            where = f"walk {abs(turn):.0f}° {'counter-clockwise' if turn > 0 else 'clockwise'} from your first photo"
        else:
            where = f"azimuth {s.azimuth_deg:.0f}°"
        height = ("high, looking down" if s.elevation_deg > 55 else "chest height, looking down" if s.elevation_deg > 30
                  else "low, almost level")
        cards.append(f'<div class="card"><img src="{t}"><div><b>Photo {i + 1}</b> · {height} '
                     f'({s.elevation_deg:.0f}°), {s.distance_m * 100:.0f} cm away<br>{where}<br>'
                     f'<span class="muted">fixes {s.gain * 100:.0f}% of the remaining physics-relevant uncertainty</span></div></div>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{asset_name}: extra photos</title><style>
body{{font:15px system-ui;margin:16px;color:#1e293b;max-width:760px}} .card{{display:flex;gap:12px;align-items:center;
border:1px solid #e2e8f0;border-radius:10px;padding:8px;margin:8px 0}} .card img{{width:180px;border-radius:6px}}
.muted{{color:#64748b;font-size:13px}} img.plan{{width:100%;max-width:420px}}</style></head><body>
<h2>{asset_name}: take these {len(suggestions)} extra photos</h2>
<p class="muted">Red areas were never seen by the camera (filled in by guesswork); they change the object's collision shape,
mass or grasps. These views cover {stats.get('covered_fraction', 0) * 100:.0f}% of that. Keep the object and lights where they were.</p>
<img class="plan" src="{plan_png}">{''.join(cards)}</body></html>"""
