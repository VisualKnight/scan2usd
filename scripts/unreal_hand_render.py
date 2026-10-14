"""The Shadow Hand placement rendered in Unreal Engine 5 with Lumen + MegaLights.

    python scripts/unreal_render.py --stages project      # once: the UE project (see unreal_render.py)
    python scripts/unreal_hand_render.py [--stages layout,build,preview,video] [--ue ...] [--project ...]

Stages:
  layout   runs the re-aimed MuJoCo placement (demo_hand_place.py, seed 0) and records, at 30 fps, the
           pose of every hand link and of the controller, and the camera of the MuJoCo cut
           (hand_motion.shot_keys). Writes output/unreal/hand_layout.usda: the hand's visual meshes (from
           the compiled Menagerie model, exact normals), the scanned holder and controller (the same USD
           assets), the tray, a wood table, ~100 shadow-casting fairy lights and a key light; and
           hand_anim.json, the per-frame tracks (Unreal units and handedness).
  build    UnrealEditor-Cmd imports the stage into /Game/Maps/HandPlace and keys every frame of the
           tracks in a level sequence (scripts/unreal/ue_build.py).
  preview  five stills from across the sequence (Movie Render Queue, Lumen + MegaLights).
  video    the whole sequence -> docs/figures/unreal_hand_place.mp4 and a poster frame.

The motion is MuJoCo's, frame for frame: Unreal renders it, it doesn't simulate it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from pxr import Gf, Kind, Usd, UsdGeom, UsdLux, UsdShade, Vt
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_hand_place as H  # noqa: E402
import hand_motion  # noqa: E402
import unreal_layout as L  # noqa: E402
from unreal_render import OUT, SETTINGS, UE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
HERE = Path(__file__).resolve().parent / "unreal"
FPS = 30
S = np.diag([1.0, -1.0, 1.0])  # USD/MuJoCo (right-handed) <-> Unreal (left-handed): Y flips
# Real Shadow Hand finish: satin black shells, brushed aluminium knuckles, dark anodized forearm.
HAND_LOOKS = {"black": ((0.025, 0.025, 0.028), 0.5, 0.0), "metallic": ((0.80, 0.80, 0.82), 0.32, 1.0),
              "gray": ((0.10, 0.105, 0.115), 0.38, 0.6)}
TABLE_STRAND = [(0.62, -0.28), (0.48, -0.18), (0.40, -0.02), (0.42, 0.16), (0.30, 0.31), (0.12, 0.40),
                (-0.07, 0.42), (-0.21, 0.33), (-0.29, 0.16), (-0.24, -0.01), (-0.10, -0.09), (0.05, -0.13),
                (0.18, -0.24)]
CAMERA_LOOK = {"focal_mm": round(hand_motion.focal_mm(20.25), 2), "fstop": 4.0}  # the shared lens, 36 x 20.25 mm back
HAND_SETTINGS = dict(SETTINGS, exposure_ev=-4.1, key_lumens=600.0, hdri_intensity=1.1)


def record():
    """Re-aimed run, seed 0: per frame, every hand link and the controller (world 4x4) and the camera."""
    ctrl, holder, R_rel, t_rel = H.load()
    su = H.Setup(0, ctrl, holder, R_rel, t_rel, "hulls", H.span_poses(), correct=True, work_tag="unreal")
    m = su.model
    wps = H.path(su)
    shots = [(t0, hand_motion.camera_track(k))
             for t0, k in hand_motion.shot_keys(su.T_start[:3, 3], su.T_target[:3, 3], wps)]
    bodies = [m.body(b).name for b in range(m.nbody) if m.body(b).name.startswith("rh_")] + ["ctrl"]
    frames = []

    def rec(d):
        poses = {n: H.body_pose(m, d, n) for n in bodies}
        eye, look = hand_motion.orbit_eye(hand_motion.camera_at(shots, d.time))
        return d.time, poses, eye, look

    frames = H.execute(su, wps, render=rec, fps=FPS, settle=3.5)
    res = H.score(ctrl, m, su.data, su.T_target)
    return su, frames, res, [t0 for t0, _ in shots[1:]]


def solid_material(stage, path: str, color, roughness: float, metallic: float = 0.0):
    """A flat-color look built like the textured wood (base color, normal and roughness textures, here
    4x4 solid PNGs): the same UsdPreviewSurface shader permutation in Unreal, so it is already compiled
    when Movie Render Queue starts. Untextured looks compile new permutations, and frames rendered
    before they finish show the default material."""
    name = path.rsplit("/", 1)[-1].lower()
    tex = OUT / "textures"
    tex.mkdir(parents=True, exist_ok=True)
    c = np.asarray(color, float)
    srgb = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    Image.new("RGB", (4, 4), tuple(int(round(255 * v)) for v in srgb)).save(tex / f"solid_{name}_diff.png")
    Image.new("L", (4, 4), int(round(255 * roughness))).save(tex / f"solid_{name}_rough.png")
    Image.new("RGB", (4, 4), (128, 128, 255)).save(tex / "solid_flat_nor_gl.png")
    return L.preview_material(stage, path, (1, 1, 1), roughness, metallic=metallic, textures={
        "diffuse": (f"./textures/solid_{name}_diff.png", "diffuseColor", "color"),
        "normal": ("./textures/solid_flat_nor_gl.png", "normal", "normal"),
        "rough": (f"./textures/solid_{name}_rough.png", "roughness", "scalar")})


def hand_meshes(stage, m, frame0):
    """One Xform per hand link (posed at frame 0) with its visual meshes in link coordinates."""
    looks = {k: solid_material(stage, f"/World/Looks/hand_{k}", *v) for k, v in HAND_LOOKS.items()}
    Usd.ModelAPI(UsdGeom.Xform.Define(stage, "/World/hand").GetPrim()).SetKind(Kind.Tokens.group)
    n = 0
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g] or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body = m.body(m.geom_bodyid[g]).name
        if not body.startswith("rh_"):
            continue
        xf = f"/World/hand/{body}"
        if not stage.GetPrimAtPath(xf):
            link = UsdGeom.Xform.Define(stage, xf)
            link.AddTransformOp().Set(Gf.Matrix4d(*frame0[body].T.flatten().tolist()))
            # A component per link: Unreal's importer makes one actor of each (kind-less prims are
            # collapsed into their parent's actor), so each link can be animated.
            Usd.ModelAPI(link.GetPrim()).SetKind(Kind.Tokens.component)
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        na, nn = m.mesh_normaladr[mid], m.mesh_normalnum[mid]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        R = R.reshape(3, 3)
        v = m.mesh_vert[va:va + vn] @ R.T + m.geom_pos[g]
        faces = m.mesh_face[fa:fa + fn]
        normals = (m.mesh_normal[na:na + nn] @ R.T)[m.mesh_facenormal[fa:fa + fn]].reshape(-1, 3)
        mesh = UsdGeom.Mesh.Define(stage, f"{xf}/geom_{g}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        mesh.CreateSubdivisionSchemeAttr("none")
        mat = m.material(m.geom_matid[g]).name if m.geom_matid[g] >= 0 else "black"
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(looks.get(mat, looks["black"]))
        n += 1
    return n


def ue_pose(T) -> list[float]:
    """World 4x4 (meters, right-handed) -> Unreal location (cm) + quaternion (x, y, z, w)."""
    q = Rotation.from_matrix(S @ T[:3, :3] @ S).as_quat()
    p = 100.0 * (S @ T[:3, 3])
    return [*p.tolist(), *q.tolist()]


def stage_layout() -> dict:
    su, frames, res, cuts = record()
    m, d = su.model, su.data
    print(f"recorded {len(frames)} frames ({len(frames) / FPS:.1f} s); placed {res['add_mm']:.1f} mm / "
          f"{res['rot_deg']:.0f} deg from the real pose", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "hand_layout.usda"
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    Usd.ModelAPI(world.GetPrim()).SetKind(Kind.Tokens.assembly)
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().documentation = "Shadow Hand placement for the Unreal render (scripts/unreal_hand_render.py)"

    L.TABLE, L.TABLE_CENTER = (2.6, 1.8, 0.035), (0.10, 0.15)  # edges out of every shot
    L.table(stage)
    frame0 = frames[0][1]
    n_geoms = hand_meshes(stage, m, frame0)

    hold = UsdGeom.Xform.Define(stage, "/World/cod_holder")
    hold.GetPrim().GetReferences().AddReference(str(L.asset("cod_holder")).replace("\\", "/"))
    hold.AddTransformOp().Set(Gf.Matrix4d(*H.body_pose(m, d, "holder").T.flatten().tolist()))
    L.static(hold.GetPrim())
    for k, v in L.LOOK_OVERRIDES["cod_holder"].items():
        UsdShade.Shader(stage.GetPrimAtPath("/World/cod_holder/Looks/M_surface/PreviewSurface")).GetInput(k).Set(v)
    ctrl = UsdGeom.Xform.Define(stage, "/World/controller")
    ctrl.GetPrim().GetReferences().AddReference(str(L.asset("controller")).replace("\\", "/"))
    ctrl.AddTransformOp().Set(Gf.Matrix4d(*frame0["ctrl"].T.flatten().tolist()))
    L.static(ctrl.GetPrim())

    stand = m.body("stand").id
    geom = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == stand)
    tray = UsdGeom.Cube.Define(stage, "/World/tray")
    tray.CreateSizeAttr(2.0)
    T_tray = H.pose(d.xmat[stand].reshape(3, 3), d.xpos[stand])
    tray.AddTransformOp().Set(Gf.Matrix4d(*T_tray.T.flatten().tolist()))
    tray.AddScaleOp().Set(Gf.Vec3f(*m.geom_size[geom].tolist()))
    UsdShade.MaterialBindingAPI.Apply(tray.GetPrim()).Bind(
        solid_material(stage, "/World/Looks/Tray", (0.11, 0.105, 0.10), 0.8))

    wire = L.preview_material(stage, "/World/Looks/Copper", (0.95, 0.64, 0.54), 0.35, metallic=1.0)
    bulb = L.preview_material(stage, "/World/Looks/Bulb", (1.0, 0.9, 0.75), 0.3, emissive=(20.0, 11.0, 4.5))
    table_pts = np.array([(x, y, L.WIRE_R) for x, y in TABLE_STRAND])

    def garland(a, b, n=9):
        a, b = np.asarray(a, float), np.asarray(b, float)
        u = np.linspace(0, 1, n)
        z = 0.42 - 0.14 * np.abs(np.sin(2 * np.pi * u))  # two swags
        xy = a + np.outer(u, b - a)
        return np.c_[xy, z]

    bulbs = {"table": L.strand(stage, "lights_table", L.catmull_rom(table_pts), wire, bulb, 8.0e3),
             "back_left": L.strand(stage, "lights_back_left", L.catmull_rom(garland((-0.62, -0.30), (-0.62, 0.62))),
                                   wire, bulb, 8.0e3),
             "back_right": L.strand(stage, "lights_back_right", L.catmull_rom(garland((-0.55, 0.66), (0.60, 0.66))),
                                    wire, bulb, 8.0e3)}
    key = UsdLux.RectLight.Define(stage, "/World/key")
    key.CreateWidthAttr(0.5)
    key.CreateHeightAttr(0.35)
    key.CreateIntensityAttr(1.2e3)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.88, 0.74))
    key.AddTransformOp().Set(L.look_at((0.75, -0.55, 0.75), (0.08, 0.15, 0.12)))
    eye0, look0 = frames[0][2], frames[0][3]
    cam = UsdGeom.Camera.Define(stage, "/World/camera")
    cam.AddTransformOp().Set(L.look_at(eye0, look0))
    stage.Save()

    tracks = {b: [] for b in frame0 if b != "ctrl"}
    tracks["controller"] = []
    for _, poses, _, _ in frames:
        for b, T in poses.items():
            tracks["controller" if b == "ctrl" else b].append(ue_pose(T))
    tracks = {k: v for k, v in tracks.items() if stage.GetPrimAtPath(f"/World/hand/{k}") or k == "controller"}
    anim = {"fps": FPS, "frames": len(frames), "tracks": tracks, "cuts": [int(round(t * FPS)) for t in cuts],
            "camera": {"eye": [f[2].tolist() for f in frames], "target": [f[3].tolist() for f in frames]}}
    (OUT / "hand_anim.json").write_text(json.dumps(anim))
    ev = hand_motion.event_times(H.path(su))
    meta = {"frames": len(frames), "fps": FPS, "events_s": ev, "result": res, "hand_meshes": n_geoms,
            "bulbs": {k: len(v) for k, v in bulbs.items()},
            "camera": {"eye": eye0.tolist(), "target": look0.tolist(), **CAMERA_LOOK}}
    (OUT / "hand_layout.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {path}: {n_geoms} hand meshes in {len(tracks) - 1} links, "
          f"{sum(meta['bulbs'].values())} shadow-casting fairy lights; {OUT / 'hand_anim.json'}", flush=True)
    return meta


class HandUE(UE):
    def settings(self, **extra) -> None:
        meta = json.loads((OUT / "hand_layout.json").read_text())
        cfg = dict(HAND_SETTINGS, layout=str(OUT / "hand_layout.usda").replace("\\", "/"),
                   hdri=str(OUT / "textures" / "wooden_lounge_4k.hdr").replace("\\", "/"),
                   camera=meta["camera"], frames=meta["frames"], fps=meta["fps"],
                   animation=str(OUT / "hand_anim.json").replace("\\", "/"),
                   map="/Game/Maps/HandPlace", scene_dir="/Game/HandScene", render_dir="/Game/HandRender",
                   sequence="/Game/HandRender/Shot", output_dir=str(self.renders).replace("\\", "/"), **extra)
        (self.scripts / "scan2usd_ue.json").write_text(json.dumps(cfg, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ue", type=Path, default=Path("C:/Program Files/Epic Games/UE_5.8"))
    ap.add_argument("--project", type=Path, default=Path("D:/UE/Scan2USD_UE"))
    ap.add_argument("--stages", default="layout,build,preview")
    args = ap.parse_args()
    stages = args.stages.split(",")
    ue = HandUE(args.ue, args.project)
    if not ue.uproject.exists():
        raise SystemExit("no UE project: run `python scripts/unreal_render.py --stages project` first")
    for f in ("ue_build.py", "ue_render.py"):
        shutil.copy(HERE / f, ue.scripts / f)
    if "layout" in stages:
        stage_layout()
    if "build" in stages:
        ue.settings()
        ue.build()
    meta = json.loads((OUT / "hand_layout.json").read_text())
    ev = meta["events_s"]
    # Spatial, not temporal, samples: the tracks hold one pose per frame, and temporal samples spread
    # over the shutter would mix two frames' poses into a double image.
    job = {"pass": "lumen", "resolution": [1920, 1080], "temporal_samples": 1, "spatial_samples": 8, "warm_up": 64,
           "cvars": {"r.MegaLights.Allowed": 1}}
    if "preview" in stages:
        times = [ev["pick"] + 0.6, ev["turn"] - 0.4, ev["preplace"] + 0.3, ev["seat"], ev["away"] + 2.0]
        jobs = [dict(job, name=f"hand_preview_{i}", single=True, range=[int(t * FPS), int(t * FPS) + 1])
                for i, t in enumerate(times)]
        ue.render(jobs)
        tiles = [Image.open(ue.renders / f"hand_preview_{i}.png").convert("RGB").resize((640, 360)) for i in range(5)]
        sheet = Image.new("RGB", (640 * 3, 360 * 2), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 3) * 640, (i // 3) * 360))
        sheet.save(OUT / "hand_preview.jpg", quality=88)
        print("wrote", OUT / "hand_preview.jpg")
    if "video" in stages:
        import imageio.v2 as iio

        for f in ue.renders.glob("hand_place.*.png"):
            f.unlink()
        ue.render([dict(job, name="hand_place")])
        files = sorted(ue.renders.glob("hand_place.*.png"))
        FIG.mkdir(parents=True, exist_ok=True)
        with iio.get_writer(FIG / "unreal_hand_place.mp4", fps=FPS, codec="libx264", quality=None,
                            pixelformat="yuv420p", output_params=["-crf", "23", "-preset", "slow"]) as w:
            for f in files:
                w.append_data(np.asarray(Image.open(f).convert("RGB").resize((1280, 720), Image.LANCZOS)))
        poster = files[min(len(files) - 1, int((ev["away"] + 2.0) * FPS))]
        Image.open(poster).convert("RGB").save(FIG / "unreal_hand_place.jpg", quality=90)
        print(f"wrote {FIG / 'unreal_hand_place.mp4'} ({len(files)} frames)")


if __name__ == "__main__":
    main()
