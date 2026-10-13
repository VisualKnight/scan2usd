"""Build the Unreal level from the Scan2USD layout (runs inside UnrealEditor-Cmd -run=pythonscript).

    UnrealEditor-Cmd Scan2USD_UE.uproject -run=pythonscript -script="ue_build.py"

Settings come from scan2usd_ue.json next to this script (written by scripts/unreal_render.py). The
level is rebuilt from scratch each run: /Game/Scene, /Game/Maps and /Game/Render are deleted first.
"""
import json
import math
import os

import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "scan2usd_ue.json")))
MAP = "/Game/Maps/Tabletop"
eal = unreal.EditorAssetLibrary
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def log(msg):
    unreal.log(f"[scan2usd] {msg}")


def import_file(path, dest, options=None):
    task = unreal.AssetImportTask()
    task.filename = path
    task.destination_path = dest
    task.automated = True
    task.save = True
    task.replace_existing = True
    if options is not None:
        task.options = options
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    return list(task.imported_object_paths)


def ue_location(p):
    """Scan2USD meters, Z-up right-handed -> Unreal centimeters, Z-up left-handed (Y flips)."""
    return unreal.Vector(p[0] * 100.0, -p[1] * 100.0, p[2] * 100.0)


def look_rotation(eye, target):
    d = [t - e for e, t in zip(eye, target)]
    d[1] = -d[1]
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return unreal.Rotator(roll=0.0, pitch=pitch, yaw=yaw)


for d in ("/Game/Scene", "/Game/Maps", "/Game/Render"):
    if eal.does_directory_exist(d):
        eal.delete_directory(d)
les.new_level(MAP)

# --- the USD layout: table, scans, fairy lights, key light -----------------------------------------
opts = unreal.UsdStageImportOptions()
opts.import_actors = True
opts.import_geometry = True
opts.import_materials = True
opts.import_skeletal_animations = False
opts.import_level_sequences = False
opts.purposes_to_import = 2  # render (default-purpose prims always come along); skips the "guide" colliders
opts.prim_path_folder_structure = False
opts.nanite_triangle_threshold = 200000
imported = import_file(CFG["layout"], "/Game/Scene", opts)
log(f"imported {len(imported)} assets from the layout")

actors = eas.get_all_level_actors()
n_point = 0
for a in actors:
    for c in a.get_components_by_class(unreal.PointLightComponent):
        n_point += 1
        c.set_editor_property("intensity_units", unreal.LightUnits.LUMENS)
        c.set_editor_property("intensity", CFG["bulb_lumens"])
        c.set_editor_property("attenuation_radius", CFG["bulb_attenuation_cm"])
        c.set_editor_property("cast_shadows", True)
        c.set_editor_property("allow_mega_lights", True)
        c.set_editor_property("light_color", unreal.Color(r=255, g=158, b=77, a=255))
        c.set_editor_property("specular_scale", 1.0)
    for c in a.get_components_by_class(unreal.RectLightComponent):
        c.set_editor_property("intensity_units", unreal.LightUnits.LUMENS)
        c.set_editor_property("intensity", CFG["key_lumens"])
        c.set_editor_property("allow_mega_lights", True)
        c.set_editor_property("barn_door_angle", 40.0)
        c.set_editor_property("barn_door_length", 15.0)
        c.set_editor_property("attenuation_radius", 400.0)
        log(f"key light {a.get_actor_label()} at {a.get_actor_location()}")
    for c in a.get_components_by_class(unreal.StaticMeshComponent):
        c.set_editor_property("cast_shadow", True)
log(f"{n_point} point lights, MegaLights allowed on each")
# Scan2USD normal maps are OpenGL-convention (+Y), as UsdPreviewSurface specifies; Unreal shades with
# DirectX-convention (-Y) maps. Flip green on import unless the importer already did.
n_flip = 0
if CFG.get("flip_normal_green", True):
    for path in eal.list_assets("/Game/Scene", recursive=True):
        tex = unreal.load_asset(path)
        if isinstance(tex, unreal.Texture2D) and "normal" in tex.get_name().lower():
            tex.set_editor_property("flip_green_channel", True)
            eal.save_loaded_asset(tex)
            n_flip += 1
log(f"flipped green on {n_flip} normal maps")
for a in actors:
    log(f"actor {a.get_class().get_name()} {a.get_actor_label()} {a.get_actor_location()}")

# --- environment: the HDRI as a backdrop sphere + sky light ----------------------------------------
hdr = import_file(CFG["hdri"], "/Game/Scene/HDRI")
cube = unreal.load_asset(hdr[0])
bp = unreal.EditorAssetLibrary.load_blueprint_class("/HDRIBackdrop/Blueprints/HDRIBackdrop")
backdrop = eas.spawn_actor_from_class(bp, unreal.Vector(0, 0, -3.5), unreal.Rotator(0, 0, CFG["hdri_yaw"]))


def bp_set(actor, names, value):
    for n in names:
        try:
            actor.set_editor_property(n, value)
            return
        except Exception:
            pass
    log(f"could not set {names[0]} on {actor.get_actor_label()}")


bp_set(backdrop, ["cubemap", "Cubemap"], cube)
bp_set(backdrop, ["intensity", "Intensity"], CFG["hdri_intensity"])
bp_set(backdrop, ["size", "Size"], 3000.0)
bp_set(backdrop, ["projection_center", "ProjectionCenter"], unreal.Vector(0, 0, 170.0))
bp_set(backdrop, ["lighting_distance_factor", "LightingDistanceFactor"], 0.0)
bp_set(backdrop, ["use_camera_projection", "UseCameraProjection"], False)
backdrop.set_actor_label("HDRI_wooden_lounge")
for a in eas.get_all_level_actors():
    if isinstance(a, unreal.CineCameraActor):
        eas.destroy_actor(a)  # the layout's USD camera; the render camera is set up below

# --- post: manual exposure, Lumen (hardware RT), MegaLights ----------------------------------------
ppv = eas.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0))
ppv.set_editor_property("unbound", True)
s = ppv.get_editor_property("settings")
for k, v in {
    "auto_exposure_method": unreal.AutoExposureMethod.AEM_MANUAL,
    "auto_exposure_apply_physical_camera_exposure": False,
    "auto_exposure_bias": CFG["exposure_ev"],
    "mega_lights": True,
    "lumen_final_gather_quality": 4.0,
    "lumen_scene_detail": 4.0,
    "lumen_scene_lighting_quality": 4.0,
    "lumen_reflection_quality": 4.0,
    "lumen_max_reflection_bounces": 2,
    "lumen_ray_lighting_mode": unreal.LumenRayLightingModeOverride.HIT_LIGHTING,
    "bloom_intensity": 0.35,
    "vignette_intensity": 0.25,
    "film_grain_intensity": 0.08,
    "scene_fringe_intensity": 0.15,
    "path_tracing_samples_per_pixel": CFG["path_tracer_spp"],
    "path_tracing_max_bounces": 8,
    "path_tracing_enable_denoiser": True,
}.items():
    s.set_editor_property(k, v)
    if hasattr(s, "override_" + k):
        s.set_editor_property("override_" + k, True)
    elif hasattr(s, "override_b_" + k):
        s.set_editor_property("override_b_" + k, True)
ppv.set_editor_property("settings", s)

# --- camera + a one-shot level sequence for Movie Render Queue --------------------------------------
cam_cfg = CFG["camera"]
cam = eas.spawn_actor_from_class(unreal.CineCameraActor, ue_location(cam_cfg["eye"]),
                                 look_rotation(cam_cfg["eye"], cam_cfg["target"]))
cam.set_actor_label("Scan2USD_Camera")
cam.set_editor_property("auto_activate_for_player", unreal.AutoReceiveInput.PLAYER0)  # -game runs see through it
cc = cam.get_cine_camera_component()
fb = cc.get_editor_property("filmback")
fb.set_editor_property("sensor_width", 36.0)
fb.set_editor_property("sensor_height", 20.25)
cc.set_editor_property("filmback", fb)
cc.set_editor_property("current_focal_length", cam_cfg["focal_mm"])
cc.set_editor_property("current_aperture", cam_cfg["fstop"])
fs = cc.get_editor_property("focus_settings")
fs.set_editor_property("focus_method", unreal.CameraFocusMethod.MANUAL)
fs.set_editor_property("manual_focus_distance",
                       100.0 * math.dist(cam_cfg["eye"], cam_cfg.get("focus", cam_cfg["target"])))
cc.set_editor_property("focus_settings", fs)

seq = unreal.AssetToolsHelpers.get_asset_tools().create_asset("Shot", "/Game/Render", unreal.LevelSequence,
                                                              unreal.LevelSequenceFactoryNew())
frames = CFG.get("frames", 1)
seq.set_display_rate(unreal.FrameRate(CFG.get("fps", 24), 1))
seq.set_playback_start(0)
seq.set_playback_end(frames)
binding = seq.add_possessable(cam)
cut = seq.add_track(unreal.MovieSceneCameraCutTrack)
sec = cut.add_section()
sec.set_range(0, frames)
sec.set_camera_binding_id(seq.get_binding_id(binding))

if CFG.get("animate"):
    # A slow orbit around the hero (the focus point): eye and aim point rotate together about it, 36
    # degrees, rising slightly. The hero stays at the same distance, so a fixed focus keeps it sharp.
    eye0, tgt0 = cam_cfg["eye"], cam_cfg["target"]
    pivot = cam_cfg.get("focus", tgt0)
    xf = binding.add_track(unreal.MovieScene3DTransformTrack).add_section()
    xf.set_range(0, frames)
    ch = xf.get_all_channels()  # loc x, y, z, rot x (roll), y (pitch), z (yaw), scale x, y, z

    def about_pivot(p, ang, dz=0.0):
        dx, dy = p[0] - pivot[0], p[1] - pivot[1]
        c, s = math.cos(ang), math.sin(ang)
        return (pivot[0] + c * dx - s * dy, pivot[1] + s * dx + c * dy, p[2] + dz)

    keys = 9
    for i in range(keys):
        u = i / (keys - 1)
        e = 0.5 - 0.5 * math.cos(math.pi * u)  # ease in and out
        a = math.radians(-18 + 36 * e)
        eye, tgt = about_pivot(eye0, a, 0.015 * e), about_pivot(tgt0, a)
        loc, rot = ue_location(eye), look_rotation(eye, tgt)
        t = unreal.FrameNumber(round(u * frames))
        for c, v in zip(ch[:6], (loc.x, loc.y, loc.z, 0.0, rot.pitch, rot.yaw)):
            c.add_key(t, v, interpolation=unreal.MovieSceneKeyInterpolation.AUTO)
    fs.set_editor_property("manual_focus_distance", 100.0 * math.dist(eye0, pivot))
    cc.set_editor_property("focus_settings", fs)
    log(f"animated camera: {keys} keys over {frames} frames")

eal.save_loaded_asset(seq)
les.save_current_level()
log(f"saved {MAP} and /Game/Render/Shot ({frames} frame(s))")
