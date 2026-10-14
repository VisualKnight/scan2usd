"""Render the Tabletop level with Movie Render Queue (runs inside UnrealEditor -ExecutePythonScript).

Jobs come from scan2usd_ue.json "renders": each has a name, a pass ("lumen" or "path_tracer"), a
resolution, sample counts and console variables (e.g. MegaLights on/off). The editor quits when the
queue is done.
"""
import json
import os

import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "scan2usd_ue.json")))


def log(msg):
    unreal.log(f"[scan2usd] {msg}")


subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
queue = subsystem.get_queue()
queue.delete_all_jobs()
for r in CFG["renders"]:
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.job_name = r["name"]
    job.map = unreal.SoftObjectPath(CFG.get("map", "/Game/Maps/Tabletop"))
    job.sequence = unreal.SoftObjectPath(CFG.get("sequence", "/Game/Render/Shot"))
    cfg = job.get_configuration()

    out = cfg.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    out.output_directory = unreal.DirectoryPath(CFG["output_dir"])
    out.output_resolution = unreal.IntPoint(*r["resolution"])
    out.file_name_format = r["name"] + ("" if CFG.get("frames", 1) == 1 or r.get("single") else ".{frame_number}")
    out.override_existing_output = True
    if "range" in r:  # a part of the sequence: [first frame, end frame)
        out.use_custom_playback_range = True
        out.custom_start_frame, out.custom_end_frame = r["range"]

    if r["pass"] == "path_tracer":
        cfg.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPass_PathTracer)
    else:
        cfg.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    for ext in r.get("formats", ["png"]):
        cls = {"png": unreal.MoviePipelineImageSequenceOutput_PNG, "exr": unreal.MoviePipelineImageSequenceOutput_EXR}[ext]
        cfg.find_or_add_setting_by_class(cls)

    aa = cfg.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.override_anti_aliasing = True
    aa.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_NONE if r["pass"] == "path_tracer" else unreal.AntiAliasingMethod.AAM_TSR
    aa.spatial_sample_count = r.get("spatial_samples", 1)
    aa.temporal_sample_count = r.get("temporal_samples", 1)
    aa.use_camera_cut_for_warm_up = False
    aa.engine_warm_up_count = r.get("warm_up", 64)
    aa.render_warm_up_count = r.get("warm_up", 64)
    aa.render_warm_up_frames = True

    cv = cfg.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    for k, v in r.get("cvars", {}).items():
        cv.add_or_update_console_variable(k, float(v))
    for cmd in r.get("start_commands", []):
        cv.start_console_commands.append(cmd)
    log(f"queued {r['name']}: {r['pass']} {r['resolution']} cvars {r.get('cvars', {})}")

executor = unreal.MoviePipelinePIEExecutor()


def finished(ex, ok):
    log(f"queue finished, success={ok}")
    unreal.SystemLibrary.quit_editor()


def job_done(job, ok):
    log(f"job {job.job_name} finished, success={ok}")


executor.on_executor_finished_delegate.add_callable(finished)
executor.on_individual_job_finished_delegate.add_callable(job_done)
subsystem.render_queue_with_executor_instance(executor)
log("rendering")
