"""Render the Scan2USD tabletop in Unreal Engine 5 with Lumen + MegaLights, and measure it.

    python scripts/unreal_layout.py          # the scene as USD (assets, table, 80 fairy lights, key, camera)
    python scripts/unreal_render.py [--ue "C:/Program Files/Epic Games/UE_5.8"] [--project tools/UE/Scan2USD_UE]
                                    [--stages project,build,stills,timing,figure] [--video]

Stages:
  project  creates a minimal UE project (Python, USD importer, Movie Render Queue, HDRI backdrop), with
           Lumen on hardware ray tracing, MegaLights and the path tracer enabled; copies ue_build.py and
           ue_render.py (scripts/unreal/) into it.
  build    UnrealEditor-Cmd imports output/unreal/layout.usda through the USD importer into a level, adds
           the HDRI backdrop (Poly Haven wooden_lounge, CC0), a post-process volume and a cine camera.
  stills   Movie Render Queue: the hero frame (Lumen + MegaLights), the same view with MegaLights off,
           and the path tracer (4,096 samples, no denoiser) as the reference.
  timing   the level in -game mode at 1080p, 900 frames with GPU stats: median GPU time with MegaLights
           on and off (the first third of the frames is dropped as warm-up).
  figure   docs/figures/unreal_render.jpg (hero), unreal_megalights.jpg (comparison strip with timings
           and PSNR against the path tracer) and unreal_render.json.
  --video  a 10 s camera move (MRQ, Lumen + MegaLights) -> docs/figures/unreal_flythrough.mp4.

Needs Unreal Engine 5.6+ (5.8 used here) and an RTX GPU. Textures and the HDRI are fetched from Poly
Haven into output/unreal/textures/ on first run.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "unreal"
FIG = ROOT / "docs" / "figures"
HERE = Path(__file__).resolve().parent / "unreal"
TEXTURES = {
    "wood_table_001_diff.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/wood_table_001/wood_table_001_diff_4k.jpg",
    "wood_table_001_nor_gl.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/wood_table_001/wood_table_001_nor_gl_4k.jpg",
    "wood_table_001_rough.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/wood_table_001/wood_table_001_rough_4k.jpg",
    "wooden_lounge_4k.hdr": "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/wooden_lounge_4k.hdr",
}
UPROJECT = {
    "FileVersion": 3, "EngineAssociation": "", "Description": "Scan2USD assets rendered in Unreal Engine",
    "Plugins": [{"Name": n, "Enabled": True} for n in
                ("PythonScriptPlugin", "EditorScriptingUtilities", "USDImporter", "MovieRenderPipeline", "HDRIBackdrop")],
}
ENGINE_INI = """[/Script/EngineSettings.GameMapsSettings]
GameDefaultMap=/Game/Maps/Tabletop.Tabletop
EditorStartupMap=/Game/Maps/Tabletop.Tabletop

[/Script/Engine.RendererSettings]
r.AllowStaticLighting=False
r.Shadow.Virtual.Enable=1
r.GenerateMeshDistanceFields=True
r.DynamicGlobalIlluminationMethod=1
r.ReflectionMethod=1
r.RayTracing=True
r.Lumen.HardwareRayTracing=True
r.Lumen.HardwareRayTracing.LightingMode=2
r.MegaLights.EnableForProject=True
r.PathTracing=True
r.SkinCache.CompileShaders=True
r.DefaultFeature.AutoExposure=False
r.DefaultFeature.MotionBlur=False
r.Nanite.ProjectEnabled=True

[/Script/WindowsTargetPlatform.WindowsTargetSettings]
DefaultGraphicsRHI=DefaultGraphicsRHI_DX12
-D3D12TargetedShaderFormats=PCD3D_SM5
+D3D12TargetedShaderFormats=PCD3D_SM6
"""
# Everything the in-editor scripts read (scripts/unreal/ue_build.py, ue_render.py).
SETTINGS = {
    "hdri_yaw": 40.0, "hdri_intensity": 0.6,
    "bulb_lumens": 3.0, "bulb_attenuation_cm": 120.0, "key_lumens": 800.0,
    "exposure_ev": -4.7, "path_tracer_spp": 4096, "frames": 1, "fps": 24,
}
STILLS = [
    {"name": "ue_hero", "pass": "lumen", "resolution": [2560, 1440], "temporal_samples": 16, "warm_up": 96,
     "cvars": {"r.MegaLights.Allowed": 1}},
    {"name": "cmp_megalights", "pass": "lumen", "resolution": [1920, 1080], "temporal_samples": 8, "warm_up": 96,
     "cvars": {"r.MegaLights.Allowed": 1, "r.FilmGrain": 0}},
    {"name": "cmp_no_megalights", "pass": "lumen", "resolution": [1920, 1080], "temporal_samples": 8, "warm_up": 96,
     "cvars": {"r.MegaLights.Allowed": 0, "r.FilmGrain": 0}},
    {"name": "cmp_path_traced", "pass": "path_tracer", "resolution": [1920, 1080], "spatial_samples": 4096,
     "temporal_samples": 1, "warm_up": 8, "cvars": {"r.PathTracing.Denoiser": 0, "r.FilmGrain": 0}},
]


def run(cmd: list[str], log: Path | None = None, timeout: float = 3600) -> None:
    print(" ".join(str(c) for c in cmd[:3]), "...", flush=True)
    with open(log or os.devnull, "w", encoding="utf-8", errors="replace") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=False)


class UE:
    def __init__(self, engine: Path, project: Path):
        self.bin = engine / "Engine" / "Binaries" / "Win64"
        self.project = project
        self.uproject = project / f"{project.name}.uproject"
        self.scripts = project / "Scripts"
        self.renders = project / "Saved" / "Scan2USD"

    def settings(self, **extra) -> None:
        cfg = dict(SETTINGS, layout=str(OUT / "layout.usda").replace("\\", "/"),
                   hdri=str(OUT / "textures" / "wooden_lounge_4k.hdr").replace("\\", "/"),
                   camera=json.loads((OUT / "layout.json").read_text())["camera"],
                   output_dir=str(self.renders).replace("\\", "/"), **extra)
        (self.scripts / "scan2usd_ue.json").write_text(json.dumps(cfg, indent=2))

    def build(self) -> None:
        run([str(self.bin / "UnrealEditor-Cmd.exe"), str(self.uproject), "-run=pythonscript",
             f"-script={self.scripts / 'ue_build.py'}", "-unattended", "-nosplash", "-nopause", "-stdout"],
            OUT / "ue_build.log")
        # unreal.log() lines go to the project's log file, not the commandlet's stdout.
        log = (self.project / "Saved" / "Logs" / f"{self.project.name}.log").read_text(encoding="utf-8", errors="replace")
        if "[scan2usd] saved" not in log or "LogPython: Error" in log:
            raise RuntimeError(f"level build failed, see {self.project / 'Saved' / 'Logs'}")

    def render(self, jobs: list[dict], res=(1280, 720), **extra) -> None:
        self.settings(renders=jobs, **extra)
        # -ExecCmds, not -ExecutePythonScript: the latter quits the editor as soon as the script returns,
        # which ends the play session Movie Render Queue renders in.
        run([str(self.bin / "UnrealEditor.exe"), str(self.uproject), "/Game/Maps/Tabletop",
             f"-ExecCmds=py {(self.scripts / 'ue_render.py').as_posix()}", "-windowed", f"-ResX={res[0]}",
             f"-ResY={res[1]}", "-nosplash", "-log=scan2usd_render.log"], timeout=4 * 3600)
        for j in jobs:
            if not list(self.renders.glob(j["name"] + "*")):
                raise RuntimeError(f"render {j['name']} missing; see {self.project / 'Saved/Logs/scan2usd_render.log'}")

    def gpu_time(self, megalights: int, frames: int = 900) -> dict:
        """Median GPU frame time (ms) of the level in -game mode, with Unreal's CSV profiler."""
        appdata = Path(os.path.expandvars(r"%LOCALAPPDATA%\UnrealEngine"))
        before = set(glob.glob(str(appdata / "*" / "Saved" / "Profiling" / "CSV" / "*.csv")))
        log_name = f"scan2usd_perf_ml{megalights}.log"
        cmds = f"r.MegaLights.Allowed {megalights},t.MaxFPS 0,r.VSync 0,r.GPUCsvStatsEnabled 1"
        proc = subprocess.Popen([str(self.bin / "UnrealEditor.exe"), str(self.uproject), "/Game/Maps/Tabletop", "-game",
                                 "-windowed", "-ResX=1920", "-ResY=1080", "-nosplash", f"-ExecCmds={cmds}",
                                 f"-csvCaptureFrames={frames}", "-csvGpuStats", f"-log={log_name}"])
        t0, new = time.time(), set()
        try:
            while time.time() - t0 < 600:
                new = set(glob.glob(str(appdata / "*" / "Saved" / "Profiling" / "CSV" / "*.csv"))) - before
                logs = glob.glob(str(appdata / "*" / "Saved" / "Logs" / log_name))
                # The CSV file appears when the capture starts; the log says when it's written.
                if new and logs and "Capture Ended" in Path(logs[0]).read_text(encoding="utf-8", errors="replace"):
                    time.sleep(3)
                    break
                time.sleep(2)
        finally:
            proc.kill()  # -csvExitOnCompletion doesn't end an editor -game session
        if not new:
            raise RuntimeError("no GPU profile captured")
        return parse_csv(max(new, key=os.path.getmtime))


def parse_csv(path: str) -> dict:
    csv.field_size_limit(10 ** 9)
    rows = list(csv.reader(open(path, encoding="utf-8", errors="replace")))
    header_rows = [i for i, r in enumerate(rows) if r and r[0] == "EVENTS"]
    head = rows[header_rows[-1]]  # the header is repeated at the end, with every stat that appeared
    data = []
    for r in rows[1:header_rows[-1]]:
        try:
            float(r[head.index("FrameTime")])
            data.append(r + [""] * (len(head) - len(r)))
        except (ValueError, IndexError):
            pass
    keep = data[len(data) // 3:]

    def med(col):
        i = head.index(col)
        v = [float(r[i]) for r in keep if r[i] != ""]
        return statistics.median(v) if v else 0.0

    passes = {c[4:]: round(med(c), 3) for c in dict.fromkeys(head) if c.startswith("GPU/")}
    top = dict(sorted(passes.items(), key=lambda kv: -kv[1])[:8])
    return {"frames": len(keep), "gpu_ms": round(med("GPUTime"), 2), "frame_ms": round(med("FrameTime"), 2),
            "passes_ms": top}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / np.mean((a - b) ** 2)))


def stage_project(ue: UE) -> None:
    ue.project.mkdir(parents=True, exist_ok=True)
    (ue.project / "Config").mkdir(exist_ok=True)
    ue.scripts.mkdir(exist_ok=True)
    proj = dict(UPROJECT, EngineAssociation=ue.bin.parents[2].name.replace("UE_", ""))
    ue.uproject.write_text(json.dumps(proj, indent=2))
    (ue.project / "Config" / "DefaultEngine.ini").write_text(ENGINE_INI)
    for f in ("ue_build.py", "ue_render.py"):
        shutil.copy(HERE / f, ue.scripts / f)
    (OUT / "textures").mkdir(parents=True, exist_ok=True)
    for name, url in TEXTURES.items():
        dst = OUT / "textures" / name
        if not dst.exists():
            print("fetching", name)
            urllib.request.urlretrieve(url, dst)
    print("project:", ue.uproject)


def stage_build(ue: UE) -> None:
    if not (OUT / "layout.usda").exists():
        raise SystemExit("run scripts/unreal_layout.py first")
    ue.settings()
    ue.build()


def stage_stills(ue: UE) -> None:
    ue.render(STILLS)


def stage_timing(ue: UE, results: dict, repeats: int = 3) -> None:
    # Alternate on/off runs: GPU clocks and background load drift by ~10% between runs.
    runs = {"on": [], "off": []}
    for _ in range(repeats):
        runs["on"].append(ue.gpu_time(1))
        runs["off"].append(ue.gpu_time(0))
    timing = {"resolution": [1920, 1080], "repeats": repeats}
    for k, rs in runs.items():
        ms = sorted(r["gpu_ms"] for r in rs)
        mid = min(rs, key=lambda r: abs(r["gpu_ms"] - statistics.median(ms)))
        timing[k] = dict(mid, gpu_ms=round(statistics.median(ms), 2), gpu_ms_runs=ms)
    results["timing"] = timing
    print(json.dumps({k: (v["gpu_ms_runs"] if isinstance(v, dict) else v) for k, v in timing.items()}))


def stage_figure(ue: UE, results: dict) -> None:
    from scan2usd.preview import _font

    def load(n):
        return np.asarray(Image.open(ue.renders / f"{n}.png").convert("RGB"), np.float64) / 255.0

    ref = load("cmp_path_traced")
    on, off = load("cmp_megalights"), load("cmp_no_megalights")
    results["vs_path_tracer"] = {"megalights_psnr_db": round(psnr(on, ref), 2), "no_megalights_psnr_db": round(psnr(off, ref), 2),
                                 "path_tracer_spp": SETTINGS["path_tracer_spp"]}
    results["scene"] = json.loads((OUT / "layout.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    Image.open(ue.renders / "ue_hero.png").convert("RGB").save(FIG / "unreal_render.jpg", quality=90)

    t = results.get("timing", {})
    labels = [("Lumen + MegaLights", on, t.get("on", {}).get("gpu_ms"), results["vs_path_tracer"]["megalights_psnr_db"]),
              ("Lumen, MegaLights off", off, t.get("off", {}).get("gpu_ms"), results["vs_path_tracer"]["no_megalights_psnr_db"]),
              ("Path tracer (reference)", ref, None, None)]
    w, h, pad, top = 760, 428, 16, 64
    sheet = Image.new("RGB", (3 * w + 4 * pad, h + top + pad), "white")
    d = ImageDraw.Draw(sheet)
    for k, (title, img, ms, db) in enumerate(labels):
        x = pad + k * (w + pad)
        sheet.paste(Image.fromarray((img * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS), (x, top))
        d.text((x, 8), title, fill=(30, 41, 59), font=_font(24))
        sub = (f"{ms:.1f} ms GPU at 1080p · {db:.1f} dB vs reference" if ms and db
               else f"{SETTINGS['path_tracer_spp']:,} samples per pixel, offline")
        d.text((x, 38), sub, fill=(100, 116, 139), font=_font(17))
    sheet.save(FIG / "unreal_megalights.jpg", quality=90)
    (FIG / "unreal_render.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "scene"}, indent=1))


def stage_video(ue: UE, seconds: float = 10.0, fps: int = 30) -> None:
    jobs = [{"name": "flythrough", "pass": "lumen", "resolution": [1920, 1080], "temporal_samples": 8, "warm_up": 64,
             "cvars": {"r.MegaLights.Allowed": 1}}]
    ue.settings(frames=int(seconds * fps), fps=fps, animate=True)
    ue.build()
    ue.render(jobs, frames=int(seconds * fps), fps=fps)
    import imageio.v2 as iio

    frames = sorted(ue.renders.glob("flythrough.*.png"))
    with iio.get_writer(FIG / "unreal_flythrough.mp4", fps=fps, codec="libx264", pixelformat="yuv420p",
                        macro_block_size=1, ffmpeg_params=["-crf", "23", "-preset", "slow"]) as w:
        for f in frames:
            w.append_data(np.asarray(Image.open(f).convert("RGB")))
    print(f"wrote {FIG / 'unreal_flythrough.mp4'} ({len(frames)} frames)")
    ue.settings()
    ue.build()  # back to the still shot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ue", type=Path, default=Path(r"C:\Program Files\Epic Games\UE_5.8"))
    ap.add_argument("--project", type=Path, default=ROOT / "tools" / "UE" / "Scan2USD_UE")
    ap.add_argument("--stages", default="project,build,stills,timing,figure")
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    ue = UE(args.ue, args.project)
    res_path = FIG / "unreal_render.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    stages = args.stages.split(",")
    if "project" in stages:
        stage_project(ue)
    if "build" in stages:
        stage_build(ue)
    if "stills" in stages:
        stage_stills(ue)
    if "timing" in stages:
        stage_timing(ue, results)
    if "figure" in stages:
        stage_figure(ue, results)
    if args.video:
        stage_video(ue)


if __name__ == "__main__":
    main()
