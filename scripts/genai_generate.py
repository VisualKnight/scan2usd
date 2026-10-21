"""Image-to-3D generation for study_generative.py (runs in its own venv, not the project's).

    <genai venv>\\python.exe scripts\\genai_generate.py --model prep|hunyuan21|hunyuan2mv|triposg|all [--seed 0]
    <genai venv>\\python.exe scripts\\genai_generate.py --model paint|paint2mv      # textures

Reads output/genai/inputs.json (study_generative.py --stage inputs), writes
  output/genai/inputs/<object>/<photo>.png   the photo cut out (rembg, BiRefNet), shared by every model
  output/genai/meshes/<model>/<object>.ply   raw generated mesh (model units and axes), plus a .json
                                             with time, peak GPU memory and the inputs used
                                             (seeds other than 0: meshes/<model>_s<seed>/)
  output/genai/textured/<model>/<object>.glb the mesh textured by Hunyuan3D-2.1's paint model

Setup (Windows, one 24 GB GPU, no compiler needed):
  uv venv --python 3.11 tools\\genai\\venv
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install diffusers "transformers<5" accelerate einops omegaconf scikit-image trimesh pymeshlab
                 opencv-python huggingface_hub peft jaxtyping typeguard "rembg[gpu]" timm torchdiffeq
                 xatlas embreex pytorch-lightning
  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 / Hunyuan3D-2 / VAST-AI-Research/TripoSG
       into tools\\genai\\repos (GENAI_REPOS); weights download into tools\\genai\\hf (GENAI_WEIGHTS), including
       the paint model's facebook/dinov2-giant.
Licenses: Hunyuan3D (Tencent Hunyuan Community License; not for use in the EU, UK or South Korea),
TripoSG (MIT), BiRefNet via rembg (MIT). The weights are not redistributed here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "genai"
REPOS = Path(os.environ.get("GENAI_REPOS", ROOT / "tools" / "genai" / "repos"))
WEIGHTS = Path(os.environ.get("GENAI_WEIGHTS", ROOT / "tools" / "genai" / "hf"))
os.environ.setdefault("HY3DGEN_MODELS", str(WEIGHTS / "hy3dgen"))
os.environ.setdefault("HF_HOME", str(WEIGHTS))
MV_ROLES = ("front", "left", "back", "right")


def items():
    return json.loads((OUT / "inputs.json").read_text())


def cutout_path(key, photo):
    return OUT / "inputs" / key / Path(photo).with_suffix(".png").name


def prep(size=1024, pad=0.12):
    """Cut each photo out once (rembg, BiRefNet), square-crop around the object, and save RGBA."""
    from rembg import new_session, remove

    session = new_session("birefnet-general")
    for key, it in items().items():
        photos = {it["single"]["photo"], *(m["photo"] for m in it["multiview"].values())}
        for photo in sorted(photos):
            dst = cutout_path(key, photo)
            if dst.exists():
                continue
            im = Image.open(ROOT / it["photo_dir"] / photo).convert("RGB")
            im.thumbnail((2048, 2048))
            rgba = remove(im, session=session, post_process_mask=True)
            a = np.asarray(rgba)[..., 3]
            ys, xs = np.nonzero(a > 127)
            if len(xs) == 0:
                raise RuntimeError(f"{key}/{photo}: nothing left after background removal")
            # keep the largest blob only (table texture can leave specks)
            from scipy import ndimage

            lab, n = ndimage.label(a > 127)
            if n > 1:
                big = 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))))
                arr = np.asarray(rgba).copy()
                arr[..., 3] = np.where(ndimage.binary_dilation(lab == big, iterations=3), arr[..., 3], 0)
                rgba = Image.fromarray(arr)
                ys, xs = np.nonzero(arr[..., 3] > 127)
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * (1 + pad)
            box = tuple(int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))
            sq = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (0, 0, 0, 0))
            sq.paste(rgba.crop(box), (0, 0))
            dst.parent.mkdir(parents=True, exist_ok=True)
            sq.resize((size, size), Image.LANCZOS).save(dst)
            print(f"cut out {key}/{photo}", flush=True)


def _save(model, key, mesh, t0, inputs, seed):
    import torch

    d = OUT / "meshes" / (model if seed == 0 else f"{model}_s{seed}")  # other seeds: a robustness check
    d.mkdir(parents=True, exist_ok=True)
    mesh.export(d / f"{key}.ply")
    meta = {"model": model, "object": key, "seconds": time.time() - t0, "seed": seed,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight), "inputs": inputs}
    (d / f"{key}.json").write_text(json.dumps(meta, indent=1))
    print(f"{model} {key}: {meta['seconds']:.0f} s, {meta['peak_gpu_gb']:.1f} GB, {meta['faces']:,} faces, "
          f"watertight {meta['watertight']}", flush=True)


def run_hunyuan21(seed, only):
    import torch

    sys.path.insert(0, str(REPOS / "Hunyuan3D-2.1" / "hy3dshape"))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.postprocessors import DegenerateFaceRemover, FloaterRemover

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    for key, it in items().items():
        if only and key not in only:
            continue
        img = Image.open(cutout_path(key, it["single"]["photo"])).convert("RGBA")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        mesh = pipe(image=img, generator=torch.manual_seed(seed))[0]
        mesh = DegenerateFaceRemover()(FloaterRemover()(mesh))
        _save("hunyuan21", key, mesh, t0, [it["single"]["photo"]], seed)


def run_hunyuan2mv(seed, only):
    import torch

    sys.path.insert(0, str(REPOS / "Hunyuan3D-2"))
    from hy3dgen.shapegen import DegenerateFaceRemover, FloaterRemover, Hunyuan3DDiTFlowMatchingPipeline

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv",
                                                            variant="fp16")
    for key, it in items().items():
        if only and key not in only:
            continue
        imgs = {r: Image.open(cutout_path(key, it["multiview"][r]["photo"])).convert("RGBA") for r in MV_ROLES}
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        mesh = pipe(image=imgs, num_inference_steps=50, octree_resolution=380, num_chunks=20000,
                    generator=torch.manual_seed(seed), output_type="trimesh")[0]
        mesh = DegenerateFaceRemover()(FloaterRemover()(mesh))
        _save("hunyuan2mv", key, mesh, t0, [it["multiview"][r]["photo"] for r in MV_ROLES], seed)


def run_triposg(seed, only):
    import torch
    import trimesh

    # TripoSG imports diso (a CUDA extension that needs a compiler) at module level, but only its
    # "flash" decoder uses it; the hierarchical one (scikit-image marching cubes) doesn't.
    sys.modules.setdefault("diso", types.SimpleNamespace(DiffDMC=None))
    sys.path.insert(0, str(REPOS / "TripoSG"))
    sys.path.insert(0, str(REPOS / "TripoSG" / "scripts"))
    from image_process import prepare_image
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    pipe = TripoSGPipeline.from_pretrained(str(WEIGHTS / "TripoSG")).to("cuda", torch.float16)
    for key, it in items().items():
        if only and key not in only:
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        img = prepare_image(str(cutout_path(key, it["single"]["photo"])), bg_color=np.array([1.0, 1.0, 1.0]),
                            rmbg_net=None)
        out = pipe(image=img, generator=torch.Generator(device=pipe.device).manual_seed(seed),
                   num_inference_steps=50, guidance_scale=7.0, use_flash_decoder=False).samples[0]
        mesh = trimesh.Trimesh(out[0].astype(np.float32), np.ascontiguousarray(out[1]))
        _save("triposg", key, mesh, t0, [it["single"]["photo"]], seed)



def run_paint(seed, only, geometry="hunyuan21"):
    """Texture a generated mesh with Hunyuan3D-2.1's paint model (multi-view PBR diffusion), conditioned
    on the same photo(s) its geometry came from. The pipeline's two compiled parts are replaced:
    the CUDA rasterizer by genai_raster.py (pure PyTorch), the mesh-aware texture inpainting by plain
    OpenCV inpainting; Real-ESRGAN upscaling of the views is skipped (plain resize)."""
    import torch
    import trimesh

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import genai_raster

    sys.modules["custom_rasterizer"] = genai_raster
    paint = REPOS / "Hunyuan3D-2.1" / "hy3dpaint"
    sys.path.insert(0, str(paint))
    import diffusers

    # The paint model's pipeline class is local code in the cloned repo (hunyuanpaintpbr/pipeline.py);
    # newer diffusers asks for an explicit opt-in to run it.
    _load = diffusers.DiffusionPipeline.from_pretrained.__func__
    diffusers.DiffusionPipeline.from_pretrained = classmethod(
        lambda cls, *a, **k: _load(cls, *a, **{"trust_remote_code": True, **k}))
    from DifferentiableRenderer.MeshRender import MeshRender
    from utils.multiview_utils import multiviewDiffusionNet
    from utils.pipeline_utils import ViewProcessor
    from utils.uvwrap_utils import mesh_uv_wrap

    cfg = types.SimpleNamespace(
        device="cuda", multiview_cfg_path=str(paint / "cfgs" / "hunyuan-paint-pbr.yaml"),
        custom_pipeline=str(paint / "hunyuanpaintpbr"), multiview_pretrained_path="tencent/Hunyuan3D-2.1",
        dino_ckpt_path="facebook/dinov2-giant", raster_mode="cr", bake_mode="back_sample", render_size=2048,
        texture_size=2048, max_selected_view_num=6, resolution=512, bake_exp=4, merge_method="fast",
        candidate_camera_azims=[0, 90, 180, 270, 0, 180], candidate_camera_elevs=[0, 0, 0, 0, 90, -90],
        candidate_view_weights=[1, 0.1, 0.5, 0.1, 0.05, 0.05])
    for azim in range(0, 360, 30):  # as Hunyuan3DPaintConfig
        for elev in (20, -20):
            cfg.candidate_camera_azims.append(azim)
            cfg.candidate_camera_elevs.append(elev)
            cfg.candidate_view_weights.append(0.01)
    render = MeshRender(default_resolution=cfg.render_size, texture_size=cfg.texture_size, bake_mode=cfg.bake_mode,
                        raster_mode=cfg.raster_mode)
    views = ViewProcessor(cfg, render)
    mv = multiviewDiffusionNet(cfg)
    roles = MV_ROLES if geometry == "hunyuan2mv" else ("single",)
    for key, it in items().items():
        if only and key not in only:
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        mesh = trimesh.load(str(OUT / "meshes" / geometry / f"{key}.ply"), force="mesh")
        if len(mesh.faces) > 40000:  # the pipeline's remesh step (quadric decimation to 40k faces)
            import pymeshlab

            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=40000, preservenormal=True)
            cm = ms.current_mesh()
            mesh = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
        mesh = mesh_uv_wrap(mesh)
        render.set_mesh(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
                        vtx_uv=np.asarray(mesh.visual.uv, np.float32), uv_idx=mesh.faces.astype(np.int32))
        elevs, azims, weights = views.bake_view_selection(cfg.candidate_camera_elevs, cfg.candidate_camera_azims,
                                                          cfg.candidate_view_weights, cfg.max_selected_view_num)
        normal_maps = views.render_normal_multiview(elevs, azims, use_abs_coor=True)
        position_maps = views.render_position_multiview(elevs, azims)
        photos = [it["single"]["photo"]] if roles == ("single",) else [it["multiview"][r]["photo"] for r in roles]
        refs = []
        for p in photos:
            im = Image.open(cutout_path(key, p)).resize((512, 512))
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            refs.append(bg)
        out = mv(refs, normal_maps + position_maps, prompt="high quality", custom_view_size=cfg.resolution,
                 resize_input=True)
        albedo = [v.resize((cfg.render_size, cfg.render_size), Image.LANCZOS) for v in out["albedo"]]
        tex, mask = views.bake_from_multiview(albedo, elevs, azims, weights)
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        tex_np = render.uv_inpaint(tex, mask_np, vertex_inpaint=False)  # uint8 RGB, UV space
        d = OUT / "textured" / geometry
        d.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(tex_np.astype(np.uint8))
        mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=img)
        mesh.export(d / f"{key}.glb")
        img.save(d / f"{key}_albedo.png")
        (d / f"{key}.json").write_text(json.dumps({
            "geometry": geometry, "object": key, "seconds": time.time() - t0, "faces": int(len(mesh.faces)),
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "reference_photos": photos,
            "views": [[float(e), float(a)] for e, a in zip(elevs, azims)]}, indent=1))
        print(f"paint {geometry} {key}: {time.time() - t0:.0f} s, {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
              flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated object keys")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    models = ["prep", "hunyuan21", "triposg", "hunyuan2mv"] if args.model == "all" else args.model.split(",")
    for m in models:
        if m == "prep":
            prep()
        elif m.startswith("paint"):  # paint (Hunyuan3D 2.1 geometry, 1 photo) or paint2mv (2mv geometry, 4 photos)
            run_paint(args.seed, only, "hunyuan2mv" if m == "paint2mv" else "hunyuan21")
        else:
            globals()[f"run_{m}"](args.seed, only)


if __name__ == "__main__":
    main()
