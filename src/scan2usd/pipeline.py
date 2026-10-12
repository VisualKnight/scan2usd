"""End-to-end build: scanned mesh -> calibrated, physics-ready OpenUSD asset."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from . import __version__
from .color import (
    Calibration,
    albedo_range_stats,
    apply_ccm_srgb8,
    fit_ccm,
    fit_neutral,
    sample_chart,
    sample_region,
)
from .mesh_ops import (
    clean_mesh,
    decimate,
    get_base_color_image,
    has_texture,
    has_vertex_colors,
    load_mesh,
    orient_and_scale,
)
from .physics import MATERIALS, convex_decompose, estimate_mass_properties, single_hull
from .usd_io import PhysicsMaterial, SurfaceMaterial, export_gltf, write_usd
from .validate import ValidationReport, validate_usd


@dataclass
class BuildOptions:
    name: str
    material: str = "plastic"
    density: float | None = None
    mass: float | None = None
    up_axis: str = "z"
    units: str = "m"
    scale_to: float | None = None
    scale_axis: str = "max"
    target_faces: int = 50_000
    min_component_ratio: float = 0.02
    collision: str = "coacd"  # "coacd" or "hull"
    hull_threshold: float = 0.05
    max_hulls: int = 32
    roughness: float = 0.5
    metallic: float = 0.0
    chart_image: Path | None = None
    chart_corners: list[list[float]] | None = None
    gray_card_image: Path | None = None
    gray_card_box: list[float] | None = None
    gray_reflectance: float | None = 0.18
    ccm_path: Path | None = None
    usda: bool = False
    gltf: bool = True
    run_sim: bool = True
    preview: bool = True
    bake_texture: int = 0  # texture size; >0 bakes vertex colors of the dense scan into a UV texture


@dataclass
class BuildResult:
    usd_path: Path
    gltf_path: Path | None
    report_path: Path
    validation: ValidationReport
    mesh: trimesh.Trimesh
    summary: dict = field(default_factory=dict)


def _calibration(opts: BuildOptions, out_dir: Path) -> Calibration | None:
    if opts.ccm_path is not None:
        return Calibration.load(opts.ccm_path)
    if opts.gray_card_image is not None:
        if opts.gray_card_box is None:
            raise ValueError("gray_card_box is required with gray_card_image")
        img = np.asarray(Image.open(opts.gray_card_image).convert("RGB"))
        cal = fit_neutral(sample_region(img, opts.gray_card_box), opts.gray_reflectance)
        cal.save(out_dir / "color_calibration.json")
        return cal
    if opts.chart_image is None:
        return None
    if opts.chart_corners is None:
        raise ValueError("chart_corners are required with chart_image")
    img = np.asarray(Image.open(opts.chart_image).convert("RGB"))
    cal = fit_ccm(sample_chart(img, np.asarray(opts.chart_corners, dtype=np.float64)))
    cal.save(out_dir / "color_calibration.json")
    return cal


def _surface(mesh: trimesh.Trimesh, opts: BuildOptions, cal: Calibration | None, out_dir: Path) -> tuple[SurfaceMaterial, dict]:
    info: dict = {}
    surface = SurfaceMaterial(roughness=opts.roughness, metallic=opts.metallic)
    if has_texture(mesh):
        img = np.asarray(get_base_color_image(mesh).convert("RGBA"))
        info["albedo_before"] = albedo_range_stats(img)
        if cal is not None:
            img = apply_ccm_srgb8(img, cal.matrix)
        tex_dir = out_dir / "textures"
        tex_dir.mkdir(parents=True, exist_ok=True)
        tex_path = tex_dir / f"{opts.name}_albedo.png"
        Image.fromarray(img if img[..., 3].min() < 255 else img[..., :3]).save(tex_path)
        surface.texture = tex_path
        info["source"] = "texture"
        info["albedo_after"] = albedo_range_stats(img)
    elif has_vertex_colors(mesh):
        colors = np.asarray(mesh.visual.vertex_colors)
        info["albedo_before"] = albedo_range_stats(colors)
        if cal is not None:
            mesh.visual.vertex_colors = apply_ccm_srgb8(colors, cal.matrix)
        surface.use_vertex_colors = True
        info["source"] = "vertex_colors"
        info["albedo_after"] = albedo_range_stats(np.asarray(mesh.visual.vertex_colors))
    else:
        info["source"] = "constant"
    return surface, info


def build_asset(input_path: str | Path, out_dir: str | Path, opts: BuildOptions) -> BuildResult:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(stage: str, start: float) -> float:
        now = time.perf_counter()
        timings[stage] = round(now - start, 3)
        return now

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if opts.material not in MATERIALS:
        raise ValueError(f"unknown material '{opts.material}'; choose from {', '.join(MATERIALS)}")
    mat = MATERIALS[opts.material]

    t = time.perf_counter()
    raw = load_mesh(input_path)
    t = lap("load", t)
    mesh, clean = clean_mesh(raw, opts.min_component_ratio)
    dense = mesh
    mesh, decimate_notes = decimate(mesh, opts.target_faces)
    clean.notes += decimate_notes
    if opts.bake_texture and has_vertex_colors(dense) and len(dense.faces) > len(mesh.faces):
        from .bake import bake_color

        mesh, _, bake_stats = bake_color(dense, mesh, size=opts.bake_texture)
        clean.notes.append(f"baked {len(dense.faces):,}-triangle vertex colors into a "
                           f"{opts.bake_texture}px texture ({bake_stats['atlas_coverage'] * 100:.0f}% atlas coverage)")
    del dense
    mesh, scale = orient_and_scale(mesh, opts.up_axis, opts.units, opts.scale_to, opts.scale_axis)
    t = lap("process", t)

    cal = _calibration(opts, out_dir)
    surface, albedo = _surface(mesh, opts, cal, out_dir)
    t = lap("color", t)

    hulls = (
        convex_decompose(mesh, opts.hull_threshold, opts.max_hulls)
        if opts.collision == "coacd"
        else single_hull(mesh)
    )
    t = lap("collision", t)
    if opts.mass is not None:
        mass_props = estimate_mass_properties(mesh, hulls, mass=opts.mass)
    else:
        mass_props = estimate_mass_properties(mesh, hulls, density=opts.density or mat["density"])
    t = lap("mass", t)

    metadata = {
        "source": Path(input_path).name,
        "material": opts.material,
        "scale_applied": scale,
        "clean": asdict(clean),
        "mass_source": "measured_mass" if opts.mass is not None else "density",
        "volume_source": mass_props.volume_source,
        "color_calibration": None if cal is None else {
            "delta_e_before": cal.delta_e_before,
            "delta_e_after": cal.delta_e_after,
        },
    }
    usd_path = out_dir / f"{opts.name}.{'usda' if opts.usda else 'usd'}"
    write_usd(
        usd_path, opts.name, mesh, hulls, mass_props, surface,
        PhysicsMaterial(mat["static_friction"], mat["dynamic_friction"], mat["restitution"]),
        metadata,
    )
    gltf_path = export_gltf(out_dir / f"{opts.name}.glb", mesh) if opts.gltf else None
    t = lap("export", t)

    validation = validate_usd(usd_path, run_sim=opts.run_sim)
    lap("validate", t)
    (out_dir / "validation.md").write_text(validation.to_markdown(), encoding="utf-8")
    if opts.preview:
        from .preview import preview_file

        preview_file(usd_path).save(out_dir / "preview.png")

    summary = {
        "scan2usd_version": __version__,
        "input": str(input_path),
        "outputs": {"usd": usd_path.name, "gltf": gltf_path.name if gltf_path else None},
        "clean": asdict(clean),
        "scale_applied": scale,
        "extents_m": [float(v) for v in mesh.extents],
        "render_triangles": int(len(mesh.faces)),
        "collision_hulls": len(hulls),
        "mass_properties": mass_props.to_dict(),
        "albedo": albedo,
        "color_calibration": None if cal is None else asdict(cal),
        "timings_s": {**timings, "total": round(time.perf_counter() - t0, 3)},
        "validation": validation.to_dict(),
    }
    report_path = out_dir / "build_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return BuildResult(usd_path, gltf_path, report_path, validation, mesh, summary)
