# Scan2USD

**Scan-to-Sim Asset Pipeline**: turns real-world captures (photogrammetry / 3D Gaussian splatting) into physics-ready **OpenUSD** assets for robotics simulation.

> Personal R&D project · work in progress (Oct 2026) · visual overview: [docs/overview.html](docs/overview.html)

Given a scanned mesh, Scan2USD produces a simulation-ready asset:

- **Clean geometry**: floaters removed, holes filled, Z-up, meters, origin at the base, topology-preserving decimation
- **Calibrated color**: a 3×3 color-correction matrix fitted on a ColorChecker photo removes the capture's lighting and camera cast (reported as CIEDE2000)
- **Collision**: approximate convex decomposition (CoACD), within PhysX hull limits
- **Mass properties**: mass, center of mass and principal inertia from volume × material density, or from a measured mass
- **OpenUSD output** with `UsdPhysics` schemas (rigid body, mass, colliders, physics material) and a `UsdPreviewSurface` material, plus glTF
- **Automated validation**: 33 checks on the written file, including MuJoCo drop tests for simulation stability

## Demo results

`scan2usd demo` builds a synthetic "scan" of a ceramic mug that mimics a real photogrammetry export: Y-up, centimeters, 0.2 mm noise, floating debris, 370k triangles, and a warm camera color cast. Because the ground truth is known, the results can be checked:

| Metric | Result |
|---|---|
| Color chart error, mean CIEDE2000 | 6.75 → **0.24** |
| Mug glaze vs. ground truth, CIEDE2000 | 6.08 → **0.68** |
| Triangles | 370k → 20k (watertight, volume within 0.2%) |
| Collision | 25 convex hulls, 1.135× solid volume |
| Mass (ceramic, 2400 kg/m³) | 371.9 g |
| Validation | 33 / 33 checks pass; settles in 0.14–0.61 s in 3 drop poses |
| Build time | 25.6 s (i9-13900K) |

## Quickstart

Requires Python 3.10–3.12. With [uv](https://docs.astral.sh/uv/):

```powershell
uv venv --python 3.11
uv pip install -e ".[dev]"
.venv\Scripts\scan2usd demo            # writes output\demo\ (see output\demo\preview.png)
.venv\Scripts\python -m pytest         # 45 tests
```

## Usage

```powershell
# Photogrammetry export in centimeters, Y-up; ceramic; color chart photo in the same light
scan2usd build mug.obj -o output\mug --units cm --up-axis y --material ceramic `
    --chart-image chart.jpg --chart-corners "412,310;1630,322;1622,1140;405,1128"

# Scale from a ruler measurement (meters) and a measured mass (kg)
scan2usd build shoe.glb -o output\shoe --scale-to 0.285 --scale-axis max --mass 0.41

# No ColorChecker? Use an 18% gray card (or white paper with --gray-reflectance none)
scan2usd build mug.ply -o output\mug --gray-card-image card.jpg --gray-card-box "820,410,980,560"

# Render a PNG of any mesh or USD asset (no 3D app needed)
scan2usd preview output\mug\mug.usd

# Validate any USD asset; fit a color matrix on its own
scan2usd validate output\mug\mug.usd --markdown report.md
scan2usd calibrate chart.jpg --chart-corners corners.json -o color_calibration.json
scan2usd calibrate card.jpg --neutral-box "820,410,980,560" -o color_calibration.json
scan2usd materials                     # density / friction presets
```

Chart corners are the outer corners of the 6×4 patch grid in pixels, ordered top-left (the "dark skin" patch corner), top-right, bottom-right, bottom-left.

Color reference options, measured on the demo's color cast (mean CIEDE2000; about 2 is barely noticeable):

| Reference | ΔE2000 | Corrects |
|---|---|---|
| none | 6.7 | – |
| white paper (`--gray-reflectance none`) | 3.8 | white balance |
| 18% gray card | 1.7 | white balance + exposure |
| ColorChecker Classic (24 patches) | 0.2 | white balance + exposure + hue cross-talk |

Outputs in the build directory:

| File | Contents |
|---|---|
| `<name>.usd` / `.usda` | Asset: render mesh, material, collision hulls, rigid body + mass |
| `<name>.glb` | Render mesh for web / DCC viewers (Y-up) |
| `textures/<name>_albedo.png` | Calibrated albedo (textured inputs) |
| `color_calibration.json` | CCM and ΔE statistics |
| `validation.md` | Human-readable check report |
| `preview.png` | Render mesh and collision hulls, side by side |
| `build_report.json` | All metrics, timings and validation results |

## Asset layout

```
/<name>                 Xform, kind=component, PhysicsRigidBodyAPI + PhysicsMassAPI
  Looks/M_surface       UsdPreviewSurface (texture, vertex colors or constant albedo)
  Looks/PM_contact      PhysicsMaterialAPI (friction, restitution)
  render                Mesh bound to M_surface
  collision/hull_###    Mesh, purpose=guide, CollisionAPI, MeshCollisionAPI(convexHull)
```

Stage metadata: `upAxis = Z`, `metersPerUnit = 1`, `kilogramsPerUnit = 1`.

## Code map

| Module | Role |
|---|---|
| `mesh_ops.py` | Load, clean, orient / scale, decimate |
| `color.py` | Color science, chart sampling, CCM fitting, CIEDE2000 |
| `physics.py` | Material presets, convex decomposition, mass properties |
| `usd_io.py` | USD writer / reader, glTF export |
| `simulate.py` | MuJoCo drop tests |
| `validate.py` | Quality checks and reports |
| `preview.py` | Dependency-free preview renders (numpy rasterizer) |
| `pipeline.py` | End-to-end orchestration |
| `demo.py`, `cli.py` | Synthetic capture, command line |

## Roadmap

- [x] **M1 · Pipeline core**: processing, calibration, collision, mass, USD/glTF, validation, MuJoCo drop tests
- [ ] **M2 · Real captures**: phone photogrammetry of 3–5 objects, reconstruction (RealityScan / COLMAP), chart detection, comparison with measured mass and size
- [ ] **M3 · Articulation**: revolute joints with limits for a hinged object
- [ ] **M4 · Isaac Sim**: drop / stack scenes and domain-randomized synthetic data; optional Gaussian-splat path
- [ ] **M5 · Showcase**: results, video, public release

## Notes

- Textured meshes are not decimated; their UV layout would break. Export a decimated, re-baked mesh from the reconstruction tool, or use vertex colors. Texture re-baking is planned.
- On Windows with Smart App Control enabled, `usd-core` prints a warning that `pxr.Ts` (animation splines) was blocked. Scan2USD doesn't use splines, so it's harmless.
- The default ColorChecker reference values are the commonly published sRGB values. For best accuracy, measure your own chart and pass `--reference` to `calibrate`.
