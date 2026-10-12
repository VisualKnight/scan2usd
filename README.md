# Scan2USD

**Scan-to-Sim asset pipeline**: phone photos of everyday objects in, physics-ready **OpenUSD** assets out, verified in NVIDIA **Isaac Sim** and **MuJoCo**.

> Personal R&D project · Oct 2026 · visual overview: [docs/overview.html](docs/overview.html) · benchmark: [docs/benchmark.md](docs/benchmark.md)

![Architecture](docs/figures/architecture.svg)

## What it does

| Stage | What happens | Module |
|---|---|---|
| Reconstruct | COLMAP structure-from-motion + multi-view stereo; the table plane becomes z = 0 (Z-up); the object is isolated as the densest footprint above the table; screened Poisson surface; flat base | `reconstruct.py` |
| Clean | Debris removed, holes filled, meters / Z-up, origin at the base, topology-preserving decimation | `mesh_ops.py` |
| Color | 3×3 color-correction matrix from a ColorChecker (or gray card) applied to the albedo; CIEDE2000 reported | `color.py` |
| Physics | CoACD convex decomposition; mass, center of mass and principal inertia from volume × density or a measured mass; voxel-fill solids for scans with holes | `physics.py` |
| Articulation | Split a scan by a seam plane, one rigid body per part, `UsdPhysics.RevoluteJoint` with limits under an articulation root | `articulation.py` |
| Export | OpenUSD (`UsdGeom`, `UsdShade` preview surface, `UsdPhysics` rigid body / mass / colliders / physics material) and glTF | `usd_io.py` |
| Validate | 33 checks on the written file, including MuJoCo drop tests | `validate.py`, `simulate.py` |
| Evaluate | Accuracy / completeness / F-score / volume against ground truth after alignment | `evaluate.py` |

## Results

**Photos → USD, scored against ground truth.** Each object is a 78-photo phone-style capture rendered in Blender (with a ColorChecker in the scene), reconstructed with COLMAP, and built with one ruler measurement. Full table: [docs/benchmark.md](docs/benchmark.md).

<!-- BENCHMARK_TABLE -->

| | |
|---|---|
| ![Texture matters](docs/figures/texture_matters.jpg) | Photogrammetry needs surface texture: the same mug with a plain dark glaze leaves holes; with a speckled glaze it reconstructs to 0.8 mm. |

**Robotics demos** (MuJoCo, and Isaac Sim 6 with PhysX + Replicator):

| | |
|---|---|
| ![Grasp](docs/figures/grasp_demo.jpg) | **Rim grasp.** One finger inside the cup: with Scan2USD's convex decomposition (25 pieces) the mug lifts 11.6 cm; with one convex hull the cavity is sealed and the grasp fails. `scripts/demo_grasp.py` |
| ![Isaac](docs/figures/isaac_dataset.jpg) | **Isaac Sim 6.** 8 assets referenced unmodified, settled by PhysX, rendered by Replicator with random cameras and lights: RGB, tight 2D boxes, segmentation, depth. `scripts/isaac_demo.py` |
| ![Bin](docs/figures/clutter_demo.jpg) | **Bin of scanned objects.** 14 objects, 392 convex colliders; all settle in the tote, resting penetration 0.08 mm. `scripts/demo_clutter.py` |
| ![Hinge](docs/figures/articulation_demo.jpg) | **Articulated asset.** A closed box scan + seam plane + hinge line → base and lid with a revolute joint (0–105°); the lid opens, a scanned toy drops in, the lid closes. `scripts/demo_articulation.py` |

## Quickstart

Python 3.10–3.12, with [uv](https://docs.astral.sh/uv/):

```powershell
uv venv --python 3.11
uv pip install -e ".[dev]"
.venv\Scripts\scan2usd demo            # synthetic mug -> output\demo\ (see preview.png)
.venv\Scripts\python -m pytest         # 60 tests
```

Photo reconstruction needs [COLMAP](https://github.com/colmap/colmap/releases) (CUDA build): unzip it to `tools\colmap\` or set `SCAN2USD_COLMAP`.

## Usage

```powershell
# Photos -> object mesh (Z-up, table at z=0, arbitrary scale)
scan2usd reconstruct photos\ -o work\mug

# Mesh -> simulation asset: scale from one ruler measurement, measured mass, color chart
scan2usd build work\mug\object_mesh.ply -o output\mug --scale-to 0.095 --scale-axis z --mass 0.36 `
    --material ceramic --chart-image chart.jpg --chart-corners "412,310;1630,322;1622,1140;405,1128"

# No ColorChecker? 18% gray card (or white paper with --gray-reflectance none)
scan2usd build mug.ply -o output\mug --gray-card-image card.jpg --gray-card-box "820,410,980,560"

# Hinged object: seam plane and hinge line in the asset frame (m, Z-up, origin at bottom center)
scan2usd articulate box.ply -o output\box --split "0,0,0.052,0,0,1" --hinge "0,0.05,0.052,-1,0,0" --limits 0,105

scan2usd preview output\mug\mug.usd    # PNG of render mesh + colliders, no 3D app needed
scan2usd validate output\mug\mug.usd --markdown report.md
scan2usd calibrate chart.jpg --chart-corners corners.json
scan2usd materials                     # density / friction presets
```

Chart corners are the outer corners of the 6×4 patch grid in pixels: top-left ("dark skin"), top-right, bottom-right, bottom-left.

Build outputs: `<name>.usd(a)`, `<name>.glb`, `textures/`, `color_calibration.json`, `validation.md`, `preview.png`, `build_report.json`.

## Scripts

| Script | Purpose |
|---|---|
| `make_synthetic_capture.py` | Render a phone-style capture of a ground-truth mesh in Blender (procedural mug or `--gso <model>`) |
| `evaluate_synthetic.py`, `run_benchmark.py` | Photos → COLMAP → Scan2USD → score against ground truth |
| `demo_grasp.py`, `demo_clutter.py`, `demo_articulation.py` | MuJoCo robotics demos (animations in `docs/figures/`) |
| `isaac_demo.py`, `make_isaac_figure.py` | Isaac Sim 6: PhysX settle test + Replicator dataset (run with Isaac Sim's Python) |
| `blender_inspect.py` | Open a build in Blender: raw input, asset, colliders |

## Asset layout

```
/<name>                 Xform, kind=component, PhysicsRigidBodyAPI + PhysicsMassAPI
  Looks/M_surface       UsdPreviewSurface (texture, vertex colors or constant albedo)
  Looks/PM_contact      PhysicsMaterialAPI (friction, restitution)
  render                Mesh bound to M_surface
  collision/hull_###    Mesh, purpose=guide, CollisionAPI, MeshCollisionAPI(convexHull)
```

Articulated assets put one such body per part under a root with `PhysicsArticulationRootAPI`, plus `joints/hinge` (`PhysicsRevoluteJoint`). Stage metadata: `upAxis = Z`, `metersPerUnit = 1`, `kilogramsPerUnit = 1`.

## Known limitations

- Plain, dark or glossy surfaces and deep textureless cavities don't reconstruct well (see the benchmark); a measured `--mass` corrects mass when the volume is off.
- Textured meshes aren't decimated (the UV layout would break); export a decimated, re-baked mesh from the reconstruction tool.
- Part splitting for articulation is manual (plane + hinge line).
- Windows with Smart App Control: `usd-core` warns that `pxr.Ts` (splines, unused) is blocked. Isaac Sim 5.1's RTX renderer crashed on driver 617; 6.0 works.

## Credits

Ground-truth meshes: [Google Scanned Objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research) (CC-BY 4.0; the model id `Great_Dinos_Triceratops_Toy` contains an Android figure). Tools: COLMAP, CoACD, OpenUSD, MuJoCo, NVIDIA Isaac Sim, Blender, trimesh, pymeshlab.
