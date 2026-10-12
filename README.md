# Scan2USD

**Scan-to-Sim Asset Pipeline**: turns real-world captures (photogrammetry / 3D Gaussian splatting) into physics-ready **OpenUSD** assets for robotics simulation.

> Personal R&D project · work in progress (Oct 2026)

## Planned pipeline
1. **Capture**: phone photogrammetry or Gaussian-splat capture of everyday objects (color checker in frame).
2. **Reconstruct**: mesh reconstruction from the capture.
3. **Process**: mesh cleanup and decimation, UVs, PBR textures, color-checker-calibrated albedo.
4. **Physics**: convex-decomposed collision, mass / inertia estimation, articulated joints.
5. **Export**: OpenUSD with UsdPhysics schemas (plus glTF).
6. **Validate**: automated checks for mesh integrity, scale, inertia sanity and simulation stability.
7. **Simulate**: load into a robotics simulator and render domain-randomized synthetic data.
