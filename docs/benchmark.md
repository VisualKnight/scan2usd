# Scan2USD synthetic benchmark

Each object is rendered as a 78-photo phone-style capture in Blender (3 rings + top views, warm light, ColorChecker in the scene), reconstructed with COLMAP, and built by Scan2USD using one ruler measurement (height, rounded to 1 mm). Scores compare the final USD render mesh against the ground-truth mesh after rigid alignment.

- **Accuracy**: distance from the reconstruction to the true surface. **Completeness**: distance from the true surface to the reconstruction.
- **F-score @2 mm**: share of both surfaces within 2 mm of each other (1.0 = perfect).

| Object | Photos registered | Accuracy mean / p95 (mm) | Completeness mean (mm) | F-score @2 mm | F-score @2 mm, above base | Volume error | Chart ΔE2000 | Validation | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| Speckled stoneware mug (procedural) | 78/78 | 0.78 / 2.87 | 0.64 | 0.934 | 0.921 | +11.8% | 7.6 → 1.31 | warn | 315 |
| Printed coffee mug | 78/78 | 2.00 / 10.20 | 2.38 | 0.731 | 0.748 | +191.5% | 8.0 → 1.30 | pass | 684 |
| Ceramic bowl | 78/78 | 3.03 / 9.61 | 2.34 | 0.494 | 0.524 | +82.0% | 7.8 → 1.32 | warn | 891 |
| Android figure toy* | 78/78 | 0.78 / 3.08 | 0.96 | 0.872 | 0.894 | +1.4% | 7.7 → 1.38 | warn | 648 |
| Lion figure | 78/78 | 0.60 / 2.41 | 0.62 | 0.941 | 0.935 | +8.2% | 8.0 → 1.29 | pass | 552 |
| Boat shoe | 78/78 | 1.73 / 7.20 | 2.53 | 0.747 | 0.739 | +32.2% | 8.9 → 1.31 | pass | 627 |

Ground-truth meshes: Google Scanned Objects (CC-BY 4.0) and a procedural mug. *The GSO model id `Great_Dinos_Triceratops_Toy` actually contains an Android figure.

## Metric scale from the ColorChecker (no ruler)

Scale estimated by triangulating the chart's patch centers in the photos (RANSAC over views), compared with the true scale from the ground-truth cameras.

| Object | Chart views used | Scale error |
|---|---|---|
| Speckled stoneware mug (procedural) | 20/22 | +0.96% |
| Printed coffee mug | 20/24 | +0.62% |
| Ceramic bowl | 9/12 | -0.91% |
| Android figure toy* | 23/26 | +0.42% |
| Lion figure | 24/26 | +1.43% |
| Boat shoe | 35/43 | +0.35% |

Mean absolute scale error: **0.78%** (max 1.43%).

## How many photos? (schleich_lion_action_figure)

Evenly spaced subsets of the same capture, each reconstructed from scratch. With 12 photos only 8 register and the result is unusable; 20 photos already give sub-millimeter accuracy; beyond ~40, extra photos add little accuracy but cost time.

| Photos | Registered | Accuracy (mm) | Completeness (mm) | F-score @2 mm | Volume error | COLMAP + mesh (s) |
|---|---|---|---|---|---|---|
| 12 | 8 | 8.05 | 6.71 | 0.102 | +157.8% | 40 |
| 20 | 19 | 0.80 | 0.89 | 0.896 | +6.9% | 108 |
| 39 | 39 | 0.61 | 0.65 | 0.920 | +10.1% | 238 |
| 78 | 78 | 0.60 | 0.62 | 0.941 | +8.2% | 480 |
