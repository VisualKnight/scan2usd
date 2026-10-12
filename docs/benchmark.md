# Scan2USD synthetic benchmark

Each object is rendered as a 78-photo phone-style capture in Blender (3 rings + top views, warm light, ColorChecker in the scene), reconstructed with COLMAP, and built by Scan2USD using one ruler measurement (height, rounded to 1 mm). Scores compare the final USD render mesh against the ground-truth mesh after rigid alignment.

- **Accuracy**: distance from the reconstruction to the true surface. **Completeness**: distance from the true surface to the reconstruction.
- **F-score @2 mm**: share of both surfaces within 2 mm of each other (1.0 = perfect).

| Object | Photos registered | Accuracy mean / p95 (mm) | Completeness mean (mm) | F-score @2 mm | Volume error | Chart ΔE2000 | Validation | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Speckled stoneware mug (procedural) | 78/78 | 0.78 / 2.87 | 0.64 | 0.934 | +11.8% | 7.6 → 1.31 | warn | 315 |
| Printed coffee mug | – | failed: FileNotFoundError: [WinError 3] The system cannot find the path specified: 'D:\\Local Projects\\Scan2USD\\data\\synthetic\\ace_coffee_mug_kristen_16_oz_cup\\ima | | | | | | |
| Ceramic bowl | – | failed: FileNotFoundError: [WinError 3] The system cannot find the path specified: 'D:\\Local Projects\\Scan2USD\\data\\synthetic\\cole_hardware_bowl_scirocco_yellowblu | | | | | | |
| Triceratops toy | – | failed: FileNotFoundError: [WinError 3] The system cannot find the path specified: 'D:\\Local Projects\\Scan2USD\\data\\synthetic\\great_dinos_triceratops_toy\\images' | | | | | | |
| Lion figure | – | failed: FileNotFoundError: [WinError 3] The system cannot find the path specified: 'D:\\Local Projects\\Scan2USD\\data\\synthetic\\schleich_lion_action_figure\\images' | | | | | | |
| Boat shoe | – | failed: FileNotFoundError: [WinError 3] The system cannot find the path specified: 'D:\\Local Projects\\Scan2USD\\data\\synthetic\\womens_bluefish_2eye_boat_shoe_in_tan | | | | | | |

Ground-truth meshes: Google Scanned Objects (CC-BY 4.0) and a procedural mug.
