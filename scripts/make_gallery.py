"""Interactive 3D viewer for Scan2USD assets (glTF + Google's model-viewer, no install needed).

    python scripts/make_gallery.py

Collects built assets into docs/gallery/ as compact .glb files (decimated, textures
downsized) and writes docs/gallery/index.html: orbit, zoom, and toggle each asset's
convex collision hulls. Open the HTML file in a browser (served over http:// for some
browsers; e.g. `python -m http.server -d docs`).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scan2usd.mesh_ops import decimate
from scan2usd.preview import hull_colors
from scan2usd.usd_io import read_usd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "gallery"

ASSETS = [
    ("data/work/mug_scan/asset/speckled_mug.usda", "Speckled mug",
     "78 synthetic photos → COLMAP → Scan2USD, scaled by the ColorChecker, texture baked (0.8 mm accuracy)"),
    ("output/buddha/buddha_head.usda", "Buddha head (real photos)", "67 real photos (AliceVision dataset, CC-BY 4.0)"),
    ("output/holder_fit/assets/cod_holder/cod_holder.usda", "Controller holder (own phone photos)",
     "81 iPhone photos; scaled from one tape measurement, measured mass, round base fitted as a cylinder"),
    ("output/holder_fit/assets/controller/controller.usda", "Game controller (own phone photos)",
     "78 + 38 iPhone photos in two passes, merged by similarity ICP; measured mass"),
    ("output/gso_assets/schleich_lion_action_figure/schleich_lion_action_figure.usda", "Lion figure",
     "Google Scanned Objects (CC-BY 4.0): 4K scanned texture, 32 convex colliders"),
    ("output/gso_assets/womens_bluefish_2eye_boat_shoe_in_tan/womens_bluefish_2eye_boat_shoe_in_tan.usda",
     "Boat shoe", "Google Scanned Objects (CC-BY 4.0)"),
    ("output/gso_assets/ace_coffee_mug_kristen_16_oz_cup/ace_coffee_mug_kristen_16_oz_cup.usda", "Printed mug",
     "Google Scanned Objects (CC-BY 4.0)"),
]


def to_y_up(m: trimesh.Trimesh) -> trimesh.Trimesh:
    m = m.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return m


def export_render(asset, path: Path, faces: int = 25000, tex: int = 1024) -> None:
    mesh = asset.render_mesh
    if mesh.visual.kind == "texture":
        image = mesh.visual.material.image
        if image is not None and max(image.size) > tex:
            image = image.convert("RGB").resize((tex, tex), Image.LANCZOS)
        out = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
        # glTF's default metallicFactor is 1.0: unset, a scan's albedo renders as dark metal. Scans are
        # dielectric, so say so explicitly.
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=image, metallicFactor=0.0, roughnessFactor=0.75)
        out.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=mat)
    else:
        out, _ = decimate(mesh, faces)
    to_y_up(out).export(str(path), include_normals=True)


def export_hulls(asset, path: Path) -> None:
    parts = []
    for h, c in zip(asset.hulls, hull_colors(len(asset.hulls))):
        h = h.copy()
        h.visual = trimesh.visual.ColorVisuals(h, face_colors=np.tile([*c, 255], (len(h.faces), 1)))
        parts.append(h)
    to_y_up(trimesh.util.concatenate(parts)).export(str(path))


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    items = []
    for rel, title, caption in ASSETS:
        src = ROOT / rel
        if not src.exists():
            print(f"skip (not built): {rel}")
            continue
        asset = read_usd(src)
        slug = src.stem
        export_render(asset, OUT / f"{slug}.glb")
        export_hulls(asset, OUT / f"{slug}_hulls.glb")
        ext = asset.render_mesh.extents * 100
        items.append({"slug": slug, "title": title, "caption": caption, "mass_g": round(asset.mass * 1000, 1),
                      "hulls": len(asset.hulls), "size_cm": " × ".join(f"{v:.1f}" for v in ext)})
        size = sum(p.stat().st_size for p in OUT.glob(f"{slug}*.glb")) / 1e6
        print(f"{slug}: {size:.1f} MB")
    (OUT / "index.html").write_text(PAGE.replace("__ITEMS__", json.dumps(items)), encoding="utf-8")
    print(f"wrote {OUT / 'index.html'} ({len(items)} assets)")


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scan2USD · 3D gallery</title>
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
<style>
  body { margin: 0; font: 15px/1.5 "Segoe UI", system-ui, sans-serif; color: #1e293b; background: #f8fafc; }
  .wrap { max-width: 1160px; margin: 0 auto; padding: 28px 24px 48px; }
  h1 { margin: 0 0 4px; font-size: 28px; } .muted { color: #64748b; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; margin-top: 18px; }
  .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; }
  model-viewer { width: 100%; height: 320px; background: linear-gradient(#fff, #eef2f7); }
  .meta { padding: 10px 14px 14px; } .meta h3 { margin: 0 0 2px; font-size: 16px; }
  .meta p { margin: 4px 0; font-size: 13px; color: #64748b; }
  label { font-size: 13px; cursor: pointer; user-select: none; }
</style></head>
<body><div class="wrap">
<h1>Scan2USD · 3D gallery</h1>
<p class="muted">Drag to orbit, scroll to zoom. Tick "collision" to see the convex pieces the physics engine uses.
<a href="../overview.html">Back to the overview</a> · <a href="../library/index.html">USD asset library</a></p>
<div class="grid" id="grid"></div>
</div>
<script>
const items = __ITEMS__;
const grid = document.getElementById('grid');
for (const it of items) {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `
    <model-viewer src="${it.slug}.glb" camera-controls auto-rotate shadow-intensity="0.6" exposure="1.3"
      tone-mapping="neutral" environment-image="neutral" alt="${it.title}"></model-viewer>
    <div class="meta"><h3>${it.title}</h3><p>${it.caption}</p>
      <p>${it.size_cm} cm · ${it.mass_g} g · ${it.hulls} convex colliders</p>
      <label><input type="checkbox"> collision</label></div>`;
  const mv = card.querySelector('model-viewer');
  card.querySelector('input').addEventListener('change', e => {
    mv.src = e.target.checked ? `${it.slug}_hulls.glb` : `${it.slug}.glb`;
  });
  grid.appendChild(card);
}
</script></body></html>
"""

if __name__ == "__main__":
    main()
