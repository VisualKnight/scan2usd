"""Create a synthetic photogrammetry capture with known ground truth.

    python scripts/make_synthetic_capture.py                       # speckled mug -> data/synthetic/mug
    python scripts/make_synthetic_capture.py --gso Schleich_Lion_Action_Figure
    python scripts/make_synthetic_capture.py --quick               # a few low-sample images

--gso uses a Google Scanned Objects model (CC-BY 4.0) from data/external/gso/<name>/,
keeping its scanned texture. Writes the ground truth (gt_mesh.*) next to the photos and
renders them with Blender (scripts/blender_render_capture.py). Set BLENDER to override
the Blender path.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"


def main() -> int:
    import trimesh

    from scan2usd.demo import make_mug

    p = argparse.ArgumentParser()
    p.add_argument("--gso", help="Google Scanned Objects model folder name under data/external/gso/")
    p.add_argument("--out", type=Path)
    p.add_argument("--quick", action="store_true", help="a handful of low-sample images to check the setup")
    p.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    args = p.parse_args()

    extra = []
    if args.gso:
        src = ROOT / "data" / "external" / "gso" / args.gso
        out = args.out or ROOT / "data" / "synthetic" / args.gso.lower()
        out.mkdir(parents=True, exist_ok=True)
        gt_dir = out / "gt"
        if gt_dir.exists():
            shutil.rmtree(gt_dir)
        shutil.copytree(src / "meshes", gt_dir / "meshes")
        shutil.copytree(src / "materials", gt_dir / "materials")
        # model.mtl references its textures by bare file name, next to the OBJ.
        for tex in (src / "materials" / "textures").glob("*"):
            shutil.copy2(tex, gt_dir / "meshes" / tex.name)
        gt_path = gt_dir / "meshes" / "model.obj"
        mesh = trimesh.load(str(gt_path), force="mesh")
        extra.append("--keep-materials")
    else:
        out = args.out or ROOT / "data" / "synthetic" / "mug"
        out.mkdir(parents=True, exist_ok=True)
        mesh = make_mug(sections=128).subdivide_to_size(max_edge=0.003, max_iter=10)
        gt_path = out / "gt_mesh.ply"
        mesh.export(str(gt_path))
    (out / "gt_path.txt").write_text(str(gt_path.relative_to(out)))
    size = float(max(mesh.extents))
    distance = max(0.2, 2.0 * size)
    print(f"ground truth: {gt_path} ({len(mesh.faces)} faces, {size * 100:.1f} cm), camera distance {distance:.2f} m")

    if args.quick:
        extra += ["--per-ring", "2", "--top", "1", "--samples", "16"]
    cmd = [args.blender, "-b", "--factory-startup", "--python", str(ROOT / "scripts" / "blender_render_capture.py"),
           "--", "--mesh", str(gt_path), "--out", str(out), "--distance", f"{distance:.3f}", *extra]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
