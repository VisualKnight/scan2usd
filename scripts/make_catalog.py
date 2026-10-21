"""Asset catalog page for the USD library: specs and the three collision variants per asset.

    python scripts/make_catalog.py [output/library/scan2usd_library.usda] [--out docs/library]

Writes docs/library/index.html and thumbnails (rendered with scan2usd.preview, no 3D app needed), with each
asset's SimReady result from docs/figures/simready.json (scripts/study_simready.py) when present.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from scan2usd.library import COLLISION_VARIANTS, library_entries
from scan2usd.preview import hull_colors, render
from scan2usd.usd_io import read_usd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"decomposed": "convex decomposition", "convex_hull": "one convex hull", "bounding_box": "bounding box"}
TITLES = {
    "ace_coffee_mug_kristen_16_oz_cup": "Printed mug", "cole_hardware_bowl_scirocco_yellowblue": "Ceramic bowl",
    "great_dinos_triceratops_toy": "Android figure", "schleich_lion_action_figure": "Lion figure",
    "womens_bluefish_2eye_boat_shoe_in_tan": "Boat shoe", "ceramic_mug": "Ceramic mug (synthetic scan)",
    "speckled_mug": "Speckled mug (78 photos → USD)", "keepsake_box": "Hinged box (articulated)",
    "buddha_head": "Buddha head (67 real photos)", "cod_holder": "Controller holder (own phone photos)",
    "controller": "Game controller (own phone photos)",
}

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scan2USD · asset library</title>
<style>
body {{ font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: #1e293b; background: #f8fafc; margin: 0; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 60px; }}
h1 {{ margin: 0 0 4px; }} .muted {{ color: #64748b; }}
.card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; margin: 14px 0; }}
.row {{ display: grid; grid-template-columns: 260px 1fr; gap: 18px; align-items: start; }}
.thumbs {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
.thumbs figure {{ margin: 0; text-align: center; }} .thumbs img {{ width: 100%; border-radius: 6px; border: 1px solid #f1f5f9; }}
figcaption {{ font-size: 12.5px; color: #64748b; }}
table {{ border-collapse: collapse; font-size: 14px; width: 100%; }} td {{ padding: 3px 8px 3px 0; vertical-align: top; overflow-wrap: anywhere; }}
td:first-child {{ color: #64748b; width: 70px; }}
pre {{ background: #0f172a; color: #e2e8f0; padding: 12px 14px; border-radius: 8px; overflow-x: auto; font-size: 13px; }}
code {{ font-family: ui-monospace, Consolas, monospace; }} a {{ color: #2563eb; }}
.sr {{ display: inline-block; background: #dcfce7; color: #166534; border-radius: 999px; padding: 0 8px; font-size: 12.5px; font-weight: 600; }}
@media (max-width: 820px) {{ .row {{ grid-template-columns: 1fr; }} }}
</style></head><body><main>
<p><a href="../overview.html">← Overview</a> · <a href="../gallery/index.html">3D gallery</a> · <a href="https://github.com/VisualKnight/scan2usd">GitHub</a></p>
<h1>Asset library</h1>
<p class="muted">{n} assets composed into one USD stage, <code>scan2usd_library.usda</code>. Each asset is a <b>payload</b>
(loaded on demand) with two <b>variant sets</b> authored in the library layer, so the asset files are never edited:
<b>collision</b> (decomposed · convex_hull · bounding_box) and <b>friction</b> (measured · low · high).
Total {mass:.2f} kg, {colliders} convex colliders in the default variant.{simready}</p>
<div class="card"><b>Use it</b>
<pre>scan2usd library output/assets -o library.usda        # build

from pxr import Usd
stage = Usd.Stage.Open("library.usda", Usd.Stage.LoadNone)   # instant: nothing loaded yet
mug = stage.GetPrimAtPath("/Library/ceramic_mug")
stage.Load(mug.GetPath())                                     # load just this asset
mug.GetVariantSet("collision").SetVariantSelection("convex_hull")
mug.GetVariantSet("friction").SetVariantSelection("low")</pre>
<p class="muted small">Why variants: the same scene can be simulated with cheap or accurate contact geometry (ablations, speed vs. fidelity),
and contact properties can be randomized for robust policies, without copying assets.</p></div>
{cards}
<p class="muted">Renders: Scan2USD's numpy preview renderer. Third-party meshes: Google Scanned Objects (CC-BY 4.0); Buddha: AliceVision dataset (CC-BY 4.0).</p>
</main></body></html>
"""


def simready_results() -> tuple[dict, dict]:
    """Per-asset SimReady results from scripts/study_simready.py, keyed by asset name."""
    p = ROOT / "docs" / "figures" / "simready.json"
    if not p.exists():
        return {}, {}
    d = json.loads(p.read_text())
    return d["summary"], {r["name"]: r for r in d["assets"]}


def simready_rows(summary: dict, r: dict | None) -> list[tuple[str, str]]:
    if r is None or not r["package"]["passed"]:
        return []
    qcode = html.escape(r["qcode"])
    return [("class", f'{html.escape(r["category"])} · <a href="https://www.wikidata.org/wiki/{qcode}">{qcode}</a>'),
            ("SimReady", f'<span class="sr">✓ {html.escape(summary["profile"])} '
                         f'{html.escape(summary["version"])}</span>')]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("library", type=Path, nargs="?", default=ROOT / "output" / "library" / "scan2usd_library.usda")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "library")
    p.add_argument("--size", type=int, default=240)
    args = p.parse_args()
    (args.out / "thumbs").mkdir(parents=True, exist_ok=True)

    sr_summary, sr_assets = simready_results()
    cards, total_mass, total_cols = [], 0.0, 0
    for e in library_entries(args.library):
        name = e["name"]
        variants = {v: read_usd(args.library, e["path"], {"collision": v}) for v in COLLISION_VARIANTS}
        base = variants["decomposed"]
        bounds = np.array([base.render_mesh.bounds.min(0), base.render_mesh.bounds.max(0)])
        view = {"size": args.size, "bounds": bounds, "azimuth": -55, "elevation": 24}
        render([base.render_mesh], **view).save(args.out / "thumbs" / f"{name}.jpg", quality=88)
        figs = []
        for v, a in variants.items():
            render(a.hulls, colors=hull_colors(len(a.hulls)), **view).save(
                args.out / "thumbs" / f"{name}_{v}.jpg", quality=88)
            figs.append(f'<figure><img src="thumbs/{name}_{v}.jpg" alt="{v}" loading="lazy">'
                        f'<figcaption>{LABELS[v]} · {len(a.hulls)}</figcaption></figure>')
        size_cm = " × ".join(f"{v * 100:.1f}" for v in e["size_m"])
        mat = base.physics_material
        inertia = base.diagonal_inertia
        rows = [
            ("size", f"{size_cm} cm"),
            ("mass", f"{e['mass_kg'] * 1000:.0f} g"),
            ("inertia", "articulated (per body)" if inertia is None else
             " · ".join(f"{v * 1e6:.0f}" for v in inertia) + " g·cm²"),
            ("friction", f"μs {mat.get('static_friction', 0):.2f} · μd {mat.get('dynamic_friction', 0):.2f}"),
            ("prim", f"<code>{html.escape(e['path'])}</code>"),
        ]
        rows += simready_rows(sr_summary, sr_assets.get(name))
        table = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
        cards.append(f'<div class="card"><div class="row"><div><img src="thumbs/{name}.jpg" alt="{name}" '
                     f'style="width:100%;border-radius:6px"><h3 style="margin:6px 0 4px">'
                     f'{html.escape(TITLES.get(name, name.replace("_", " ")))}</h3>'
                     f'<table>{table}</table></div><div class="thumbs">{"".join(figs)}</div></div></div>')
        total_mass += e["mass_kg"]
        total_cols += e["colliders"]["decomposed"]
        print(f"  {name}")

    simready = ""
    if sr_summary:
        simready = (f'<br>Each asset also ships as an <a href="https://github.com/NVIDIA/simready-foundation">NVIDIA '
                    f'SimReady</a> package (<code>scan2usd simready</code>): {sr_summary["package_passed"]}/{len(cards)} '
                    f'pass the {html.escape(sr_summary["profile"])} {html.escape(sr_summary["version"])} profile '
                    f'with NVIDIA\'s validator.')
    page = PAGE.format(n=len(cards), mass=total_mass, colliders=total_cols, cards="\n".join(cards),
                       simready=simready)
    (args.out / "index.html").write_text(page, encoding="utf-8")
    (args.out / "library.json").write_text(json.dumps(library_entries(args.library), indent=2))
    print(f"wrote {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
