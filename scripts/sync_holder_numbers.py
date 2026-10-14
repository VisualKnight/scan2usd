"""Write result numbers into README.md and docs/overview.html from the result files.

    python scripts/sync_holder_numbers.py

Every synced number in the docs sits inside a marker such as <!--n:sim_add-->9.9<!--/n--> (HTML
comments are invisible on GitHub and in the page). This script rewrites them from
output/holder_fit/state.json and docs/figures/ (isaac_holder_fit, holder_place, hand_place, unreal_render .json),
so the published text always matches the published figures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def values() -> dict[str, str]:
    st = json.loads((ROOT / "output" / "holder_fit" / "state.json").read_text())
    isaac = json.loads((ROOT / "docs" / "figures" / "isaac_holder_fit.json").read_text())["results"]
    sim, hull, tr = st["sim"]["decomposed"], st["sim"]["one hull"], st["truth"]
    asset = st["assets"]
    return {
        "sim_add": f"{sim['add_mm_median']:.1f}", "sim_rot": f"{sim['rot_deg_median']:.0f}",
        "sim_held": f"{sim['held_in_holder']}/{sim['of']}", "hull_held": f"{hull['held_in_holder']}/{hull['of']}",
        "release": f"{sim['start_add_mm_median']:.0f}",
        "isaac_add": f"{isaac['decomposed']['add_mm_median']:.1f}",
        "isaac_rot": f"{isaac['decomposed']['rot_deg_median']:.0f}",
        "isaac_held": f"{isaac['decomposed']['held_in_holder']}/{isaac['decomposed']['of']}",
        "isaac_hull_held": f"{isaac['convex_hull']['held_in_holder']}/{isaac['convex_hull']['of']}",
        "merge_rmse": f"{st['merge']['rmse_mm']:.1f}",
        "truth_holder": f"{tr['holder']['rmse_mm']:.1f}", "truth_ctrl": f"{tr['controller']['rmse_mm']:.1f}",
        "scale_agree": f"{abs(tr['holder']['free_scale'] - 1) * 100:.1f}",
        "holder_hulls": str(asset["cod_holder"]["hulls"]), "ctrl_hulls": str(asset["controller"]["hulls"]),
        "base_r": f"{st['base']['radius_mm']:.1f}", "base_rms": f"{st['base']['rim_rms_mm']:.2f}",
        **unreal_values(),
        **place_values(),
        **hand_values(),
    }


def place_values() -> dict[str, str]:
    p = ROOT / "docs" / "figures" / "holder_place.json"
    if not p.exists():
        return {}
    s = json.loads(p.read_text())["summary"]
    return {"place_held": f"{s['hulls']['held']}/{s['hulls']['trials']}",
            "place_add": f"{s['hulls']['add_mm_median']:.1f}", "place_rot": f"{s['hulls']['rot_deg_median']:.0f}",
            "place_hull_held": f"{s['single']['held']}/{s['single']['trials']}"}


def hand_values() -> dict[str, str]:
    p = ROOT / "docs" / "figures" / "hand_place.json"
    if not p.exists():
        return {}
    s = json.loads(p.read_text())["summary"]
    ok, op, hull = s["hulls_reaimed"], s["hulls_open_loop"], s["single_reaimed"]
    return {"hand_held": f"{ok['held']}/{ok['trials']}", "hand_add": f"{ok['add_mm_median']:.1f}",
            "hand_rot": f"{ok['rot_deg_median']:.0f}", "hand_open_held": f"{op['held']}/{op['trials']}",
            "hand_open_add": f"{op['add_mm_median']:.1f}", "hand_open_rot": f"{op['rot_deg_median']:.0f}",
            "hand_hull_held": f"{hull['held']}/{hull['trials']}"}


def unreal_values() -> dict[str, str]:
    p = ROOT / "docs" / "figures" / "unreal_render.json"
    if not p.exists():
        return {}
    ue = json.loads(p.read_text())
    return {"ue_on": f"{ue['timing']['on']['gpu_ms']:.1f}", "ue_off": f"{ue['timing']['off']['gpu_ms']:.1f}",
            "ue_on_db": f"{ue['vs_path_tracer']['megalights_psnr_db']:.1f}",
            "ue_off_db": f"{ue['vs_path_tracer']['no_megalights_psnr_db']:.1f}"}


def main() -> None:
    v = values()
    for rel in ("README.md", "docs/overview.html"):
        p = ROOT / rel
        s = p.read_text(encoding="utf-8")
        used = set()

        def sub(m):
            used.add(m.group(1))
            return f"<!--n:{m.group(1)}-->{v[m.group(1)]}<!--/n-->"

        s = re.sub(r"<!--n:(\w+)-->.*?<!--/n-->", sub, s)
        p.write_text(s, encoding="utf-8")
        print(f"{rel}: {len(used)} numbers synced ({', '.join(sorted(used))})")


if __name__ == "__main__":
    main()
