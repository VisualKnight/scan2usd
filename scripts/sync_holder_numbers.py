"""Write result numbers into README.md and docs/overview.html from the result files.

    python scripts/sync_holder_numbers.py

Every synced number in the docs sits inside a marker such as <!--n:sim_add-->9.9<!--/n--> (HTML
comments are invisible on GitHub and in the page). This script rewrites them from
output/holder_fit/state.json and docs/figures/ (isaac_holder_fit, holder_place, hand_place, isaac_hand_place, unreal_render,
genai .json),
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
        **isaac_hand_values(),
        **reel_values(),
        **learned_values(),
        **progress_values(),
        **simready_values(),
        **vision_values(),
        **genai_values(),
    }


def genai_values() -> dict[str, str]:
    """Generated vs scanned (study_generative.py -> docs/figures/genai.json)."""
    p = ROOT / "docs" / "figures" / "genai.json"
    if not p.exists():
        return {}
    src = json.loads(p.read_text())["sources"]
    short = {"photogrammetry": "pg", "hunyuan2mv": "mv", "hunyuan21": "h21", "triposg": "tsg"}
    out = {}
    for k, s in src.items():
        t = short[k]
        out[f"gen_f_{t}"] = f"{s['fscore_2mm_mean']:.3f}"
        out[f"gen_mass_{t}"] = f"{s['mass_abs_error_pct_median']:.0f}"
        out[f"gen_t_{t}"] = f"{s['seconds_mean']:.0f}"
        out[f"gen_wins_{t}"] = f"{s['wins_fscore_vs_photogrammetry']}/{s['objects']}"
        if "real_drop" in s:
            d = s["real_drop"]
            out[f"gen_held_{t}"] = f"{d['held']}/{d['of']}"
            out[f"gen_add_{t}"] = f"{d['add_mm_median']:.1f}"
        if "real_drop_seeds" in s:
            seeds = s["real_drop_seeds"]
            out[f"gen_seeds_{t}"] = f"{sum(x['held'] for x in seeds)}/{sum(x['of'] for x in seeds)}"
            out[f"gen_nseeds_{t}"] = str(len(seeds))
        if "look" in s:
            out[f"gen_de_{t}"] = f"{s['look']['delta_e2000_wb_median']:.1f}"
            out[f"gen_psnr_{t}"] = f"{s['look']['psnr_db']:.1f}"
    return out


def place_values() -> dict[str, str]:
    p = ROOT / "docs" / "figures" / "holder_place.json"
    if not p.exists():
        return {}
    s = json.loads(p.read_text())["summary"]
    out = {}
    ip = ROOT / "docs" / "figures" / "isaac_holder_place.json"
    if ip.exists():
        d = json.loads(ip.read_text())["summary"]["decomposed"]
        h = json.loads(ip.read_text())["summary"]["convex_hull"]
        out.update({"iplace_held": f"{d['held']}/{d['trials']}", "iplace_add": f"{d['add_mm_median']:.1f}",
                    "iplace_rot": f"{d['rot_deg_median']:.0f}", "iplace_hull_held": f"{h['held']}/{h['trials']}"})
    return {**out, "place_held": f"{s['hulls']['held']}/{s['hulls']['trials']}",
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


def isaac_hand_values() -> dict[str, str]:
    p = ROOT / "docs" / "figures" / "isaac_hand_place.json"
    if not p.exists():
        return {}
    s = json.loads(p.read_text())["summary"]
    ok, op, hull = s["decomposed_reaimed"], s["decomposed_open_loop"], s["convex_hull_reaimed"]
    return {"ihand_held": f"{ok['held']}/{ok['trials']}", "ihand_add": f"{ok['add_mm_median']:.1f}",
            "ihand_rot": f"{ok['rot_deg_median']:.0f}", "ihand_open_held": f"{op['held']}/{op['trials']}",
            "ihand_open_add": f"{op['add_mm_median']:.1f}", "ihand_open_rot": f"{op['rot_deg_median']:.0f}",
            "ihand_hull_held": f"{hull['held']}/{hull['trials']}"}


def learned_values() -> dict[str, str]:
    """The learned hand-place policy (scripts/rl/train_policy.py -> docs/figures/learned_hand_place.json)."""
    p = ROOT / "docs" / "figures" / "learned_hand_place.json"
    if not p.exists():
        return {}
    s = json.loads(p.read_text())
    ev, film = s["eval"], s["film"]
    return {"learn_placed": f"{ev['placed']}/{ev['envs']}", "learn_add": f"{ev['add_mm_median']:.1f}",
            "learn_rot": f"{ev['rot_deg_median']:.0f}", "learn_demos": f"{s['demos']:,}", "learn_rounds": str(s["rounds"]),
            "learn_min": f"{s['train_minutes']:.0f}", "learn_film": f"{film['placed']}/{film['envs']}",
            "learn_power": f"{s['power_max_w']:.0f}", "teach_add": f"{s['teacher']['add_mm_median']:.1f}",
            "teach_rot": f"{s['teacher']['rot_deg_median']:.0f}"}


def progress_values() -> dict[str, str]:
    """The learned policy before and beyond (scripts/rl/eval_sweep.py + plot_learning.py)."""
    p = ROOT / "docs" / "figures" / "learned_progress.json"
    if not p.exists():
        return {}
    s = json.loads(p.read_text())
    cur = {j["label"]: j for j in s["curve"]}
    jit = {(j["label"], j["scale"]): j for j in s["jitter"]}

    def frac(j):
        return f"{j['placed']}/{j['envs']}"

    out = {"prog_bc10k": frac(cur["bc10000"]), "prog_bc": frac(cur["bc40000"]), "prog_r0": frac(cur["r0"]),
           "prog_r1": frac(cur["r1"])}
    for sc in (2.0, 3.0, 4.0):
        out[f"jit{sc:g}_net"], out[f"jit{sc:g}_teach"] = frac(jit[("final", sc)]), frac(jit[("teacher", sc)])
    for k, f in s.get("films", {}).items():
        out[f"film_{k.removeprefix('stage_')}"] = frac(f)
    return out


def simready_values() -> dict[str, str]:
    """NVIDIA SimReady conformance of the library (scripts/study_simready.py)."""
    p = ROOT / "docs" / "figures" / "simready.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    s, rows = d["summary"], d["assets"]
    n = len(rows)

    def met(fid):
        return sum(r["package"]["features"].get(fid, {}).get("met", False) for r in rows)

    return {"sr_pkg": f"{s['package_passed']}/{n}", "sr_built": f"{s['as_built_passed']}/{n}",
            "sr_req": str(len(s["required_features"])), "sr_opt": str(len(s["optional_features"])),
            "sr_version": s["version"], "sr_validator": s["validator"],
            "sr_grasp": f"{met('FET_005_STANDARD')}/{n}", "sr_disp": f"{met('FET_010_STANDARD')}/{n}",
            "sr_physx": f"{met('FET_003_PHYSX')}/{n}"}


def vision_values() -> dict[str, str]:
    """The vision-based hand policy (scripts/rl/vision_summary.py): every run as vis_<run> (placed/400) and
    vis_<run>_add (median mm); the pose networks' closed-loop estimate errors as vis_<run>_<scene|start|hand>_mm."""
    p = ROOT / "docs" / "figures" / "vision_policy.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = {}
    for k, r in d["runs"].items():
        out[f"vis_{k}"] = f"{r['placed']}/{r['envs']}"
        out[f"vis_{k}_add"] = f"{r['add_mm_median']:.1f}"
        out[f"vis_{k}_rot"] = f"{r['rot_deg_median']:.0f}"
        e = r.get("estimates", {})
        for name, key in (("goal", "target_mm_median"), ("start", "start_mm_median"), ("hand", "hand_mm_median")):
            if key in e:
                out[f"vis_{k}_{name}_mm"] = f"{e[key]:.1f}"
    for k, v in d.get("nets", {}).items():
        if "vision" in v and "mm_median" in v["vision"]:
            out[f"vis_net_{k}_mm"] = f"{v['vision']['mm_median']:.1f}"
        elif "vision" in v:
            out[f"vis_net_{k}_mm"] = f"{v['vision']['target']['mm_median']:.1f}"
    return out


def reel_values() -> dict[str, str]:
    p = ROOT / "docs" / "figures" / "demo_reel.json"
    return {"reel_s": str(json.loads(p.read_text())["seconds"])} if p.exists() else {}


def version_videos(text: str, absolute_only: bool = False) -> str:
    """Append ?v=<content hash> to every linked MP4, so browsers fetch a re-rendered video instead of
    a cached one under the same URL (GitHub Pages lets browsers cache for 10 minutes)."""
    import hashlib

    def sub(m):
        url = m.group(2)
        if absolute_only and not url.startswith("http"):  # repo-relative README links: no cache to bust
            return m.group(1) + url
        local = url.split("visualknight.github.io/scan2usd/")[-1]
        for base in (ROOT / "docs", ROOT):
            f = base / local
            if f.exists():
                return f"{m.group(1)}{url}?v={hashlib.sha1(f.read_bytes()).hexdigest()[:8]}"
        return m.group(0)

    return re.sub(r'(["(])([^"()\s]*?\.mp4)(\?v=[0-9a-f]+)?(?=[")])', sub, text)


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
        s = version_videos(s, absolute_only=rel == "README.md")
        p.write_text(s, encoding="utf-8")
        print(f"{rel}: {len(used)} numbers synced ({', '.join(sorted(used))})")


if __name__ == "__main__":
    main()
