"""The architecture diagram: a layered system view.

    python scripts/make_architecture.py

Four stages in the page's colors (Capture, Build, Verify, Use) around one hub: the OpenUSD asset,
the single file every simulator and renderer reads. One feedback loop: what the asset doesn't know
tells the camera which photos to take next.

Writes docs/figures/architecture.svg (the README's static image) and the same vector diagram inline
into docs/overview.html (between <!--arch:start--> and <!--arch:end-->), where hovering or tapping a
box shows its operations, numbers and scripts. The numbers there are sync markers, kept current by
scripts/sync_holder_numbers.py.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "architecture.svg"
PAGE = ROOT / "docs" / "overview.html"
W, H = 1180, 640
COL = {"cap": "#0e7490", "bld": "#6d28d9", "ver": "#047857", "use": "#c2410c", "ink": "#1e293b", "soft": "#64748b"}
TINT = {"cap": "#ecfeff", "bld": "#f5f3ff", "ver": "#ecfdf5", "use": "#fff7ed"}

parts: list[str] = []


def text(x, y, s, size=14, color=COL["ink"], anchor="start", weight="normal"):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                 f'font-weight="{weight}">{escape(s)}</text>')


def box(key, x, y, w, h, stage, title, lines=(), small=None, fill=None):
    parts.append(f'<g class="node" data-k="{key}" data-stage="{stage}" tabindex="0">')
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill or "#ffffff"}" '
                 f'stroke="{COL[stage]}" stroke-width="2.2" filter="url(#arch_rough)"/>')
    text(x + 14, y + 26, title, 16, COL["ink"], weight="bold")
    for i, s in enumerate(lines):
        text(x + 14, y + 48 + 19 * i, s, 13, COL["ink"])
    if small:
        text(x + 14, y + h - 12, small, 11, COL["soft"])
    parts.append("</g>")


def arrow(d, stage="ink", dash=False, width=2.2):
    color = COL[stage]
    dashes = 'stroke-dasharray="8 6" ' if dash else ""
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
                 f'{dashes}marker-end="url(#arch_a_{stage})"/>')


def lane(x, w, stage, n, label):
    parts.append(f'<rect x="{x}" y="96" width="{w}" height="{H - 150}" rx="14" fill="{TINT[stage]}" opacity="0.7"/>')
    text(x + 14, 122, f"{n} · {label.upper()}", 13, COL[stage], weight="bold")


# lanes
lane(20, 210, "cap", 1, "Capture")
lane(250, 300, "bld", 2, "Build")
lane(570, 230, "ver", 3, "Verify")
lane(820, 340, "use", 4, "Use")

# 1 capture
box("photos", 36, 150, 178, 100, "cap", "Phone photos", ["60–150 shots", "of an everyday object"])
box("refs", 36, 274, 178, 100, "cap", "References", ["color chart · weight", "tape (optional)"])
box("guide", 36, 444, 178, 100, "cap", "Capture guide", ["which photos", "to take next"])

# 2 build: pipeline + asset hub
box("pipeline", 266, 150, 268, 210, "bld", "scan2usd pipeline", [
    "reconstruct (COLMAP → mesh)",
    "clean, bake textures",
    "calibrate color, metric scale",
    "convex hulls, mass, inertia",
    "joints for articulated objects",
])
box("asset", 266, 392, 268, 152, "bld", "OpenUSD asset", [
    "look · colliders · mass",
    "+ tested grasps, uncertainty,",
    "   Gaussian-splat look",
    "one file every engine reads",
], fill="#ede9fe")

# 3 verify
box("validator", 600, 150, 184, 100, "ver", "Validator", ["33 checks on the", "written file"])
box("truth", 600, 272, 184, 100, "ver", "Ground truth", ["surface error vs", "true meshes"])
box("realcheck", 600, 444, 184, 100, "ver", "Real-world check", ["scan vs the real", "objects, posed"])

# 4 use
box("physics", 836, 150, 308, 118, "use", "Physics + robots", ["MuJoCo · Isaac Sim PhysX", "Franka Panda · Shadow Hand", "pick, place, drop tests"])
box("perception", 836, 288, 308, 100, "use", "Perception", ["Replicator synthetic data →", "detector → pose → pick"])
box("render", 836, 408, 308, 136, "use", "Rendering + library", ["Isaac Sim RTX path tracing", "Unreal Engine 5.8 (MegaLights)", "USD asset library, 3D gallery"])

# capture -> build
arrow("M214,200 C240,200 244,230 262,230", "cap")
arrow("M214,324 C240,324 244,296 262,292", "cap")
arrow("M400,360 L400,388", "bld", width=2.6)
# asset -> verify: one bus at x=574
parts.append('<path d="M534,430 L574,430 M574,200 L574,494" fill="none" stroke="#047857" stroke-width="2.2" stroke-linecap="round"/>')
for y in (200, 322, 494):
    arrow(f"M574,{y} L596,{y}", "ver")
# asset -> use: under the verify lane, one bus at x=812
parts.append('<path d="M400,544 L400,566 L812,566 L812,209" fill="none" stroke="#c2410c" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>')
for y in (209, 338, 476):
    arrow(f"M812,{y} L832,{y}", "use")
text(606, 584, "same file in every engine", 12, COL["use"])
# feedback: asset uncertainty -> capture guide -> photos
arrow("M266,494 L218,494", "cap", dash=True)
arrow("M30,494 L30,206 L32,206", "cap", dash=True)
parts.append('<path d="M36,494 L30,494" fill="none" stroke="#0e7490" stroke-width="2.2" stroke-dasharray="8 6"/>')
text(222, 482, "unknowns", 11, COL["cap"])

MARKERS = "".join(f'    <marker id="arch_a_{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
                  f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10" fill="none" stroke="{c}" '
                  f'stroke-width="1.8"/></marker>\n' for k, c in COL.items())
BODY = "\n".join("  " + p for p in parts)


def svg(footer: str, inline: bool = False) -> str:
    size = ' style="width:100%;height:auto;display:block"' if inline else ""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"{size} font-family="'Segoe Print', 'Comic Sans MS', cursive" role="img" aria-label="Scan2USD system view">
  <defs>
    <filter id="arch_rough" x="-5%" y="-5%" width="110%" height="110%">
      <feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="2" seed="3"/>
      <feDisplacementMap in="SourceGraphic" scale="2.5"/>
    </filter>
{MARKERS}  </defs>
  <rect width="{W}" height="{H}" fill="#fbfbf7"/>
  <text x="24" y="42" font-size="24" fill="{COL["ink"]}">Scan2USD · system view</text>
  <text x="24" y="68" font-size="13" fill="{COL["soft"]}">phone photos in, one physics-ready OpenUSD asset out; every simulator and renderer reads the same file</text>
{BODY}
  <text x="{W - 24}" y="{H - 18}" font-size="11" fill="{COL["soft"]}" text-anchor="end">{escape(footer)}</text>
</svg>
'''


# Hover details per box: (what happens, numbers, scripts). Numbers in <!--n:key--> markers are kept
# current by sync_holder_numbers.py; the values here are only the first fill.
def n(key: str) -> str:
    return f"<!--n:{key}-->{NUM.get(key, '?')}<!--/n-->"


try:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sync_holder_numbers import values

    NUM = values()
except (OSError, KeyError, ValueError, ImportError):
    NUM = {}

DETAILS = {
    "photos": (
        "Phone photos",
        "Orbit the object at two or three heights with lots of overlap, on a textured table. "
        "The benchmark renders the same kind of capture in Blender, so every scan has a true mesh to score against.",
        ["78 photos per benchmark object, 6 objects", "0.78 mm mean surface error (speckled mug)"],
        ["blender_render_capture.py", "make_synthetic_capture.py", "study_photo_count.py"],
    ),
    "refs": (
        "References",
        "A ColorChecker (or an 18% gray card) in some photos is found automatically. Its patch spacing also gives "
        "metric scale, no ruler needed. A kitchen scale gives mass; a tape measure is the fallback for size.",
        ["chart color error ΔE2000 ~8 → 1.3", "size error ≤ 1.43% from the chart alone"],
        ["color.py", "chart.py", "eval_chart_scale.py"],
    ),
    "guide": (
        "Capture guide",
        "Reads the asset's uncertainty and lists camera positions that would see the physics-relevant surface "
        "no photo saw twice. Work in progress.",
        ["8 extra photos, rendered where asked", "helps the bowl; not the printed mug (its white inside is visible but textureless)"],
        ["guide.py", "study_capture_guide.py", "scan2usd guide"],
    ),
    "pipeline": (
        "scan2usd pipeline",
        "COLMAP structure-from-motion and multi-view stereo → table plane to z = 0 → object isolated → Poisson "
        "surface. Then repair, meters and Z-up, decimate, bake 2K albedo + normal maps, correct color, set metric "
        "scale, split into convex hulls (CoACD) and compute mass, center of mass and inertia. Optional: carve out "
        "space the photos see through, or split a scan into parts on a revolute joint.",
        [f"controller holder: {n('holder_hulls')} convex pieces", "one command: scan2usd scan photos\\ -o out\\"],
        ["pipeline.py", "reconstruct.py", "mesh_ops.py", "bake.py", "color.py", "physics.py", "carve.py", "articulation.py"],
    ),
    "asset": (
        "OpenUSD asset",
        "One file: render mesh with a PBR look, convex-hull colliders with a physics material, rigid body, mass and "
        "inertia. Codeless USD schemas add tested grasps, property intervals (mass, scale, friction) and per-vertex "
        "observed / inferred / sealed labels. Optional Gaussian splat for the look. Many assets become one library "
        "with collision and friction variant sets.",
        ["true mass inside the stored interval: 5/6 objects", "756 grasp re-tests stored", "also exported as glTF and MuJoCo MJCF"],
        ["usd_io.py", "schemas.py", "uncertainty.py", "splat.py", "library.py", "mjcf.py"],
    ),
    "validator": (
        "Validator",
        "Opens the written file the way a simulator would and checks it: USD units and up axis, textures resolve; "
        "geometry watertight, consistent winding, plausible size, resting on z = 0; hulls convex, ≤ 255 vertices, "
        "matching the render mesh; mass, inertia (triangle inequality), center of mass, density; drop tests settle.",
        ["33 checks on every asset"],
        ["validate.py", "simulate.py", "scan2usd validate"],
    ),
    "truth": (
        "Ground truth",
        "Aligns each scan to its true mesh, then measures surface accuracy, completeness, F-score and volume, "
        "and color error on the chart before and after correction. Image-to-3D meshes are scored the same way.",
        ["mean error 0.60–3.03 mm over 6 objects", "F-score @2 mm up to 0.94",
         f"one photo, Hunyuan3D 2.1: F {n('gen_f_h21')} vs {n('gen_f_pg')} (78 photos)"],
        ["evaluate.py", "run_benchmark.py", "evaluate_synthetic.py", "study_generative.py"],
    ),
    "realcheck": (
        "Real-world check",
        "A controller holder and a controller, scanned alone and together. Registering the scans gives the real "
        "resting pose: the answer key. The scanned controller is dropped into the scanned holder, then placed by robots.",
        [f"drop: {n('sim_held')} held, median {n('sim_add')} mm (MuJoCo); {n('isaac_held')} in Isaac Sim",
         f"holder as one convex hull: {n('hull_held')}",
         f"scans agree on scale within {n('scale_agree')}%"],
        ["study_holder_fit.py", "register.py", "isaac_holder_fit.py"],
    ),
    "physics": (
        "Physics + robots",
        "MuJoCo and Isaac Sim PhysX run the same assets. A Franka Panda pinches the controller and seats it in the "
        "holder (MuJoCo IK, replayed on Isaac's Franka); a Shadow Hand pinches it top to bottom as a person does; the Panda "
        "picks the scanned mug; an articulated lid opens on its joint.",
        [f"Panda: {n('place_held')} MuJoCo, {n('iplace_held')} Isaac Sim ({n('iplace_add')} mm)",
         f"Shadow Hand: {n('hand_held')} MuJoCo, {n('ihand_held')} Isaac Sim",
         "holder as one convex hull: every placement fails"],
        ["demo_holder_place.py", "isaac_holder_place.py", "demo_hand_pinch.py", "isaac_hand_place.py", "demo_panda.py", "isaac_franka_pick.py"],
    ),
    "perception": (
        "Perception",
        "Isaac Sim Replicator renders the scanned assets under random lights, poses and distractors, with labels. "
        "A detector trained only on those finds the object in real photos, then drives a see-locate-pick loop.",
        ["67/67 real photos (59/67 without distractors)", "10/10 picks, 0.6 mm median pose error"],
        ["isaac_synthdata.py", "replicator_to_coco.py", "train_detector.py", "demo_perception_pick.py"],
    ),
    "render": (
        "Rendering + library",
        "Isaac Sim RTX path traces the assets with PBR look presets. Unreal Engine 5.8 imports the same USD stage "
        "as is (Lumen + MegaLights, Sequencer). The Gaussian splat gives a photo-real look. The library catalog "
        "and 3D gallery are on the web.",
        [f"MegaLights: {n('ue_on')} ms per frame vs {n('ue_off')} ms", "splat beats the mesh on 18/18 unseen photos"],
        ["isaac_pathtrace.py", "unreal_render.py", "unreal_hand_render.py", "make_gallery.py", "make_catalog.py"],
    ),
}

CSS = """<style>
.arch { position: relative; }
.arch .node { cursor: pointer; outline: none; transition: opacity .15s; }
.arch .node rect { transition: stroke-width .15s; }
.arch.hovering .node:not(.on) { opacity: .45; }
.arch .node.on rect { stroke-width: 3.6; }
.arch .hint { margin: 8px 4px 0; color: var(--soft); font-size: 13px; }
.archtip { position: absolute; z-index: 5; width: 330px; max-width: calc(100% - 16px); background: #fff; border: 1px solid var(--rule);
  border-top: 4px solid var(--c); border-radius: 6px; box-shadow: 0 10px 28px rgba(15, 23, 42, .18); padding: 12px 14px 12px;
  font-size: 13.5px; line-height: 1.45; pointer-events: none; }
.archtip h4 { margin: 0 0 6px; font-size: 15px; color: var(--c); }
.archtip p { margin: 0 0 8px; }
.archtip ul { margin: 0 0 8px; padding-left: 18px; }
.archtip li { margin: 2px 0; font-weight: 600; }
.archtip code { display: inline-block; margin: 2px 4px 0 0; padding: 1px 6px; border-radius: 4px; background: #f1f5f9; font-size: 12px; }
</style>"""

JS = """<script>
(() => {
  const box = document.querySelector('.arch'); if (!box) return;
  const tip = box.querySelector('.archtip'), nodes = box.querySelectorAll('.node');
  let pinned = null;
  const show = (g) => {
    const d = box.querySelector('.archd[data-k="' + g.dataset.k + '"]'); if (!d) return;
    nodes.forEach((x) => x.classList.toggle('on', x === g)); box.classList.add('hovering');
    tip.style.setProperty('--c', g.querySelector('rect').getAttribute('stroke'));
    tip.innerHTML = d.innerHTML; tip.hidden = false;
    const b = box.getBoundingClientRect(), r = g.getBoundingClientRect(), tw = tip.offsetWidth, th = tip.offsetHeight;
    let x = r.right - b.left + 10, y = r.top - b.top;
    if (x + tw > b.width) x = r.left - b.left - tw - 10;
    if (x < 0) { x = 8; y = b.height + 6; }  // narrow screens: under the diagram, which stays visible
    else y = Math.max(0, Math.min(y, b.height - th));
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  };
  const hide = () => { if (pinned) return; tip.hidden = true; box.classList.remove('hovering'); nodes.forEach((x) => x.classList.remove('on')); };
  nodes.forEach((g) => {
    g.addEventListener('mouseenter', () => { if (!pinned) show(g); });
    g.addEventListener('mouseleave', hide);
    g.addEventListener('focus', () => show(g)); g.addEventListener('blur', () => { pinned = null; hide(); });
    g.addEventListener('click', (e) => { e.stopPropagation(); pinned = pinned === g ? null : g; show(g); if (!pinned) hide(); });
  });
  document.addEventListener('click', () => { pinned = null; hide(); });
})();
</script>"""


def detail_html(key) -> str:
    title, what, nums, files = DETAILS[key]
    return (f'<div class="archd" data-k="{key}" hidden><h4>{escape(title)}</h4><p>{escape(what)}</p>'
            f'<ul>{"".join(f"<li>{s}</li>" for s in nums)}</ul>'
            f'<div>{"".join(f"<code>{escape(f)}</code>" for f in files)}</div></div>')


OUT.write_text(svg('module-level map: README, "What it does"'), encoding="utf-8")
print("wrote", OUT)

block = ("<!--arch:start-->\n" + CSS + '\n<div class="board arch">\n'
         + svg("", inline=True)
         + '<div class="archtip" hidden></div>\n' + "\n".join(detail_html(k) for k in DETAILS)
         + '\n<p class="hint">Hover (or tap) a box for its operations, numbers and scripts.</p>\n</div>\n'
         + JS + "\n<!--arch:end-->")
page = PAGE.read_text(encoding="utf-8")
old_img = '<img class="board" src="figures/architecture.svg" alt="Whiteboard architecture diagram">'
if "<!--arch:start-->" in page:
    a, b = page.index("<!--arch:start-->"), page.index("<!--arch:end-->") + len("<!--arch:end-->")
    page = page[:a] + block + page[b:]
else:
    assert page.count(old_img) == 1, "overview: architecture image not found"
    page = page.replace(old_img, block)
PAGE.write_text(page, encoding="utf-8")
print("wrote", PAGE, "(inline, interactive)")
