"""The overview's architecture diagram: a layered system view (docs/figures/architecture.svg).

    python scripts/make_architecture.py

Four stages in the page's colors (Capture, Build, Verify, Use) around one hub: the OpenUSD asset,
the single file every simulator and renderer reads. One feedback loop: what the asset doesn't know
tells the camera which photos to take next. Module-level detail lives in the README's stage table.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "architecture.svg"
W, H = 1180, 640
COL = {"cap": "#0e7490", "bld": "#6d28d9", "ver": "#047857", "use": "#c2410c", "ink": "#1e293b", "soft": "#64748b"}
TINT = {"cap": "#ecfeff", "bld": "#f5f3ff", "ver": "#ecfdf5", "use": "#fff7ed"}

parts: list[str] = []


def text(x, y, s, size=14, color=COL["ink"], anchor="start", weight="normal"):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                 f'font-weight="{weight}">{escape(s)}</text>')


def box(x, y, w, h, stage, title, lines=(), small=None, fill=None):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill or "#ffffff"}" '
                 f'stroke="{COL[stage]}" stroke-width="2.2" filter="url(#marker)"/>')
    text(x + 14, y + 26, title, 16, COL["ink"], weight="bold")
    for i, s in enumerate(lines):
        text(x + 14, y + 48 + 19 * i, s, 13, COL["ink"])
    if small:
        text(x + 14, y + h - 12, small, 11, COL["soft"])


def arrow(d, stage="ink", dash=False, width=2.2):
    color = COL[stage]
    dashes = 'stroke-dasharray="8 6" ' if dash else ""
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
                 f'{dashes}marker-end="url(#a_{stage})"/>')


def lane(x, w, stage, n, label):
    parts.append(f'<rect x="{x}" y="96" width="{w}" height="{H - 150}" rx="14" fill="{TINT[stage]}" opacity="0.7"/>')
    text(x + 14, 122, f"{n} · {label.upper()}", 13, COL[stage], weight="bold")


# lanes
lane(20, 210, "cap", 1, "Capture")
lane(250, 300, "bld", 2, "Build")
lane(570, 230, "ver", 3, "Verify")
lane(820, 340, "use", 4, "Use")

# 1 capture
box(36, 150, 178, 100, "cap", "Phone photos", ["60–150 shots", "of an everyday object"])
box(36, 274, 178, 100, "cap", "References", ["color chart · weight", "tape (optional)"])
box(36, 444, 178, 100, "cap", "Capture guide", ["which photos", "to take next"])

# 2 build: pipeline + asset hub
box(266, 150, 268, 210, "bld", "scan2usd pipeline", [
    "reconstruct (COLMAP → mesh)",
    "clean, bake textures",
    "calibrate color, metric scale",
    "convex hulls, mass, inertia",
    "joints for articulated objects",
])
box(266, 392, 268, 152, "bld", "OpenUSD asset", [
    "look · colliders · mass",
    "+ tested grasps, uncertainty,",
    "   Gaussian-splat look",
    "one file every engine reads",
], fill="#ede9fe")

# 3 verify
box(600, 150, 184, 100, "ver", "Validator", ["33 checks on the", "written file"])
box(600, 272, 184, 100, "ver", "Ground truth", ["surface error vs", "true meshes"])
box(600, 444, 184, 100, "ver", "Real-world check", ["scan vs the real", "objects, posed"])

# 4 use
box(836, 150, 308, 118, "use", "Physics + robots", ["MuJoCo · Isaac Sim PhysX", "Franka Panda · Shadow Hand", "pick, place, drop tests"])
box(836, 288, 308, 100, "use", "Perception", ["Replicator synthetic data →", "detector → pose → pick"])
box(836, 408, 308, 136, "use", "Rendering + library", ["Isaac Sim RTX path tracing", "Unreal Engine 5.8 (MegaLights)", "USD asset library, 3D gallery"])

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

MARKERS = "".join(f'    <marker id="a_{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
                  f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10" fill="none" stroke="{c}" '
                  f'stroke-width="1.8"/></marker>\n' for k, c in COL.items())
BODY = "\n".join("  " + p for p in parts)
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="'Segoe Print', 'Comic Sans MS', cursive">
  <defs>
    <filter id="marker" x="-5%" y="-5%" width="110%" height="110%">
      <feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="2" seed="3"/>
      <feDisplacementMap in="SourceGraphic" scale="2.5"/>
    </filter>
{MARKERS}  </defs>
  <rect width="{W}" height="{H}" fill="#fbfbf7"/>
  <text x="24" y="42" font-size="24" fill="{COL["ink"]}">Scan2USD · system view</text>
  <text x="24" y="68" font-size="13" fill="{COL["soft"]}">phone photos in, one physics-ready OpenUSD asset out; every simulator and renderer reads the same file</text>
{BODY}
  <text x="{W - 24}" y="{H - 18}" font-size="11" fill="{COL["soft"]}" text-anchor="end">module-level map: README, "What it does"</text>
</svg>
'''
OUT.write_text(svg, encoding="utf-8")
print("wrote", OUT)
