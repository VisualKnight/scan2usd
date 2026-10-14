"""Second hero strip: the real pair -> a Shadow Hand picks, places and seats the scanned controller.

    <isaac-sim>\\python.exe scripts\\isaac_hand_place.py --stills   # path-traced stills (Isaac Sim 6)
    python scripts/make_hand_hero.py                                # -> docs/figures/hand_hero.jpg

Same layout as the real-world-check strip (study_holder_fit.py, holder_fit_lead.jpg): panel 1 is the
same iPhone photo of the real pair; panels 2-4 are Isaac Sim 6 path-traced frames of the re-aimed run
(seed 0), paused at the grasp, the insertion and after the release. PhysX simulates the run; the
green ghost in panel 4 is where the real controller rests, measured in a scan of the real pair.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_holder_fit import real_pair_photo  # noqa: E402

from scan2usd.preview import _font  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STILLS = ROOT / "output" / "isaac_hand_stills"
FIG = ROOT / "docs" / "figures"


def main() -> None:
    S = 420
    meta = json.loads((STILLS / "stills.json").read_text())
    res = meta["result"]

    def still(name):
        return Image.open(STILLS / f"{name}.png").convert("RGB").resize((S, S), Image.LANCZOS)

    result = still("result")
    d = ImageDraw.Draw(result)
    txt = "green: where it really rests (photo 1)"
    d.rectangle([0, S - 40, 20 + d.textlength(txt, font=_font(17)), S], fill="white")
    d.text((10, S - 34), txt, fill=(21, 128, 61), font=_font(17))
    panels = [(real_pair_photo(S), "1 - The real pair", "iPhone photo"),
              (still("grasp"), "2 - Shadow Hand picks the scan", "Isaac Sim 6, PhysX, path traced"),
              (still("place"), "3 - Turns it upright, slides it in", "re-aimed from the in-hand pose"),
              (result, "4 - Placed", f"{res['add_mm']:.1f} mm / {res['rot_deg']:.0f}\u00b0 from the real pose")]
    gap, top_h = 50, 70
    out = Image.new("RGB", (4 * S + 3 * gap + 40, S + top_h + 20), "white")
    d = ImageDraw.Draw(out)
    for k, (img, t, sub) in enumerate(panels):
        x = 20 + k * (S + gap)
        d.text((x, 8), t, fill=(30, 41, 59), font=_font(24))
        d.text((x, 40), sub, fill=(100, 116, 139), font=_font(17))
        out.paste(img, (x, top_h))
        d.rectangle([x, top_h, x + S - 1, top_h + S - 1], outline=(226, 232, 240))
        if k < 3:
            ax, ay = x + S + 10, top_h + S // 2
            d.polygon([(ax, ay - 12), (ax + 28, ay), (ax, ay + 12)], fill=(37, 99, 235))
    out.save(FIG / "hand_hero.jpg", quality=90)  # re-encoded: no EXIF / GPS
    print(f"wrote {FIG / 'hand_hero.jpg'}")


if __name__ == "__main__":
    main()
