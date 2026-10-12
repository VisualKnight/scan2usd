"""Automated quality checks for a Scan2USD asset, run against the written USD file."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .physics import MAX_HULL_VERTICES, union_volume_mesh
from .usd_io import read_usd

PASS, WARN, FAIL = "pass", "warn", "fail"

MIN_SIZE_M, MAX_SIZE_M = 0.005, 5.0
MIN_DENSITY, MAX_DENSITY = 10.0, 22000.0
MAX_RENDER_TRIANGLES = 200_000
MAX_HULLS = 64


@dataclass
class Check:
    category: str
    name: str
    status: str
    detail: str = ""


@dataclass
class ValidationReport:
    asset: str
    checks: list[Check] = field(default_factory=list)
    drop_tests: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, category: str, name: str, ok: bool | str, detail: str = "") -> None:
        status = ok if isinstance(ok, str) else (PASS if ok else FAIL)
        self.checks.append(Check(category, name, status, detail))

    @property
    def status(self) -> str:
        states = {c.status for c in self.checks}
        return FAIL if FAIL in states else WARN if WARN in states else PASS

    @property
    def counts(self) -> dict:
        return {s: sum(c.status == s for c in self.checks) for s in (PASS, WARN, FAIL)}

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "status": self.status,
            "counts": self.counts,
            "stats": self.stats,
            "checks": [asdict(c) for c in self.checks],
            "drop_tests": self.drop_tests,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def to_markdown(self) -> str:
        icon = {PASS: "✅", WARN: "⚠️", FAIL: "❌"}
        c = self.counts
        lines = [
            f"# Validation report: {self.asset}",
            "",
            f"**Result: {icon[self.status]} {self.status.upper()}** "
            f"({c[PASS]} passed, {c[WARN]} warnings, {c[FAIL]} failed)",
            "",
        ]
        if self.stats:
            lines += ["## Asset stats", "", "| | |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in self.stats.items()]
            lines.append("")
        lines += ["## Checks", "", "| | Category | Check | Detail |", "|---|---|---|---|"]
        lines += [f"| {icon[x.status]} | {x.category} | {x.name} | {x.detail} |" for x in self.checks]
        if self.drop_tests:
            lines += [
                "",
                "## MuJoCo drop tests",
                "",
                "| Pose | Stable | Settled | Settle time (s) | Impact penetration (mm) | Resting penetration (mm) | Note |",
                "|---|---|---|---|---|---|---|",
            ]
            for d in self.drop_tests:
                t = "" if d["settle_time_s"] is None else f"{d['settle_time_s']:.2f}"
                lines.append(
                    f"| {d['pose']} | {'yes' if d['stable'] else 'NO'} | {'yes' if d['settled'] else 'no'} | "
                    f"{t} | {d['max_penetration_m'] * 1000:.2f} | {d['resting_penetration_m'] * 1000:.3f} | {d['message']} |"
                )
        return "\n".join(lines) + "\n"


def _fmt(v: float, unit: str = "") -> str:
    return f"{v:.4g}{unit}"


def validate_usd(path: str | Path, run_sim: bool = True) -> ValidationReport:
    path = Path(path)
    rep = ValidationReport(asset=path.name)
    try:
        asset = read_usd(path)
    except Exception as exc:  # report unreadable assets instead of raising
        rep.add("usd", "readable", False, str(exc))
        return rep
    rep.add("usd", "readable", True, f"defaultPrim /{asset.name}")

    # Stage conventions expected by robotics simulators (Isaac Sim: Z-up, meters, kg).
    si = asset.stage_info
    rep.add("usd", "up axis is Z", si["up_axis"] == "Z", si["up_axis"])
    rep.add("usd", "meters per unit = 1", abs(si["meters_per_unit"] - 1.0) < 1e-9, _fmt(si["meters_per_unit"]))
    rep.add("usd", "kilograms per unit = 1", abs(si["kilograms_per_unit"] - 1.0) < 1e-9, _fmt(si["kilograms_per_unit"]))
    rep.add("usd", "render material bound", asset.has_render_material)
    missing = [p for p in asset.texture_paths if not p.exists()]
    rep.add("usd", "textures resolvable", not missing,
            ", ".join(str(p) for p in missing) or f"{len(asset.texture_paths)} texture(s)")

    # Render geometry.
    mesh = asset.render_mesh
    tris = len(mesh.faces)
    finite = bool(np.all(np.isfinite(mesh.vertices)))
    rep.add("geometry", "finite vertex positions", finite)
    rep.add("geometry", "triangle budget", PASS if tris <= MAX_RENDER_TRIANGLES else WARN,
            f"{tris:,} triangles (budget {MAX_RENDER_TRIANGLES:,})")
    check = mesh.copy()
    check.merge_vertices(merge_tex=True, merge_norm=True)
    watertight = bool(check.is_watertight)
    rep.add("geometry", "watertight", PASS if watertight else WARN,
            "closed manifold" if watertight else "open or non-manifold surface")
    rep.add("geometry", "consistent winding", PASS if check.is_winding_consistent else WARN)
    ext = mesh.extents
    size = float(ext.max())
    rep.add("geometry", "plausible real-world size", PASS if MIN_SIZE_M <= size <= MAX_SIZE_M else WARN,
            "extents " + " x ".join(f"{e * 100:.1f}" for e in ext) + " cm")
    rep.add("geometry", "rests on ground plane (min z = 0)",
            PASS if abs(mesh.bounds[0][2]) < 1e-3 * max(size, 1e-3) else WARN, _fmt(float(mesh.bounds[0][2]), " m"))

    # Collision geometry.
    hulls = asset.hulls
    rep.add("collision", "has colliders", bool(hulls), f"{len(hulls)} hull(s)")
    if hulls:
        rep.add("collision", "hull count", PASS if len(hulls) <= MAX_HULLS else WARN, f"{len(hulls)} (max {MAX_HULLS})")
        convex = all(h.is_convex for h in hulls)
        rep.add("collision", "all hulls convex", convex)
        max_v = max(len(h.vertices) for h in hulls)
        rep.add("collision", "hull vertex limit", max_v <= 255,
                f"max {max_v} vertices (PhysX limit 255, target {MAX_HULL_VERTICES})")
        approx = {c["approximation"] for c in asset.collider_info}
        rep.add("collision", "convexHull approximation", approx == {"convexHull"}, ", ".join(sorted(approx)))
        rep.add("collision", "physics material bound", bool(asset.physics_material),
                ", ".join(f"{k}={_fmt(v)}" for k, v in asset.physics_material.items()))
        hull_union = union_volume_mesh(hulls)
        if watertight and check.volume > 0:
            ratio = float(hull_union.volume / check.volume)
            rep.add("collision", "hull volume vs render volume",
                    PASS if 0.9 <= ratio <= 1.35 else WARN, f"{ratio:.3f}x (expect 0.9-1.35)")
        hb_lo, hb_hi = hull_union.bounds
        mb_lo, mb_hi = mesh.bounds
        dev = float(np.max(np.abs(np.r_[hb_lo - mb_lo, hb_hi - mb_hi])))
        rep.add("collision", "hull bounds match render bounds",
                PASS if dev <= 0.03 * size + 1e-4 else WARN, f"max deviation {dev * 1000:.1f} mm")

    # Mass properties.
    has_mass = asset.mass is not None and asset.diagonal_inertia is not None
    rep.add("physics", "RigidBodyAPI applied", "PhysicsRigidBodyAPI" in asset.root_schemas)
    rep.add("physics", "MassAPI authored", has_mass)
    inertia = None
    if has_mass:
        m = float(asset.mass)
        d = np.asarray(asset.diagonal_inertia, dtype=np.float64)
        rep.add("physics", "mass > 0", m > 0, _fmt(m, " kg"))
        rep.add("physics", "principal inertia > 0", bool(np.all(d > 0)), " ".join(_fmt(v) for v in d))
        a, b, c = np.sort(d)
        tri_ok = a + b >= c * (1 - 1e-6)
        rep.add("physics", "inertia triangle inequality", bool(tri_ok), f"{_fmt(a)} + {_fmt(b)} >= {_fmt(c)}")
        com = asset.center_of_mass
        lo, hi = mesh.bounds
        tol = 0.01 * size
        com_ok = bool(np.all(com >= lo - tol) and np.all(com <= hi + tol))
        rep.add("physics", "center of mass inside bounds", com_ok, "(" + ", ".join(f"{v * 100:.2f}" for v in com) + ") cm")
        solid = check if watertight else (union_volume_mesh(hulls) if hulls else None)
        if solid is not None and solid.volume > 0:
            rho = m / float(solid.volume)
            rep.add("physics", "plausible density", PASS if MIN_DENSITY <= rho <= MAX_DENSITY else WARN,
                    f"{rho:.0f} kg/m^3")
            rep.stats["volume (cm^3)"] = f"{solid.volume * 1e6:.1f}"
        inertia = asset.inertia_tensor()
        rep.stats["mass (kg)"] = f"{m:.4f}"
    rep.stats["render triangles"] = f"{tris:,}"
    rep.stats["collision hulls"] = str(len(hulls))
    rep.stats["extents (cm)"] = " x ".join(f"{e * 100:.1f}" for e in ext)

    if run_sim and hulls and has_mass and rep.status != FAIL:
        from .simulate import DROP_POSES, drop_test

        friction = float(asset.physics_material.get("dynamic_friction", 0.5) or 0.5)
        for pose in DROP_POSES:
            res = drop_test(hulls, float(asset.mass), asset.center_of_mass, inertia, friction=friction, pose=pose)
            rep.drop_tests.append(res.to_dict())
            rep.add("simulation", f"drop test ({pose}) stable", res.stable, res.message if not res.stable else "")
            rep.add("simulation", f"drop test ({pose}) settles", PASS if res.settled else WARN,
                    f"{res.settle_time_s:.2f} s" if res.settled else res.message)
    return rep
