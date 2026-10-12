"""Articulated assets: split a scan into parts and join them with a hinge (revolute joint).

Typical cases: box lids, laptop screens, cabinet doors. The split is given as a plane
(the seam between parts) and the hinge as an axis line with angle limits; automatic part
segmentation is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from . import __version__
from .physics import MATERIALS, MassProperties, convex_decompose, estimate_mass_properties, single_hull
from .usd_io import PhysicsMaterial, SurfaceMaterial, _prim_name, author_body, define_physics_material


@dataclass
class Hinge:
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower_deg: float = 0.0
    upper_deg: float = 110.0


@dataclass
class Part:
    name: str
    mesh: trimesh.Trimesh
    hulls: list[trimesh.Trimesh] = field(default_factory=list)
    mass: MassProperties | None = None


def split_mesh(mesh: trimesh.Trimesh, origin, normal) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Cut a closed mesh along a plane; returns (part below, part above the plane), both capped."""
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    below = mesh.slice_plane(origin, -n, cap=True)
    above = mesh.slice_plane(origin, n, cap=True)
    if len(below.faces) == 0 or len(above.faces) == 0:
        raise ValueError("the split plane does not cut the mesh")
    return below, above


def _x_to(axis) -> Gf.Quatf:
    a = np.asarray(axis, dtype=np.float64)
    a /= np.linalg.norm(a)
    rot, _ = Rotation.align_vectors([a], [[1.0, 0.0, 0.0]])
    x, y, z, w = rot.as_quat()
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def build_parts(mesh: trimesh.Trimesh, origin, normal, material: str = "wood", names=("base", "lid"),
                collision: str = "coacd") -> list[Part]:
    below, above = split_mesh(mesh, origin, normal)
    rho = MATERIALS[material]["density"]
    parts = []
    for name, m in zip(names, (below, above)):
        hulls = convex_decompose(m, threshold=0.05, max_hulls=16) if collision == "coacd" else single_hull(m)
        parts.append(Part(name, m, hulls, estimate_mass_properties(m, hulls, density=rho)))
    return parts


def write_articulated_usd(path: str | Path, name: str, parts: list[Part], hinge: Hinge,
                          surface: SurfaceMaterial, physics_material: PhysicsMaterial,
                          fixed_base: bool = False) -> Path:
    """One USD asset: articulation root, one rigid body per part, a revolute joint between them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root_path = Sdf.Path(f"/{_prim_name(name)}")
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)
    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)
    root.SetAssetInfo({"name": name, "version": __version__})
    UsdPhysics.ArticulationRootAPI.Apply(root)

    looks = UsdGeom.Scope.Define(stage, root_path.AppendChild("Looks")).GetPath()
    phys_mat = define_physics_material(stage, looks.AppendChild("PM_contact"), physics_material)
    body_paths = []
    for part in parts:
        body = UsdGeom.Xform.Define(stage, root_path.AppendChild(_prim_name(part.name))).GetPrim()
        author_body(stage, body, path, part.mesh, part.hulls, part.mass, surface,
                    looks.AppendChild(f"M_{_prim_name(part.name)}"), phys_mat)
        body_paths.append(body.GetPath())

    joints = UsdGeom.Scope.Define(stage, root_path.AppendChild("joints")).GetPath()
    if fixed_base:
        fix = UsdPhysics.FixedJoint.Define(stage, joints.AppendChild("anchor"))
        fix.CreateBody1Rel().SetTargets([body_paths[0]])
    joint = UsdPhysics.RevoluteJoint.Define(stage, joints.AppendChild("hinge"))
    joint.CreateBody0Rel().SetTargets([body_paths[0]])
    joint.CreateBody1Rel().SetTargets([body_paths[1]])
    joint.CreateAxisAttr(UsdPhysics.Tokens.x)
    q = _x_to(hinge.axis)
    o = Gf.Vec3f(*[float(v) for v in hinge.origin])
    joint.CreateLocalPos0Attr(o)
    joint.CreateLocalRot0Attr(q)
    joint.CreateLocalPos1Attr(o)
    joint.CreateLocalRot1Attr(q)
    joint.CreateLowerLimitAttr(float(hinge.lower_deg))
    joint.CreateUpperLimitAttr(float(hinge.upper_deg))
    stage.GetRootLayer().Save()
    return path


def read_articulation(path: str | Path) -> dict:
    """Bodies (as AssetData-like dicts) and revolute joints from an articulated USD."""
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    bodies, joints = {}, []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies[str(prim.GetPath())] = prim
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(prim)
            q = j.GetLocalRot0Attr().Get()
            rot = Rotation.from_quat([*q.GetImaginary(), q.GetReal()])
            axis_local = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[j.GetAxisAttr().Get()]
            joints.append({
                "path": str(prim.GetPath()),
                "body0": str(j.GetBody0Rel().GetTargets()[0]),
                "body1": str(j.GetBody1Rel().GetTargets()[0]),
                "origin": list(j.GetLocalPos0Attr().Get()),
                "axis": rot.apply(axis_local).tolist(),
                "lower_deg": j.GetLowerLimitAttr().Get(),
                "upper_deg": j.GetUpperLimitAttr().Get(),
            })
    return {"articulation_root": root.HasAPI(UsdPhysics.ArticulationRootAPI), "bodies": bodies, "joints": joints,
            "stage": stage}


def make_keepsake_box(size=(0.14, 0.10, 0.07), lid_height=0.018, wall=0.008, hollow=False) -> tuple[trimesh.Trimesh, dict]:
    """A closed wooden box 'scan' plus the split plane and hinge that turn it into box + lid.

    hollow=True adds the inner cavity (as if scanned open and closed), so the parts become
    an open tray and a shallow lid.
    """
    sx, sy, sz = size
    box = trimesh.creation.box((sx, sy, sz))
    box.apply_translation([0, 0, sz / 2])
    if hollow:
        cavity = trimesh.creation.box((sx - 2 * wall, sy - 2 * wall, sz - 2 * wall))
        cavity.apply_translation([0, 0, sz / 2])
        box = trimesh.boolean.difference([box, cavity], engine="manifold")
    box = box.subdivide_to_size(0.006)
    seam_z = sz - lid_height
    hinge = Hinge(origin=(0.0, sy / 2, seam_z), axis=(-1.0, 0.0, 0.0), lower_deg=0.0, upper_deg=105.0)
    return box, {"plane_origin": (0, 0, seam_z), "plane_normal": (0, 0, 1), "hinge": hinge}


def mujoco_xml(path: str | Path, fixed_base: bool = True) -> tuple[str, dict]:
    """MJCF for an articulated Scan2USD asset: base body, child part on a limited hinge."""
    art = read_articulation(path)
    j = art["joints"][0]
    files, assets, bodies = {}, [], {}
    for body_path in (j["body0"], j["body1"]):
        prim = art["bodies"][body_path]
        name = prim.GetName()
        hull_meshes = []
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                pts = np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get(), dtype=np.float64)
                hull_meshes.append(trimesh.convex.convex_hull(pts))
        geoms = []
        for k, h in enumerate(hull_meshes):
            fname = f"{name}_hull_{k}.stl"
            files[fname] = h.export(file_type="stl")
            assets.append(f'<mesh name="{name}_hull_{k}" file="{fname}"/>')
            geoms.append(f'<geom type="mesh" mesh="{name}_hull_{k}" rgba="0.62 0.42 0.25 1" solref="0.005 1"/>')
        m = UsdPhysics.MassAPI(prim)
        com = m.GetCenterOfMassAttr().Get()
        diag = m.GetDiagonalInertiaAttr().Get()
        q = m.GetPrincipalAxesAttr().Get()
        bodies[name] = {
            "geoms": "".join(geoms),
            "inertial": (f'<inertial pos="{com[0]:.6g} {com[1]:.6g} {com[2]:.6g}" mass="{m.GetMassAttr().Get():.6g}" '
                         f'diaginertia="{diag[0]:.6g} {diag[1]:.6g} {diag[2]:.6g}" '
                         f'quat="{q.GetReal():.6g} {q.GetImaginary()[0]:.6g} {q.GetImaginary()[1]:.6g} '
                         f'{q.GetImaginary()[2]:.6g}"/>'),
        }
    base, lid = (Sdf.Path(j["body0"]).name, Sdf.Path(j["body1"]).name)
    o, a = j["origin"], j["axis"]
    # Degrees: MuJoCo's default <compiler angle="degree">.
    joint_xml = (f'<joint name="hinge" type="hinge" pos="{o[0]:.6g} {o[1]:.6g} {o[2]:.6g}" '
                 f'axis="{a[0]:.6g} {a[1]:.6g} {a[2]:.6g}" limited="true" '
                 f'range="{j["lower_deg"]:.6g} {j["upper_deg"]:.6g}" damping="0.002" solreflimit="0.004 1"/>')
    body_xml = (f'<body name="{base}">{"" if fixed_base else "<freejoint/>"}{bodies[base]["inertial"]}'
                f'{bodies[base]["geoms"]}<body name="{lid}">{joint_xml}{bodies[lid]["inertial"]}'
                f'{bodies[lid]["geoms"]}</body></body>')
    # Parts touch along the seam; a welded (fixed) base would otherwise collide with its child.
    exclude = f'<contact><exclude body1="{base}" body2="{lid}"/></contact>'
    return body_xml, {"assets": "".join(assets), "files": files, "joint": j, "contact": exclude}
