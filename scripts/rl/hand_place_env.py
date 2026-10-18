"""Isaac Lab task: a floating Shadow Hand places the scanned controller into the scanned holder.

Runs in the Isaac Lab venv (setup: scripted_rollout.py). One env is the cell of scripts/isaac_hand_place.py,
on GPU PhysX and cloned N times:

  hand        the Menagerie Shadow Hand as imported by Isaac Sim's MJCF importer (output/isaac_hand/),
              its forearm welded to a heavy, gravity-free body that follows the wrist target (MuJoCo: the
              mocap weld). GPU PhysX has no kinematic targets in its tensor API, and a weld to a kinematic
              body moved by pose writes keeps the hand where the body was, so the body is dynamic (100 kg)
              and gets, every substep, the velocity that brings it to the target by the next one. Finger
              servos as in MuJoCo (stiffness, damping, armature, torque limits; a tendon-coupled J1/J2
              pair gets one drive each at twice the stiffness).
  controller  the scanned controller (convex decomposition), dynamic.
  holder      the scanned holder (convex decomposition), kinematic: posed per env at reset.
  tray        a kinematic box under the middle of the controller (its sides overhang for the fingers).

Control: every physics substep the wrist target (pose) and 24 joint position targets are applied from
two buffers. The scripted teacher (hand_teacher.py) fills them from a sampled plan; the learned policy
will fill them from its actions. Scene poses are relative to each env's origin.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_box_minus
from isaaclab_physx.physics import PhysxCfg

import hand_teacher as T

ROOT = T.ROOT
FIT = os.path.join(ROOT, "output", "holder_fit", "assets")
HOLDER_USD = os.path.join(FIT, "cod_holder", "cod_holder.usda")
CTRL_USD = os.path.join(FIT, "controller", "controller.usda")
HAND_USD = os.path.join(ROOT, "output", "isaac_hand", "right_hand", "right_hand.usda")
SPEC = T.Spec()
PARK = (0.0, -0.6, 0.6)  # where the hand waits (relative to the env origin) while the scene settles
# (color, roughness, metallic) as isaac_hand_place.py: glossy light grey shells, darker forearm, metal knuckles.
LOOKS = {"shell": ((0.78, 0.78, 0.79), 0.28, 0.0), "forearm": ((0.62, 0.62, 0.64), 0.3, 0.0),
         "metal": ((0.55, 0.56, 0.58), 0.3, 0.8), "tray": ((0.70, 0.69, 0.67), 0.32, 0.0)}


def _servo_actuators() -> dict:
    sv = SPEC.servo
    kp = {n: s["kp"] * (2.0 if s["tendon"] else 1.0) for n, s in sv.items()}
    return {"hand": ImplicitActuatorCfg(
        joint_names_expr=list(sv),
        stiffness=kp,
        damping={n: s["damping"] for n, s in sv.items()},
        armature={n: s["armature"] for n, s in sv.items()},
        effort_limit_sim={n: max(abs(v) for v in s["force_range"]) for n, s in sv.items()},
    )}


def _kinematic(**kw):
    return sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True, **kw)


@configclass
class HandPlaceEnvCfg(DirectRLEnvCfg):
    decimation = 8  # 240 Hz physics, 30 Hz control
    episode_length_s = 60.0
    action_space = 6 + 24
    observation_space = 1
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=8,
        physics_material=RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
        physics=PhysxCfg(gpu_max_rigid_contact_count=2**23, gpu_max_rigid_patch_count=2**22),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=12, env_spacing=1.0, replicate_physics=True,
                                                     clone_in_fabric=False)

    hand: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HAND_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=PARK, rot=(0.0, 0.0, 0.0, 1.0)),
        actuators=_servo_actuators(),
        soft_joint_pos_limit_factor=1.0,
    )
    wrist: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/WristTarget",
        spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2),  # inertia ~0.7 kg m^2: contacts barely turn it
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,
                                                                               linear_damping=0.0,
                                                                               angular_damping=0.0,
                                                                               max_depenetration_velocity=1.0),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                                  collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                                  visible=False),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PARK, rot=(0.0, 0.0, 0.0, 1.0)),
    )
    controller: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Controller",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CTRL_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=32,
                                                         solver_velocity_iteration_count=4,
                                                         max_depenetration_velocity=1.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.05, 0.16)),
    )
    holder: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Holder",
        spawn=sim_utils.UsdFileCfg(usd_path=HOLDER_USD, rigid_props=_kinematic()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.25, 0.0)),
    )
    tray: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Tray",
        spawn=sim_utils.CuboidCfg(
            size=tuple(float(2 * v) for v in SPEC.tray_half), rigid_props=_kinematic(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=LOOKS["tray"][0], roughness=LOOKS["tray"][1])),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.05, SPEC.tray_h / 2)),
    )
    skin_friction = 1.0  # hand skin, combined with "max" (MuJoCo takes the larger coefficient too)


class HandPlaceEnv(DirectRLEnv):
    cfg: HandPlaceEnvCfg

    def __init__(self, cfg: HandPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        self.joint_names = list(self.hand.joint_names)
        self.forearm_idx = self.hand.body_names.index("rh_forearm")
        self.keys = SPEC.joint_targets(self.joint_names)
        self.wrist_target = torch.zeros((n, 7), device=self.device)
        self.joint_target = torch.zeros((n, len(self.joint_names)), device=self.device)
        self._all = torch.arange(n, device=self.device, dtype=torch.int32)
        # scripted teacher: padded sampled plans (relative poses), per-env cursor and re-aim index
        self.teacher_on = False
        self.on_substep = None  # optional callback(env) each substep (the re-aim)
        park = torch.tensor([*PARK, 0.0, 0.0, 0.0, 1.0], device=self.device)
        self.wrist_target[:] = park
        self.wrist_target[:, :3] += self.scene.env_origins
        self.joint_target[:] = torch.tensor(self.keys[SPEC.open_key], device=self.device)

    # --- scene ------------------------------------------------------------------------------------

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.hand)
        self.wrist = RigidObject(self.cfg.wrist)
        self.controller = RigidObject(self.cfg.controller)
        self.holder = RigidObject(self.cfg.holder)
        self.tray = RigidObject(self.cfg.tray)
        # a plain light floor, as the project's other Isaac renders (the stock ground plane is a blue grid)
        floor = sim_utils.CuboidCfg(size=(200.0, 200.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.54),
                                                                                roughness=0.6))
        floor.func("/World/ground", floor, translation=(0.0, 0.0, -0.05))
        self._author_env0()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["hand"] = self.hand
        for k in ("wrist", "controller", "holder", "tray"):
            self.scene.rigid_objects[k] = getattr(self, k)
        dome = sim_utils.DomeLightCfg(intensity=700.0, color=(1.0, 1.0, 1.0))
        dome.func("/World/Dome", dome)
        # a sun from the front-left, the same for every cell (a point key light only lights env 0)
        sun = sim_utils.DistantLightCfg(intensity=750.0, angle=1.5, color=(1.0, 0.97, 0.92))
        from scipy.spatial.transform import Rotation

        d = np.array([0.35, 0.55, -0.76]) / np.linalg.norm([0.35, 0.55, -0.76])  # light travels down, to +x +y
        q = Rotation.align_vectors([d], [[0.0, 0.0, -1.0]])[0].as_quat()  # the light shines along its -z
        sun.func("/World/Sun", sun, orientation=tuple(float(v) for v in q))

    def _author_env0(self):
        """In env 0, before cloning: the forearm weld, the hand's skin friction and its studio look."""
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade

        stage = self.scene.stage
        env0 = "/World/envs/env_0"
        hand = stage.GetPrimAtPath(f"{env0}/Hand")
        forearm = next(p for p in Usd.PrimRange(hand) if p.GetName() == "rh_forearm")
        weld = UsdPhysics.FixedJoint.Define(stage, f"{env0}/WristWeld")
        weld.CreateBody0Rel().SetTargets([Sdf.Path(f"{env0}/WristTarget")])
        weld.CreateBody1Rel().SetTargets([forearm.GetPath()])
        weld.CreateExcludeFromArticulationAttr(True)
        for a in ("CreateLocalPos0Attr", "CreateLocalPos1Attr"):
            getattr(weld, a)(Gf.Vec3f(0, 0, 0))
        for a in ("CreateLocalRot0Attr", "CreateLocalRot1Attr"):
            getattr(weld, a)(Gf.Quatf(1, 0, 0, 0))
        skin = UsdShade.Material.Define(stage, "/World/Looks/skin")
        pm = UsdPhysics.MaterialAPI.Apply(skin.GetPrim())
        pm.CreateStaticFrictionAttr(self.cfg.skin_friction)
        pm.CreateDynamicFrictionAttr(self.cfg.skin_friction)
        pm.CreateRestitutionAttr(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(skin.GetPrim()).CreateFrictionCombineModeAttr("max")
        looks = {}
        for k, (c, r, m) in LOOKS.items():
            mat = UsdShade.Material.Define(stage, f"/World/Looks/{k}")
            sh = UsdShade.Shader.Define(stage, f"/World/Looks/{k}/PreviewSurface")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*c))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(r)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(m)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            looks[k] = mat
        for p in Usd.PrimRange(hand):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(skin, UsdShade.Tokens.strongerThanDescendants, "physics")
            if p.IsInstance():
                name = p.GetName()
                look = looks["metal"] if "knuckle" in name else looks["forearm"] if "forearm" in name else looks["shell"]
                UsdShade.MaterialBindingAPI.Apply(p).Bind(look, UsdShade.Tokens.strongerThanDescendants)

    # --- control ----------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()  # the learned policy maps these into the targets (later)

    def _apply_action(self) -> None:
        if self.on_substep is not None:
            self.on_substep(self)
        # The wrist body's velocity: reach the target by the end of this substep.
        cur = self.wrist.data.root_link_pose_w.torch
        v = (self.wrist_target[:, :3] - cur[:, :3]) / self.physics_dt
        w = quat_box_minus(self.wrist_target[:, 3:7], cur[:, 3:7]) / self.physics_dt
        self.wrist.write_root_velocity_to_sim_index(root_velocity=torch.cat([v, w], -1), env_ids=self._all.long())
        self.hand.set_joint_position_target_index(target=self.joint_target)

    # --- state helpers (poses relative to each env origin, (x, y, z, qx, qy, qz, qw)) ----------------

    def rel(self, pose_w: torch.Tensor) -> torch.Tensor:
        out = pose_w.clone()
        out[:, :3] -= self.scene.env_origins
        return out

    def controller_pose(self) -> torch.Tensor:
        return self.rel(self.controller.data.root_link_pose_w.torch)

    def forearm_pose(self) -> torch.Tensor:
        return self.rel(self.hand.data.body_link_pose_w.torch[:, self.forearm_idx])

    def place_scene(self, scenes: list[dict], env_ids: torch.Tensor | None = None, hand_poses=None) -> None:
        """Pose the holder, tray and controller from teacher scenes (4x4, env-relative), and park the hand
        (at hand_poses, 4x4 env-relative, open; default: PARK)."""
        ids = self._all.long() if env_ids is None else env_ids.long()
        o = self.scene.env_origins[ids]

        def w(key):
            p = torch.tensor(np.array([T.pose7(s[key]) for s in scenes]), dtype=torch.float32, device=self.device)
            p[:, :3] += o
            return p

        self.holder.write_root_pose_to_sim_index(root_pose=w("holder"), env_ids=ids)
        self.tray.write_root_pose_to_sim_index(root_pose=w("tray"), env_ids=ids)
        self.controller.write_root_pose_to_sim_index(root_pose=w("controller_start"), env_ids=ids)
        self.controller.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(ids), 6), device=self.device),
                                                         env_ids=ids)
        if hand_poses is None:
            park = torch.tensor([*PARK, 0.0, 0.0, 0.0, 1.0], device=self.device).repeat(len(ids), 1)
        else:
            park = torch.tensor(np.array([T.pose7(h) for h in hand_poses]), dtype=torch.float32, device=self.device)
        park[:, :3] += o
        self.teleport_hand(park, torch.tensor(self.keys[SPEC.open_key], device=self.device).repeat(len(ids), 1), ids)

    def teleport_hand(self, pose_w: torch.Tensor, q: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Hand, its wrist target and joint targets to a pose (world) and joint angles, at rest."""
        ids = env_ids.long()
        self.hand.write_root_pose_to_sim_index(root_pose=pose_w, env_ids=ids)
        self.hand.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(ids), 6), device=self.device),
                                                   env_ids=ids)
        self.hand.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=ids)
        self.wrist.write_root_pose_to_sim_index(root_pose=pose_w, env_ids=ids)
        self.wrist.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(ids), 6), device=self.device),
                                                    env_ids=ids)
        self.wrist_target[ids] = pose_w
        self.joint_target[ids] = q

    # --- RL interface (filled in with the policy; the scripted rollout only steps) ----------------------

    def _get_observations(self) -> dict:
        return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)


def quat_angle_deg(qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
    d = torch.abs((qa * qb).sum(-1)).clamp(max=1.0)
    return torch.rad2deg(2 * torch.acos(d))


__all__ = ["HandPlaceEnv", "HandPlaceEnvCfg", "SPEC", "quat_angle_deg"]
