"""The hand-place task (hand_place_env.py) seen by fixed RGB-D cameras, for the vision-based policy.

Two cameras per cell (Isaac Lab's Camera: every env's views rendered in one batched pass), on either side of
the table, looking down at the workspace between the tray and the holder: colour plus depth (distance to the
image plane), like a RealSense or an iPhone's LiDAR. The cells are spaced wider than for the state-based
policy so neighbouring cells stay out of view.

Looks are randomized per episode (randomize_looks): the dome and sun (intensity, colour, the sun's direction),
the floor's and the tray's colour. The scanned assets keep their own textures: they are what is being seen.

Runs in the Isaac Lab venv, with the app launched with enable_cameras=True.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.configclass import configclass

from hand_place_env import HandPlaceEnv, HandPlaceEnvCfg

# (eye, look-at) per camera, relative to the env origin: the tray is at (0.25, 0.05), the holder at (0, 0.25);
# the hand comes in from -y, so the cameras look from the sides (+x and -x), not past its forearm.
CAMS = {
    "cam_a": ((0.78, -0.10, 0.55), (0.13, 0.15, 0.06)),
    "cam_b": ((-0.52, -0.05, 0.55), (0.13, 0.15, 0.06)),
}
IMG_W, IMG_H = 160, 120
SPACING = 2.5


def look_quat(eye, target) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a camera at `eye` looking at `target`, z up, in Isaac Lab's "world" convention
    (forward +x, up +z)."""
    f = np.subtract(target, eye).astype(float)
    f /= np.linalg.norm(f)
    left = np.cross([0.0, 0.0, 1.0], f)
    left /= np.linalg.norm(left)
    up = np.cross(f, left)
    return tuple(float(v) for v in Rotation.from_matrix(np.c_[f, left, up]).as_quat())


def camera_cfg(name: str, width: int = IMG_W, height: int = IMG_H, eye=None, target=None, focal: float = 24.0,
               data_types=("rgb", "distance_to_image_plane")) -> CameraCfg:
    if eye is None:
        eye, target = CAMS[name]
    return CameraCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        offset=CameraCfg.OffsetCfg(pos=eye, rot=look_quat(eye, target), convention="world"),
        data_types=list(data_types),
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal, focus_distance=1.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 5.0)),
        width=width, height=height,
    )


@configclass
class VisionHandPlaceEnvCfg(HandPlaceEnvCfg):
    cam_a: CameraCfg = camera_cfg("cam_a")
    cam_b: CameraCfg = camera_cfg("cam_b")
    film: CameraCfg | None = None  # an extra colour camera for filming (vision_film.py)

    def __post_init__(self):
        self.scene.env_spacing = SPACING


class VisionHandPlaceEnv(HandPlaceEnv):
    cfg: VisionHandPlaceEnvCfg
    CAM_NAMES = tuple(CAMS)

    def _author_env0(self):
        super()._author_env0()
        self.cams = {k: Camera(getattr(self.cfg, k)) for k in self.CAM_NAMES}  # in env 0, before cloning
        if self.cfg.film is not None:
            self.cams["film"] = Camera(self.cfg.film)

    def _setup_scene(self):
        import carb

        # auto exposure (like a phone or a RealSense): without it the studio lights clip the light-grey
        # hand, tray and floor to near-white and the randomized lighting barely shows
        carb.settings.get_settings().set("/rtx/post/histogram/enabled", True)
        super()._setup_scene()
        for k, c in self.cams.items():
            self.scene.sensors[k] = c

    def images(self) -> torch.Tensor:
        """(n, views, 4, H, W) float: RGB in [0, 1] and depth in m (0 where nothing was hit)."""
        out = []
        for k in self.CAM_NAMES:
            d = self.cams[k].data.output
            rgb = d["rgb"][..., :3].float() / 255.0
            dep = d["distance_to_image_plane"].float()
            dep = torch.nan_to_num(dep, nan=0.0, posinf=0.0, neginf=0.0)
            out.append(torch.cat([rgb, dep], -1).permute(0, 3, 1, 2))
        return torch.stack(out, 1)

    # --- looks ---------------------------------------------------------------------------------------

    def randomize_looks(self, rng: np.random.Generator | None):
        """New lighting and colours for every cell (rng None: the nominal look)."""
        from pxr import Gf, UsdShade

        stage = self.scene.stage
        dome, sun = stage.GetPrimAtPath("/World/Dome"), stage.GetPrimAtPath("/World/Sun")
        if rng is None:
            dome_i, sun_i, warm, d = 700.0, 750.0, 0.0, np.array([0.35, 0.55, -0.76])
            floor, tray = (0.55, 0.55, 0.54), (0.70, 0.69, 0.67)
        else:
            dome_i, sun_i = rng.uniform(250, 1200), rng.uniform(0, 1500)
            warm = rng.uniform(-0.15, 0.15)
            az, el = rng.uniform(0, 2 * np.pi), rng.uniform(np.deg2rad(25), np.deg2rad(80))
            d = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            floor = tuple(np.clip(rng.uniform(0.15, 0.85) * (1 + rng.uniform(-0.25, 0.25, 3)), 0, 1))
            tray = tuple(np.clip(rng.uniform(0.15, 0.90) * (1 + rng.uniform(-0.3, 0.3, 3)), 0, 1))
        tint = Gf.Vec3f(1.0 + warm, 1.0, 1.0 - warm)
        dome.GetAttribute("inputs:intensity").Set(float(dome_i))
        dome.GetAttribute("inputs:color").Set(tint)
        sun.GetAttribute("inputs:intensity").Set(float(sun_i))
        sun.GetAttribute("inputs:color").Set(tint)
        d = d / np.linalg.norm(d)
        q = Rotation.align_vectors([d], [[0.0, 0.0, -1.0]])[0].as_quat()
        xf = sun.GetAttribute("xformOp:orient")
        xf.Set(type(xf.Get())(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
        floor_n = tray_n = 0
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            is_floor = p.startswith("/World/ground")
            is_tray = p.startswith("/World/envs/env_") and "/Tray/" in p
            if not (is_floor or is_tray) or not prim.IsA(UsdShade.Shader):
                continue
            inp = UsdShade.Shader(prim).GetInput("diffuseColor")
            if inp:
                inp.Set(Gf.Vec3f(*[float(c) for c in (floor if is_floor else tray)]))
                floor_n += is_floor
                tray_n += is_tray
        self.looks = {"dome": float(dome_i), "sun": float(sun_i), "warm": float(warm), "floor": list(floor),
                      "tray": list(tray), "shaders": [floor_n, tray_n]}


__all__ = ["VisionHandPlaceEnv", "VisionHandPlaceEnvCfg", "CAMS", "IMG_W", "IMG_H", "look_quat"]
