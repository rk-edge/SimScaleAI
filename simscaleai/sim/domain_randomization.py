"""Domain randomization pipeline for sim-to-real transfer.

Provides systematic randomization of visual and physics parameters
to improve policy robustness across the sim-to-real gap.

Randomization targets:
  Visual: lighting direction/color, camera pose, material colors/reflectance
  Physics: friction, mass, damping, actuator gains
  Geometry: object size, table position
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization.

    Each field defines the randomization range as (scale_low, scale_high)
    multiplicative factors applied to the nominal value, or absolute ranges
    where specified.
    """

    # ── Visual Randomization ──────────────────────────────────────────
    randomize_lighting: bool = True
    light_direction_range: float = 1.0       # uniform [-range, range] for xyz
    light_diffuse_range: tuple[float, float] = (0.3, 1.0)
    light_ambient_range: tuple[float, float] = (0.0, 0.5)

    randomize_camera: bool = True
    camera_pos_noise: float = 0.05           # metres
    camera_fovy_range: tuple[float, float] = (50.0, 70.0)

    randomize_materials: bool = True
    color_noise: float = 0.15                # uniform noise per RGB channel

    # ── Physics Randomization ─────────────────────────────────────────
    randomize_friction: bool = True
    friction_scale: tuple[float, float] = (0.7, 1.3)

    randomize_mass: bool = True
    mass_scale: tuple[float, float] = (0.5, 2.0)  # for manipulated objects only

    randomize_damping: bool = True
    damping_scale: tuple[float, float] = (0.8, 1.2)

    randomize_gains: bool = True
    kp_scale: tuple[float, float] = (0.8, 1.2)

    # ── Geometry Randomization ────────────────────────────────────────
    randomize_object_size: bool = True
    size_scale: tuple[float, float] = (0.7, 1.5)

    randomize_table_pos: bool = False
    table_pos_noise: float = 0.02

    # ── Dynamics Randomization ────────────────────────────────────────
    randomize_gravity: bool = False
    gravity_noise: float = 0.5               # m/s² noise on gravity z

    randomize_timestep: bool = False
    timestep_scale: tuple[float, float] = (0.8, 1.2)


def apply_domain_randomization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    config: DomainRandomizationConfig | None = None,
    nominal_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply domain randomization to a MuJoCo model.

    Args:
        model: MuJoCo model to randomize
        data: MuJoCo data
        rng: Random number generator
        config: Randomization config (default = all enabled)
        nominal_state: Cached nominal values for relative randomization.
                      If None, current values are treated as nominal.

    Returns:
        Dict of applied randomizations (for logging/debugging)
    """
    config = config or DomainRandomizationConfig()
    applied: dict[str, Any] = {}

    # Cache nominal values on first call
    if nominal_state is None:
        nominal_state = _cache_nominal(model)

    # ── Visual ────────────────────────────────────────────────────────
    if config.randomize_lighting and model.nlight > 0:
        for i in range(model.nlight):
            model.light_dir[i] = rng.uniform(
                -config.light_direction_range,
                config.light_direction_range,
                size=3,
            )
            model.light_diffuse[i] = rng.uniform(
                *config.light_diffuse_range, size=3
            )
        applied["lighting"] = True

    if config.randomize_camera:
        for i in range(model.ncam):
            noise = rng.uniform(
                -config.camera_pos_noise, config.camera_pos_noise, size=3
            )
            model.cam_pos[i] = nominal_state["cam_pos"][i] + noise
            model.cam_fovy[i] = rng.uniform(*config.camera_fovy_range)
        applied["camera"] = True

    if config.randomize_materials:
        for i in range(model.nmat):
            noise = rng.uniform(-config.color_noise, config.color_noise, size=4)
            noise[3] = 0  # Don't randomize alpha
            model.mat_rgba[i] = np.clip(
                nominal_state["mat_rgba"][i] + noise, 0, 1
            )
        applied["materials"] = True

    # ── Physics ───────────────────────────────────────────────────────
    if config.randomize_friction:
        for i in range(model.ngeom):
            scale = rng.uniform(*config.friction_scale, size=3)
            model.geom_friction[i] = nominal_state["geom_friction"][i] * scale
        applied["friction"] = True

    if config.randomize_mass:
        # Only randomize free-body objects (have free joints)
        for i in range(model.nbody):
            if model.body_jntnum[i] == 1:
                jnt_idx = model.body_jntadr[i]
                if model.jnt_type[jnt_idx] == mujoco.mjtJoint.mjJNT_FREE:
                    # Randomize all geoms of this body
                    start = model.body_geomadr[i]
                    for g in range(model.body_geomnum[i]):
                        scale = rng.uniform(*config.mass_scale)
                        model.body_mass[i] = nominal_state["body_mass"].get(
                            i, model.body_mass[i]
                        ) * scale
        applied["mass"] = True

    if config.randomize_damping:
        for i in range(model.njnt):
            scale = rng.uniform(*config.damping_scale)
            model.dof_damping[i] = nominal_state["dof_damping"][i] * scale
        applied["damping"] = True

    if config.randomize_gains:
        for i in range(model.nu):
            scale = rng.uniform(*config.kp_scale)
            model.actuator_gainprm[i, 0] = (
                nominal_state["actuator_gainprm"][i, 0] * scale
            )
        applied["gains"] = True

    # ── Geometry ──────────────────────────────────────────────────────
    if config.randomize_object_size:
        for i in range(model.nbody):
            if model.body_jntnum[i] == 1:
                jnt_idx = model.body_jntadr[i]
                if model.jnt_type[jnt_idx] == mujoco.mjtJoint.mjJNT_FREE:
                    start = model.body_geomadr[i]
                    for g in range(model.body_geomnum[i]):
                        gidx = start + g
                        scale = rng.uniform(*config.size_scale)
                        model.geom_size[gidx] = (
                            nominal_state["geom_size"][gidx] * scale
                        )
        applied["object_size"] = True

    if config.randomize_table_pos:
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and "table" in name.lower():
                noise = rng.uniform(
                    -config.table_pos_noise, config.table_pos_noise, size=3
                )
                noise[2] = 0  # Don't move table vertically
                model.body_pos[i] = nominal_state.get("table_pos", model.body_pos[i]) + noise
        applied["table_pos"] = True

    # ── Dynamics ──────────────────────────────────────────────────────
    if config.randomize_gravity:
        model.opt.gravity[2] = -9.81 + rng.uniform(
            -config.gravity_noise, config.gravity_noise
        )
        applied["gravity"] = model.opt.gravity[2]

    if config.randomize_timestep:
        scale = rng.uniform(*config.timestep_scale)
        model.opt.timestep = nominal_state["timestep"] * scale
        applied["timestep"] = model.opt.timestep

    return applied


def _cache_nominal(model: mujoco.MjModel) -> dict[str, Any]:
    """Cache nominal model values for relative randomization."""
    nominal = {
        "cam_pos": model.cam_pos.copy(),
        "cam_fovy": model.cam_fovy.copy(),
        "mat_rgba": model.mat_rgba.copy(),
        "geom_friction": model.geom_friction.copy(),
        "geom_size": model.geom_size.copy(),
        "dof_damping": model.dof_damping.copy(),
        "actuator_gainprm": model.actuator_gainprm.copy(),
        "timestep": model.opt.timestep,
        "body_mass": {i: model.body_mass[i] for i in range(model.nbody)},
    }
    return nominal
