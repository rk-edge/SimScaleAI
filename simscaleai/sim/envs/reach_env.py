"""Reach environment — robot must move end-effector to a target position.

Simplest manipulation task. Good starting point for testing the full stack.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces

from simscaleai.sim.base_env import BaseRobotEnv, SimConfig


_ASSET_DIR = Path(__file__).parent.parent / "assets"


class ReachEnv(BaseRobotEnv):
    """Franka Panda arm reaching a target 3D position.

    Observation:
        - joint_pos (7,): robot joint angles
        - joint_vel (7,): robot joint velocities
        - ee_pos (3,): end-effector position
        - target_pos (3,): goal position
        - image (H, W, 3): RGB camera image (if cameras configured)

    Action:
        - delta_pos (3,): end-effector position delta (x, y, z)
        - gripper (1,): gripper open/close (unused in reach, kept for compatibility)

    Reward:
        - Negative L2 distance from end-effector to target
        - +1.0 bonus for being within 0.05m of target
    """

    def __init__(self, config: SimConfig | None = None, render_mode: str | None = None):
        if config is None:
            config = SimConfig()

        # Use default scene if no XML specified
        if not config.xml_path:
            config.xml_path = str(_ASSET_DIR / "panda_reach.xml")

        # Generate default XML if it doesn't exist
        if not os.path.exists(config.xml_path):
            os.makedirs(os.path.dirname(config.xml_path), exist_ok=True)
            _generate_default_scene(config.xml_path)

        super().__init__(config, render_mode)

        # Target position (set during reset)
        self._target_pos = np.zeros(3)

        # Cache body/site IDs
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

    def _build_observation_space(self) -> spaces.Space:
        obs_dict = {
            "joint_pos": spaces.Box(-np.pi, np.pi, shape=(7,), dtype=np.float64),
            "joint_vel": spaces.Box(-10.0, 10.0, shape=(7,), dtype=np.float64),
            "ee_pos": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float64),
            "target_pos": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float64),
        }
        # Add image observation if cameras configured
        if self.config.cameras:
            cam = self.config.cameras[0]
            obs_dict["image"] = spaces.Box(
                0, 255, shape=(cam.height, cam.width, 3), dtype=np.uint8
            )
        return spaces.Dict(obs_dict)

    def _build_action_space(self) -> spaces.Space:
        # 3D end-effector delta + gripper
        return spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float64)

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = {
            "joint_pos": self.data.qpos[:7].copy(),
            "joint_vel": self.data.qvel[:7].copy(),
            "ee_pos": self.data.site_xpos[self._ee_site_id].copy(),
            "target_pos": self._target_pos.copy(),
        }
        if self.config.cameras:
            camera_data = self.render_camera()
            obs["image"] = camera_data["rgb"]
        return obs

    def _get_reward(self) -> float:
        ee_pos = self.data.site_xpos[self._ee_site_id]
        dist = np.linalg.norm(ee_pos - self._target_pos)
        reward = -float(dist)
        # Bonus for reaching target
        if dist < 0.05:
            reward += 1.0
        return reward

    def _is_terminated(self) -> bool:
        ee_pos = self.data.site_xpos[self._ee_site_id]
        dist = np.linalg.norm(ee_pos - self._target_pos)
        return bool(dist < 0.02)  # Success threshold

    def _reset_task(self, rng: np.random.Generator) -> None:
        # Random target position in workspace
        self._target_pos = np.array([
            rng.uniform(0.3, 0.7),   # x: forward
            rng.uniform(-0.3, 0.3),  # y: left-right
            rng.uniform(0.1, 0.5),   # z: height
        ])
        # Move target visual body
        self.model.body_pos[self._target_body_id] = self._target_pos

    def _apply_action(self, action: np.ndarray) -> None:
        """Map 4D action (delta_pos[3] + gripper[1]) to 9 actuators via Jacobian."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta_pos = action[:3] * 0.05  # Scale delta to metres
        gripper_cmd = action[3]

        # Compute positional Jacobian at end-effector site
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._ee_site_id)

        # Use Jacobian transpose to convert EE delta → joint torques
        j_arm = jacp[:, :7]  # Only arm joints (first 7 DoF)
        joint_delta = j_arm.T @ delta_pos

        # Set arm control (position offset from current)
        self.data.ctrl[:7] = self.data.qpos[:7] + joint_delta
        # Set gripper fingers
        self.data.ctrl[7] = gripper_cmd * 0.04
        self.data.ctrl[8] = gripper_cmd * 0.04

    def _get_info(self) -> dict:
        ee_pos = self.data.site_xpos[self._ee_site_id]
        dist = float(np.linalg.norm(ee_pos - self._target_pos))
        return {
            "step": self._step_count,
            "distance": dist,
            "success": dist < 0.05,
        }


def _generate_default_scene(xml_path: str) -> None:
    """Generate a minimal Franka Panda reach scene MJCF XML."""
    xml = """<mujoco model="panda_reach">
  <compiler angle="radian" meshdir="." autolimits="true"/>

  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <default>
    <default class="panda">
      <joint damping="1.0" armature="0.1"/>
      <geom condim="4" friction="1 0.5 0.01" margin="0.001"/>
      <position kp="100" kv="20"/>
    </default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="1" type="sphere" size="0.02" rgba="0.9 0.2 0.2 1"/>
    </default>
  </default>

  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.8 0.8 0.8" rgb2="0.6 0.6 0.6"
             width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4" reflectance="0.1"/>
    <material name="robot_mat" rgba="0.7 0.7 0.7 1"/>
    <material name="target_mat" rgba="0.2 0.8 0.2 0.5"/>
  </asset>

  <worldbody>
    <!-- Floor -->
    <geom name="floor" type="plane" size="2 2 0.01" material="grid_mat"/>

    <!-- Table -->
    <body name="table" pos="0.5 0 0.2">
      <geom type="box" size="0.4 0.5 0.2" rgba="0.6 0.4 0.2 1" mass="100"/>
    </body>

    <!-- Simplified Panda Arm (7-DOF) -->
    <body name="link0" pos="0.1 0 0.4" childclass="panda">
      <geom type="cylinder" size="0.06 0.05" rgba="0.9 0.9 0.9 1" mass="2"/>

      <body name="link1" pos="0 0 0.05">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.05" material="robot_mat" mass="2"/>

        <body name="link2" pos="0 0 0.15">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1.76 1.76"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.04" material="robot_mat" mass="2"/>

          <body name="link3" pos="0 0 0.15">
            <joint name="j3" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.04" material="robot_mat" mass="1.5"/>

            <body name="link4" pos="0 0 0.12">
              <joint name="j4" type="hinge" axis="0 -1 0" range="-3.07 0.07"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.035" material="robot_mat" mass="1.5"/>

              <body name="link5" pos="0 0 0.12">
                <joint name="j5" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.035" material="robot_mat" mass="1"/>

                <body name="link6" pos="0 0 0.1">
                  <joint name="j6" type="hinge" axis="0 1 0" range="-0.02 3.75"/>
                  <geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.03" material="robot_mat" mass="1"/>

                  <body name="link7" pos="0 0 0.08">
                    <joint name="j7" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
                    <geom type="cylinder" size="0.04 0.02" material="robot_mat" mass="0.5"/>

                    <!-- End-effector site -->
                    <site name="ee_site" pos="0 0 0.04" size="0.01" rgba="1 0 0 1"/>

                    <!-- Simple gripper -->
                    <body name="left_finger" pos="0 0.02 0.04">
                      <joint name="finger_left" type="slide" axis="0 1 0" range="0 0.04"/>
                      <geom type="box" size="0.01 0.005 0.03" rgba="0.5 0.5 0.5 1" mass="0.1"/>
                    </body>
                    <body name="right_finger" pos="0 -0.02 0.04">
                      <joint name="finger_right" type="slide" axis="0 -1 0" range="0 0.04"/>
                      <geom type="box" size="0.01 0.005 0.03" rgba="0.5 0.5 0.5 1" mass="0.1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Target (visual only) -->
    <body name="target" pos="0.5 0 0.5">
      <geom type="sphere" size="0.03" material="target_mat" contype="0" conaffinity="0"/>
    </body>

    <!-- Lights -->
    <light name="top_light" pos="0.5 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light name="side_light" pos="0 1 1" dir="0 -0.5 -0.5" diffuse="0.5 0.5 0.5"/>

    <!-- Camera -->
    <camera name="wrist_cam" pos="0.8 0 0.8" xyaxes="0 1 0 -0.5 0 0.5" fovy="60"/>
    <camera name="front_cam" pos="1.5 0 0.8" xyaxes="0 1 0 -0.5 0 0.8" fovy="60"/>
  </worldbody>

  <actuator>
    <position class="panda" name="act_j1" joint="j1"/>
    <position class="panda" name="act_j2" joint="j2"/>
    <position class="panda" name="act_j3" joint="j3"/>
    <position class="panda" name="act_j4" joint="j4"/>
    <position class="panda" name="act_j5" joint="j5"/>
    <position class="panda" name="act_j6" joint="j6"/>
    <position class="panda" name="act_j7" joint="j7"/>
    <position name="act_finger_left" joint="finger_left" kp="50"/>
    <position name="act_finger_right" joint="finger_right" kp="50"/>
  </actuator>
</mujoco>
"""
    with open(xml_path, "w") as f:
        f.write(xml)
