"""Juggle environment — robot must keep three balls in the air.

The Franka Panda arm has a flat paddle attached to the end-effector.
Three balls are placed on the paddle and the robot must toss and catch them,
keeping all three airborne in a juggling pattern.

This is a challenging dynamic manipulation task that tests:
- Fast reactive control
- Multi-object tracking
- Precise timing and velocity control
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces

from simscaleai.sim.base_env import BaseRobotEnv, SimConfig


_ASSET_DIR = Path(__file__).parent.parent / "assets"

# Ball names in the MJCF
_BALL_NAMES = ["ball_1", "ball_2", "ball_3"]
_BALL_COLORS = [
    [1.0, 0.2, 0.2, 1.0],  # red
    [0.2, 0.8, 0.2, 1.0],  # green
    [0.2, 0.3, 1.0, 1.0],  # blue
]


class JuggleEnv(BaseRobotEnv):
    """Franka Panda arm juggling three balls with a paddle.

    Observation:
        - joint_pos (7,): robot joint angles
        - joint_vel (7,): robot joint velocities
        - ee_pos (3,): end-effector (paddle) position
        - ee_vel (3,): end-effector velocity
        - ball_pos (9,): positions of 3 balls (3 x 3)
        - ball_vel (9,): velocities of 3 balls (3 x 3)
        - image (H,W,3): RGB camera image (if cameras configured)

    Action:
        - delta_pos (3,): end-effector position delta (x, y, z)
        - paddle_tilt (1,): wrist tilt angle for directing tosses

    Reward:
        - +1.0 for each ball above the paddle height
        - +5.0 bonus if all 3 balls are airborne simultaneously
        - -10.0 if any ball hits the floor
        - Small reward for keeping balls close to center (catchable)
    """

    def __init__(self, config: SimConfig | None = None, render_mode: str | None = None):
        if config is None:
            config = SimConfig()

        if not config.xml_path:
            config.xml_path = str(_ASSET_DIR / "panda_juggle.xml")

        if not os.path.exists(config.xml_path):
            os.makedirs(os.path.dirname(config.xml_path), exist_ok=True)
            _generate_juggle_scene(config.xml_path)

        # Juggling needs faster control for reactive catching
        if config.control_dt == 0.05:
            config.control_dt = 0.02  # 50Hz control

        super().__init__(config, render_mode)

        # Cache body/site IDs
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_site"
        )
        self._ball_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in _BALL_NAMES
        ]
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self._ball_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
            for name in _BALL_NAMES
        ]

        # Track balls that have fallen
        self._balls_dropped = set()

    def _build_observation_space(self) -> spaces.Space:
        obs_dict = {
            "joint_pos": spaces.Box(-np.pi, np.pi, shape=(7,), dtype=np.float64),
            "joint_vel": spaces.Box(-10.0, 10.0, shape=(7,), dtype=np.float64),
            "ee_pos": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float64),
            "ee_vel": spaces.Box(-5.0, 5.0, shape=(3,), dtype=np.float64),
            "ball_pos": spaces.Box(-2.0, 2.0, shape=(9,), dtype=np.float64),
            "ball_vel": spaces.Box(-10.0, 10.0, shape=(9,), dtype=np.float64),
        }
        if self.config.cameras:
            cam = self.config.cameras[0]
            obs_dict["image"] = spaces.Box(
                0, 255, shape=(cam.height, cam.width, 3), dtype=np.uint8
            )
        return spaces.Dict(obs_dict)

    def _build_action_space(self) -> spaces.Space:
        # 3D EE delta + wrist tilt
        return spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float64)

    def _get_obs(self) -> dict[str, np.ndarray]:
        # End-effector position and velocity
        ee_pos = self.data.site_xpos[self._ee_site_id].copy()

        # Compute EE velocity via Jacobian
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._ee_site_id)
        ee_vel = jacp @ self.data.qvel

        # Ball states
        ball_positions = []
        ball_velocities = []
        for bid in self._ball_body_ids:
            # Body com position
            ball_positions.append(self.data.xpos[bid].copy())
            # Body com velocity (from qvel of their free joints)
            # Each free joint has 6 dof: 3 trans + 3 rot
            joint_id = self.model.body_jntadr[bid]
            if joint_id >= 0:
                qvel_adr = self.model.jnt_dofadr[joint_id]
                ball_velocities.append(self.data.qvel[qvel_adr:qvel_adr + 3].copy())
            else:
                ball_velocities.append(np.zeros(3))

        obs = {
            "joint_pos": self.data.qpos[:7].copy(),
            "joint_vel": self.data.qvel[:7].copy(),
            "ee_pos": ee_pos,
            "ee_vel": ee_vel,
            "ball_pos": np.concatenate(ball_positions),
            "ball_vel": np.concatenate(ball_velocities),
        }
        if self.config.cameras:
            camera_data = self.render_camera()
            obs["image"] = camera_data["rgb"]
        return obs

    def _get_reward(self) -> float:
        ee_pos = self.data.site_xpos[self._ee_site_id]
        paddle_z = ee_pos[2]
        reward = 0.0

        balls_airborne = 0
        for i, bid in enumerate(self._ball_body_ids):
            ball_pos = self.data.xpos[bid]
            ball_z = ball_pos[2]

            # Check if ball is above paddle level (airborne)
            if ball_z > paddle_z + 0.02:
                reward += 1.0
                balls_airborne += 1

            # Reward for keeping balls within catchable horizontal range
            horiz_dist = np.linalg.norm(ball_pos[:2] - ee_pos[:2])
            reward += max(0, 0.5 - horiz_dist)

            # Check floor contact
            if ball_z < 0.42:  # table height ~ 0.4
                if i not in self._balls_dropped:
                    reward -= 10.0
                    self._balls_dropped.add(i)

        # Bonus for all 3 airborne simultaneously
        if balls_airborne == 3:
            reward += 5.0

        return reward

    def _is_terminated(self) -> bool:
        # Terminate if all 3 balls dropped
        return len(self._balls_dropped) >= 3

    def _reset_task(self, rng: np.random.Generator) -> None:
        self._balls_dropped = set()

        # Position paddle arm upward (good initial catching pose)
        # Set initial joint angles for a ~level paddle position
        init_qpos = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.8])
        self.data.qpos[:7] = init_qpos
        self.data.ctrl[:7] = init_qpos

        # Place 3 balls slightly above the paddle in a triangle pattern
        paddle_pos = np.array([0.35, 0.0, 0.85])  # approximate EE pos
        offsets = [
            np.array([0.0, 0.0, 0.05]),
            np.array([0.03, 0.02, 0.10]),
            np.array([-0.03, 0.02, 0.15]),
        ]
        for i, bid in enumerate(self._ball_body_ids):
            joint_id = self.model.body_jntadr[bid]
            if joint_id >= 0:
                qadr = self.model.jnt_qposadr[joint_id]
                ball_start = paddle_pos + offsets[i]
                # Add small random perturbation
                ball_start[:2] += rng.uniform(-0.01, 0.01, size=2)
                self.data.qpos[qadr:qadr + 3] = ball_start
                self.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]  # identity quat
                # Small random initial velocity
                qvel_adr = self.model.jnt_dofadr[joint_id]
                self.data.qvel[qvel_adr:qvel_adr + 3] = rng.uniform(-0.1, 0.1, size=3)
                self.data.qvel[qvel_adr + 3:qvel_adr + 6] = 0

    def _apply_action(self, action: np.ndarray) -> None:
        """Map 4D action (delta_pos[3] + tilt[1]) to actuators via Jacobian."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta_pos = action[:3] * 0.03  # Smaller scale for fast control
        tilt = action[3] * 0.1  # Wrist tilt

        # Jacobian-transpose IK for position
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._ee_site_id)
        j_arm = jacp[:, :7]
        joint_delta = j_arm.T @ delta_pos

        # Apply to arm joints
        self.data.ctrl[:7] = self.data.qpos[:7] + joint_delta
        # Wrist tilt affects last joint
        self.data.ctrl[6] += tilt
        # Gripper closed (paddle)
        self.data.ctrl[7] = 0.0
        self.data.ctrl[8] = 0.0

    def _get_info(self) -> dict:
        balls_airborne = 0
        for bid in self._ball_body_ids:
            if self.data.xpos[bid][2] > 0.5:
                balls_airborne += 1
        return {
            "step": self._step_count,
            "balls_airborne": balls_airborne,
            "balls_dropped": len(self._balls_dropped),
            "success": balls_airborne == 3 and len(self._balls_dropped) == 0,
        }


def _generate_juggle_scene(xml_path: str) -> None:
    """Generate a Franka Panda juggling scene with paddle and 3 balls."""
    xml = """<mujoco model="panda_juggle">
  <compiler angle="radian" meshdir="." autolimits="true"/>

  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast">
    <flag contact="enable"/>
  </option>

  <default>
    <default class="panda">
      <joint damping="1.0" armature="0.1"/>
      <geom condim="4" friction="1 0.5 0.01" margin="0.001"/>
      <position kp="200" kv="30"/>
    </default>
    <default class="ball">
      <geom type="sphere" size="0.025" mass="0.05" condim="4"
            friction="0.8 0.3 0.01" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    </default>
  </default>

  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.8 0.8 0.8" rgb2="0.6 0.6 0.6"
             width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4" reflectance="0.1"/>
    <material name="robot_mat" rgba="0.7 0.7 0.7 1"/>
    <material name="paddle_mat" rgba="0.3 0.3 0.3 1"/>
    <material name="ball_red" rgba="1.0 0.2 0.2 1"/>
    <material name="ball_green" rgba="0.2 0.8 0.2 1"/>
    <material name="ball_blue" rgba="0.2 0.3 1.0 1"/>
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

                    <!-- Paddle (flat disc for juggling) -->
                    <body name="paddle" pos="0 0 0.04">
                      <geom name="paddle_geom" type="cylinder" size="0.08 0.005"
                            material="paddle_mat" mass="0.2"
                            friction="1.5 0.5 0.01" condim="4"/>
                      <!-- Slight rim to help catch -->
                      <geom name="rim1" type="capsule" fromto="-0.07 0 0.005 0.07 0 0.005"
                            size="0.005" material="paddle_mat" mass="0.01"/>
                      <geom name="rim2" type="capsule" fromto="0 -0.07 0.005 0 0.07 0.005"
                            size="0.005" material="paddle_mat" mass="0.01"/>
                      <site name="paddle_site" pos="0 0 0.01" size="0.01" rgba="1 1 0 1"/>
                    </body>

                    <!-- Gripper actuators kept for compatibility -->
                    <body name="left_finger" pos="0 0.02 0.04">
                      <joint name="finger_left" type="slide" axis="0 1 0" range="0 0.04"/>
                      <geom type="box" size="0.001 0.001 0.001" rgba="0.5 0.5 0.5 0" mass="0.01" contype="0" conaffinity="0"/>
                    </body>
                    <body name="right_finger" pos="0 -0.02 0.04">
                      <joint name="finger_right" type="slide" axis="0 -1 0" range="0 0.04"/>
                      <geom type="box" size="0.001 0.001 0.001" rgba="0.5 0.5 0.5 0" mass="0.01" contype="0" conaffinity="0"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Three juggling balls (free joints = 6 DOF each) -->
    <body name="ball_1" pos="0.35 0.0 0.90">
      <joint name="ball_1_free" type="free"/>
      <geom name="ball_1_geom" class="ball" material="ball_red"/>
    </body>

    <body name="ball_2" pos="0.38 0.02 0.95">
      <joint name="ball_2_free" type="free"/>
      <geom name="ball_2_geom" class="ball" material="ball_green"/>
    </body>

    <body name="ball_3" pos="0.32 0.02 1.00">
      <joint name="ball_3_free" type="free"/>
      <geom name="ball_3_geom" class="ball" material="ball_blue"/>
    </body>

    <!-- Lights -->
    <light name="top_light" pos="0.5 0 2.0" dir="0 0 -1" diffuse="0.9 0.9 0.9" castshadow="true"/>
    <light name="side_light" pos="0 1.5 1.5" dir="0 -0.5 -0.5" diffuse="0.4 0.4 0.4"/>

    <!-- Cameras -->
    <camera name="wrist_cam" pos="0.8 0 1.0" xyaxes="0 1 0 -0.5 0 0.5" fovy="70"/>
    <camera name="front_cam" pos="1.5 0 0.9" xyaxes="0 1 0 -0.3 0 0.8" fovy="60"/>
    <camera name="side_cam" pos="0.4 1.2 0.9" xyaxes="-1 0 0 0 -0.3 0.8" fovy="60"/>
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
