"""Pick-and-place environment — robot must grasp an object and place it at a goal.

The core manipulation task for Amazon's warehouse robotics.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces

from simscaleai.sim.base_env import BaseRobotEnv, SimConfig


_ASSET_DIR = Path(__file__).parent.parent / "assets"


class PickPlaceEnv(BaseRobotEnv):
    """Franka Panda arm picking up an object and placing it at a target.

    Observation:
        - joint_pos (7,): robot joint angles
        - joint_vel (7,): robot joint velocities
        - ee_pos (3,): end-effector position
        - object_pos (3,): object position
        - object_quat (4,): object orientation (quaternion)
        - target_pos (3,): goal placement position
        - gripper_state (1,): gripper open/close
        - image (H, W, 3): RGB camera image (if cameras configured)

    Action:
        - delta_pos (3,): end-effector position delta (x, y, z)
        - gripper (1,): gripper command (-1=close, 1=open)

    Reward (shaped, multi-stage):
        - Stage 1: Reach — negative distance ee → object
        - Stage 2: Grasp — bonus for grasping object
        - Stage 3: Lift — bonus for lifting object above table
        - Stage 4: Place — negative distance object → target + bonus for placement
    """

    def __init__(self, config: SimConfig | None = None, render_mode: str | None = None):
        if config is None:
            config = SimConfig()

        if not config.xml_path:
            config.xml_path = str(_ASSET_DIR / "panda_pick_place.xml")

        if not os.path.exists(config.xml_path):
            os.makedirs(os.path.dirname(config.xml_path), exist_ok=True)
            _generate_pick_place_scene(config.xml_path)

        super().__init__(config, render_mode)

        self._target_pos = np.zeros(3)

        # Cache body/site IDs
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "place_target"
        )

        # Object free-joint address
        self._obj_qpos_addr = self.model.jnt_qposadr[
            self.model.body_jntadr[self._object_body_id]
        ]
        self._obj_qvel_addr = self.model.jnt_dofadr[
            self.model.body_jntadr[self._object_body_id]
        ]

        # Grasp state
        self._object_grasped = False
        self._object_lifted = False
        self._grasp_locked = False  # True when object is kinematically attached
        self._grasp_offset = np.zeros(3)  # EE → object offset when grasped
        self._gripper_cmd = 1.0  # Last gripper command

    def _build_observation_space(self) -> spaces.Space:
        obs_dict = {
            "joint_pos": spaces.Box(-np.pi, np.pi, shape=(7,), dtype=np.float64),
            "joint_vel": spaces.Box(-10.0, 10.0, shape=(7,), dtype=np.float64),
            "ee_pos": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float64),
            "object_pos": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float64),
            "object_quat": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float64),
            "target_pos": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float64),
            "gripper_state": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
        }
        if self.config.cameras:
            cam = self.config.cameras[0]
            obs_dict["image"] = spaces.Box(
                0, 255, shape=(cam.height, cam.width, 3), dtype=np.uint8
            )
        return spaces.Dict(obs_dict)

    def _build_action_space(self) -> spaces.Space:
        return spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float64)

    def _get_obs(self) -> dict[str, np.ndarray]:
        a = self._obj_qpos_addr
        obs = {
            "joint_pos": self.data.qpos[:7].copy(),
            "joint_vel": self.data.qvel[:7].copy(),
            "ee_pos": self.data.site_xpos[self._ee_site_id].copy(),
            "object_pos": self.data.xpos[self._object_body_id].copy(),
            "object_quat": self.data.qpos[a + 3: a + 7].copy(),
            "target_pos": self._target_pos.copy(),
            "gripper_state": np.array([self.data.qpos[7]]),
        }
        if self.config.cameras:
            camera_data = self.render_camera()
            obs["image"] = camera_data["rgb"]
        return obs

    def _apply_action(self, action: np.ndarray) -> None:
        """Map 4D action (delta_pos[3] + gripper[1]) to 9 actuators via damped pseudoinverse IK."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta_pos = action[:3] * 0.1  # Scale delta to metres
        self._gripper_cmd = action[3]  # Save for grasp lock logic

        # Compute positional Jacobian at end-effector site
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._ee_site_id)

        # Damped least-squares pseudoinverse: J^T (J J^T + λ²I)^{-1} Δx
        j_arm = jacp[:, :7]
        damping = 0.05
        jjt = j_arm @ j_arm.T + damping**2 * np.eye(3)
        joint_delta = j_arm.T @ np.linalg.solve(jjt, delta_pos)

        # Position control
        self.data.ctrl[:7] = self.data.qpos[:7] + joint_delta
        # Gripper: -1 = close (0), +1 = open (0.04)
        gripper_pos = (self._gripper_cmd + 1.0) / 2.0 * 0.04
        self.data.ctrl[7] = gripper_pos
        self.data.ctrl[8] = gripper_pos

    def step(self, action):
        """Override step to apply grasp lock after each physics substep."""
        self._apply_action(action)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            # Re-attach object to EE every substep while grasped
            if self._grasp_locked:
                ee_pos = self.data.site_xpos[self._ee_site_id]
                a = self._obj_qpos_addr
                self.data.qpos[a: a + 3] = ee_pos + self._grasp_offset
                v = self._obj_qvel_addr
                self.data.qvel[v: v + 6] = 0

        # Check grasp lock transitions after physics
        ee_pos = self.data.site_xpos[self._ee_site_id]
        obj_pos = self.data.xpos[self._object_body_id]
        reach_dist = float(np.linalg.norm(ee_pos - obj_pos))

        if not self._grasp_locked:
            if self._gripper_cmd < -0.5 and reach_dist < 0.08:
                self._grasp_locked = True
                self._object_grasped = True
                self._grasp_offset = obj_pos - ee_pos
        else:
            if self._gripper_cmd > 0.5:
                self._grasp_locked = False

        self._step_count += 1
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._step_count >= self.config.max_episode_steps
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _is_grasping(self) -> bool:
        """Detect grasp via finger-object contact forces."""
        # Get geom IDs by name
        obj_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        left_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_geom")
        right_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_geom")

        left_contact = False
        right_contact = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if obj_geom in pair and left_geom in pair:
                left_contact = True
            if obj_geom in pair and right_geom in pair:
                right_contact = True
        return left_contact and right_contact

    def _get_reward(self) -> float:
        ee_pos = self.data.site_xpos[self._ee_site_id]
        obj_pos = self.data.xpos[self._object_body_id]
        target_pos = self._target_pos

        reward = 0.0

        # Stage 1: Reach toward object
        reach_dist = float(np.linalg.norm(ee_pos - obj_pos))
        reward += -reach_dist

        # Stage 2: Grasp bonus
        if self._grasp_locked:
            self._object_grasped = True
            reward += 1.0
        elif reach_dist < 0.08:
            reward += 0.25  # proximity bonus

        # Stage 3: Lift bonus
        if self._object_grasped and obj_pos[2] > 0.48:
            self._object_lifted = True
            reward += 1.0

        # Stage 4: Place — distance object → target
        if self._object_lifted:
            place_dist = float(np.linalg.norm(obj_pos - target_pos))
            reward += -place_dist
            if place_dist < 0.05:
                reward += 5.0  # Big bonus for successful placement

        return reward

    def _is_terminated(self) -> bool:
        obj_pos = self.data.xpos[self._object_body_id]
        place_dist = float(np.linalg.norm(obj_pos - self._target_pos))
        return bool(self._object_lifted and place_dist < 0.03)

    def _reset_task(self, rng: np.random.Generator) -> None:
        self._object_grasped = False
        self._object_lifted = False
        self._grasp_locked = False
        self._grasp_offset = np.zeros(3)

        # Randomize object position on table
        obj_pos = np.array([
            rng.uniform(0.3, 0.6),
            rng.uniform(-0.2, 0.2),
            0.45,  # On table surface
        ])

        # Set object position via free joint
        a = self._obj_qpos_addr
        self.data.qpos[a: a + 3] = obj_pos
        self.data.qpos[a + 3: a + 7] = [1, 0, 0, 0]  # Identity quat

        # Randomize target position (different from object)
        self._target_pos = np.array([
            rng.uniform(0.3, 0.6),
            rng.uniform(-0.2, 0.2),
            0.45,
        ])
        # Ensure target is sufficiently far from object
        while np.linalg.norm(self._target_pos[:2] - obj_pos[:2]) < 0.1:
            self._target_pos[:2] = rng.uniform([0.3, -0.2], [0.6, 0.2])

        self.model.body_pos[self._target_body_id] = self._target_pos

    def _get_info(self) -> dict:
        ee_pos = self.data.site_xpos[self._ee_site_id]
        obj_pos = self.data.xpos[self._object_body_id]
        return {
            "step": self._step_count,
            "reach_dist": float(np.linalg.norm(ee_pos - obj_pos)),
            "place_dist": float(np.linalg.norm(obj_pos - self._target_pos)),
            "grasped": self._object_grasped,
            "lifted": self._object_lifted,
            "is_grasping": self._is_grasping(),
            "success": self._is_terminated(),
        }


def _generate_pick_place_scene(xml_path: str) -> None:
    """Generate a pick-and-place scene with an object and target."""
    xml = """<mujoco model="panda_pick_place">
  <compiler angle="radian" meshdir="." autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <default>
    <default class="panda">
      <joint damping="1.0" armature="0.1"/>
      <geom condim="4" friction="1 0.5 0.01" margin="0.001"/>
      <position kp="200" kv="30"/>
    </default>
  </default>

  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.8 0.8 0.8" rgb2="0.6 0.6 0.6"
             width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4" reflectance="0.1"/>
    <material name="robot_mat" rgba="0.7 0.7 0.7 1"/>
    <material name="object_mat" rgba="0.8 0.2 0.2 1"/>
    <material name="target_mat" rgba="0.2 0.8 0.2 0.3"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.01" material="grid_mat"/>

    <body name="table" pos="0.5 0 0.2">
      <geom type="box" size="0.4 0.5 0.2" rgba="0.6 0.4 0.2 1" mass="100"/>
    </body>

    <!-- Panda Arm -->
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
                    <site name="ee_site" pos="0 0 0.04" size="0.01" rgba="1 0 0 1"/>
                    <!-- Wider fingers with named geoms for contact detection -->
                    <body name="left_finger" pos="0 0.04 0.04">
                      <joint name="finger_left" type="slide" axis="0 -1 0" range="0 0.04"/>
                      <geom name="left_finger_geom" type="box" size="0.015 0.005 0.04"
                            rgba="0.5 0.5 0.5 1" mass="0.1" friction="2 0.5 0.01" condim="4"/>
                    </body>
                    <body name="right_finger" pos="0 -0.04 0.04">
                      <joint name="finger_right" type="slide" axis="0 1 0" range="0 0.04"/>
                      <geom name="right_finger_geom" type="box" size="0.015 0.005 0.04"
                            rgba="0.5 0.5 0.5 1" mass="0.1" friction="2 0.5 0.01" condim="4"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Manipulable object (small cube, 3cm) -->
    <body name="object" pos="0.5 0 0.45">
      <freejoint name="object_joint"/>
      <geom name="object_geom" type="box" size="0.015 0.015 0.015" material="object_mat"
            mass="0.05" friction="1.5 0.5 0.01" condim="4"/>
    </body>

    <!-- Placement target (visual only) -->
    <body name="place_target" pos="0.5 0.2 0.45">
      <geom type="box" size="0.03 0.03 0.005" material="target_mat" contype="0" conaffinity="0"/>
    </body>

    <light name="top_light" pos="0.5 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light name="side_light" pos="0 1 1" dir="0 -0.5 -0.5" diffuse="0.5 0.5 0.5"/>

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
    <position name="act_finger_left" joint="finger_left" kp="100"/>
    <position name="act_finger_right" joint="finger_right" kp="100"/>
  </actuator>
</mujoco>
"""
    with open(xml_path, "w") as f:
        f.write(xml)
