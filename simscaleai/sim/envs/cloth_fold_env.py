"""Cloth folding environment — robot must fold a deformable cloth in half.

Uses MuJoCo 3.x flexcomp for physically-accurate cloth simulation.
Genuinely frontier research: deformable object manipulation with
learned policies on a real physics engine.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces

from simscaleai.sim.base_env import BaseRobotEnv, SimConfig


_ASSET_DIR = Path(__file__).parent.parent / "assets"

# Cloth grid dimensions
_CLOTH_ROWS = 8
_CLOTH_COLS = 8
_N_VERTS = _CLOTH_ROWS * _CLOTH_COLS  # 64


class ClothFoldEnv(BaseRobotEnv):
    """Franka Panda arm folding a deformable cloth in half.

    The cloth is an 8×8 flexcomp grid (~17.5cm × 17.5cm) on a table.
    The task: pick up one edge and fold it onto the opposite edge.

    Observation:
        - joint_pos (7,): robot joint angles
        - joint_vel (7,): robot joint velocities
        - ee_pos (3,): end-effector position
        - gripper_state (1,): gripper opening
        - cloth_vertices (N_VERTS * 3 = 192,): all cloth vertex positions (flattened)
        - cloth_center (3,): cloth centroid
        - grasp_edge (3,): current grasp-edge midpoint (edge to pick up)
        - target_edge (3,): target fold-edge midpoint (where to place)
        - image (H, W, 3): camera image (optional)

    Action:
        - delta_pos (3,): end-effector position delta
        - gripper (1,): gripper command (-1=close, 1=open)

    Reward (multi-stage):
        - Stage 1: Reach — move EE to grasp edge
        - Stage 2: Grasp — close gripper on cloth edge
        - Stage 3: Lift — lift edge above table
        - Stage 4: Fold — bring edge to target edge position
        - Stage 5: Release — open gripper, cloth should stay folded
    """

    def __init__(self, config: SimConfig | None = None, render_mode: str | None = None):
        if config is None:
            config = SimConfig()

        if not config.xml_path:
            config.xml_path = str(_ASSET_DIR / "panda_cloth_fold.xml")

        super().__init__(config, render_mode)

        # Cache IDs
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )

        # Flex vertex info
        self._flex_id = 0
        self._vert_start = int(self.model.flex_vertadr[self._flex_id])
        self._vert_count = int(self.model.flex_vertnum[self._flex_id])
        assert self._vert_count == _N_VERTS, (
            f"Expected {_N_VERTS} cloth vertices, got {self._vert_count}"
        )

        # Robot has 9 qpos (7 joints + 2 finger slides)
        # Cloth vertices start at qpos[9] with 3 DOFs each
        self._cloth_qpos_start = 9
        self._cloth_qpos_end = 9 + _N_VERTS * 3

        # Fold configuration:
        # We fold along the X-axis (far edge onto near edge, toward robot base)
        # This aligns with the arm's natural extension/retraction direction.
        # Grasp edge: last row (vertices 56–63) — far from base (high X)
        # Target edge: first row (vertices 0–7) — near base (low X)
        self._grasp_vert_ids = list(range((_CLOTH_ROWS - 1) * _CLOTH_COLS, _N_VERTS))  # [56..63]
        self._target_vert_ids = list(range(0, _CLOTH_COLS))  # [0..7]

        # Cache body rest positions for each grasp vertex — needed for qpos<->world mapping
        # For flex vertices: world_pos = body_rest_pos + qpos
        self._grasp_body_rest = []
        vert_start = self._vert_start
        for vid in self._grasp_vert_ids:
            body_id = int(self.model.flex_vertbodyid[vert_start + vid])
            self._grasp_body_rest.append(self.model.body_pos[body_id].copy())

        # State tracking
        self._cloth_grasped = False
        self._cloth_lifted = False
        self._cloth_folded = False
        self._grasp_locked = False
        self._grasp_offsets: list[np.ndarray] = []
        self._gripper_cmd = 1.0

        # Settle steps on reset
        self._settle_steps = 200

    def _build_observation_space(self) -> spaces.Space:
        obs_dict = {
            "joint_pos": spaces.Box(-np.pi, np.pi, shape=(7,), dtype=np.float64),
            "joint_vel": spaces.Box(-10.0, 10.0, shape=(7,), dtype=np.float64),
            "ee_pos": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float64),
            "gripper_state": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
            "cloth_vertices": spaces.Box(
                -2.0, 2.0, shape=(_N_VERTS * 3,), dtype=np.float64
            ),
            "cloth_center": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float64),
            "grasp_edge": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float64),
            "target_edge": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float64),
            "fold_target": spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float64),
        }
        if self.config.cameras:
            cam = self.config.cameras[0]
            obs_dict["image"] = spaces.Box(
                0, 255, shape=(cam.height, cam.width, 3), dtype=np.uint8
            )
        return spaces.Dict(obs_dict)

    def _build_action_space(self) -> spaces.Space:
        return spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float64)

    # --- Cloth vertex helpers ---

    def _get_cloth_vertices(self) -> np.ndarray:
        """Get all cloth vertex world positions as (N_VERTS, 3)."""
        return self.data.flexvert_xpos[
            self._vert_start : self._vert_start + self._vert_count
        ].copy()

    def _get_edge_midpoint(self, vert_ids: list[int]) -> np.ndarray:
        """Get midpoint of a set of vertex IDs."""
        verts = self._get_cloth_vertices()
        edge_verts = verts[vert_ids]
        return edge_verts.mean(axis=0)

    def _get_cloth_center(self) -> np.ndarray:
        """Get cloth centroid."""
        return self._get_cloth_vertices().mean(axis=0)

    # --- Core env methods ---

    def _get_obs(self) -> dict[str, np.ndarray]:
        verts = self._get_cloth_vertices()
        obs = {
            "joint_pos": self.data.qpos[:7].copy(),
            "joint_vel": self.data.qvel[:7].copy(),
            "ee_pos": self.data.site_xpos[self._ee_site_id].copy(),
            "gripper_state": np.array([self.data.qpos[7]]),
            "cloth_vertices": verts.flatten(),
            "cloth_center": verts.mean(axis=0),
            "grasp_edge": self._get_edge_midpoint(self._grasp_vert_ids),
            "target_edge": self._get_edge_midpoint(self._target_vert_ids),
            "fold_target": self._initial_target_mid.copy(),
        }
        if self.config.cameras:
            camera_data = self.render_camera()
            obs["image"] = camera_data["rgb"]
        return obs

    def _apply_action(self, action: np.ndarray) -> None:
        """Map 4D action to actuators via damped pseudoinverse IK."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta_pos = action[:3] * 0.1
        self._gripper_cmd = action[3]

        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._ee_site_id)

        j_arm = jacp[:, :7]
        damping = 0.05
        jjt = j_arm @ j_arm.T + damping**2 * np.eye(3)
        joint_delta = j_arm.T @ np.linalg.solve(jjt, delta_pos)

        self.data.ctrl[:7] = self.data.qpos[:7] + joint_delta
        gripper_pos = (self._gripper_cmd + 1.0) / 2.0 * 0.04
        self.data.ctrl[7] = gripper_pos
        self.data.ctrl[8] = gripper_pos

    def _zero_cloth_vel(self, vert_ids: list[int]) -> None:
        """Zero out qvel for specific cloth vertex ids to prevent flyoff."""
        # Robot has nv_robot qvel entries, then cloth DOFs
        nv_robot = self.model.nv - _N_VERTS * 3
        for vid in vert_ids:
            vidx = nv_robot + vid * 3
            self.data.qvel[vidx : vidx + 3] = 0.0

    def step(self, action):
        """Step with cloth grasp lock: kinematically attach edge vertices to EE."""
        self._apply_action(action)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

            # Grasp lock: move grasped edge vertices with EE
            if self._grasp_locked:
                ee_pos = self.data.site_xpos[self._ee_site_id]
                for i, vid in enumerate(self._grasp_vert_ids):
                    qidx = self._cloth_qpos_start + vid * 3
                    # desired world pos = ee_pos + offset_from_ee
                    desired_world = ee_pos + self._grasp_offsets[i]
                    # qpos = world_pos - body_rest_pos
                    self.data.qpos[qidx : qidx + 3] = desired_world - self._grasp_body_rest[i]
                # Zero velocities so they don't accumulate
                self._zero_cloth_vel(self._grasp_vert_ids)

        # Run forward pass to update flexvert_xpos after any kinematic changes
        mujoco.mj_forward(self.model, self.data)

        # Check grasp transitions
        ee_pos = self.data.site_xpos[self._ee_site_id].copy()
        grasp_edge_pos = self._get_edge_midpoint(self._grasp_vert_ids)
        dist_to_edge = float(np.linalg.norm(ee_pos - grasp_edge_pos))

        if not self._grasp_locked:
            # Lock if gripper closed and near edge
            if self._gripper_cmd < -0.5 and dist_to_edge < 0.06:
                self._grasp_locked = True
                self._cloth_grasped = True
                # Store offset from EE to each grasp vertex
                verts = self._get_cloth_vertices()
                self._grasp_offsets = [
                    verts[vid] - ee_pos for vid in self._grasp_vert_ids
                ]
        else:
            # Release if gripper opens
            if self._gripper_cmd > 0.5:
                # Zero velocities before releasing to prevent flyoff
                self._zero_cloth_vel(self._grasp_vert_ids)
                self._grasp_locked = False

        self._step_count += 1
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._step_count >= self.config.max_episode_steps
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_reward(self) -> float:
        """Multi-stage reward for cloth folding."""
        ee_pos = self.data.site_xpos[self._ee_site_id]
        grasp_mid = self._get_edge_midpoint(self._grasp_vert_ids)
        target_mid = self._get_edge_midpoint(self._target_vert_ids)
        verts = self._get_cloth_vertices()

        reward = 0.0

        # Stage 1: Reach — move to grasp edge
        reach_dist = float(np.linalg.norm(ee_pos - grasp_mid))
        reward += -reach_dist

        # Stage 2: Grasp bonus
        if self._grasp_locked:
            reward += 1.0

        # Stage 3: Lift bonus — grasp edge should be above table
        if self._cloth_grasped:
            grasp_z = grasp_mid[2]
            if grasp_z > 0.406:  # just above resting height on table
                self._cloth_lifted = True
                reward += 1.0

        # Stage 4: Fold — bring grasp edge to overlap INITIAL target edge
        if self._cloth_lifted:
            fold_target = self._initial_target_mid.copy()
            fold_target[2] += 0.01  # slightly above table
            fold_dist = float(np.linalg.norm(grasp_mid - fold_target))
            reward += -fold_dist
            if fold_dist < 0.05:
                reward += 3.0
                self._cloth_folded = True

        # Stage 5: Release bonus — if folded and gripper open
        if self._cloth_folded and not self._grasp_locked:
            reward += 5.0

        return reward

    def _is_terminated(self) -> bool:
        """Episode succeeds when cloth is folded and released."""
        return self._cloth_folded and not self._grasp_locked

    def _compute_fold_quality(self) -> float:
        """Compute fold quality: overlap between grasp edge and target edge.

        Returns a value in [0, 1] where 1 = perfect fold.
        """
        verts = self._get_cloth_vertices()
        grasp_verts = verts[self._grasp_vert_ids]  # (8, 3)
        target_verts = verts[self._target_vert_ids]  # (8, 3)

        # For a perfect fold, each grasp vertex should be directly above
        # the corresponding target vertex (matching X, same Y, slightly higher Z)
        dists = np.linalg.norm(
            grasp_verts[:, :2] - target_verts[:, :2], axis=1
        )
        mean_dist = float(dists.mean())
        # Normalize: 0 dist → quality 1.0; dist 0.1 → quality ~0
        quality = max(0.0, 1.0 - mean_dist / 0.1)
        return quality

    def _reset_task(self, rng: np.random.Generator) -> None:
        """Reset cloth and task state."""
        self._cloth_grasped = False
        self._cloth_lifted = False
        self._cloth_folded = False
        self._grasp_locked = False
        self._grasp_offsets = []
        self._gripper_cmd = 1.0

        # Let cloth settle on table
        for _ in range(self._settle_steps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Cache the INITIAL target edge midpoint (before any manipulation)
        # This is the fold destination and must not move with the cloth.
        self._initial_target_mid = self._get_edge_midpoint(self._target_vert_ids).copy()

    def _get_info(self) -> dict:
        ee_pos = self.data.site_xpos[self._ee_site_id]
        grasp_mid = self._get_edge_midpoint(self._grasp_vert_ids)
        fold_quality = self._compute_fold_quality()
        fold_dist = float(np.linalg.norm(grasp_mid - self._initial_target_mid))

        return {
            "step": self._step_count,
            "reach_dist": float(np.linalg.norm(ee_pos - grasp_mid)),
            "fold_dist": fold_dist,
            "grasped": self._cloth_grasped,
            "lifted": self._cloth_lifted,
            "folded": self._cloth_folded,
            "fold_quality": fold_quality,
            "success": self._is_terminated(),
        }
