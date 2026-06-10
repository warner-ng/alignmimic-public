from __future__ import annotations

import math
import numpy as np
import os
import torch
import trimesh
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu", quat_order: str = "wxyz"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        assert quat_order in ("wxyz", "xyzw"), f"Unsupported motion quat order: {quat_order}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        if quat_order == "xyzw":
            body_quat_w = body_quat_w[..., [3, 0, 1, 2]]
        self._body_quat_w = body_quat_w
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class ObjectMotionLoader:
    def __init__(self, object_motion_file: str, device: str = "cpu"):
        assert os.path.isfile(object_motion_file), f"Invalid file path: {object_motion_file}"
        data = np.load(object_motion_file)
        self.root_pos = torch.tensor(data["trans"], dtype=torch.float32, device=device)
        root_quat_xyzw = torch.tensor(data["rot"], dtype=torch.float32, device=device)
        self.root_quat = root_quat_xyzw[:, [3, 0, 1, 2]]
        self.time_step_total = self.root_pos.shape[0]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        self.object: RigidObject | None = None
        self.object_motion: ObjectMotionLoader | None = None
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(
            self.cfg.motion_file, self.body_indexes, device=self.device, quat_order=self.cfg.motion_quat_order
        )
        if self.cfg.object_asset_name is not None and self.cfg.object_motion_file is not None:
            self.object = env.scene[self.cfg.object_asset_name]
            self.object_motion = ObjectMotionLoader(self.cfg.object_motion_file, device=self.device)
            assert self.object_motion.time_step_total == self.motion.time_step_total, (
                "Object motion frames must match human motion frames: "
                f"{self.object_motion.time_step_total} != {self.motion.time_step_total}"
            )
            self.object_points = self._load_object_points()
            if self.cfg.debug_vis and hasattr(self, "current_anchor_visualizer"):
                self._create_object_point_visualizers()
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        if self.has_object_motion:
            self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_object_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def has_object_motion(self) -> bool:
        return getattr(self, "object", None) is not None and getattr(self, "object_motion", None) is not None

    def _offset_quat(self, offset_deg: Sequence[float], count: int, dtype: torch.dtype) -> torch.Tensor:
        offset = torch.tensor(offset_deg, device=self.device, dtype=dtype) * (torch.pi / 180.0)
        axes = torch.eye(3, device=self.device, dtype=dtype).unsqueeze(0).expand(count, -1, -1)
        qx = self._axis_angle_quat(offset[0], axes[:, 0, :], dtype)
        qy = self._axis_angle_quat(offset[1], axes[:, 1, :], dtype)
        qz = self._axis_angle_quat(offset[2], axes[:, 2, :], dtype)
        return quat_mul(qx, quat_mul(qy, qz))

    def _has_offset(self, offset: Sequence[float]) -> bool:
        return any(abs(value) > 1e-8 for value in offset)

    def _human_root_rot_offset_quat(self, root_quat: torch.Tensor) -> torch.Tensor:
        offset = self.cfg.human_root_rot_offset_deg
        if not self._has_offset(offset):
            identity = torch.zeros((root_quat.shape[0], 4), device=self.device, dtype=root_quat.dtype)
            identity[:, 0] = 1.0
            return identity
        return self._offset_quat(offset, root_quat.shape[0], root_quat.dtype)

    def _object_root_rot_offset_quat(self, root_quat: torch.Tensor) -> torch.Tensor:
        offset = self.cfg.object_root_rot_offset_deg
        if not self._has_offset(offset):
            identity = torch.zeros((root_quat.shape[0], 4), device=self.device, dtype=root_quat.dtype)
            identity[:, 0] = 1.0
            return identity
        offset_rad = torch.tensor(offset, device=self.device, dtype=root_quat.dtype) * (torch.pi / 180.0)
        local_axes = torch.eye(3, device=self.device, dtype=root_quat.dtype).unsqueeze(0).expand(
            root_quat.shape[0], -1, -1
        )
        root_quat_axes = root_quat.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 4)
        world_axes = quat_apply(root_quat_axes, local_axes.reshape(-1, 3)).reshape(root_quat.shape[0], 3, 3)
        qy_axis = self._axis_angle_quat(offset_rad[1], world_axes[:, 1, :], root_quat.dtype)
        qx_axis = self._axis_angle_quat(offset_rad[0], world_axes[:, 0, :], root_quat.dtype)
        qz_axis = self._axis_angle_quat(offset_rad[2], world_axes[:, 2, :], root_quat.dtype)
        return quat_mul(qz_axis, quat_mul(qy_axis, qx_axis))

    def _axis_angle_quat(self, angle: torch.Tensor, axis: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        angle_batch = angle.repeat(axis.shape[0]).to(dtype=dtype)
        half_angle = 0.5 * angle_batch
        quat = torch.zeros((axis.shape[0], 4), device=self.device, dtype=dtype)
        quat[:, 0] = torch.cos(half_angle)
        quat[:, 1:] = axis * torch.sin(half_angle).unsqueeze(1)
        return quat

    def _pair_root_rot_offset_quat(self, count: int, dtype: torch.dtype) -> torch.Tensor:
        offset = self.cfg.motion_global_rot_offset_deg
        if not self._has_offset(offset):
            identity = torch.zeros((count, 4), device=self.device, dtype=dtype)
            identity[:, 0] = 1.0
            return identity
        return self._offset_quat(offset, count, dtype)

    def _load_object_points(self) -> torch.Tensor:
        assert self.cfg.object_mesh_file is not None
        mesh_obj = trimesh.load(self.cfg.object_mesh_file, force="mesh")
        center = np.mean(mesh_obj.vertices, axis=0)
        object_points, _ = trimesh.sample.sample_surface_even(
            mesh_obj, count=self.cfg.object_point_count, seed=2024
        )
        object_points = torch.tensor(
            (object_points - center) * self.cfg.object_scale, dtype=torch.float32, device=self.device
        )
        if object_points.shape[0] < self.cfg.object_point_count:
            repeat_count = self.cfg.object_point_count - object_points.shape[0]
            object_points = torch.cat([object_points, object_points[:repeat_count]], dim=0)
        return object_points

    def _motion_body_pos_quat_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        body_pos = self.motion.body_pos_w[self.time_steps].clone()
        body_quat = self.motion.body_quat_w[self.time_steps].clone()
        root_pos = body_pos[:, self.motion_anchor_body_index]
        root_quat = body_quat[:, self.motion_anchor_body_index]
        human_rot = self._human_root_rot_offset_quat(root_quat)
        body_pos = root_pos[:, None, :] + quat_apply(
            human_rot[:, None, :].expand(-1, body_pos.shape[1], -1).reshape(-1, 4),
            (body_pos - root_pos[:, None, :]).reshape(-1, 3),
        ).reshape_as(body_pos)
        human_rot_body = human_rot[:, None, :].expand(-1, body_quat.shape[1], -1)
        body_quat = quat_mul(human_rot_body.reshape(-1, 4), body_quat.reshape(-1, 4)).reshape_as(body_quat)
        pair_rot = self._pair_root_rot_offset_quat(body_pos.shape[0], body_pos.dtype)
        body_pos = quat_apply(
            pair_rot[:, None, :].expand(-1, body_pos.shape[1], -1).reshape(-1, 4), body_pos.reshape(-1, 3)
        ).reshape_as(body_pos)
        pair_rot_body = pair_rot[:, None, :].expand(-1, body_quat.shape[1], -1)
        body_quat = quat_mul(pair_rot_body.reshape(-1, 4), body_quat.reshape(-1, 4)).reshape_as(body_quat)
        pair_trans = torch.tensor(self.cfg.motion_global_pos_offset, device=self.device, dtype=body_pos.dtype)
        body_pos = body_pos + pair_trans.view(1, 1, 3)
        return body_pos + self._env.scene.env_origins[:, None, :], body_quat

    def _motion_body_vel_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        body_lin_vel = self.motion.body_lin_vel_w[self.time_steps].clone()
        body_ang_vel = self.motion.body_ang_vel_w[self.time_steps].clone()
        root_quat = self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]
        human_rot = self._human_root_rot_offset_quat(root_quat)
        pair_rot = self._pair_root_rot_offset_quat(body_lin_vel.shape[0], body_lin_vel.dtype)
        total_rot = quat_mul(pair_rot, human_rot)
        rot = total_rot[:, None, :].expand(-1, body_lin_vel.shape[1], -1).reshape(-1, 4)
        body_lin_vel = quat_apply(rot, body_lin_vel.reshape(-1, 3)).reshape_as(body_lin_vel)
        body_ang_vel = quat_apply(rot, body_ang_vel.reshape(-1, 3)).reshape_as(body_ang_vel)
        return body_lin_vel, body_ang_vel

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        body_pos, _ = self._motion_body_pos_quat_w()
        return body_pos

    @property
    def body_quat_w(self) -> torch.Tensor:
        _, body_quat = self._motion_body_pos_quat_w()
        return body_quat

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        body_lin_vel, _ = self._motion_body_vel_w()
        return body_lin_vel

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        _, body_ang_vel = self._motion_body_vel_w()
        return body_ang_vel

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def object_root_pos_w(self) -> torch.Tensor:
        assert self.object_motion is not None
        object_pos = self.object_motion.root_pos[self.time_steps].clone()
        object_quat = self.object_motion.root_quat[self.time_steps].clone()
        object_rot = self._object_root_rot_offset_quat(object_quat)
        object_quat = quat_mul(object_rot, object_quat)
        local_offset = torch.tensor(self.cfg.object_root_pos_offset, device=self.device, dtype=object_pos.dtype)
        object_pos = object_pos + quat_apply(object_quat, local_offset.unsqueeze(0).expand(object_pos.shape[0], -1))
        pair_rot = self._pair_root_rot_offset_quat(object_pos.shape[0], object_pos.dtype)
        pair_trans = torch.tensor(self.cfg.motion_global_pos_offset, device=self.device, dtype=object_pos.dtype)
        object_pos = quat_apply(pair_rot, object_pos) + pair_trans.view(1, 3)
        return object_pos + self._env.scene.env_origins

    @property
    def object_root_quat_w(self) -> torch.Tensor:
        assert self.object_motion is not None
        object_quat = self.object_motion.root_quat[self.time_steps].clone()
        object_rot = self._object_root_rot_offset_quat(object_quat)
        object_quat = quat_mul(object_rot, object_quat)
        pair_rot = self._pair_root_rot_offset_quat(object_quat.shape[0], object_quat.dtype)
        return quat_mul(pair_rot, object_quat)

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_object_root_pos_w(self) -> torch.Tensor:
        assert self.object is not None
        return self.object.data.root_pos_w

    @property
    def robot_object_root_quat_w(self) -> torch.Tensor:
        assert self.object is not None
        return self.object.data.root_quat_w

    def object_points_world(self, root_pos: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "object_points")
        object_points = self.object_points.unsqueeze(0).expand(root_pos.shape[0], -1, -1)
        object_quat = root_quat.unsqueeze(1).expand(-1, object_points.shape[1], -1).reshape(-1, 4)
        object_points_world = quat_apply(object_quat, object_points.reshape(-1, 3)).reshape_as(object_points)
        return object_points_world + root_pos.unsqueeze(1)

    def object_points_world_for_vis(self, root_pos: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "object_points")
        max_points = max(1, self.cfg.object_point_visualizer_max_points)
        stride = max(1, self.object_points.shape[0] // max_points)
        object_points = self.object_points[::stride][:max_points]
        object_points = object_points.unsqueeze(0).expand(root_pos.shape[0], -1, -1)
        object_quat = root_quat.unsqueeze(1).expand(-1, object_points.shape[1], -1).reshape(-1, 4)
        object_points_world = quat_apply(object_quat, object_points.reshape(-1, 3)).reshape_as(object_points)
        return object_points_world + root_pos.unsqueeze(1)

    def object_point_cloud_dist(self) -> torch.Tensor:
        actual_points = self.object_points_world(self.robot_object_root_pos_w, self.robot_object_root_quat_w)
        ref_points = self.object_points_world(self.object_root_pos_w, self.object_root_quat_w)
        return torch.norm(actual_points - ref_points, dim=-1).mean(dim=-1)

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        if self.has_object_motion:
            self.metrics["error_object_pos"] = torch.norm(self.object_root_pos_w - self.robot_object_root_pos_w, dim=-1)
            self.metrics["error_object_rot"] = quat_error_magnitude(
                self.object_root_quat_w, self.robot_object_root_quat_w
            )

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        if self.has_object_motion:
            object_state = torch.zeros((len(env_ids), 13), dtype=torch.float32, device=self.device)
            object_state[:, :3] = self.object_root_pos_w[env_ids]
            object_state[:, 2] += self.cfg.object_root_z_bias
            object_state[:, 3:7] = self.object_root_quat_w[env_ids]
            self.object.write_root_state_to_sim(object_state, env_ids=env_ids)

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(self.cfg.current_anchor_visualizer_cfg)
                self.goal_anchor_visualizer = VisualizationMarkers(self.cfg.goal_anchor_visualizer_cfg)
                if self.has_object_motion:
                    self._create_object_point_visualizers()

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.current_body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/current/" + name
                            )
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.goal_body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            if self.has_object_motion:
                if not hasattr(self, "current_object_points_visualizer"):
                    self._create_object_point_visualizers()
                self.current_object_points_visualizer.set_visibility(True)
                self.goal_object_points_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                if self.has_object_motion:
                    self.current_object_points_visualizer.set_visibility(False)
                    self.goal_object_points_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _create_object_point_visualizers(self):
        self.current_object_points_visualizer = VisualizationMarkers(self.cfg.current_object_points_visualizer_cfg)
        self.goal_object_points_visualizer = VisualizationMarkers(self.cfg.goal_object_points_visualizer_cfg)
        self.current_object_points_visualizer.set_visibility(True)
        self.goal_object_points_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])

        if self.has_object_motion:
            vis_env_count = min(self.num_envs, max(1, self.cfg.object_point_visualizer_env_count))
            current_points = self.object_points_world_for_vis(
                self.robot_object_root_pos_w[:vis_env_count], self.robot_object_root_quat_w[:vis_env_count]
            )
            goal_points = self.object_points_world_for_vis(
                self.object_root_pos_w[:vis_env_count], self.object_root_quat_w[:vis_env_count]
            )
            self.current_object_points_visualizer.visualize(current_points.reshape(-1, 3))
            self.goal_object_points_visualizer.visualize(goal_points.reshape(-1, 3))


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    motion_quat_order: str = "wxyz"
    object_asset_name: str | None = None
    object_motion_file: str | None = None
    object_root_z_bias: float = 0.0
    object_root_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_root_rot_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_mesh_file: str | None = None
    object_scale: float = 1.0
    object_point_count: int = 1024
    object_point_visualizer_max_points: int = 128
    object_point_visualizer_env_count: int = 2
    human_root_rot_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    motion_global_rot_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    motion_global_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    current_anchor_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current/anchor",
        markers={
            "current": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )
    goal_anchor_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal/anchor",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )
    current_body_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current/body",
        markers={
            "current": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )
    goal_body_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal/body",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )
    current_object_points_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current/object_points",
        markers={
            "current": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            )
        },
    )
    goal_object_points_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal/object_points",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )
