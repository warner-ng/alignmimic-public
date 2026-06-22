#!/usr/bin/env python3
"""Export a G1 human pkl to a self-consistent BeyondMimic motion.npz."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Export G1 pkl to BeyondMimic motion.npz.")
parser.add_argument("--human_pkl", required=True, help="Aligned G1 human pkl.")
parser.add_argument("--output", required=True, help="Output BeyondMimic motion.npz.")
parser.add_argument(
    "--whole_body_tracking_root",
    default="/home/warner/_projects/whole_body_tracking",
    help="whole_body_tracking checkout root.",
)
parser.add_argument("--root_quat_order", choices=("xyzw", "wxyz"), default="xyzw", help="Input pkl root_rot order.")
parser.add_argument(
    "--output_quat_order",
    choices=("xyzw", "wxyz"),
    default="xyzw",
    help="Stored body_quat_w order. Keep xyzw for Tracking-Flat-G1-Bike-HOI-v0.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

whole_body_tracking_root = Path(args_cli.whole_body_tracking_root).resolve()
sys.path.insert(0, str(whole_body_tracking_root / "source"))
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


PKL_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class ExportSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return quat[..., [1, 2, 3, 0]]


def finite_difference(values: torch.Tensor, dt: float) -> torch.Tensor:
    out = torch.zeros_like(values)
    if values.shape[0] == 1:
        return out
    out[0] = (values[1] - values[0]) / dt
    out[-1] = (values[-1] - values[-2]) / dt
    if values.shape[0] > 2:
        out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return out


def angular_velocity_from_quat(quat_wxyz: torch.Tensor, dt: float) -> torch.Tensor:
    out = torch.zeros((quat_wxyz.shape[0], 3), dtype=quat_wxyz.dtype, device=quat_wxyz.device)
    if quat_wxyz.shape[0] == 1:
        return out
    q_rel_first = quat_mul(quat_wxyz[1:2], quat_conjugate(quat_wxyz[0:1]))
    q_rel_last = quat_mul(quat_wxyz[-1:], quat_conjugate(quat_wxyz[-2:-1]))
    out[0:1] = axis_angle_from_quat(q_rel_first) / dt
    out[-1:] = axis_angle_from_quat(q_rel_last) / dt
    if quat_wxyz.shape[0] > 2:
        q_rel = quat_mul(quat_wxyz[2:], quat_conjugate(quat_wxyz[:-2]))
        out[1:-1] = axis_angle_from_quat(q_rel) / (2.0 * dt)
    return out


def load_human_motion(path: str, device: str) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    for key in ("root_pos", "root_rot", "dof_pos"):
        if key not in data:
            raise KeyError(f"Missing key '{key}' in human motion: {path}")
    fps = int(data.get("fps", 30))
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)
    if args_cli.root_quat_order == "xyzw":
        root_rot = xyzw_to_wxyz(root_rot)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    return (
        fps,
        torch.tensor(root_pos, dtype=torch.float32, device=device),
        torch.tensor(root_rot, dtype=torch.float32, device=device),
        torch.tensor(dof_pos, dtype=torch.float32, device=device),
    )


def run_export(sim: SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    fps, root_pos, root_quat, dof_pos_pkl = load_human_motion(args_cli.human_pkl, sim.device)
    dt = 1.0 / float(fps)
    root_lin_vel = finite_difference(root_pos, dt)
    root_ang_vel = angular_velocity_from_quat(root_quat, dt)

    robot_joint_names = list(robot.data.joint_names)
    joint_pos = robot.data.default_joint_pos[0].repeat(dof_pos_pkl.shape[0], 1)
    for pkl_index, name in enumerate(PKL_DOF_NAMES):
        if name not in robot_joint_names:
            raise KeyError(f"Missing robot joint '{name}' in IsaacLab G1.")
        joint_pos[:, robot_joint_names.index(name)] = dof_pos_pkl[:, pkl_index]
    joint_vel = finite_difference(joint_pos, dt)

    log = {
        "fps": np.asarray([fps], dtype=np.int64),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    sim_dt = sim.get_physics_dt()
    frame = 0
    while simulation_app.is_running() and frame < root_pos.shape[0]:
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = root_pos[frame : frame + 1]
        root_state[:, 3:7] = root_quat[frame : frame + 1]
        root_state[:, 7:10] = root_lin_vel[frame : frame + 1]
        root_state[:, 10:] = root_ang_vel[frame : frame + 1]
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos[frame : frame + 1], joint_vel[frame : frame + 1])
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        body_quat_w = robot.data.body_quat_w[0].cpu().numpy().copy()
        if args_cli.output_quat_order == "xyzw":
            body_quat_w = wxyz_to_xyzw(body_quat_w)
        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(body_quat_w)
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())
        frame += 1

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[key] = np.stack(log[key], axis=0).astype(np.float32)
    np.savez(output, **log)
    print(f"[OK] wrote BeyondMimic motion: {output}")


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / 50.0
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ExportSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    run_export(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
