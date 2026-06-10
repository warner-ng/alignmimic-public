"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry.")
parser.add_argument("--motion_file", type=str, default=None, help="Local human motion.npz path.")
parser.add_argument("--motion_quat_order", choices=("wxyz", "xyzw"), default="wxyz", help="Human motion quaternion order.")
parser.add_argument("--object_motion_file", type=str, default=None, help="Optional object motion .npz path.")
parser.add_argument("--object_urdf", type=str, default=None, help="Optional object URDF path.")
parser.add_argument("--object_scale", type=float, default=1.0, help="Object asset scale.")
parser.add_argument("--object_root_z_bias", type=float, default=0.0, help="Object root z spawn bias.")
parser.add_argument("--object_root_pos_offset", nargs=3, type=float, default=(0.0, 0.0, 0.0), help="Local xyz object root offset.")
parser.add_argument(
    "--object_root_rot_offset_deg", nargs=3, type=float, default=(0.0, 0.0, 0.0), help="Local-axis object root rpy offset."
)
parser.add_argument("--human_root_rot_offset_deg", nargs=3, type=float, default=(0.0, 0.0, 0.0), help="Human root rpy offset.")
parser.add_argument("--motion_global_rot_offset_deg", nargs=3, type=float, default=(0.0, 0.0, 0.0), help="Global rpy offset.")
parser.add_argument("--motion_global_pos_offset", nargs=3, type=float, default=(0.0, 0.0, 0.0), help="Global xyz offset.")
parser.add_argument("--max_frames", type=int, default=0, help="Maximum replay frames before exit. 0 means run forever.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader, ObjectMotionLoader


def _has_offset(offset: tuple[float, float, float] | list[float]) -> bool:
    return any(abs(value) > 1e-8 for value in offset)


def _axis_angle_quat(angle: torch.Tensor, axis: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    angle_batch = angle.repeat(axis.shape[0]).to(dtype=dtype)
    half_angle = 0.5 * angle_batch
    quat = torch.zeros((axis.shape[0], 4), device=device, dtype=dtype)
    quat[:, 0] = torch.cos(half_angle)
    quat[:, 1:] = axis * torch.sin(half_angle).unsqueeze(1)
    return quat


def _offset_quat(offset_deg: tuple[float, float, float] | list[float], count: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    offset = torch.tensor(offset_deg, device=device, dtype=dtype) * (torch.pi / 180.0)
    axes = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(count, -1, -1)
    qx = _axis_angle_quat(offset[0], axes[:, 0, :], device, dtype)
    qy = _axis_angle_quat(offset[1], axes[:, 1, :], device, dtype)
    qz = _axis_angle_quat(offset[2], axes[:, 2, :], device, dtype)
    return quat_mul(qx, quat_mul(qy, qz))


def _object_root_rot_offset_quat(
    root_quat: torch.Tensor, offset_deg: tuple[float, float, float] | list[float], device: str
) -> torch.Tensor:
    if not _has_offset(offset_deg):
        identity = torch.zeros((root_quat.shape[0], 4), device=device, dtype=root_quat.dtype)
        identity[:, 0] = 1.0
        return identity
    offset_rad = torch.tensor(offset_deg, device=device, dtype=root_quat.dtype) * (torch.pi / 180.0)
    local_axes = torch.eye(3, device=device, dtype=root_quat.dtype).unsqueeze(0).expand(root_quat.shape[0], -1, -1)
    root_quat_axes = root_quat.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 4)
    world_axes = quat_apply(root_quat_axes, local_axes.reshape(-1, 3)).reshape(root_quat.shape[0], 3, 3)
    qx_axis = _axis_angle_quat(offset_rad[0], world_axes[:, 0, :], device, root_quat.dtype)
    qy_axis = _axis_angle_quat(offset_rad[1], world_axes[:, 1, :], device, root_quat.dtype)
    qz_axis = _axis_angle_quat(offset_rad[2], world_axes[:, 2, :], device, root_quat.dtype)
    return quat_mul(qz_axis, quat_mul(qy_axis, qx_axis))


def _apply_human_offset(
    root_pos: torch.Tensor, root_quat: torch.Tensor, root_lin_vel: torch.Tensor, root_ang_vel: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    human_rot = _offset_quat(args_cli.human_root_rot_offset_deg, root_quat.shape[0], root_quat.device, root_quat.dtype)
    pair_rot = _offset_quat(args_cli.motion_global_rot_offset_deg, root_quat.shape[0], root_quat.device, root_quat.dtype)
    total_rot = quat_mul(pair_rot, human_rot)
    pair_trans = torch.tensor(args_cli.motion_global_pos_offset, device=root_pos.device, dtype=root_pos.dtype)
    return (
        quat_apply(pair_rot, root_pos) + pair_trans.view(1, 3),
        quat_mul(total_rot, root_quat),
        quat_apply(total_rot, root_lin_vel),
        quat_apply(total_rot, root_ang_vel),
    )


def _apply_object_offset(object_pos: torch.Tensor, object_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    object_rot = _object_root_rot_offset_quat(object_quat, args_cli.object_root_rot_offset_deg, object_quat.device)
    object_quat = quat_mul(object_rot, object_quat)
    local_offset = torch.tensor(args_cli.object_root_pos_offset, device=object_pos.device, dtype=object_pos.dtype)
    object_pos = object_pos + quat_apply(object_quat, local_offset.unsqueeze(0).expand(object_pos.shape[0], -1))
    pair_rot = _offset_quat(args_cli.motion_global_rot_offset_deg, object_pos.shape[0], object_pos.device, object_pos.dtype)
    pair_trans = torch.tensor(args_cli.motion_global_pos_offset, device=object_pos.device, dtype=object_pos.dtype)
    object_pos = quat_apply(pair_rot, object_pos) + pair_trans.view(1, 3)
    object_quat = quat_mul(pair_rot, object_quat)
    return object_pos, object_quat


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            # texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object: RigidObjectCfg | None = (
        RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.UrdfFileCfg(
                asset_path=args_cli.object_urdf,
                fix_base=False,
                merge_fixed_joints=True,
                joint_drive=None,
                scale=(args_cli.object_scale, args_cli.object_scale, args_cli.object_scale),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                ),
            ),
        )
        if args_cli.object_urdf is not None
        else None
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    object_asset: RigidObject | None = scene["object"] if args_cli.object_urdf is not None else None
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    motion_file = args_cli.motion_file
    if motion_file is None:
        assert args_cli.registry_name is not None, "Either --motion_file or --registry_name must be provided."
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
        quat_order=args_cli.motion_quat_order,
    )
    object_motion = ObjectMotionLoader(args_cli.object_motion_file, sim.device) if args_cli.object_motion_file is not None else None
    assert (object_asset is None) == (object_motion is None), "--object_urdf and --object_motion_file must be provided together."
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    frame_count = 0
    # Simulation loop
    while simulation_app.is_running():
        if args_cli.max_frames > 0 and frame_count >= args_cli.max_frames:
            break
        frame_count += 1
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_pos, root_quat, root_lin_vel, root_ang_vel = _apply_human_offset(
            motion.body_pos_w[time_steps][:, 0],
            motion.body_quat_w[time_steps][:, 0],
            motion.body_lin_vel_w[time_steps][:, 0],
            motion.body_ang_vel_w[time_steps][:, 0],
        )
        root_states[:, :3] = root_pos + scene.env_origins
        root_states[:, 3:7] = root_quat
        root_states[:, 7:10] = root_lin_vel
        root_states[:, 10:] = root_ang_vel

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        if object_asset is not None and object_motion is not None:
            object_states = object_asset.data.default_root_state.clone()
            object_pos, object_quat = _apply_object_offset(
                object_motion.root_pos[time_steps],
                object_motion.root_quat[time_steps],
            )
            object_states[:, :3] = object_pos + scene.env_origins
            object_states[:, 2] += args_cli.object_root_z_bias
            object_states[:, 3:7] = object_quat
            object_asset.write_root_state_to_sim(object_states)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
