"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import pickle
import torch
import trimesh

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry.")
parser.add_argument("--motion_file", type=str, default=None, help="Local human motion.npz path.")
parser.add_argument("--human_pkl", type=str, default=None, help="ResMimic-style human motion .pkl path.")
parser.add_argument("--motion_quat_order", choices=("wxyz", "xyzw"), default="wxyz", help="Human motion quaternion order.")
parser.add_argument("--object_motion_file", type=str, default=None, help="Optional object motion .npz path.")
parser.add_argument("--object_urdf", type=str, default=None, help="Optional object URDF path.")
parser.add_argument("--object_usd", type=str, default=None, help="Optional object USD path.")
parser.add_argument("--object_mesh", type=str, default=None, help="Optional object mesh path for pair leveling.")
parser.add_argument("--object_scale", type=float, default=1.0, help="Object asset scale.")
parser.add_argument("--object_mesh_scale", type=float, default=1.0, help="Object mesh scale for pair leveling.")
parser.add_argument("--human_root_z_bias", type=float, default=0.0, help="Human root z bias.")
parser.add_argument("--object_root_z_bias", type=float, default=0.0, help="Object root z spawn bias.")
parser.add_argument("--object_root_pos_offset", nargs=3, type=float, default=(0.0, 0.0, 0.0), help="Local xyz object root offset.")
parser.add_argument(
    "--object_root_rot_offset_deg", nargs=3, type=float, default=(0.0, 0.0, 0.0), help="Local-axis object root rpy offset."
)
parser.add_argument("--human_root_rot_offset_deg", nargs=3, type=float, default=(0.0, 0.0, 0.0), help="Human root rpy offset.")
parser.add_argument("--motion_global_rot_offset_deg", nargs=3, type=float, default=(0.0, 0.0, 0.0), help="Global rpy offset.")
parser.add_argument("--motion_global_pos_offset", nargs=3, type=float, default=(0.0, 0.0, 0.0), help="Global xyz offset.")
parser.add_argument("--enable_runtime_pair_leveling", action="store_true", help="Match viser runtime pair leveling.")
parser.add_argument("--runtime_pair_level_target_z", type=float, default=0.0, help="Pair leveling target z.")
parser.add_argument("--debug_marker_frames", type=int, default=0, help="Print red/green marker positions for N frames.")
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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader, ObjectMotionLoader


CURRENT_BODY_VIS_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Replay/current/body",
    markers={
        "current": sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    },
)
GOAL_BODY_VIS_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Replay/goal/body",
    markers={
        "goal": sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )
    },
)

REPLAY_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


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


def _apply_body_offset(body_pos: torch.Tensor, body_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    root_pos = body_pos[:, 0]
    root_quat = body_quat[:, 0]
    human_rot = _offset_quat(args_cli.human_root_rot_offset_deg, root_quat.shape[0], root_quat.device, root_quat.dtype)
    body_pos = root_pos[:, None, :] + quat_apply(
        human_rot[:, None, :].expand(-1, body_pos.shape[1], -1).reshape(-1, 4),
        (body_pos - root_pos[:, None, :]).reshape(-1, 3),
    ).reshape_as(body_pos)
    human_rot_body = human_rot[:, None, :].expand(-1, body_quat.shape[1], -1)
    body_quat = quat_mul(human_rot_body.reshape(-1, 4), body_quat.reshape(-1, 4)).reshape_as(body_quat)
    pair_rot = _offset_quat(args_cli.motion_global_rot_offset_deg, body_pos.shape[0], body_pos.device, body_pos.dtype)
    body_pos = quat_apply(
        pair_rot[:, None, :].expand(-1, body_pos.shape[1], -1).reshape(-1, 4), body_pos.reshape(-1, 3)
    ).reshape_as(body_pos)
    pair_rot_body = pair_rot[:, None, :].expand(-1, body_quat.shape[1], -1)
    body_quat = quat_mul(pair_rot_body.reshape(-1, 4), body_quat.reshape(-1, 4)).reshape_as(body_quat)
    pair_trans = torch.tensor(args_cli.motion_global_pos_offset, device=body_pos.device, dtype=body_pos.dtype)
    body_pos = body_pos + pair_trans.view(1, 1, 3)
    return body_pos, body_quat


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


def _xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return quat[:, [3, 0, 1, 2]]


def _debug_list(tensor: torch.Tensor) -> list[float]:
    return [round(float(value), 6) for value in tensor.detach().cpu().tolist()]


def _load_object_points_for_leveling(path: str, scale: float, device: str, dtype: torch.dtype) -> torch.Tensor:
    mesh = trimesh.load(path, force="mesh", process=False)
    object_points = np.asarray(mesh.vertices, dtype=np.float32) * float(scale)
    object_points = object_points - object_points.mean(axis=0, keepdims=True)
    return torch.tensor(object_points, dtype=dtype, device=device)


def _compute_pair_level_transform(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    local_body_pos: torch.Tensor,
    foot_ids: list[int],
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    object_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = root_pos.dtype
    device = root_pos.device
    identity = torch.zeros((1, 4), device=device, dtype=dtype)
    identity[:, 0] = 1.0

    human_support_points = quat_apply(
        root_quat[:1].expand(len(foot_ids), -1), local_body_pos[0, foot_ids]
    ) + root_pos[:1]
    human_support = human_support_points[torch.argmin(human_support_points[:, 2])]
    object_support_points = quat_apply(
        object_quat[:1].expand(object_points.shape[0], -1), object_points
    ) + object_pos[:1]
    object_support = object_support_points[torch.argmin(object_support_points[:, 2])]

    d = human_support - object_support
    h = d.clone()
    h[2] = 0.0
    level_rot = identity
    d_norm = torch.linalg.norm(d)
    h_norm = torch.linalg.norm(h)
    if h_norm > 1e-8 and d_norm > 1e-8 and torch.abs(d[2]) > 1e-8:
        axis = torch.cross(d, h, dim=0)
        axis_norm = torch.linalg.norm(axis)
        if axis_norm > 1e-8:
            cos_angle = torch.clamp(torch.dot(d, h) / (d_norm * h_norm), -1.0, 1.0)
            angle = torch.acos(cos_angle)
            level_rot = _axis_angle_quat(angle, (axis / axis_norm).view(1, 3), device, dtype)

    midpoint = 0.5 * (human_support + object_support)
    level_trans = midpoint - quat_apply(level_rot, midpoint.view(1, 3))[0]
    human_support_after = quat_apply(level_rot, human_support.view(1, 3))[0] + level_trans
    object_support_after = quat_apply(level_rot, object_support.view(1, 3))[0] + level_trans
    level_trans[2] += args_cli.runtime_pair_level_target_z - 0.5 * (
        human_support_after[2] + object_support_after[2]
    )
    return level_rot, level_trans


def _apply_pair_level_transform(
    root_pos: torch.Tensor, root_quat: torch.Tensor, level_rot: torch.Tensor, level_trans: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    level_rot_batch = level_rot.expand(root_pos.shape[0], -1)
    return quat_apply(level_rot_batch, root_pos) + level_trans.view(1, 3), quat_mul(level_rot_batch, root_quat)


def _apply_body_level_transform(
    body_pos: torch.Tensor, body_quat: torch.Tensor, level_rot: torch.Tensor, level_trans: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    level_rot_body = level_rot[:, None, :].expand(body_pos.shape[0], body_pos.shape[1], -1)
    body_pos = quat_apply(level_rot_body.reshape(-1, 4), body_pos.reshape(-1, 3)).reshape_as(body_pos)
    body_pos = body_pos + level_trans.view(1, 1, 3)
    body_quat = quat_mul(level_rot_body.reshape(-1, 4), body_quat.reshape(-1, 4)).reshape_as(body_quat)
    return body_pos, body_quat


def _load_human_pkl(
    path: str, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, list[str]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    for key in ("root_pos", "root_rot", "dof_pos"):
        if key not in data:
            raise KeyError(f"Missing key '{key}' in human motion: {path}")
    root_pos = torch.tensor(np.asarray(data["root_pos"], dtype=np.float32), device=device)
    root_rot = _xyzw_to_wxyz(torch.tensor(np.asarray(data["root_rot"], dtype=np.float32), device=device))
    dof_pos = torch.tensor(np.asarray(data["dof_pos"], dtype=np.float32), device=device)
    dof_vel = torch.zeros_like(dof_pos)
    if "dof_vel" in data:
        dof_vel = torch.tensor(np.asarray(data["dof_vel"], dtype=np.float32), device=device)
    local_body_pos = None
    if "local_body_pos" in data:
        local_body_pos = torch.tensor(np.asarray(data["local_body_pos"], dtype=np.float32), device=device)
    link_body_list = list(data.get("link_body_list", []))
    return root_pos, root_rot, dof_pos, dof_vel, local_body_pos, link_body_list


def _make_object_spawn_cfg():
    object_file_scale = (args_cli.object_scale, args_cli.object_scale, args_cli.object_scale)
    object_collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.02,
        rest_offset=0.0,
    )
    object_rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=5.0,
    )
    if args_cli.object_usd is not None:
        return sim_utils.UsdFileCfg(
            usd_path=args_cli.object_usd,
            scale=object_file_scale,
            collision_props=object_collision_props,
            rigid_props=object_rigid_props,
        )
    if args_cli.object_urdf is not None:
        return sim_utils.UrdfFileCfg(
            asset_path=args_cli.object_urdf,
            fix_base=False,
            merge_fixed_joints=True,
            # Keep IsaacLab's default convex hull path; convex_decomposition can break PhysX scene creation.
            # collider_type="convex_decomposition",
            joint_drive=None,
            scale=object_file_scale,
            collision_props=object_collision_props,
            rigid_props=object_rigid_props,
        )
    return None


OBJECT_SPAWN_CFG = _make_object_spawn_cfg()


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
            spawn=OBJECT_SPAWN_CFG,
        )
        if OBJECT_SPAWN_CFG is not None
        else None
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    object_asset: RigidObject | None = scene["object"] if (args_cli.object_usd is not None or args_cli.object_urdf is not None) else None
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    motion_file = args_cli.motion_file
    if motion_file is None and args_cli.human_pkl is None:
        assert args_cli.registry_name is not None, "Either --motion_file or --registry_name must be provided."
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    motion = None
    root_pos_seq = None
    root_quat_seq = None
    dof_pos_seq = None
    dof_vel_seq = None
    local_body_pos_seq = None
    link_body_list = []
    motion_body_ids, motion_body_names = robot.find_bodies(REPLAY_BODY_NAMES, preserve_order=True)
    motion_body_ids_tensor = torch.tensor(motion_body_ids, dtype=torch.long, device=sim.device)
    if motion_file is not None:
        motion = MotionLoader(
            motion_file,
            motion_body_ids_tensor,
            sim.device,
            quat_order=args_cli.motion_quat_order,
        )
    if args_cli.human_pkl is not None:
        root_pos_seq, root_quat_seq, dof_pos_seq, dof_vel_seq, local_body_pos_seq, link_body_list = _load_human_pkl(
            args_cli.human_pkl, sim.device
        )
    object_motion = ObjectMotionLoader(args_cli.object_motion_file, sim.device) if args_cli.object_motion_file is not None else None
    assert (object_asset is None) == (
        object_motion is None
    ), "--object_usd/--object_urdf and --object_motion_file must be provided together."
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    time_step_total = motion.time_step_total if motion is not None else int(root_pos_seq.shape[0])
    level_rot = None
    level_trans = None
    if args_cli.enable_runtime_pair_leveling:
        assert args_cli.human_pkl is not None and local_body_pos_seq is not None and link_body_list
        assert object_motion is not None and args_cli.object_mesh is not None
        foot_ids = [
            index
            for index, name in enumerate(link_body_list)
            if any(key in name.lower() for key in ("ankle", "toe", "foot"))
        ]
        if not foot_ids:
            raise ValueError("No ankle/toe/foot links found in human motion for runtime pair leveling.")
        preview_root_pos, preview_root_quat, _, _ = _apply_human_offset(
            root_pos_seq,
            root_quat_seq,
            torch.zeros((time_step_total, 3), dtype=torch.float32, device=sim.device),
            torch.zeros((time_step_total, 3), dtype=torch.float32, device=sim.device),
        )
        preview_object_pos, preview_object_quat = _apply_object_offset(object_motion.root_pos, object_motion.root_quat)
        object_points = _load_object_points_for_leveling(
            args_cli.object_mesh, args_cli.object_mesh_scale, sim.device, preview_root_pos.dtype
        )
        level_rot, level_trans = _compute_pair_level_transform(
            preview_root_pos,
            preview_root_quat,
            local_body_pos_seq,
            foot_ids,
            preview_object_pos,
            preview_object_quat,
            object_points,
        )
    marker_body_ids = []
    marker_source_ids = []
    current_body_visualizers = []
    goal_body_visualizers = []
    if motion is not None:
        marker_body_ids = motion_body_ids
        marker_source_ids = list(range(len(motion_body_names)))
        found_names = motion_body_names
    elif local_body_pos_seq is not None and link_body_list:
        marker_names = [name for name in REPLAY_BODY_NAMES if name in link_body_list]
        marker_body_ids, found_names = robot.find_bodies(marker_names, preserve_order=True)
        marker_source_ids = [link_body_list.index(name) for name in found_names]
    else:
        found_names = []
    if found_names:
        for name in found_names:
            current_body_visualizers.append(
                VisualizationMarkers(CURRENT_BODY_VIS_CFG.replace(prim_path="/Visuals/Replay/current/" + name))
            )
            goal_body_visualizers.append(
                VisualizationMarkers(GOAL_BODY_VIS_CFG.replace(prim_path="/Visuals/Replay/goal/" + name))
            )
    if args_cli.debug_marker_frames > 0:
        print(
            "[DEBUG_MARKER] "
            f"motion_file={motion_file} human_pkl={args_cli.human_pkl} motion_quat_order={args_cli.motion_quat_order}",
            flush=True,
        )
        print(
            "[DEBUG_MARKER] "
            f"marker_names={found_names} marker_body_ids={list(marker_body_ids)} marker_source_ids={marker_source_ids}",
            flush=True,
        )

    frame_count = 0
    # Simulation loop
    while simulation_app.is_running():
        if args_cli.max_frames > 0 and frame_count >= args_cli.max_frames:
            break
        reset_ids = time_steps >= time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        body_pos_ref = None
        if motion is not None:
            body_pos_ref, body_quat_ref = _apply_body_offset(motion.body_pos_w[time_steps], motion.body_quat_w[time_steps])
            if level_rot is not None and level_trans is not None:
                body_pos_ref, body_quat_ref = _apply_body_level_transform(body_pos_ref, body_quat_ref, level_rot, level_trans)
            body_pos_ref[:, :, 2] += args_cli.human_root_z_bias
            root_pos = body_pos_ref[:, 0]
            root_quat = body_quat_ref[:, 0]
            _, _, root_lin_vel, root_ang_vel = _apply_human_offset(
                motion.body_pos_w[time_steps][:, 0],
                motion.body_quat_w[time_steps][:, 0],
                motion.body_lin_vel_w[time_steps][:, 0],
                motion.body_ang_vel_w[time_steps][:, 0],
            )
            if level_rot is not None:
                level_rot_batch = level_rot.expand(root_lin_vel.shape[0], -1)
                root_lin_vel = quat_apply(level_rot_batch, root_lin_vel)
                root_ang_vel = quat_apply(level_rot_batch, root_ang_vel)
            joint_pos = motion.joint_pos[time_steps]
            joint_vel = motion.joint_vel[time_steps]
            local_body_pos = None
        elif args_cli.human_pkl is not None:
            root_pos, root_quat, root_lin_vel, root_ang_vel = _apply_human_offset(
                root_pos_seq[time_steps],
                root_quat_seq[time_steps],
                torch.zeros((scene.num_envs, 3), dtype=torch.float32, device=sim.device),
                torch.zeros((scene.num_envs, 3), dtype=torch.float32, device=sim.device),
            )
            if level_rot is not None and level_trans is not None:
                root_pos, root_quat = _apply_pair_level_transform(root_pos, root_quat, level_rot, level_trans)
            joint_pos = dof_pos_seq[time_steps]
            joint_vel = dof_vel_seq[time_steps]
            local_body_pos = local_body_pos_seq[time_steps] if local_body_pos_seq is not None else None
            root_pos[:, 2] += args_cli.human_root_z_bias
        root_states[:, :3] = root_pos + scene.env_origins
        root_states[:, 3:7] = root_quat
        root_states[:, 7:10] = root_lin_vel
        root_states[:, 10:] = root_ang_vel

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        if object_asset is not None and object_motion is not None:
            object_states = object_asset.data.default_root_state.clone()
            object_pos, object_quat = _apply_object_offset(
                object_motion.root_pos[time_steps],
                object_motion.root_quat[time_steps],
            )
            if level_rot is not None and level_trans is not None:
                object_pos, object_quat = _apply_pair_level_transform(object_pos, object_quat, level_rot, level_trans)
            object_pos[:, 2] += args_cli.object_root_z_bias
            object_states[:, :3] = object_pos + scene.env_origins
            # z bias is applied to object_pos above to match viser transform order.
            # object_states[:, 2] += args_cli.object_root_z_bias
            object_states[:, 3:7] = object_quat
            object_asset.write_root_state_to_sim(object_states)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)
        if marker_body_ids:
            if body_pos_ref is not None:
                goal_body_pos = body_pos_ref[:, marker_source_ids] + scene.env_origins[:, None, :]
            else:
                marker_local_body_pos = local_body_pos[:, marker_source_ids]
                goal_body_quat = root_quat[:, None, :].expand(-1, marker_local_body_pos.shape[1], -1).reshape(-1, 4)
                goal_body_pos = quat_apply(goal_body_quat, marker_local_body_pos.reshape(-1, 3)).reshape_as(
                    marker_local_body_pos
                )
                goal_body_pos = goal_body_pos + root_pos[:, None, :] + scene.env_origins[:, None, :]
            for i, body_id in enumerate(marker_body_ids):
                current_body_visualizers[i].visualize(robot.data.body_pos_w[:, body_id], robot.data.body_quat_w[:, body_id])
                goal_body_visualizers[i].visualize(goal_body_pos[:, i])
            if frame_count < args_cli.debug_marker_frames:
                current_body_pos = robot.data.body_pos_w[:, marker_body_ids]
                marker_diff = torch.linalg.norm(current_body_pos - goal_body_pos, dim=-1)
                print(
                    "[DEBUG_MARKER] "
                    f"frame={frame_count} time_step={int(time_steps[0].item())} "
                    f"root_pos={_debug_list(root_states[0, :3])} root_quat={_debug_list(root_states[0, 3:7])}",
                    flush=True,
                )
                for i, name in enumerate(found_names):
                    print(
                        "[DEBUG_MARKER] "
                        f"{name} red={_debug_list(current_body_pos[0, i])} "
                        f"green={_debug_list(goal_body_pos[0, i])} diff={float(marker_diff[0, i].item()):.6f}",
                        flush=True,
                    )
        elif frame_count < args_cli.debug_marker_frames:
            print(
                "[DEBUG_MARKER] "
                f"frame={frame_count} time_step={int(time_steps[0].item())} no marker_body_ids",
                flush=True,
            )

        # Match train/play static world-origin viewer; do not overwrite mouse camera edits every frame.
        # pos_lookat = root_states[0, :3].cpu().numpy()
        # sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)
        frame_count += 1
        time_steps += 1


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    sim.set_camera_view(np.array([1.5, 1.5, 1.5]), np.array([0.0, 0.0, 0.0]))
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
