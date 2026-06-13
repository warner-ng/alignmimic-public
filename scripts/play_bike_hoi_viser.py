#!/usr/bin/env python3
"""Non-physics viser playback for ResMimic-retargeted G1 bike HOI motions."""

from __future__ import annotations

import argparse
import pickle
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


DEFAULT_TAG = "Date03_Sub01_bike_May_31_19_34"
DEFAULT_PAIR_SUFFIX = "bikez"
DEFAULT_RESMIMIC_ROOT = "/home/warner/_projects/ResMimic"
DEFAULT_HUMAN_MOTION = (
    f"{DEFAULT_RESMIMIC_ROOT}/assets/motions/{DEFAULT_TAG}_human_upright_{DEFAULT_PAIR_SUFFIX}_aligned.pkl"
)
DEFAULT_OBJECT_MOTION = (
    f"{DEFAULT_RESMIMIC_ROOT}/assets/motions/{DEFAULT_TAG}_object_upright_{DEFAULT_PAIR_SUFFIX}_aligned.npz"
)
DEFAULT_ROBOT_URDF = f"{DEFAULT_RESMIMIC_ROOT}/assets/g1/g1_custom_collision_29dof.urdf"
DEFAULT_OBJECT_URDF = f"{DEFAULT_RESMIMIC_ROOT}/assets/bicycle_top_tube/bikered.urdf"
DEFAULT_OBJECT_MESH = f"{DEFAULT_RESMIMIC_ROOT}/assets/bikered.stl"


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw, dtype=np.float64)
    return np.stack([q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]], axis=-1)


def quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(np.asarray(q1, dtype=np.float64), -1, 0)
    x2, y2, z2, w2 = np.moveaxis(np.asarray(q2, dtype=np.float64), -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def root_rot_offset_quat_xyzw(offset_deg: tuple[float, float, float]) -> np.ndarray:
    roll_deg, pitch_deg, yaw_deg = offset_deg
    qx = R.from_rotvec(np.deg2rad(roll_deg) * np.array([1.0, 0.0, 0.0], dtype=np.float64)).as_quat()
    qy = R.from_rotvec(np.deg2rad(pitch_deg) * np.array([0.0, 1.0, 0.0], dtype=np.float64)).as_quat()
    qz = R.from_rotvec(np.deg2rad(yaw_deg) * np.array([0.0, 0.0, 1.0], dtype=np.float64)).as_quat()
    return quat_mul_xyzw(qx, quat_mul_xyzw(qy, qz))


def apply_root_rot_offset_xyzw(root_rot_xyzw: np.ndarray, offset_deg: tuple[float, float, float]) -> np.ndarray:
    if np.allclose(np.asarray(offset_deg, dtype=np.float64), 0.0):
        return root_rot_xyzw.copy()
    offset_q = root_rot_offset_quat_xyzw(offset_deg)
    return quat_mul_xyzw(offset_q[None, :], root_rot_xyzw)


def apply_root_local_rot_offset_xyzw(root_rot_xyzw: np.ndarray, offset_deg: tuple[float, float, float]) -> np.ndarray:
    if np.allclose(np.asarray(offset_deg, dtype=np.float64), 0.0):
        return root_rot_xyzw.copy()
    offset_q = root_rot_offset_quat_xyzw(offset_deg)
    return quat_mul_xyzw(root_rot_xyzw, offset_q[None, :])


def apply_root_local_axes_rot_offset_xyzw(
    root_rot_xyzw: np.ndarray, offset_deg: tuple[float, float, float]
) -> np.ndarray:
    if np.allclose(np.asarray(offset_deg, dtype=np.float64), 0.0):
        return root_rot_xyzw.copy()
    angles = np.deg2rad(np.asarray(offset_deg, dtype=np.float64))
    out = []
    for q in root_rot_xyzw:
        axes = R.from_quat(q).apply(np.eye(3, dtype=np.float64))
        qx = R.from_rotvec(angles[0] * axes[0]).as_quat()
        qy = R.from_rotvec(angles[1] * axes[1]).as_quat()
        qz = R.from_rotvec(angles[2] * axes[2]).as_quat()
        offset_q = quat_mul_xyzw(qz, quat_mul_xyzw(qy, qx))
        out.append(quat_mul_xyzw(offset_q, q))
    return np.stack(out, axis=0)


def load_human_motion(path: str) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    for key in ("root_pos", "root_rot", "dof_pos", "local_body_pos", "link_body_list"):
        if key not in data:
            raise KeyError(f"Missing key '{key}' in human motion: {path}")
    fps = int(data.get("fps", 30))
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    local_body_pos = np.asarray(data["local_body_pos"], dtype=np.float64)
    link_body_list = list(data["link_body_list"])
    return fps, root_pos, root_rot, dof_pos, local_body_pos, link_body_list


def load_object_motion(path: str, human_frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    for key in ("trans", "rot"):
        if key not in data:
            raise KeyError(f"Missing key '{key}' in object motion: {path}")
    object_pos = np.asarray(data["trans"], dtype=np.float64)
    object_rot_xyzw = np.asarray(data["rot"], dtype=np.float64)
    assert int(object_pos.shape[0]) == human_frame_count, (
        "Object motion frames must match human motion frames: "
        f"{int(object_pos.shape[0])} != {human_frame_count}"
    )
    return object_pos, object_rot_xyzw


def compute_pair_level_transform(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    local_body_pos: np.ndarray,
    foot_ids: list[int],
    object_pos: np.ndarray,
    object_rot_xyzw: np.ndarray,
    object_points: np.ndarray,
    target_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    world_feet = R.from_quat(root_rot_xyzw[0]).apply(local_body_pos[0, foot_ids]) + root_pos[0]
    human_support = world_feet[np.argmin(world_feet[:, 2])]
    world_obj = R.from_quat(object_rot_xyzw[0]).apply(object_points) + object_pos[0]
    object_support = world_obj[np.argmin(world_obj[:, 2])]
    d = human_support - object_support
    h = d.copy()
    h[2] = 0.0
    level_rot = R.identity()
    if np.linalg.norm(h) > 1e-8 and np.linalg.norm(d) > 1e-8 and abs(d[2]) > 1e-8:
        axis = np.cross(d, h)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            angle = np.arccos(np.clip(np.dot(d, h) / (np.linalg.norm(d) * np.linalg.norm(h)), -1.0, 1.0))
            level_rot = R.from_rotvec(axis / axis_norm * angle)
    midpoint = 0.5 * (human_support + object_support)
    trans = midpoint - level_rot.apply(midpoint)
    human_support_after = level_rot.apply(human_support) + trans
    object_support_after = level_rot.apply(object_support) + trans
    trans[2] += target_z - 0.5 * (human_support_after[2] + object_support_after[2])
    return level_rot.as_quat(), trans


def apply_pair_level_transform(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    object_pos: np.ndarray,
    object_rot_xyzw: np.ndarray,
    level_rot_xyzw: np.ndarray,
    level_trans: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    level_rot = R.from_quat(level_rot_xyzw)
    root_pos_out = level_rot.apply(root_pos) + level_trans[None, :]
    object_pos_out = level_rot.apply(object_pos) + level_trans[None, :]
    root_rot_out = quat_mul_xyzw(level_rot_xyzw[None, :], root_rot_xyzw)
    object_rot_out = quat_mul_xyzw(level_rot_xyzw[None, :], object_rot_xyzw)
    return root_pos_out, root_rot_out, object_pos_out, object_rot_out


def build_scaled_urdf_copy(urdf_path: str, scale: float) -> str:
    src = Path(urdf_path).resolve()
    tree = ET.parse(src)
    root = tree.getroot()
    scale = float(scale)

    def scale_vec_text(text: str) -> str:
        vals = [float(x) for x in text.split()]
        vals = [v * scale for v in vals]
        return " ".join(f"{v:.8g}" for v in vals)

    for origin in root.findall(".//origin"):
        xyz = origin.get("xyz")
        if xyz:
            origin.set("xyz", scale_vec_text(xyz))

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh_path = Path(filename)
            if not mesh_path.is_absolute():
                mesh.set("filename", str((src.parent / mesh_path).resolve()))
        scale_text = mesh.get("scale")
        vals = [1.0, 1.0, 1.0]
        if scale_text:
            parsed = [float(x) for x in scale_text.split()]
            vals = (parsed + parsed[-1:] * 3)[:3]
        vals = [v * scale for v in vals]
        mesh.set("scale", f"{vals[0]:.8g} {vals[1]:.8g} {vals[2]:.8g}")
    for box in root.findall(".//box"):
        size = box.get("size")
        if size:
            box.set("size", scale_vec_text(size))
    for sphere in root.findall(".//sphere"):
        radius = sphere.get("radius")
        if radius:
            sphere.set("radius", f"{float(radius) * scale:.8g}")
    for cylinder in root.findall(".//cylinder"):
        radius = cylinder.get("radius")
        length = cylinder.get("length")
        if radius:
            cylinder.set("radius", f"{float(radius) * scale:.8g}")
        if length:
            cylinder.set("length", f"{float(length) * scale:.8g}")
    tmp = tempfile.NamedTemporaryFile(prefix="bike_hoi_object_scaled_", suffix=".urdf", delete=False)
    tree.write(tmp.name, encoding="utf-8", xml_declaration=True)
    return tmp.name


def build_qpos_sequence(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    object_pos: np.ndarray,
    object_rot_xyzw: np.ndarray,
) -> np.ndarray:
    n = int(min(len(root_pos), len(root_rot_xyzw), len(dof_pos), len(object_pos), len(object_rot_xyzw)))
    if n <= 0:
        raise ValueError("No valid frames found in inputs.")
    robot_dof = int(dof_pos.shape[1])
    qpos = np.zeros((n, 7 + robot_dof + 7), dtype=np.float64)
    qpos[:, 0:3] = root_pos[:n]
    qpos[:, 3:7] = xyzw_to_wxyz(root_rot_xyzw[:n])
    qpos[:, 7 : 7 + robot_dof] = dof_pos[:n]
    qpos[:, -7:-4] = object_pos[:n]
    qpos[:, -4:] = xyzw_to_wxyz(object_rot_xyzw[:n])
    return qpos


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    return q if norm == 0.0 else q / norm


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(q0 + u * (q1 - q0))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - u) * theta) * q0 + np.sin(u * theta) * q1) / sin_theta


def start_playback(
    server: viser.ViserServer,
    robot: ViserUrdf,
    robot_base,
    object_base,
    qpos: np.ndarray,
    fps: int,
) -> None:
    robot_dof = int(qpos.shape[1] - 14)
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, qpos.shape[0] - 1), step=1, initial_value=0)
        play_button = server.gui.add_button("Play / Pause")
        fps_input = server.gui.add_number("FPS", initial_value=int(fps), min=1, max=240, step=1)
        interp_input = server.gui.add_number("Visual FPS multiplier", initial_value=2, min=1, max=8, step=1)
    playing = {"value": False}
    frame_float = {"value": 0.0}
    programmatic = {"value": False}

    def apply_frame(q: np.ndarray) -> None:
        robot.update_cfg(q[7 : 7 + robot_dof])
        robot_base.position = q[0:3]
        robot_base.wxyz = q[3:7]
        object_base.position = q[-7:-4]
        object_base.wxyz = q[-4:]

    def apply_index(index: int) -> None:
        apply_frame(qpos[int(np.clip(index, 0, qpos.shape[0] - 1))])

    @play_button.on_click
    def _(_event) -> None:
        playing["value"] = not playing["value"]
        frame_float["value"] = float(frame_slider.value)

    @frame_slider.on_update
    def _(_event) -> None:
        if not programmatic["value"]:
            playing["value"] = False
            frame_float["value"] = float(frame_slider.value)
            apply_index(int(frame_slider.value))

    def loop() -> None:
        while True:
            if playing["value"] and qpos.shape[0] > 1:
                mult = max(1, int(interp_input.value))
                frame_float["value"] = (frame_float["value"] + 1.0 / float(mult)) % float(qpos.shape[0])
                k0 = int(np.floor(frame_float["value"]))
                k1 = (k0 + 1) % qpos.shape[0]
                u = float(frame_float["value"] - k0)
                q = qpos[k0].copy()
                q[0:3] = (1.0 - u) * qpos[k0, 0:3] + u * qpos[k1, 0:3]
                q[3:7] = slerp_wxyz(qpos[k0, 3:7], qpos[k1, 3:7], u)
                q[7 : 7 + robot_dof] = (1.0 - u) * qpos[k0, 7 : 7 + robot_dof] + u * qpos[
                    k1, 7 : 7 + robot_dof
                ]
                q[-7:-4] = (1.0 - u) * qpos[k0, -7:-4] + u * qpos[k1, -7:-4]
                q[-4:] = slerp_wxyz(qpos[k0, -4:], qpos[k1, -4:], u)
                apply_frame(q)
                programmatic["value"] = True
                frame_slider.value = k0
                programmatic["value"] = False
                time.sleep(1.0 / float(max(1, int(fps_input.value)) * mult))
            else:
                time.sleep(0.02)

    threading.Thread(target=loop, daemon=True).start()
    apply_index(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-physics viser preview for ResMimic G1 bike HOI.")
    parser.add_argument("--human", default=DEFAULT_HUMAN_MOTION)
    parser.add_argument("--object", default=DEFAULT_OBJECT_MOTION)
    parser.add_argument("--robot-urdf", default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--object-urdf", default=DEFAULT_OBJECT_URDF)
    parser.add_argument("--object-mesh", default=DEFAULT_OBJECT_MESH)
    parser.add_argument("--human-root-rot-roll-deg", type=float, default=0.0)
    parser.add_argument("--human-root-rot-pitch-deg", type=float, default=0.0)
    parser.add_argument("--human-root-rot-yaw-deg", type=float, default=0.0)
    parser.add_argument("--object-root-rot-roll-deg", type=float, default=-90.0)
    parser.add_argument("--object-root-rot-pitch-deg", type=float, default=60.0)
    parser.add_argument("--object-root-rot-yaw-deg", type=float, default=0.0)
    parser.add_argument("--object-root-local-rot-roll-deg", type=float, default=0.0)
    parser.add_argument("--object-root-local-rot-pitch-deg", type=float, default=0.0)
    parser.add_argument("--object-root-local-rot-yaw-deg", type=float, default=0.0)
    parser.add_argument("--object-root-pos-offset-x", type=float, default=-0.65)
    parser.add_argument("--object-root-pos-offset-y", type=float, default=-0.25)
    parser.add_argument("--object-root-pos-offset-z", type=float, default=0.0)
    parser.add_argument("--pair-root-rot-roll-deg", type=float, default=-85.0)
    parser.add_argument("--pair-root-rot-pitch-deg", type=float, default=90.0)
    parser.add_argument("--pair-root-rot-yaw-deg", type=float, default=0.0)
    parser.add_argument("--pair-root-pos-offset-x", type=float, default=-3.0)
    parser.add_argument("--pair-root-pos-offset-y", type=float, default=0.0)
    parser.add_argument("--pair-root-pos-offset-z", type=float, default=1.0)
    parser.add_argument("--human-root-z-bias", type=float, default=0.0)
    parser.add_argument("--object-root-z-bias", type=float, default=0.0)
    parser.add_argument("--object-mesh-scale", type=float, default=0.6)
    parser.add_argument("--object-root-node", default="/world/object_base/object_visual")
    parser.add_argument("--enable-runtime-pair-leveling", action="store_true")
    parser.add_argument("--runtime-pair-level-target-z", type=float, default=0.0)
    parser.add_argument("--show-ground", action="store_true")
    parser.add_argument("--ground-size", type=float, default=8.0)
    parser.add_argument("--ground-cell-size", type=float, default=0.5)
    parser.add_argument("--ground-plane-z", type=float, default=0.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--no-server", action="store_true", help="Only build qpos and print diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fps_file, root_pos, root_rot_xyzw, dof_pos, local_body_pos, link_body_list = load_human_motion(args.human)
    object_pos, object_quat_xyzw = load_object_motion(args.object, int(root_pos.shape[0]))
    mesh = trimesh.load(args.object_mesh, force="mesh", process=False)
    object_points = np.asarray(mesh.vertices, dtype=np.float64) * float(args.object_mesh_scale)
    object_points = object_points - object_points.mean(axis=0, keepdims=True)

    root_rot_xyzw = apply_root_rot_offset_xyzw(
        root_rot_xyzw,
        (args.human_root_rot_roll_deg, args.human_root_rot_pitch_deg, args.human_root_rot_yaw_deg),
    )
    object_quat_xyzw = apply_root_local_axes_rot_offset_xyzw(
        object_quat_xyzw,
        (args.object_root_rot_roll_deg, args.object_root_rot_pitch_deg, args.object_root_rot_yaw_deg),
    )
    object_quat_xyzw = apply_root_local_rot_offset_xyzw(
        object_quat_xyzw,
        (
            args.object_root_local_rot_roll_deg,
            args.object_root_local_rot_pitch_deg,
            args.object_root_local_rot_yaw_deg,
        ),
    )
    object_offset = np.array(
        [args.object_root_pos_offset_x, args.object_root_pos_offset_y, args.object_root_pos_offset_z],
        dtype=np.float64,
    )
    object_pos = object_pos + R.from_quat(object_quat_xyzw).apply(object_offset)

    pair_rot_offset = (args.pair_root_rot_roll_deg, args.pair_root_rot_pitch_deg, args.pair_root_rot_yaw_deg)
    if not np.allclose(np.asarray(pair_rot_offset, dtype=np.float64), 0.0):
        pair_rot_q = root_rot_offset_quat_xyzw(pair_rot_offset)
        pair_rot = R.from_quat(pair_rot_q)
        root_pos = pair_rot.apply(root_pos)
        object_pos = pair_rot.apply(object_pos)
        root_rot_xyzw = quat_mul_xyzw(pair_rot_q[None, :], root_rot_xyzw)
        object_quat_xyzw = quat_mul_xyzw(pair_rot_q[None, :], object_quat_xyzw)

    pair_pos = np.array(
        [args.pair_root_pos_offset_x, args.pair_root_pos_offset_y, args.pair_root_pos_offset_z],
        dtype=np.float64,
    )
    root_pos = root_pos + pair_pos[None, :]
    object_pos = object_pos + pair_pos[None, :]

    if args.enable_runtime_pair_leveling:
        foot_ids = [
            index
            for index, name in enumerate(link_body_list)
            if any(key in name.lower() for key in ("ankle", "toe", "foot"))
        ]
        if not foot_ids:
            raise ValueError("No ankle/toe/foot links found in human motion for runtime pair leveling.")
        level_rot_xyzw, level_trans = compute_pair_level_transform(
            root_pos,
            root_rot_xyzw,
            local_body_pos,
            foot_ids,
            object_pos,
            object_quat_xyzw,
            object_points,
            args.runtime_pair_level_target_z,
        )
        root_pos, root_rot_xyzw, object_pos, object_quat_xyzw = apply_pair_level_transform(
            root_pos,
            root_rot_xyzw,
            object_pos,
            object_quat_xyzw,
            level_rot_xyzw,
            level_trans,
        )

    root_pos = root_pos.copy()
    object_pos = object_pos.copy()
    root_pos[:, 2] += float(args.human_root_z_bias)
    object_pos[:, 2] += float(args.object_root_z_bias)
    fps = int(args.fps) if int(args.fps) > 0 else max(1, int(fps_file))
    qpos = build_qpos_sequence(root_pos, root_rot_xyzw, dof_pos, object_pos, object_quat_xyzw)

    print(f"[INFO] qpos_shape={qpos.shape} fps={fps}")
    print(f"[INFO] first_robot_root={qpos[0, 0:7].round(6).tolist()}")
    print(f"[INFO] first_object_root={qpos[0, -7:].round(6).tolist()}")
    print(f"[INFO] object_mesh_vertices={len(mesh.vertices)} object_mesh_scale={args.object_mesh_scale:.3f}")
    if args.no_server:
        return

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_frame("/world", wxyz=np.array([1.0, 0.0, 0.0, 0.0]), position=np.zeros(3))
    if args.show_ground:
        server.scene.add_grid(
            "/world/ground",
            width=args.ground_size,
            height=args.ground_size,
            plane="xy",
            cell_color=(180, 180, 180),
            section_color=(120, 120, 120),
            cell_size=args.ground_cell_size,
            section_size=max(args.ground_cell_size, 1.0),
            position=(0.0, 0.0, args.ground_plane_z),
            plane_opacity=0.15,
            plane_color=(220, 220, 220),
        )
    robot_base = server.scene.add_frame("/world/robot_base")
    object_base = server.scene.add_frame("/world/object_base")
    server.scene.add_frame("/world/object_base/object_visual")
    robot = ViserUrdf(server, Path(args.robot_urdf), root_node_name="/world/robot_base/robot")
    object_urdf_path = Path(args.object_urdf)
    if abs(float(args.object_mesh_scale) - 1.0) > 1e-8:
        object_urdf_path = Path(build_scaled_urdf_copy(str(object_urdf_path), float(args.object_mesh_scale)))
    ViserUrdf(server, object_urdf_path, root_node_name=args.object_root_node)
    start_playback(server, robot, robot_base, object_base, qpos, fps)
    print(f"[READY] viewer: http://{args.host}:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
