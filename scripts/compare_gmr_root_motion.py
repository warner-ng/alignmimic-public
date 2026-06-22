#!/usr/bin/env python3
"""Compare pre-GMR and post-GMR human root motion.

This script is analysis-only: it does not modify any motion files.
It focuses on:
1. root position drift before vs after retargeting
2. root rotation drift before vs after retargeting
3. post-retarget human root vs object root horizontal alignment
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


import numpy.core as _np_core

sys.modules.setdefault("numpy._core", _np_core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def load_preretarget_human(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    data = np.load(path, allow_pickle=True)
    required = ("trans", "root_orient", "mocap_frame_rate")
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in pre-retarget file: {path}")

    root_pos = np.asarray(data["trans"], dtype=np.float64)
    root_rot = R.from_rotvec(np.asarray(data["root_orient"], dtype=np.float64)).as_quat()
    fps = float(np.asarray(data["mocap_frame_rate"]).item())
    return root_pos, root_rot, fps


def load_postretarget_human(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    required = ("root_pos", "root_rot", "fps")
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in post-retarget file: {path}")

    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)
    fps = float(data["fps"])
    return root_pos, root_rot, fps


def load_object_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = ("trans", "rot")
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in object motion file: {path}")

    return np.asarray(data["trans"], dtype=np.float64), np.asarray(data["rot"], dtype=np.float64)


def align_lengths(*arrays: np.ndarray) -> list[np.ndarray]:
    n = min(arr.shape[0] for arr in arrays)
    return [arr[:n] for arr in arrays]


def quat_angle_deg(q_a_xyzw: np.ndarray, q_b_xyzw: np.ndarray) -> np.ndarray:
    rel = R.from_quat(q_a_xyzw).inv() * R.from_quat(q_b_xyzw)
    return np.rad2deg(rel.magnitude())


def summarize_vector(name: str, vec: np.ndarray) -> list[str]:
    norms = np.linalg.norm(vec, axis=1)
    mean_xyz = vec.mean(axis=0)
    abs_mean_xyz = np.abs(vec).mean(axis=0)
    max_abs_xyz = np.abs(vec).max(axis=0)
    return [
        f"{name}:",
        f"  mean delta xyz = [{mean_xyz[0]:.6f}, {mean_xyz[1]:.6f}, {mean_xyz[2]:.6f}]",
        f"  mean |delta| xyz = [{abs_mean_xyz[0]:.6f}, {abs_mean_xyz[1]:.6f}, {abs_mean_xyz[2]:.6f}]",
        f"  max |delta| xyz = [{max_abs_xyz[0]:.6f}, {max_abs_xyz[1]:.6f}, {max_abs_xyz[2]:.6f}]",
        f"  mean L2 = {norms.mean():.6f}",
        f"  p95 L2 = {np.percentile(norms, 95):.6f}",
        f"  max L2 = {norms.max():.6f}",
    ]


def summarize_angles(name: str, angles_deg: np.ndarray) -> list[str]:
    return [
        f"{name}:",
        f"  mean angle deg = {angles_deg.mean():.6f}",
        f"  p95 angle deg = {np.percentile(angles_deg, 95):.6f}",
        f"  max angle deg = {angles_deg.max():.6f}",
    ]


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    k = min(k, values.shape[0])
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.argsort(values)[-k:][::-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GMR pre/post root motion and post-human/object alignment.")
    parser.add_argument("--pre-human", type=Path, required=True, help="Path to pre-retarget SMPL-X npz.")
    parser.add_argument("--post-human", type=Path, required=True, help="Path to post-retarget human pkl.")
    parser.add_argument("--object", type=Path, required=True, help="Path to object motion npz.")
    parser.add_argument("--topk", type=int, default=5, help="Number of worst frames to print.")
    parser.add_argument("--quiet", action="store_true", help="Only print minimal summary lines.")
    args = parser.parse_args()

    pre_root_pos, pre_root_rot, pre_fps = load_preretarget_human(args.pre_human)
    post_root_pos, post_root_rot, post_fps = load_postretarget_human(args.post_human)
    object_root_pos, _ = load_object_motion(args.object)

    pre_root_pos, pre_root_rot, post_root_pos, post_root_rot, object_root_pos = align_lengths(
        pre_root_pos, pre_root_rot, post_root_pos, post_root_rot, object_root_pos
    )
    n_frames = pre_root_pos.shape[0]

    root_pos_delta = post_root_pos - pre_root_pos
    root_rot_angle_deg = quat_angle_deg(pre_root_rot, post_root_rot)

    post_vs_object_xyz = post_root_pos - object_root_pos
    post_vs_object_xy = post_vs_object_xyz[:, :2]
    post_vs_object_xy_norm = np.linalg.norm(post_vs_object_xy, axis=1)

    print(f"Frames compared: {n_frames}")
    print(f"Pre-retarget FPS: {pre_fps}")
    print(f"Post-retarget FPS: {post_fps}")

    if args.quiet:
        return

    for line in summarize_vector("Post human minus pre human root_pos", root_pos_delta):
        print(line)
    print()

    for line in summarize_angles("Pre/post human root_rot relative angle", root_rot_angle_deg):
        print(line)
    print()

    for line in summarize_vector("Post human minus object root_pos", post_vs_object_xyz):
        print(line)
    print("Post human minus object root_pos (horizontal XY only):")
    print(f"  mean delta xy = [{post_vs_object_xy[:, 0].mean():.6f}, {post_vs_object_xy[:, 1].mean():.6f}]")
    print(f"  mean XY distance = {post_vs_object_xy_norm.mean():.6f}")
    print(f"  p95 XY distance = {np.percentile(post_vs_object_xy_norm, 95):.6f}")
    print(f"  max XY distance = {post_vs_object_xy_norm.max():.6f}")
    print()

    print(f"Top {min(args.topk, n_frames)} frames with largest pre/post root_pos L2 drift:")
    pos_norm = np.linalg.norm(root_pos_delta, axis=1)
    for idx in topk_indices(pos_norm, args.topk):
        delta = root_pos_delta[idx]
        print(
            f"  frame={idx} delta_xyz=[{delta[0]:.6f}, {delta[1]:.6f}, {delta[2]:.6f}] "
            f"L2={pos_norm[idx]:.6f}"
        )
    print()

    print(f"Top {min(args.topk, n_frames)} frames with largest pre/post root_rot angle drift:")
    for idx in topk_indices(root_rot_angle_deg, args.topk):
        print(f"  frame={idx} angle_deg={root_rot_angle_deg[idx]:.6f}")
    print()

    print(f"Top {min(args.topk, n_frames)} frames with largest post-human/object XY drift:")
    for idx in topk_indices(post_vs_object_xy_norm, args.topk):
        delta_xy = post_vs_object_xy[idx]
        print(
            f"  frame={idx} delta_xy=[{delta_xy[0]:.6f}, {delta_xy[1]:.6f}] "
            f"xy_dist={post_vs_object_xy_norm[idx]:.6f} z_delta={post_vs_object_xyz[idx, 2]:.6f}"
        )


if __name__ == "__main__":
    main()
