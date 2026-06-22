#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import sys
import shutil

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import numpy.core as _np_core

sys.modules.setdefault("numpy._core", _np_core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def parse_args():
    parser = argparse.ArgumentParser(description="Export CARI4D pth into SMPL-X and object motion intermediates.")
    parser.add_argument("--cari4d_pth", type=str, required=True)
    parser.add_argument("--split", type=str, default="in", choices=["in", "pr", "gt"])
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--project_root", type=str, default="")
    parser.add_argument("--cari4d_root", type=str, default="")
    parser.add_argument("--motion_dir", type=str, default="")
    parser.add_argument("--gender", type=str, default="male", choices=["male", "female", "neutral"])
    parser.add_argument("--pair_suffix", type=str, default="")
    parser.add_argument("--auto_pair_suffixes", type=str, default="bikez,chairz,suitcasez")
    parser.add_argument("--object_offset_x", type=float, default=0.0)
    parser.add_argument("--object_offset_y", type=float, default=0.0)
    parser.add_argument("--object_offset_z", type=float, default=0.0)
    parser.add_argument("--object_rot_roll_deg", type=float, default=0.0)
    parser.add_argument("--object_rot_pitch_deg", type=float, default=0.0)
    parser.add_argument("--object_rot_yaw_deg", type=float, default=0.0)
    return parser.parse_args()


def _frame_time(frame_name: str) -> float:
    frame_name = frame_name.strip()
    t_token = frame_name.split("/")[-1]
    if t_token.isdigit():
        return float(t_token)
    if not t_token.startswith("t"):
        raise ValueError(f"Unexpected frame token: {t_token}")
    return float(t_token[1:])


def infer_fps(frames):
    if len(frames) > 0:
        sample_token = str(frames[0]).strip().split("/")[-1]
        if sample_token.isdigit():
            return 30
    ts = np.array([_frame_time(x) for x in frames], dtype=np.float64)
    if len(ts) < 2:
        return 30
    dt = np.diff(ts)
    dt = dt[dt > 1e-6]
    if len(dt) == 0:
        return 30
    return int(np.round(1.0 / np.median(dt)))


def infer_cari4d_root(cari4d_pth: str):
    p = osp.abspath(cari4d_pth)
    cur = osp.dirname(p)
    while True:
        if osp.isdir(osp.join(cur, "learning")):
            return cur
        parent = osp.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _collect_pair_suffixes(pair_suffix: str, auto_pair_suffixes: str):
    suffixes = []
    if pair_suffix and pair_suffix.strip():
        suffixes.append(pair_suffix.strip())
    if auto_pair_suffixes:
        for s in auto_pair_suffixes.split(","):
            s = s.strip()
            if s:
                suffixes.append(s)
    uniq, seen = [], set()
    for s in suffixes:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def pad_betas_to_gmr_dim(betas_10: np.ndarray, target_dim: int = 16) -> np.ndarray:
    betas_10 = np.asarray(betas_10, dtype=np.float32).reshape(-1)
    if betas_10.shape[0] > target_dim:
        return betas_10[:target_dim].copy()
    if betas_10.shape[0] == target_dim:
        return betas_10.copy()
    out = np.zeros((target_dim,), dtype=np.float32)
    out[: betas_10.shape[0]] = betas_10
    return out


def main():
    args = parse_args()
    project_root = osp.abspath(args.project_root) if args.project_root else osp.abspath(osp.join(osp.dirname(__file__), ".."))
    cari4d_root = osp.abspath(args.cari4d_root) if args.cari4d_root else None
    if cari4d_root is None:
        cari4d_root = infer_cari4d_root(args.cari4d_pth)
    if cari4d_root and cari4d_root not in sys.path:
        sys.path.insert(0, cari4d_root)

    if args.motion_dir:
        motion_dir = osp.abspath(args.motion_dir)
    else:
        motion_dir = osp.join(project_root, "assets", "motions")
    os.makedirs(motion_dir, exist_ok=True)

    data = torch.load(args.cari4d_pth, map_location="cpu", weights_only=False)
    if args.split not in data:
        raise KeyError(f"Split '{args.split}' not found in {args.cari4d_pth}; available: {list(data.keys())}")
    split_data = data[args.split]

    smpl_pose = split_data["smpl_pose"].detach().cpu().numpy()
    smpl_t = split_data["smpl_t"].detach().cpu().numpy()
    betas = split_data["betas"].detach().cpu().numpy()
    frames = split_data["frames"]
    pose_abs = split_data["pose_abs"].detach().cpu().numpy()
    src_fps = infer_fps(frames)

    smplx_npz_path = osp.join(motion_dir, f"{args.tag}_smplx_input.npz")
    betas_16 = pad_betas_to_gmr_dim(np.mean(betas, axis=0).astype(np.float32), target_dim=16)

    np.savez(
        smplx_npz_path,
        pose_body=smpl_pose[:, 3:66].astype(np.float32),
        betas=betas_16,
        expression=np.zeros((smpl_pose.shape[0], 10), dtype=np.float32),
        root_orient=smpl_pose[:, :3].astype(np.float32),
        trans=smpl_t.astype(np.float32),
        mocap_frame_rate=np.array(src_fps, dtype=np.float32),
        gender=np.array(args.gender),
    )

    object_trans = pose_abs[:, :3, 3].astype(np.float32)
    object_offset = np.array([args.object_offset_x, args.object_offset_y, args.object_offset_z], dtype=np.float32)
    object_trans = object_trans + object_offset[None, :]
    object_rot_m = R.from_matrix(pose_abs[:, :3, :3])
    object_rot_offset = R.from_euler(
        "xyz",
        [args.object_rot_roll_deg, args.object_rot_pitch_deg, args.object_rot_yaw_deg],
        degrees=True,
    )
    object_rot = (object_rot_m * object_rot_offset).as_quat().astype(np.float32)

    object_npz_path = osp.join(motion_dir, f"{args.tag}_object.npz")
    np.savez(object_npz_path, trans=object_trans, rot=object_rot)

    pair_suffixes = _collect_pair_suffixes(args.pair_suffix, args.auto_pair_suffixes)
    emitted_pair_files = []
    for suffix in pair_suffixes:
        object_pair_path = osp.join(motion_dir, f"{args.tag}_object_upright_{suffix}.npz")
        shutil.copyfile(object_npz_path, object_pair_path)
        emitted_pair_files.append(object_pair_path)

    print("[Done] Exported CARI4D intermediates:")
    print("  SMPLX input:", smplx_npz_path)
    print("  Object motion:", object_npz_path)
    print("  Object translation offset:", object_offset.tolist())
    print(
        "  Object rotation offset deg:",
        [float(args.object_rot_roll_deg), float(args.object_rot_pitch_deg), float(args.object_rot_yaw_deg)],
    )
    if emitted_pair_files:
        print("  Upright object files:")
        for p in emitted_pair_files:
            print("   -", p)
    print("  Frames:", int(object_trans.shape[0]))
    print("  FPS:", int(np.round(src_fps)))


if __name__ == "__main__":
    main()
