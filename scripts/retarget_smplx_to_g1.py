#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import pickle
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_args():
    parser = argparse.ArgumentParser(description="Retarget SMPL-X npz to G1 human motion using GMR.")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--tgt_fps", type=int, default=30)
    parser.add_argument("--project_root", type=str, default="")
    parser.add_argument("--gmr_root", type=str, default="/home/warner/_projects/GMR")
    parser.add_argument("--motion_dir", type=str, default="")
    parser.add_argument("--pair_suffix", type=str, default="")
    parser.add_argument("--auto_pair_suffixes", type=str, default="bikez,chairz,suitcasez")
    return parser.parse_args()


def ensure_import_paths(project_root, gmr_root):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if gmr_root not in sys.path:
        sys.path.insert(0, gmr_root)


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


def main():
    args = parse_args()
    project_root = osp.abspath(args.project_root) if args.project_root else osp.abspath(osp.join(osp.dirname(__file__), ".."))
    gmr_root = osp.abspath(args.gmr_root)
    ensure_import_paths(project_root, gmr_root)

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
    import mujoco as mj

    if args.motion_dir:
        motion_dir = osp.abspath(args.motion_dir)
    else:
        motion_dir = osp.join(project_root, "assets", "motions")

    smplx_npz_path = osp.join(motion_dir, f"{args.tag}_smplx_input.npz")
    object_npz_path = osp.join(motion_dir, f"{args.tag}_object.npz")
    if not osp.isfile(smplx_npz_path):
        raise FileNotFoundError(f"SMPL-X input not found: {smplx_npz_path}")
    if not osp.isfile(object_npz_path):
        raise FileNotFoundError(f"Object motion not found: {object_npz_path}")

    smplx_body_model_path = osp.join(gmr_root, "assets", "body_models")
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(smplx_npz_path, smplx_body_model_path)
    smplx_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps
    )

    retarget = GMR(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        verbose=False,
        use_velocity_limit=False,
    )

    carry_pkl = osp.join(project_root, "assets", "motions", "carry.pkl")
    with open(carry_pkl, "rb") as f:
        carry_data = pickle.load(f)
    link_body_list = carry_data["link_body_list"]

    body_ids = []
    for name in link_body_list:
        bid = mj.mj_name2id(retarget.model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' from carry.pkl not found in GMR robot model {args.robot}")
        body_ids.append(bid)

    qpos_list = []
    local_body_pos_list = []
    for frame_data in smplx_frames:
        qpos = retarget.retarget(frame_data)
        qpos_list.append(qpos.copy())
        retarget.configuration.data.qpos[:] = qpos
        mj.mj_forward(retarget.model, retarget.configuration.data)

        root_pos = qpos[:3]
        root_quat_xyzw = np.array([qpos[4], qpos[5], qpos[6], qpos[3]], dtype=np.float64)
        rot_inv = R.from_quat(root_quat_xyzw).inv()
        body_global = retarget.configuration.data.xpos[body_ids]
        body_local = rot_inv.apply(body_global - root_pos[None, :])
        local_body_pos_list.append(body_local.astype(np.float32))

    qpos_arr = np.asarray(qpos_list, dtype=np.float32)
    local_body_pos_arr = np.asarray(local_body_pos_list, dtype=np.float32)

    root_pos = qpos_arr[:, :3]
    root_rot = np.stack([qpos_arr[:, 4], qpos_arr[:, 5], qpos_arr[:, 6], qpos_arr[:, 3]], axis=-1).astype(np.float32)
    dof_pos = qpos_arr[:, 7:]

    object_data = np.load(object_npz_path)
    n = min(len(root_pos), len(object_data["trans"]))

    human_motion = {
        "fps": int(np.round(aligned_fps)),
        "root_pos": root_pos[:n].astype(np.float32),
        "root_rot": root_rot[:n].astype(np.float32),
        "dof_pos": dof_pos[:n].astype(np.float32),
        "local_body_pos": local_body_pos_arr[:n].astype(np.float32),
        "link_body_list": link_body_list,
    }

    human_pkl_path = osp.join(motion_dir, f"{args.tag}_human.pkl")
    with open(human_pkl_path, "wb") as f:
        pickle.dump(human_motion, f)

    pair_suffixes = _collect_pair_suffixes(args.pair_suffix, args.auto_pair_suffixes)
    emitted_pair_files = []
    for suffix in pair_suffixes:
        human_pair_path = osp.join(motion_dir, f"{args.tag}_human_upright_{suffix}.pkl")
        object_pair_path = osp.join(motion_dir, f"{args.tag}_object_upright_{suffix}.npz")
        shutil.copyfile(human_pkl_path, human_pair_path)
        if not osp.isfile(object_pair_path):
            shutil.copyfile(object_npz_path, object_pair_path)
        emitted_pair_files.append((human_pair_path, object_pair_path))

    print("[Done] Retargeted SMPL-X to G1 human motion:")
    print("  SMPLX input:", smplx_npz_path)
    print("  Human motion:", human_pkl_path)
    if emitted_pair_files:
        print("  Upright pair files:")
        for h, o in emitted_pair_files:
            print("   -", h)
            print("   -", o)
    print("  Frames:", int(n))
    print("  FPS:", int(np.round(aligned_fps)))


if __name__ == "__main__":
    main()
