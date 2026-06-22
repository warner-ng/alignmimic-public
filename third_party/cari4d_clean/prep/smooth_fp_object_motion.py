import argparse
import os
import os.path as osp

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_args():
    parser = argparse.ArgumentParser(description="Smooth FoundationPose object motion")
    parser.add_argument("--input", required=True, help="Input FoundationPose *_all.pkl")
    parser.add_argument("--output", required=True, help="Output smoothed pkl")
    parser.add_argument("--alpha_trans", type=float, default=0.25)
    parser.add_argument("--alpha_rot", type=float, default=0.35)
    return parser.parse_args()


def lowpass_translations(translations, alpha):
    smoothed = translations.copy()
    for i in range(1, len(smoothed)):
        smoothed[i] = smoothed[i - 1] * (1.0 - alpha) + translations[i] * alpha
    return smoothed


def lowpass_rotations(rotations, alpha):
    smoothed = rotations.copy()
    for i in range(1, len(smoothed)):
        prev = R.from_matrix(smoothed[i - 1])
        cur = R.from_matrix(rotations[i])
        rel = prev.inv() * cur
        smoothed[i] = (prev * R.from_rotvec(rel.as_rotvec() * alpha)).as_matrix()
    return smoothed


def smooth_pose_sequence(poses, alpha_trans, alpha_rot):
    smoothed = poses.copy()
    smoothed[:, :3, :3] = lowpass_rotations(poses[:, :3, :3], alpha_rot)
    smoothed[:, :3, 3] = lowpass_translations(poses[:, :3, 3], alpha_trans)
    return smoothed


def mean_translation_jitter(poses):
    trans = poses[:, :3, 3]
    if len(trans) < 4:
        return 0.0
    jerk = trans[3:] - 3.0 * trans[2:-1] + 3.0 * trans[1:-2] - trans[:-3]
    return float(np.linalg.norm(jerk, axis=1).mean())


def main():
    args = parse_args()
    data = joblib.load(args.input)
    poses = data["fp_poses"].copy()
    before = mean_translation_jitter(poses[:, 0])
    for kid in range(poses.shape[1]):
        poses[:, kid] = smooth_pose_sequence(poses[:, kid], args.alpha_trans, args.alpha_rot)
    data["fp_poses"] = poses
    data["smoothing"] = {
        "alpha_trans": args.alpha_trans,
        "alpha_rot": args.alpha_rot,
        "source": args.input,
    }
    after = mean_translation_jitter(poses[:, 0])
    os.makedirs(osp.dirname(args.output), exist_ok=True)
    joblib.dump(data, args.output)
    print(f"saved smoothed FoundationPose motion: {args.output}")
    print(f"translation jitter mean: {before:.6f} -> {after:.6f}")


if __name__ == "__main__":
    main()
