import argparse

import joblib
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Check GVHMR human and FoundationPose object sequence lengths")
    parser.add_argument("--gvhmr_results", required=True, help="GVHMR hmr4d_results.pt")
    parser.add_argument("--fp_file", required=True, help="FoundationPose object motion pkl")
    return parser.parse_args()


def main():
    args = parse_args()
    human_data = torch.load(args.gvhmr_results, map_location="cpu")
    object_data = joblib.load(args.fp_file)

    human_len = int(human_data["smpl_params_global"]["body_pose"].shape[0])
    object_len = int(object_data["fp_poses"].shape[0])
    object_frame_len = len(object_data["frames"])

    print(f"GVHMR human length: {human_len}")
    print(f"FoundationPose object length: {object_len}")
    print(f"FoundationPose frame list length: {object_frame_len}")

    if object_len != object_frame_len:
        raise SystemExit(f"FoundationPose pose/frame length mismatch: poses={object_len}, frames={object_frame_len}")
    if human_len != object_len:
        raise SystemExit(f"Human/object length mismatch: human={human_len}, object={object_len}")

    print(f"Human/object length check passed: {human_len} frames")


if __name__ == "__main__":
    main()
