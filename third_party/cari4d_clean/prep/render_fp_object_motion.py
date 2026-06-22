import argparse
import os
import os.path as osp

import cv2
import imageio
import joblib
import numpy as np
import pyrender
import trimesh


CVCAM_IN_GLCAM = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Render FoundationPose object motion to MP4")
    parser.add_argument("--video", required=True, help="Input RGB video matching the FP frame ids")
    parser.add_argument("--fp_file", required=True, help="FoundationPose *_all.pkl")
    parser.add_argument("--mesh_file", required=True, help="Metric object mesh used by FoundationPose")
    parser.add_argument("--out_video", required=True, help="Output render MP4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--render_scale", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--kid", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--side_by_side", action="store_true")
    return parser.parse_args()


def load_camera(video_path, render_scale):
    cam_file = video_path.replace(".mp4", ".pkl")
    cam = joblib.load(cam_file)
    return np.array(
        [
            [float(cam["fx"]) * render_scale, 0.0, float(cam["cx"]) * render_scale],
            [0.0, float(cam["fy"]) * render_scale, float(cam["cy"]) * render_scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def make_scene(mesh_file, camera_k, height, width):
    scene = pyrender.Scene(ambient_light=[0.7, 0.7, 0.7], bg_color=[0, 0, 0, 0])
    camera = pyrender.IntrinsicsCamera(
        fx=camera_k[0, 0],
        fy=camera_k[1, 1],
        cx=camera_k[0, 2],
        cy=camera_k[1, 2],
        znear=0.001,
        zfar=20.0,
    )
    scene.add(camera, pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=np.eye(4))
    mesh = trimesh.load(mesh_file, process=False, force="mesh")
    mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(width, height)
    return scene, mesh_node, renderer


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_index}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def compose_frame(rgb, rendered, depth, alpha, side_by_side):
    mask = depth > 0
    out = rgb.copy()
    out[mask] = (rgb[mask] * (1.0 - alpha) + rendered[mask] * alpha).astype(np.uint8)
    if side_by_side:
        return np.concatenate([rgb, out], axis=1)
    return out


def main():
    args = parse_args()
    fp_data = joblib.load(args.fp_file)
    poses = fp_data["fp_poses"][:, args.kid]
    frames = fp_data["frames"]
    if args.max_frames > 0:
        poses = poses[: args.max_frames]
        frames = frames[: args.max_frames]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(src_w * args.render_scale)
    height = int(src_h * args.render_scale)
    camera_k = load_camera(args.video, args.render_scale)
    scene, mesh_node, renderer = make_scene(args.mesh_file, camera_k, height, width)

    os.makedirs(osp.dirname(args.out_video), exist_ok=True)
    writer = imageio.get_writer(args.out_video, fps=args.fps)
    for frame_name, pose in zip(frames, poses):
        frame_idx = int(frame_name)
        rgb = read_frame(cap, frame_idx)
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        mesh_node.matrix = CVCAM_IN_GLCAM @ pose
        rendered, depth = renderer.render(scene)
        frame = compose_frame(rgb, rendered[:, :, :3], depth, args.alpha, args.side_by_side)
        writer.append_data(frame)
    writer.close()
    renderer.delete()
    cap.release()
    print(f"saved render video: {args.out_video}")


if __name__ == "__main__":
    main()
