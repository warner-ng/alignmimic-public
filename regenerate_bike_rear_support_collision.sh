#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESMIMIC_ROOT="${RESMIMIC_ROOT:-/home/warner/_projects/ResMimic}"
URDF="${URDF:-$RESMIMIC_ROOT/assets/bicycle_top_tube/bikered.urdf}"
BASE_MESH="${BASE_MESH:-$RESMIMIC_ROOT/assets/bikered.stl}"
OUTPUT_MESH="${OUTPUT_MESH:-$RESMIMIC_ROOT/assets/bikered_with_rear_support_collision.stl}"
KEEP_SUPPORT_ONLY_MESH="${KEEP_SUPPORT_ONLY_MESH:-$RESMIMIC_ROOT/assets/bikered_rear_support_only_collision.stl}"

source /mnt/projects_ext4/conda/miniconda3/etc/profile.d/conda.sh
conda activate beyondmimic

python - "$URDF" "$BASE_MESH" "$OUTPUT_MESH" "$KEEP_SUPPORT_ONLY_MESH" <<'PY'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

urdf_path = Path(sys.argv[1]).resolve()
base_mesh_path = Path(sys.argv[2]).resolve()
output_mesh_path = Path(sys.argv[3]).resolve()
support_only_mesh_path = Path(sys.argv[4]).resolve()


def parse_vec(text: str, expected: int, label: str) -> tuple[float, ...]:
    values = tuple(float(x) for x in text.split())
    if len(values) != expected:
        raise ValueError(f"{label} must have {expected} values: {text}")
    return values


def load_support(name: str) -> dict[str, tuple[float, ...]]:
    root = ET.parse(urdf_path).getroot()
    link = root.find(f".//link[@name='{name}']")
    if link is None:
        raise ValueError(f"Missing link: {name}")
    collision = link.find("collision")
    if collision is None:
        raise ValueError(f"Missing collision for link: {name}")
    origin = collision.find("origin")
    box = collision.find("geometry/box")
    if origin is None or box is None:
        raise ValueError(f"{name} collision must use origin + box geometry")
    xyz = parse_vec(origin.get("xyz", "0 0 0"), 3, f"{name} origin xyz")
    rpy = parse_vec(origin.get("rpy", "0 0 0"), 3, f"{name} origin rpy")
    size = parse_vec(box.get("size", ""), 3, f"{name} box size")
    return {"xyz": xyz, "rpy": rpy, "size": size}


support_meshes = []

for support_name in ("rear_support_left", "rear_support_right"):
    support = load_support(support_name)
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler("xyz", support["rpy"]).as_matrix()
    transform[:3, 3] = np.asarray(support["xyz"], dtype=np.float64)
    support_meshes.append(trimesh.creation.box(extents=support["size"], transform=transform))
    print(
        f"[INFO] {support_name}: xyz={support['xyz']} rpy={support['rpy']} size={support['size']}",
        flush=True,
    )

support_only = trimesh.util.concatenate(support_meshes)
base_mesh = trimesh.load_mesh(base_mesh_path, force="mesh")
combined = trimesh.util.concatenate([base_mesh, support_only])
output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
support_only.export(support_only_mesh_path)
combined.export(output_mesh_path)
print(f"[OK] kept support-only mesh {support_only_mesh_path}", flush=True)
print(f"[OK] wrote {output_mesh_path}", flush=True)
print(f"[OK] vertices={len(combined.vertices)} faces={len(combined.faces)}", flush=True)
PY
