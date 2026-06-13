#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESMIMIC_ROOT="${RESMIMIC_ROOT:-/home/warner/_projects/ResMimic}"
URDF="${URDF:-$RESMIMIC_ROOT/assets/bicycle_top_tube/bikered.urdf}"
BASE_MESH="${BASE_MESH:-$RESMIMIC_ROOT/assets/bikered.stl}"
OUTPUT_MESH="${OUTPUT_MESH:-$RESMIMIC_ROOT/assets/bikered_with_rear_support_collision.stl}"
KEEP_SUPPORT_ONLY_MESH="${KEEP_SUPPORT_ONLY_MESH:-$RESMIMIC_ROOT/assets/bikered_rear_support_only_collision.stl}"
OUTPUT_USD="${OUTPUT_USD:-$RESMIMIC_ROOT/assets/bicycle_top_tube/bikered_rigid_compound.usd}"
USD_BUILD_DIR="${USD_BUILD_DIR:-/tmp/bikered_rigid_compound_build}"
BIKE_MESH_COLLISION_APPROXIMATION="${BIKE_MESH_COLLISION_APPROXIMATION:-convexDecomposition}"

source /mnt/projects_ext4/conda/miniconda3/etc/profile.d/conda.sh
conda activate beyondmimic
export PYTHONPATH="/home/warner/IsaacLab/source/isaaclab:${PYTHONPATH:-}"

python - "$URDF" "$BASE_MESH" "$OUTPUT_MESH" "$KEEP_SUPPORT_ONLY_MESH" "$OUTPUT_USD" "$USD_BUILD_DIR" "$BIKE_MESH_COLLISION_APPROXIMATION" <<'PY'
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
output_usd_path = Path(sys.argv[5]).resolve()
usd_build_dir = Path(sys.argv[6]).resolve()
bike_mesh_collision_approximation = sys.argv[7]


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

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

converter_cfg = UrdfConverterCfg(
    asset_path=str(urdf_path),
    usd_dir=str(usd_build_dir),
    usd_file_name="bikered_import.usd",
    fix_base=False,
    merge_fixed_joints=True,
    joint_drive=None,
    collision_from_visuals=False,
    collider_type="convex_hull",
)
converter = UrdfConverter(converter_cfg)
source_base_usd = Path(converter.usd_path).parent / "configuration" / f"{Path(converter.usd_path).stem}_base.usd"
source_stage = Usd.Stage.Open(str(source_base_usd))
target_stage = Usd.Stage.CreateNew(str(output_usd_path))
UsdGeom.SetStageUpAxis(target_stage, UsdGeom.Tokens.z)
target_stage.SetMetadata("metersPerUnit", 1.0)

root = UsdGeom.Xform.Define(target_stage, "/bikered")
target_stage.SetDefaultPrim(root.GetPrim())
body = UsdGeom.Xform.Define(target_stage, "/bikered/bike_root")
visuals = UsdGeom.Xform.Define(target_stage, "/bikered/bike_root/visuals")
collisions = UsdGeom.Xform.Define(target_stage, "/bikered/bike_root/collisions")
UsdGeom.Scope.Define(target_stage, "/meshes")

source_layer = source_stage.GetRootLayer()
target_layer = target_stage.GetRootLayer()

for child in source_stage.GetPrimAtPath("/meshes").GetChildren():
    Sdf.CopySpec(source_layer, child.GetPath(), target_layer, Sdf.Path("/meshes").AppendChild(child.GetName()))

for child in source_stage.GetPrimAtPath("/visuals/bike_root").GetChildren():
    Sdf.CopySpec(source_layer, child.GetPath(), target_layer, visuals.GetPath().AppendChild(child.GetName()))

copied_collider_keys = set()
for child in source_stage.GetPrimAtPath("/colliders/bike_root").GetChildren():
    gprim_type = ""
    for descendant in Usd.PrimRange(child):
        if descendant.IsA(UsdGeom.Gprim):
            gprim_type = descendant.GetTypeName()
            break
    key = (
        gprim_type,
        str(child.GetAttribute("xformOp:translate").Get()),
        str(child.GetAttribute("xformOp:orient").Get()),
        str(child.GetAttribute("xformOp:scale").Get()),
    )
    if key in copied_collider_keys:
        continue
    copied_collider_keys.add(key)
    Sdf.CopySpec(source_layer, child.GetPath(), target_layer, collisions.GetPath().AppendChild(child.GetName()))

UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
mass_api.CreateMassAttr(8.6)
mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.08, 0.08, 0.08))
mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

for prim in target_stage.Traverse():
    if prim.GetPath().HasPrefix(collisions.GetPath()) and prim.IsA(UsdGeom.Gprim):
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.GetTypeName() == "Mesh":
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            # A single convex hull fills the bike frame's empty space; decomposition keeps the collision closer to the STL.
            mesh_collision.CreateApproximationAttr(bike_mesh_collision_approximation)

target_stage.Save()
simulation_app.close()

print(f"[OK] wrote compound USD {output_usd_path}", flush=True)
print(f"[OK] bike mesh collision approximation={bike_mesh_collision_approximation}", flush=True)
PY
