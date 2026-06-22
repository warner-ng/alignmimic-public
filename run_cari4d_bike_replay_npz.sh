#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_MODE=config source "$ROOT_DIR/run_cari4d_bike_data_preparation.sh"

source /mnt/projects_ext4/conda/miniconda3/etc/profile.d/conda.sh
conda activate beyondmimic

REPLAY_ARGS=(
  scripts/replay_npz.py
  --motion_file "${HUMAN_MOTION}"
  --human_pkl "${HUMAN_PKL}"
  --motion_quat_order xyzw
  --debug_marker_frames "${DEBUG_MARKER_FRAMES:-5}"
  --object_motion_file "${OBJECT_MOTION}"
  --object_usd "${OBJECT_USD}"
  --object_urdf "${OBJECT_URDF}"
  --object_mesh "${OBJECT_MESH}"
  --object_scale "${OBJECT_SCALE}"
  --object_mesh_scale "${OBJECT_SCALE}"
  --human_root_z_bias "${VIEWER_HUMAN_ROOT_Z_BIAS}"
  --object_root_z_bias "${VIEWER_OBJECT_ROOT_Z_BIAS}"
  --object_root_pos_offset "${OBJECT_ROOT_POS_OFFSET_X}" "${OBJECT_ROOT_POS_OFFSET_Y}" "${OBJECT_ROOT_POS_OFFSET_Z}"
  --object_root_rot_offset_deg "${OBJECT_ROOT_ROT_ROLL_DEG}" "${OBJECT_ROOT_ROT_PITCH_DEG}" "${OBJECT_ROOT_ROT_YAW_DEG}"
  --human_root_rot_offset_deg "${HUMAN_ROOT_ROT_ROLL_DEG}" "${HUMAN_ROOT_ROT_PITCH_DEG}" "${HUMAN_ROOT_ROT_YAW_DEG}"
  --motion_global_rot_offset_deg "${HUMAN_OBJECT_ROOT_ROT_ROLL_DEG}" "${HUMAN_OBJECT_ROOT_ROT_PITCH_DEG}" "${HUMAN_OBJECT_ROOT_ROT_YAW_DEG}"
  --motion_global_pos_offset "${HUMAN_OBJECT_ROOT_TRANS_X}" "${HUMAN_OBJECT_ROOT_TRANS_Y}" "${HUMAN_OBJECT_ROOT_TRANS_Z}"
)

if [[ "${ENABLE_RUNTIME_PAIR_LEVELING}" == "1" ]]; then
  REPLAY_ARGS+=(--enable_runtime_pair_leveling --runtime_pair_level_target_z "${RUNTIME_PAIR_LEVEL_TARGET_Z}")
fi

python "${REPLAY_ARGS[@]}" "$@"
