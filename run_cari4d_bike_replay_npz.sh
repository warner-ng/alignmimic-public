#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_MODE=config source "$ROOT_DIR/run_cari4d_bike_resmimic.sh"

source /mnt/projects_ext4/conda/miniconda3/etc/profile.d/conda.sh
conda activate beyondmimic

REPLAY_ARGS=(
  scripts/replay_npz.py
  --motion_file "${HUMAN_MOTION}"
  --motion_quat_order xyzw
  --object_motion_file "${OBJECT_MOTION}"
  --object_urdf "${OBJECT_URDF}"
  --object_scale "${OBJECT_SCALE}"
  --object_root_z_bias "${OBJECT_SPAWN_Z_BIAS}"
  --object_root_pos_offset "${OBJECT_ROOT_POS_OFFSET_X}" "${OBJECT_ROOT_POS_OFFSET_Y}" "${OBJECT_ROOT_POS_OFFSET_Z}"
  --object_root_rot_offset_deg "${OBJECT_ROOT_ROT_ROLL_DEG}" "${OBJECT_ROOT_ROT_PITCH_DEG}" "${OBJECT_ROOT_ROT_YAW_DEG}"
  --human_root_rot_offset_deg "${HUMAN_ROOT_ROT_ROLL_DEG}" "${HUMAN_ROOT_ROT_PITCH_DEG}" "${HUMAN_ROOT_ROT_YAW_DEG}"
  --motion_global_rot_offset_deg "${HUMAN_OBJECT_ROOT_ROT_ROLL_DEG}" "${HUMAN_OBJECT_ROOT_ROT_PITCH_DEG}" "${HUMAN_OBJECT_ROOT_ROT_YAW_DEG}"
  --motion_global_pos_offset "${HUMAN_OBJECT_ROOT_TRANS_X}" "${HUMAN_OBJECT_ROOT_TRANS_Y}" "${HUMAN_OBJECT_ROOT_TRANS_Z}"
)

python "${REPLAY_ARGS[@]}" "$@"
