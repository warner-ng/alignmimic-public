#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_SCRIPT="${CONFIG_SCRIPT:-$ROOT_DIR/run_cari4d_bike_train.sh}"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-/home/warner/_projects/whole_body_tracking/logs/rsl_rl/g1_flat/2026-06-11_21-56-50_carrying_bike_rack_g1_hoi/model_1500.pt}}"
NUM_ENVS="${NUM_ENVS:-1}"
TASK="${TASK:-Tracking-Flat-G1-Bike-HOI-v0}"
LOGGER="${LOGGER:-tensorboard}"
HEADLESS="${HEADLESS:-0}"
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"

# 这里负责关闭物体加载
DISABLE_OBJECT_LOADING="${DISABLE_OBJECT_LOADING:-0}"

[[ -f "$CONFIG_SCRIPT" ]] || { echo "[ERROR] missing CONFIG_SCRIPT=$CONFIG_SCRIPT" >&2; exit 1; }
[[ -f "$CHECKPOINT_PATH" ]] || { echo "[ERROR] missing CHECKPOINT_PATH=$CHECKPOINT_PATH" >&2; exit 1; }

RUN_MODE=config source "$CONFIG_SCRIPT"

RUN_DIR="$(basename "$(dirname "$CHECKPOINT_PATH")")"
CHECKPOINT_FILE="$(basename "$CHECKPOINT_PATH")"

source /mnt/projects_ext4/conda/miniconda3/etc/profile.d/conda.sh
conda activate beyondmimic

PLAY_ARGS=(
  scripts/rsl_rl/play.py
  --task="$TASK"
  --num_envs="$NUM_ENVS"
  --load_run="$RUN_DIR"
  --checkpoint="$CHECKPOINT_FILE"
  --motion_file="$HUMAN_MOTION"
  --object_motion_file="$OBJECT_MOTION"
  --object_scale="$OBJECT_SCALE"
  --object_root_z_bias="$OBJECT_SPAWN_Z_BIAS"
  --object_root_pos_offset "$OBJECT_ROOT_POS_OFFSET_X" "$OBJECT_ROOT_POS_OFFSET_Y" "$OBJECT_ROOT_POS_OFFSET_Z"
  --object_root_rot_offset_deg "$OBJECT_ROOT_ROT_ROLL_DEG" "$OBJECT_ROOT_ROT_PITCH_DEG" "$OBJECT_ROOT_ROT_YAW_DEG"
  --human_root_rot_offset_deg "$HUMAN_ROOT_ROT_ROLL_DEG" "$HUMAN_ROOT_ROT_PITCH_DEG" "$HUMAN_ROOT_ROT_YAW_DEG"
  --motion_global_rot_offset_deg "$HUMAN_OBJECT_ROOT_ROT_ROLL_DEG" "$HUMAN_OBJECT_ROOT_ROT_PITCH_DEG" "$HUMAN_OBJECT_ROOT_ROT_YAW_DEG"
  --motion_global_pos_offset "$HUMAN_OBJECT_ROOT_TRANS_X" "$HUMAN_OBJECT_ROOT_TRANS_Y" "$HUMAN_OBJECT_ROOT_TRANS_Z"
  --logger="$LOGGER"
  env.terminations.anchor_pos=null
  env.terminations.anchor_ori=null
  env.terminations.ee_body_pos=null
  env.terminations.object_far=null
)

if [[ "$DISABLE_OBJECT_LOADING" == "1" ]]; then
  PLAY_ARGS+=(--disable_object_loading)
fi

if [[ "$HEADLESS" == "1" ]]; then
  PLAY_ARGS+=(--headless)
fi

if [[ "$VIDEO" == "1" ]]; then
  PLAY_ARGS+=(--video --video_length "$VIDEO_LENGTH")
fi

cd "$ROOT_DIR"
python "${PLAY_ARGS[@]}"
