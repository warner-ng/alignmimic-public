#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PREP_SCRIPT="${DATA_PREP_SCRIPT:-$ROOT_DIR/run_cari4d_bike_data_preparation.sh}"

[[ -f "$DATA_PREP_SCRIPT" ]] || { echo "[ERROR] missing DATA_PREP_SCRIPT=$DATA_PREP_SCRIPT" >&2; exit 1; }
RUN_MODE=config source "$DATA_PREP_SCRIPT"

TASK="${TASK:-Tracking-Flat-G1-Bike-HOI-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
LOGGER="${LOGGER:-wandb}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-carrying_bike_rack}"
RUN_NAME="${RUN_NAME:-carrying_bike_rack_g1_hoi}"
RECORD_VIDEO="${RECORD_VIDEO:-1}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
VIDEO_INTERVAL_ITERATIONS="${VIDEO_INTERVAL_ITERATIONS:-500}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
DISABLE_OBJECT_LOADING="${DISABLE_OBJECT_LOADING:-0}"
START_AT_ZERO_ON_RESAMPLE="${START_AT_ZERO_ON_RESAMPLE:-1}"

[[ -f "$HUMAN_MOTION" ]] || { echo "[ERROR] missing HUMAN_MOTION=$HUMAN_MOTION; run data preparation first." >&2; exit 1; }
[[ -f "$OBJECT_MOTION" ]] || { echo "[ERROR] missing OBJECT_MOTION=$OBJECT_MOTION; run data preparation first." >&2; exit 1; }

echo "[INFO] training with prepared motion data"
echo "[INFO] HUMAN_MOTION=${HUMAN_MOTION}"
echo "[INFO] HUMAN_PKL=${HUMAN_PKL}"
echo "[INFO] OBJECT_MOTION=${OBJECT_MOTION}"
echo "[INFO] object scale=${OBJECT_SCALE} rot=${OBJECT_ROOT_ROT_ROLL_DEG},${OBJECT_ROOT_ROT_PITCH_DEG},${OBJECT_ROOT_ROT_YAW_DEG} pos=${OBJECT_ROOT_POS_OFFSET_X},${OBJECT_ROOT_POS_OFFSET_Y},${OBJECT_ROOT_POS_OFFSET_Z}"
echo "[INFO] pair rot=${HUMAN_OBJECT_ROOT_ROT_ROLL_DEG},${HUMAN_OBJECT_ROOT_ROT_PITCH_DEG},${HUMAN_OBJECT_ROOT_ROT_YAW_DEG} pos=${HUMAN_OBJECT_ROOT_TRANS_X},${HUMAN_OBJECT_ROOT_TRANS_Y},${HUMAN_OBJECT_ROOT_TRANS_Z}"
echo "[INFO] human rot=${HUMAN_ROOT_ROT_ROLL_DEG},${HUMAN_ROOT_ROT_PITCH_DEG},${HUMAN_ROOT_ROT_YAW_DEG} spawn_z=${HUMAN_SPAWN_Z_BIAS} object_spawn_z=${OBJECT_SPAWN_Z_BIAS}"

source /mnt/projects_ext4/conda/miniconda3/etc/profile.d/conda.sh
conda activate beyondmimic

cd "$ROOT_DIR"
TRAIN_ARGS=(
  scripts/rsl_rl/train.py
  --task "$TASK"
  --motion_file "$HUMAN_MOTION"
  --object_motion_file "$OBJECT_MOTION"
  --num_envs "$NUM_ENVS"
  --headless
  --logger "$LOGGER"
  --log_project_name "$LOG_PROJECT_NAME"
  --run_name "$RUN_NAME"
  --object_scale "$OBJECT_SCALE"
  --object_root_z_bias "$OBJECT_SPAWN_Z_BIAS"
  --object_root_pos_offset "$OBJECT_ROOT_POS_OFFSET_X" "$OBJECT_ROOT_POS_OFFSET_Y" "$OBJECT_ROOT_POS_OFFSET_Z"
  --object_root_rot_offset_deg "$OBJECT_ROOT_ROT_ROLL_DEG" "$OBJECT_ROOT_ROT_PITCH_DEG" "$OBJECT_ROOT_ROT_YAW_DEG"
  --human_root_rot_offset_deg "$HUMAN_ROOT_ROT_ROLL_DEG" "$HUMAN_ROOT_ROT_PITCH_DEG" "$HUMAN_ROOT_ROT_YAW_DEG"
  --motion_global_rot_offset_deg "$HUMAN_OBJECT_ROOT_ROT_ROLL_DEG" "$HUMAN_OBJECT_ROOT_ROT_PITCH_DEG" "$HUMAN_OBJECT_ROOT_ROT_YAW_DEG"
  --motion_global_pos_offset "$HUMAN_OBJECT_ROOT_TRANS_X" "$HUMAN_OBJECT_ROOT_TRANS_Y" "$HUMAN_OBJECT_ROOT_TRANS_Z"
)

if [[ "$RECORD_VIDEO" == "1" ]]; then
  TRAIN_ARGS+=(--video --video_interval "$VIDEO_INTERVAL" --video_interval_iterations "$VIDEO_INTERVAL_ITERATIONS" --video_length "$VIDEO_LENGTH")
fi
if [[ "$START_AT_ZERO_ON_RESAMPLE" == "1" ]]; then
  TRAIN_ARGS+=(--start_at_zero_on_resample)
fi
if [[ -n "$MAX_ITERATIONS" ]]; then
  TRAIN_ARGS+=(--max_iterations "$MAX_ITERATIONS")
fi
if [[ "$DISABLE_OBJECT_LOADING" == "1" ]]; then
  TRAIN_ARGS+=(--disable_object_loading)
fi

python "${TRAIN_ARGS[@]}"
