#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GMR_ROOT="${GMR_ROOT:-/home/warner/_projects/GMR}"
CARI4D_ROOT="${CARI4D_ROOT:-$ROOT_DIR/third_party/cari4d_clean}"
CARI4D_PYTHON="${CARI4D_PYTHON:-/home/warner/miniconda3/envs/cari4d/bin/python}"
GMR_PYTHON="${GMR_PYTHON:-/home/warner/miniconda3/envs/gmr/bin/python}"
LOCAL_ASSETS_DIR="${LOCAL_ASSETS_DIR:-$ROOT_DIR/assets}"

TAG="${TAG:-Date03_Sub01_bike_May_31_19_34}"
SPLIT="${SPLIT:-in}"
PAIR_SUFFIX="${PAIR_SUFFIX:-bikez}"
CARI4D_PTH="${CARI4D_PTH:-$CARI4D_ROOT/output/opt/cari4d-release+step031397_demo-hy3d3-optv2/Flat_Sub01_bike.pth}"
MOTION_DIR="${MOTION_DIR:-$ROOT_DIR/artifacts/${TAG}_motions}"
HUMAN_MOTION_DIR="${HUMAN_MOTION_DIR:-$ROOT_DIR/artifacts/${TAG}_g1}"

HUMAN_MOTION="${HUMAN_MOTION:-$HUMAN_MOTION_DIR/motion.npz}"
HUMAN_PKL="${HUMAN_PKL:-$MOTION_DIR/${TAG}_human_upright_${PAIR_SUFFIX}_aligned.pkl}"
OBJECT_MOTION="${OBJECT_MOTION:-$MOTION_DIR/${TAG}_object_upright_${PAIR_SUFFIX}_aligned.npz}"
OBJECT_URDF="${OBJECT_URDF:-$LOCAL_ASSETS_DIR/bike_cari4d/bicycle_top_tube/bikered.urdf}"
OBJECT_USD="${OBJECT_USD:-$LOCAL_ASSETS_DIR/bike_cari4d/bicycle_top_tube/bikered_rigid_compound.usd}"
OBJECT_MESH="${OBJECT_MESH:-$LOCAL_ASSETS_DIR/bike_cari4d/bikered.stl}"
PRE_HUMAN="$MOTION_DIR/${TAG}_smplx_input.npz"
HUMAN_PAIR="$MOTION_DIR/${TAG}_human_upright_${PAIR_SUFFIX}.pkl"
OBJECT_PAIR="$MOTION_DIR/${TAG}_object_upright_${PAIR_SUFFIX}.npz"
ALIGNED_HUMAN="$MOTION_DIR/${TAG}_human_upright_${PAIR_SUFFIX}_aligned.pkl"
ALIGNED_OBJECT="$MOTION_DIR/${TAG}_object_upright_${PAIR_SUFFIX}_aligned.npz"

# 修改顺序：先改 object scale，再改 object rpy，这样更容易保证 motion 方向合理。
# rpy：从法向轴看向平面，顺时针为正。
# 红轴=X=roll，绿轴=Y=pitch，蓝轴=Z=yaw。
#
# object scale 统一缩放系数：同时作用于 IsaacLab 与 viser。
OBJECT_SCALE="${OBJECT_SCALE:-0.5}"

# object 单独 rpy 旋转。
# 沿 object root 当前局部轴旋转；内部取当前三根局部轴生成 axis-angle quaternion，不走 Euler 角。
OBJECT_ROOT_ROT_ROLL_DEG="${OBJECT_ROOT_ROT_ROLL_DEG:--110.0}" # object 局部红轴 / X / roll
OBJECT_ROOT_ROT_PITCH_DEG="${OBJECT_ROOT_ROT_PITCH_DEG:-60.0}" # object 局部绿轴 / Y / pitch
OBJECT_ROOT_ROT_YAW_DEG="${OBJECT_ROOT_ROT_YAW_DEG:--20.0}" # object 局部蓝轴 / Z / yaw

# object 单独 xyz 平移。
# 沿 object root 的局部坐标轴平移；IsaacLab 和 viser 同时生效。
OBJECT_ROOT_POS_OFFSET_X="${OBJECT_ROOT_POS_OFFSET_X:--0.5}" # object 局部红轴 / X
OBJECT_ROOT_POS_OFFSET_Y="${OBJECT_ROOT_POS_OFFSET_Y:--0.25}" # object 局部绿轴 / Y
OBJECT_ROOT_POS_OFFSET_Z="${OBJECT_ROOT_POS_OFFSET_Z:--0.3}" # object 局部蓝轴 / Z

# human + object 总体 rpy 旋转。
# pair-level：绕世界坐标系同时旋转 human/object 的 root 位置和姿态。
HUMAN_OBJECT_ROOT_ROT_ROLL_DEG="${HUMAN_OBJECT_ROOT_ROT_ROLL_DEG:--85.0}"
HUMAN_OBJECT_ROOT_ROT_PITCH_DEG="${HUMAN_OBJECT_ROOT_ROT_PITCH_DEG:-90.0}" # 世界绿轴 / Y / pitch
HUMAN_OBJECT_ROOT_ROT_YAW_DEG="${HUMAN_OBJECT_ROOT_ROT_YAW_DEG:-0.0}" # 世界蓝轴 / Z / yaw

# human + object 总体 xyz 平移。
# pair-level 全局平移，同时作用于 human/object。
# RUNTIME_PAIR_LEVEL_TARGET_Z 仅控制 pair leveling 后的落地高度（若开启 leveling）。
HUMAN_OBJECT_ROOT_TRANS_X="${HUMAN_OBJECT_ROOT_TRANS_X:--3.0}"
HUMAN_OBJECT_ROOT_TRANS_Y="${HUMAN_OBJECT_ROOT_TRANS_Y:-0.0}"
HUMAN_OBJECT_ROOT_TRANS_Z="${HUMAN_OBJECT_ROOT_TRANS_Z:-1.0}"

# human 单独 rpy，一般不用修，修物体即可。
# 这组只修机器人 root 姿态；IsaacLab 和 viser 同时生效。
HUMAN_ROOT_ROT_ROLL_DEG="${HUMAN_ROOT_ROT_ROLL_DEG:-0.0}"
HUMAN_ROOT_ROT_PITCH_DEG="${HUMAN_ROOT_ROT_PITCH_DEG:-0.0}"
HUMAN_ROOT_ROT_YAW_DEG="${HUMAN_ROOT_ROT_YAW_DEG:-0.0}"

# IsaacLab 生成高度。
# OBJECT_SPAWN_Z_BIAS 只影响 IsaacLab reset 时真实 object actor 的 spawn 高度，不改 motion/ref 高度。
HUMAN_SPAWN_Z_BIAS="${HUMAN_SPAWN_Z_BIAS:-0.0}"
OBJECT_SPAWN_Z_BIAS="${OBJECT_SPAWN_Z_BIAS:-0.2}"
VIEWER_HUMAN_ROOT_Z_BIAS="${VIEWER_HUMAN_ROOT_Z_BIAS:-0.0}"
VIEWER_OBJECT_ROOT_Z_BIAS="${VIEWER_OBJECT_ROOT_Z_BIAS:-0.0}"
ENABLE_RUNTIME_PAIR_LEVELING="${ENABLE_RUNTIME_PAIR_LEVELING:-0}"
RUNTIME_PAIR_LEVEL_TARGET_Z="${RUNTIME_PAIR_LEVEL_TARGET_Z:-0.0}"
SHOW_GROUND="${SHOW_GROUND:-1}"

RUN_MODE="${RUN_MODE:-viewer}" # all | retarget | viewer | config
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
LOGGER="${LOGGER:-wandb}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-carrying_bike_rack}"
RUN_NAME="${RUN_NAME:-carrying_bike_rack_g1_hoi}"
RECORD_VIDEO="${RECORD_VIDEO:-1}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
VIDEO_INTERVAL_ITERATIONS="${VIDEO_INTERVAL_ITERATIONS:-500}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
# whether or not load object (debug)
DISABLE_OBJECT_LOADING="${DISABLE_OBJECT_LOADING:-0}"
# 0 means no, 1 means from 1st frame
START_AT_ZERO_ON_RESAMPLE="${START_AT_ZERO_ON_RESAMPLE:-1}" 
VIEWER_PORT="${VIEWER_PORT:-8080}"
VIEWER_PORT_START="${VIEWER_PORT_START:-$VIEWER_PORT}"
VIEWER_PORT_END="${VIEWER_PORT_END:-8099}"
VIEWER_HOST="${VIEWER_HOST:-0.0.0.0}"
NO_SERVER="${NO_SERVER:-0}"
VIEWER_REQUIRE_EXACT_PORT="${VIEWER_REQUIRE_EXACT_PORT:-0}"

if [[ "${RUN_MODE}" == "config" ]]; then
  return 0 2>/dev/null || exit 0
fi

echo "[INFO] RUN_MODE=${RUN_MODE}"
echo "[INFO] HUMAN_MOTION_NPZ=${HUMAN_MOTION}"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] HUMAN_PKL=${HUMAN_PKL}"
echo "[INFO] OBJECT_MOTION=${OBJECT_MOTION}"
echo "[INFO] object scale=${OBJECT_SCALE} rot=${OBJECT_ROOT_ROT_ROLL_DEG},${OBJECT_ROOT_ROT_PITCH_DEG},${OBJECT_ROOT_ROT_YAW_DEG} pos=${OBJECT_ROOT_POS_OFFSET_X},${OBJECT_ROOT_POS_OFFSET_Y},${OBJECT_ROOT_POS_OFFSET_Z}"
echo "[INFO] pair rot=${HUMAN_OBJECT_ROOT_ROT_ROLL_DEG},${HUMAN_OBJECT_ROOT_ROT_PITCH_DEG},${HUMAN_OBJECT_ROOT_ROT_YAW_DEG} pos=${HUMAN_OBJECT_ROOT_TRANS_X},${HUMAN_OBJECT_ROOT_TRANS_Y},${HUMAN_OBJECT_ROOT_TRANS_Z}"
echo "[INFO] human rot=${HUMAN_ROOT_ROT_ROLL_DEG},${HUMAN_ROOT_ROT_PITCH_DEG},${HUMAN_ROOT_ROT_YAW_DEG} spawn_z=${HUMAN_SPAWN_Z_BIAS} object_spawn_z=${OBJECT_SPAWN_Z_BIAS}"
echo "[INFO] disable_object_loading=${DISABLE_OBJECT_LOADING}"
echo "[INFO] start_at_zero_on_resample=${START_AT_ZERO_ON_RESAMPLE}"
echo "[INFO] object_far termination threshold=1.0; human tracking terminations remain enabled"

viewer_port_in_use() {
  ss -ltn "sport = :$1" | awk 'NR > 1 { found = 1 } END { exit found ? 0 : 1 }'
}

find_available_tcp_port() {
  local port="${1:-8080}"
  local max_port="${2:-8099}"

  while (( port <= max_port )); do
    if ! viewer_port_in_use "$port"; then
      echo "$port"
      return 0
    fi
    port=$((port + 1))
  done

  return 1
}

case "${RUN_MODE}" in
  all)
    echo "[INFO] preparing motion data..."
    RUN_MODE=retarget bash "$ROOT_DIR/run_cari4d_bike_data_preparation.sh"
    echo "[INFO] launching viewer for offset tuning..."
    RUN_MODE=viewer bash "$ROOT_DIR/run_cari4d_bike_data_preparation.sh"
    ;;
  retarget)
    [[ -f "$ROOT_DIR/scripts/export_cari4d_intermediate.py" ]] || { echo "[ERROR] missing export_cari4d_intermediate.py" >&2; exit 1; }
    [[ -f "$ROOT_DIR/scripts/retarget_smplx_to_g1.py" ]] || { echo "[ERROR] missing retarget_smplx_to_g1.py" >&2; exit 1; }
    [[ -f "$ROOT_DIR/scripts/export_beyondmimic_motion_npz.py" ]] || { echo "[ERROR] missing export_beyondmimic_motion_npz.py" >&2; exit 1; }
    [[ -f "$ROOT_DIR/assets/motions/carry.pkl" ]] || { echo "[ERROR] missing $ROOT_DIR/assets/motions/carry.pkl" >&2; exit 1; }
    [[ -f "$CARI4D_PTH" ]] || { echo "[ERROR] missing CARI4D_PTH=$CARI4D_PTH" >&2; exit 1; }
    [[ -x "$CARI4D_PYTHON" ]] || { echo "[ERROR] CARI4D_PYTHON is not executable: $CARI4D_PYTHON" >&2; exit 1; }
    [[ -x "$GMR_PYTHON" ]] || { echo "[ERROR] GMR_PYTHON is not executable: $GMR_PYTHON" >&2; exit 1; }
    mkdir -p "$MOTION_DIR" "$HUMAN_MOTION_DIR"
    cd "$ROOT_DIR"

    echo "[1/5] CARI4D pth -> smplx/object intermediate"
    "$CARI4D_PYTHON" "$ROOT_DIR/scripts/export_cari4d_intermediate.py" \
      --cari4d_pth "$CARI4D_PTH" \
      --split "$SPLIT" \
      --tag "$TAG" \
      --project_root "$ROOT_DIR" \
      --cari4d_root "$CARI4D_ROOT" \
      --motion_dir "$MOTION_DIR" \
      --pair_suffix "$PAIR_SUFFIX"

    echo "[2/5] GMR smplx -> G1 pkl"
    "$GMR_PYTHON" "$ROOT_DIR/scripts/retarget_smplx_to_g1.py" \
      --tag "$TAG" \
      --robot unitree_g1 \
      --tgt_fps 30 \
      --project_root "$ROOT_DIR" \
      --gmr_root "$GMR_ROOT" \
      --motion_dir "$MOTION_DIR" \
      --pair_suffix "$PAIR_SUFFIX"

    echo "[3/5] verify paired motion"
    [[ -f "$PRE_HUMAN" ]] || { echo "[ERROR] missing $PRE_HUMAN" >&2; exit 1; }
    [[ -f "$HUMAN_PAIR" ]] || { echo "[ERROR] missing $HUMAN_PAIR" >&2; exit 1; }
    [[ -f "$OBJECT_PAIR" ]] || { echo "[ERROR] missing $OBJECT_PAIR" >&2; exit 1; }
    "$GMR_PYTHON" "$ROOT_DIR/scripts/compare_gmr_root_motion.py" \
      --pre-human "$PRE_HUMAN" \
      --post-human "$HUMAN_PAIR" \
      --object "$OBJECT_PAIR"

    echo "[4/5] copy paired motion to aligned names"
    cp -f "$HUMAN_PAIR" "$ALIGNED_HUMAN"
    cp -f "$OBJECT_PAIR" "$ALIGNED_OBJECT"
    echo "[OK] aligned human: $ALIGNED_HUMAN"
    echo "[OK] aligned object: $ALIGNED_OBJECT"

    echo "[5/5] export BeyondMimic motion.npz"
    source /mnt/projects_ext4/conda/miniconda3/etc/profile.d/conda.sh
    conda activate beyondmimic
    python "$ROOT_DIR/scripts/export_beyondmimic_motion_npz.py" \
      --human_pkl "$ALIGNED_HUMAN" \
      --output "$HUMAN_MOTION" \
      --whole_body_tracking_root "$ROOT_DIR" \
      --root_quat_order xyzw \
      --output_quat_order xyzw \
      --headless
    echo "[OK] BeyondMimic motion: $HUMAN_MOTION"
    ;;
  viewer)
    if [[ "${NO_SERVER}" != "1" && "${VIEWER_REQUIRE_EXACT_PORT}" == "1" ]] && viewer_port_in_use "${VIEWER_PORT}"; then
      echo "[ERROR] viewer port ${VIEWER_PORT} is already in use; refusing random viser fallback." >&2
      echo "[HINT] kill the old viewer process or run with VIEWER_PORT=<free_port>." >&2
      exit 1
    fi
    if [[ "${NO_SERVER}" != "1" && "${VIEWER_REQUIRE_EXACT_PORT}" != "1" ]]; then
      if ! VIEWER_PORT="$(find_available_tcp_port "${VIEWER_PORT_START}" "${VIEWER_PORT_END}")"; then
        echo "[ERROR] no available viewer port in ${VIEWER_PORT_START}-${VIEWER_PORT_END}." >&2
        exit 1
      fi
      if [[ "${VIEWER_PORT}" != "${VIEWER_PORT_START}" ]]; then
        echo "[WARN] ${VIEWER_PORT_START} is busy, viewer will use port ${VIEWER_PORT}."
      fi
    fi
    VIEWER_ARGS=(
      scripts/play_bike_hoi_viser.py
      --human "${HUMAN_PKL}"
      --object "${OBJECT_MOTION}"
      --object-urdf "${OBJECT_URDF}"
      --object-mesh "${OBJECT_MESH}"
      --human-root-rot-roll-deg "${HUMAN_ROOT_ROT_ROLL_DEG}"
      --human-root-rot-pitch-deg "${HUMAN_ROOT_ROT_PITCH_DEG}"
      --human-root-rot-yaw-deg "${HUMAN_ROOT_ROT_YAW_DEG}"
      --object-root-rot-roll-deg "${OBJECT_ROOT_ROT_ROLL_DEG}"
      --object-root-rot-pitch-deg "${OBJECT_ROOT_ROT_PITCH_DEG}"
      --object-root-rot-yaw-deg "${OBJECT_ROOT_ROT_YAW_DEG}"
      --object-root-local-rot-roll-deg 0.0
      --object-root-local-rot-pitch-deg 0.0
      --object-root-local-rot-yaw-deg 0.0
      --object-root-pos-offset-x "${OBJECT_ROOT_POS_OFFSET_X}"
      --object-root-pos-offset-y "${OBJECT_ROOT_POS_OFFSET_Y}"
      --object-root-pos-offset-z "${OBJECT_ROOT_POS_OFFSET_Z}"
      --pair-root-rot-roll-deg "${HUMAN_OBJECT_ROOT_ROT_ROLL_DEG}"
      --pair-root-rot-pitch-deg "${HUMAN_OBJECT_ROOT_ROT_PITCH_DEG}"
      --pair-root-rot-yaw-deg "${HUMAN_OBJECT_ROOT_ROT_YAW_DEG}"
      --pair-root-pos-offset-x "${HUMAN_OBJECT_ROOT_TRANS_X}"
      --pair-root-pos-offset-y "${HUMAN_OBJECT_ROOT_TRANS_Y}"
      --pair-root-pos-offset-z "${HUMAN_OBJECT_ROOT_TRANS_Z}"
      --human-root-z-bias "${VIEWER_HUMAN_ROOT_Z_BIAS}"
      --object-root-z-bias "${VIEWER_OBJECT_ROOT_Z_BIAS}"
      --object-mesh-scale "${OBJECT_SCALE}"
      --host "${VIEWER_HOST}"
      --port "${VIEWER_PORT}"
    )
    if [[ "${ENABLE_RUNTIME_PAIR_LEVELING}" == "1" ]]; then
      VIEWER_ARGS+=(--enable-runtime-pair-leveling --runtime-pair-level-target-z "${RUNTIME_PAIR_LEVEL_TARGET_Z}")
    fi
    if [[ "${SHOW_GROUND}" == "1" ]]; then
      VIEWER_ARGS+=(--show-ground)
    fi
    if [[ "${NO_SERVER}" == "1" ]]; then
      VIEWER_ARGS+=(--no-server)
    fi
    if [[ "${NO_SERVER}" != "1" ]]; then
      echo "[INFO] viewer url=http://localhost:${VIEWER_PORT}/"
    fi
    /home/warner/miniconda3/envs/rl-motion/bin/python "${VIEWER_ARGS[@]}"
    ;;
  *)
    echo "[ERROR] RUN_MODE must be all, retarget, viewer, or config, got: ${RUN_MODE}" >&2
    exit 2
    ;;
esac
