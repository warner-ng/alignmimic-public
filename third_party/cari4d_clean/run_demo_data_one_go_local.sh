#!/usr/bin/env bash

# see data/videogen for the actual preparation result of your custom data
#  It works best when both the person and object are mostly visible

set -euo pipefail



# shared
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VIDEO_IN="${PROJECT_ROOT}/flat_bike.mov"
SEQ_NAME="flat_bike"

# step 1: SAM3 masks
SAM3_ENV="sam3"
MASKS_ROOT="${PROJECT_ROOT}/data/cari4d-demo/videogen/masks"
MASKS_OUT="${MASKS_ROOT}/${SEQ_NAME}_masks_k0.h5"
SAM3_VIS_OUT="${MASKS_ROOT}/${SEQ_NAME}_sam3_vis.mp4"
HUMAN_PROMPT="person"
OBJECT_PROMPT="bicycle"
CHUNK_SIZE=300

# step 2: Hunyuan3D mesh
HY3D_ENV="hy3d"
HY3D_ROOT="${PROJECT_ROOT}/data/cari4d-demo/meshes"
HY3D_FRAME_INDEX=330 # NOTE: if first frame is occluded, should use later frame to reconstuct better
HY3D_FRAME_TAG="$(printf '%03d' "${HY3D_FRAME_INDEX}")"
HY3D_OUT_DIR="${HY3D_ROOT}/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba"
HY3D_RGBA_OUT="${HY3D_OUT_DIR}/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba.png"
HY3D_GLB_OUT="${HY3D_OUT_DIR}/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba.glb"
HY3D_OBJ_OUT="${HY3D_OUT_DIR}/${SEQ_NAME}_${HY3D_FRAME_TAG}_align.obj"
BLENDER_PATH="/home/warner/tools/blender-3.6.17-linux-x64/blender"
GLB2OBJ_SCRIPT="${PROJECT_ROOT}/prep/glb2obj.py"

# step 3: Sapiens 2D keypoints
CARI4D_ENV="cari4d"
PACKED_ROOT="${PROJECT_ROOT}/data/cari4d-demo/videogen/packed"
SAPIENS_PACKED_OUT="${PACKED_ROOT}/${SEQ_NAME}_GT-packed.pkl"

# step 4: full CARI4D pipeline
CARI4D_DOCKER="cari4d"
PIPELINE_SEQ_NAME="Flat_Sub01_bike"
WILD_VIDEO_DIR="${PROJECT_ROOT}/data/cari4d-demo/wild/videos"
PIPELINE_VIDEO="${WILD_VIDEO_DIR}/${PIPELINE_SEQ_NAME}.0.color.mp4"
PIPELINE_VIDEO_DIR="${WILD_VIDEO_DIR}"
PIPELINE_ALIGNED_DIR="${PROJECT_ROOT}/data/cari4d-demo/wild/videos-aligned"
PIPELINE_ALIGNED_VIDEO="${PIPELINE_ALIGNED_DIR}/${PIPELINE_SEQ_NAME}.0.color.mp4"
PIPELINE_HY3D_ROOT="${PROJECT_ROOT}/data/cari4d-demo/videogen/meshes"
PIPELINE_HY3D_LINK_DIR="${PIPELINE_HY3D_ROOT}/${PIPELINE_SEQ_NAME}_${HY3D_FRAME_TAG}_rgba"
PIPELINE_HY3D_LINK="${PIPELINE_HY3D_LINK_DIR}/${PIPELINE_SEQ_NAME}_${HY3D_FRAME_TAG}_align.obj"
PIPELINE_HY3D_METRIC_ROOT="${PIPELINE_HY3D_ROOT}-metric"
PIPELINE_MASKS_OUT="${MASKS_ROOT}/${PIPELINE_SEQ_NAME}_masks_k0.h5"
PIPELINE_PACKED_OUT="${PACKED_ROOT}/${PIPELINE_SEQ_NAME}_GT-packed.pkl"
PIPELINE_DEPTH_OUT="${PIPELINE_VIDEO_DIR}/${PIPELINE_SEQ_NAME}.0.depth-reg.mp4"
PIPELINE_INTRINSICS_OUT="${PIPELINE_VIDEO_DIR}/${PIPELINE_SEQ_NAME}.0.color.pkl"
PIPELINE_NLF_ROOT="${PROJECT_ROOT}/data/cari4d-demo/videogen/nlf"
PIPELINE_NLF_OUT="${PIPELINE_NLF_ROOT}/${PIPELINE_SEQ_NAME}_params.pkl"
PIPELINE_NLF_OPT_ROOT="${PIPELINE_NLF_ROOT}-opt"
PIPELINE_NLF_OPT_OUT="${PIPELINE_NLF_OPT_ROOT}/${PIPELINE_SEQ_NAME}_params.pkl"
PIPELINE_ALIGNED_DEPTH_OUT="${PIPELINE_ALIGNED_DIR}/${PIPELINE_SEQ_NAME}.0.depth-reg.mp4"
PIPELINE_SCALE_OUT="${PIPELINE_HY3D_METRIC_ROOT}/${PIPELINE_SEQ_NAME}_scale.json"
PIPELINE_METRIC_OBJ="${PIPELINE_HY3D_METRIC_ROOT}/${PIPELINE_SEQ_NAME}_${HY3D_FRAME_TAG}_align/${PIPELINE_SEQ_NAME}_${HY3D_FRAME_TAG}_align.obj"
PIPELINE_FP_ROOT="${PROJECT_ROOT}/data/cari4d-demo/videogen/fp-hy3d3-track"
PIPELINE_FP_OUT="${PIPELINE_FP_ROOT}/${PIPELINE_SEQ_NAME}_all.pkl"
PIPELINE_FP_VIZ_FILTER="${PIPELINE_FP_ROOT}/${PIPELINE_SEQ_NAME}_000000-000448_k0_filter_k0.mp4"
PIPELINE_FP_VIZ_RAW="${PIPELINE_FP_ROOT}/${PIPELINE_SEQ_NAME}_all_k0_k0.mp4"
PIPELINE_COCONET_OUT="${PROJECT_ROOT}/output/coconet"
PIPELINE_COCONET_PTH="${PIPELINE_COCONET_OUT}/cari4d-release+step031397_demo/${PIPELINE_SEQ_NAME}.pth"
PIPELINE_OPT_OUT="${PROJECT_ROOT}/output/opt/cari4d-release+step031397_demo-hy3d3-optv2/${PIPELINE_SEQ_NAME}.pth"
ALIGN_RENDER_CHUNK_SIZE=8 # step 4.4 OOM prevention

SCALE_CANDIDATES="${SCALE_CANDIDATES:-30}" # step 4.5 可以提高匹配质量，多送几个尺度匹配候选
RENDER_BATCH="${RENDER_BATCH:-8}" # step 4.5/4.6 OOM prevention; original default is 512
REFINE_MODEL_BATCH="${REFINE_MODEL_BATCH:-128}" # step 4.5/4.6 model forward batch; original default is 1024
SCORE_MODEL_BATCH="${SCORE_MODEL_BATCH:-}" # step 4.5/4.6 score model batch; empty keeps original full-batch behavior

FP_ANGULAR_VELO="${FP_ANGULAR_VELO:-2.5}" # step 4.6 temporal rotation threshold; original default is 2.5; smaller means smoother
FP_OCC_FRAMES_ALLOWED="${FP_OCC_FRAMES_ALLOWED:-15}" # step 4.6 original default is 15; keep temporal filter through short occlusions
FP_MAX_ATTEMPTS="${FP_MAX_ATTEMPTS:-5}" # step 4.6 original default is 5; retry count when temporal filter finds no candidate

COCONET_CLIP_LEN="${COCONET_CLIP_LEN:-90}" # step 4.7 temporal window; original config is 96

OPT_VIZ_BATCH="${OPT_VIZ_BATCH:-64}" # step 4.8 visualization render batch; does not change optimization batch
OPT_TEMP_WEIGHT="${OPT_TEMP_WEIGHT:-1000}" # match the Jun 14 okay output setup






# step 1: prepare SAM3 masks for the video (3 min)
# code and checkpoint download needed (see docs/custom_video.md)
# outputs:
#   ${MASKS_ROOT}/${SEQ_NAME}_masks_k0.h5
#   ${MASKS_ROOT}/${SEQ_NAME}_sam3_vis.mp4

echo "=== Step 1: SAM3 masks ==="

cd "${PROJECT_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${SAM3_ENV}"
set -u

mkdir -p "${MASKS_ROOT}"

if [[ -f "${MASKS_OUT}" ]]; then
  echo "Skip SAM3 masks: ${MASKS_OUT} already exists"
else
  if [[ -z "${HF_TOKEN:-}" ]]; then
    HF_TOKEN="$(python - <<'PY'
from huggingface_hub import get_token
print(get_token() or "")
PY
)"
  fi

  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set. Run: export HF_TOKEN=<your_huggingface_token>"
    exit 1
  fi
  export HF_TOKEN

  python prep/run_sam3_masks.py \
    --video "${VIDEO_IN}" \
    --human_prompt "${HUMAN_PROMPT}" \
    --object_prompt "${OBJECT_PROMPT}" \
    --output_dir "${MASKS_ROOT}" \
    --chunk_size "${CHUNK_SIZE}" \
    --visualize
fi

echo "SAM3 masks ready: ${MASKS_OUT}"
echo "SAM3 visualization ready: ${SAM3_VIS_OUT}"












# step 2: reconstruct object mesh with Hunyuan3D （11 min）
# environment needed: hy3d
# outputs: （example）
#   data/cari4d-demo/meshes/flat_bike_320_rgba/flat_bike_320_rgba.png
#   data/cari4d-demo/meshes/flat_bike_320_rgba/flat_bike_320_rgba.glb
#   data/cari4d-demo/meshes/flat_bike_320_rgba/flat_bike_320_align.obj

echo "=== Step 2: Hunyuan3D mesh ==="

set +u
conda activate "${HY3D_ENV}"
set -u

mkdir -p "${HY3D_ROOT}"

if [[ -f "${HY3D_OBJ_OUT}" ]]; then
  echo "Skip Hunyuan3D: ${HY3D_OBJ_OUT} already exists"
elif [[ -f "${HY3D_GLB_OUT}" ]]; then
  echo "Resume Hunyuan3D: convert existing GLB to OBJ"
  "${BLENDER_PATH}" -b -P "${GLB2OBJ_SCRIPT}" -- "${HY3D_OUT_DIR}" "${HY3D_OUT_DIR}"
  PRODUCED_OBJ="${HY3D_OUT_DIR}/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba.obj"
  if [[ -f "${PRODUCED_OBJ}" ]]; then
    mv "${PRODUCED_OBJ}" "${HY3D_OBJ_OUT}"
  fi
  test -f "${HY3D_OBJ_OUT}"
else
  python prep/run_hy3d_recon.py \
    --video "${VIDEO_IN}" \
    --masks_root "${MASKS_ROOT}" \
    --hy3d_root "${HY3D_ROOT}" \
    --frame_index "${HY3D_FRAME_INDEX}" \
    --blender_path "${BLENDER_PATH}" \
    --skip_glb2obj
  "${BLENDER_PATH}" -b -P "${GLB2OBJ_SCRIPT}" -- "${HY3D_OUT_DIR}" "${HY3D_OUT_DIR}"
  PRODUCED_OBJ="${HY3D_OUT_DIR}/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba.obj"
  if [[ -f "${PRODUCED_OBJ}" ]]; then
    mv "${PRODUCED_OBJ}" "${HY3D_OBJ_OUT}"
  fi
  test -f "${HY3D_OBJ_OUT}"
fi

echo "Hunyuan3D RGBA ready: ${HY3D_RGBA_OUT}"
echo "Hunyuan3D GLB ready: ${HY3D_GLB_OUT}"
echo "Hunyuan3D OBJ ready: ${HY3D_OBJ_OUT}"









# step 3: detect 2D human keypoints with Sapiens (3 min)
# environment needed: cari4d
# outputs:
#   data/cari4d-demo/videogen/packed/flat_bike_GT-packed.pkl

echo "=== Step 3: Sapiens 2D keypoints ==="

set +u
conda activate "${CARI4D_ENV}"
set -u

mkdir -p "${PACKED_ROOT}"

if [[ -f "${SAPIENS_PACKED_OUT}" ]]; then
  echo "Skip Sapiens pose: ${SAPIENS_PACKED_OUT} already exists"
else
  python prep/run_sapiens_pose.py \
    --video "${VIDEO_IN}" \
    --masks_root "${MASKS_ROOT}" \
    --packed_root "${PACKED_ROOT}"
fi

echo "Sapiens packed keypoints ready: ${SAPIENS_PACKED_OUT}"








# step 4: run the full CARI4D pipeline
# environment needed: cari4d
# outputs:
#   data/cari4d-demo/wild/videos/flat_bike.0.color.mp4
#   ${PIPELINE_DEPTH_OUT}
#   ${PIPELINE_NLF_OUT}
#   ${PIPELINE_NLF_OPT_OUT}
#   ${PIPELINE_ALIGNED_VIDEO}
#   ${PIPELINE_SCALE_OUT}
#   ${PIPELINE_FP_OUT}
#   ${PIPELINE_COCONET_PTH}
#   ${PIPELINE_OPT_OUT}

run_pipeline_cmd() {
  docker start "${CARI4D_DOCKER}" >/dev/null
  docker exec -e PYTHONPATH="${PROJECT_ROOT}" -w "${PROJECT_ROOT}" "${CARI4D_DOCKER}" "$@"
}

mkdir -p "${WILD_VIDEO_DIR}"

if [[ -f "${PIPELINE_VIDEO}" ]]; then
  echo "Pipeline MP4 ready: ${PIPELINE_VIDEO}"
else
  ffmpeg -y -i "${VIDEO_IN}" -c:v libx264 -pix_fmt yuv420p -an "${PIPELINE_VIDEO}"
fi

mkdir -p "${PIPELINE_HY3D_LINK_DIR}"
if [[ ! -e "${PIPELINE_HY3D_LINK}" ]]; then
  ln -s "${HY3D_OBJ_OUT}" "${PIPELINE_HY3D_LINK}"
fi
cp -f "${HY3D_OUT_DIR}/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba.mtl" "${PIPELINE_HY3D_LINK_DIR}/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba.mtl"
if ! grep -q '^map_Kd ' "${PIPELINE_HY3D_LINK_DIR}/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba.mtl"; then
  printf 'map_Kd %s\n' "${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba.png" >> "${PIPELINE_HY3D_LINK_DIR}/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba.mtl"
fi
if [[ ! -e "${PIPELINE_HY3D_LINK_DIR}/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba.png" ]]; then
  ln -s "${HY3D_RGBA_OUT}" "${PIPELINE_HY3D_LINK_DIR}/${SEQ_NAME}_${HY3D_FRAME_TAG}_rgba.png"
fi

if [[ ! -e "${PIPELINE_PACKED_OUT}" ]]; then
  ln -s "${SAPIENS_PACKED_OUT}" "${PIPELINE_PACKED_OUT}"
fi

if [[ ! -f "${PIPELINE_MASKS_OUT}" ]]; then
  python - <<PY
import h5py
src = "${MASKS_OUT}"
dst = "${PIPELINE_MASKS_OUT}"
src_group = "${SEQ_NAME}"
dst_group = "${PIPELINE_SEQ_NAME}"
with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
    fout.copy(fin[src_group], dst_group)
PY
fi

for required_path in \
  "${PROJECT_ROOT}/unidepth" \
  "${PROJECT_ROOT}/VolumetricSMPL" \
  "${PROJECT_ROOT}/weights" \
  "${PROJECT_ROOT}/experiments" \
  "${PROJECT_ROOT}/data/assets" \
  "${PROJECT_ROOT}/data/smpl"
do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required CARI4D dependency: ${required_path}"
    echo "Follow README setup before running Step 4."
    exit 1
  fi
done

# Original black-box call is kept in scripts/demo-custom.sh.bak:
# bash scripts/demo-custom.sh "${PIPELINE_VIDEO}"

# Step 1: run Unidepth estimation
echo "=== Step 4.1: run Unidepth estimation ==="
if [[ -f "${PIPELINE_DEPTH_OUT}" && -f "${PIPELINE_INTRINSICS_OUT}" ]]; then
  echo "Skip Unidepth: ${PIPELINE_DEPTH_OUT} already exists"
else
  run_pipeline_cmd python prep/unidepth_behave.py --wild_video --video "${PIPELINE_VIDEO}" -o "${PIPELINE_VIDEO_DIR}"
fi

# Step 2: run NLF
echo "=== Step 4.2: run NLF ==="
if [[ -f "${PIPELINE_NLF_OUT}" ]]; then
  echo "Skip NLF: ${PIPELINE_NLF_OUT} already exists"
else
  run_pipeline_cmd env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CARI4D_NLF_CHUNK_SIZE=60 python prep/run_nlf_sepK.py -o "${PIPELINE_NLF_ROOT}" --masks_root "${MASKS_ROOT}" --video "${PIPELINE_VIDEO}" --wild_video
fi

# Step 2 (alternative): run SAM3D-body instead of NLF
# python prep/run_sam3d_sepK.py -o "${PIPELINE_NLF_ROOT}" --masks_root "${MASKS_ROOT}" --video "${PIPELINE_VIDEO}" --wild_video

# Step 3: run SMPLH fitting to get globally consistent human pose and translation
echo "=== Step 4.3: run SMPLH fitting to get globally consistent human pose and translation ==="
if [[ -f "${PIPELINE_NLF_OPT_OUT}" ]]; then
  echo "Skip SMPLH fitting: ${PIPELINE_NLF_OPT_OUT} already exists"
else
  run_pipeline_cmd python prep/fit_smplh_global.py --wild_video --video "${PIPELINE_VIDEO}" --packed_root "${PACKED_ROOT}" --masks_root "${MASKS_ROOT}" \
    --nlf_path="${PIPELINE_NLF_ROOT}"
fi

# Step 4: align Unidepth to GENMO human
echo "=== Step 4.4: align Unidepth to GENMO human ==="
if [[ -f "${PIPELINE_ALIGNED_VIDEO}" && -f "${PIPELINE_ALIGNED_DEPTH_OUT}" ]]; then
  echo "Skip depth alignment: ${PIPELINE_ALIGNED_VIDEO} already exists"
else
  run_pipeline_cmd python prep/align_monod2hum.py --wild_video --nlf_path "${PIPELINE_NLF_OPT_ROOT}" \
    --masks_root "${MASKS_ROOT}" \
    --video "${PIPELINE_VIDEO}" \
    --render_chunk_size "${ALIGN_RENDER_CHUNK_SIZE}"
fi

# Update the video path, pointing to the new video with aligned depth.

# Step 5: estimate metric scale of the object
echo "=== Step 4.5: estimate metric scale of the object ==="
if [[ -f "${PIPELINE_METRIC_OBJ}" ]]; then
  echo "Skip metric scale: ${PIPELINE_METRIC_OBJ} already exists"
else
  run_pipeline_cmd env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CARI4D_SCALE_CANDIDATES="${SCALE_CANDIDATES}" CARI4D_RENDER_BATCH="${RENDER_BATCH}" CARI4D_MODEL_BATCH="${REFINE_MODEL_BATCH}" CARI4D_SCORE_MODEL_BATCH="${SCORE_MODEL_BATCH}" python tools/estimate_scale_video.py --wild_video --video "${PIPELINE_ALIGNED_VIDEO}" --masks_root "${MASKS_ROOT}" --hy3d_root "${PIPELINE_HY3D_ROOT}" -o "${PIPELINE_HY3D_METRIC_ROOT}"
fi

# Step 6: run FP in tracking mode
echo "=== Step 4.6: run FP in tracking mode ==="
if [[ -f "${PIPELINE_FP_OUT}" ]]; then
  echo "Skip FoundationPose tracking: ${PIPELINE_FP_OUT} already exists"
else
  run_pipeline_cmd env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CARI4D_RENDER_BATCH="${RENDER_BATCH}" CARI4D_MODEL_BATCH="${REFINE_MODEL_BATCH}" CARI4D_SCORE_MODEL_BATCH="${SCORE_MODEL_BATCH}" python prep/fp_hy3d_track.py --viz_path x --wild_video --kid 0 \
    --masks_root "${MASKS_ROOT}" --hy3d_root="${PIPELINE_HY3D_METRIC_ROOT}" \
    --video "${PIPELINE_ALIGNED_VIDEO}" -o "${PIPELINE_FP_ROOT}"
fi
if [[ -f "${PIPELINE_FP_VIZ_FILTER}" ]]; then
  echo "FoundationPose visualization ready: ${PIPELINE_FP_VIZ_FILTER}"
elif [[ -f "${PIPELINE_FP_VIZ_RAW}" ]]; then
  echo "FoundationPose visualization ready: ${PIPELINE_FP_VIZ_RAW}"
else
  echo "FoundationPose visualization not found under ${PIPELINE_FP_ROOT}"
fi

# Step 7: run CoCoNet to refine human + object
echo "=== Step 4.7: run CoCoNet to refine human + object ==="
if [[ -f "${PIPELINE_COCONET_PTH}" ]]; then
  echo "Skip CoCoNet: ${PIPELINE_COCONET_PTH} already exists"
else
  run_pipeline_cmd env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-behave.json \
    use_sel_view=True render_video=True identifier=_demo use_intermediate=False data_name=test-only \
    clip_len="${COCONET_CLIP_LEN}" window="${COCONET_CLIP_LEN}" \
    hy3d_meshes_root="${PIPELINE_HY3D_METRIC_ROOT}" \
    masks_root="${MASKS_ROOT}" \
    fp_root="${PIPELINE_FP_ROOT}" \
    nlf_root="${PIPELINE_NLF_OPT_ROOT}" \
    video="${PIPELINE_ALIGNED_VIDEO}" cam_id=0 wild_video=True \
    outpath="${PIPELINE_COCONET_OUT}"
fi

# Step 8: run joint optimization
echo "=== Step 4.8: run joint optimization ==="
if [[ -f "${PIPELINE_OPT_OUT}" ]]; then
  echo "Skip joint optimization: ${PIPELINE_OPT_OUT} already exists"
else
  run_pipeline_cmd env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CARI4D_OPT_VIZ_BATCH="${OPT_VIZ_BATCH}" python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300 save_name=optv2 batch_size=64 opt_rot=True \
    opt_trans=True w_temp="${OPT_TEMP_WEIGHT}" w_sil=0.002 w_contact=200.0 w_pen=2.0 w_j2d=0.03 opt_smpl_trans=False opt_betas=False \
    pth_file="${PIPELINE_COCONET_PTH}" wild_video=True use_input=True \
    video_root="$(dirname "${PIPELINE_ALIGNED_VIDEO}")" \
    packed_root="${PACKED_ROOT}" \
    masks_root="${MASKS_ROOT}" \
    hy3d_meshes_root="${PIPELINE_HY3D_METRIC_ROOT}" outpath=output/opt
fi

echo "CARI4D pipeline video ready: ${PIPELINE_VIDEO}"
echo "CARI4D aligned video ready: ${PIPELINE_ALIGNED_VIDEO}"
echo "CARI4D CoCoNet ready: ${PIPELINE_COCONET_PTH}"
echo "CARI4D optimized result ready: ${PIPELINE_OPT_OUT}"
