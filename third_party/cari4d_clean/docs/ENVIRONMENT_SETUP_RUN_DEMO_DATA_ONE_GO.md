# Environment setup for `run_demo_data_one_go.sh`

This file documents the environments used by:

```bash
bash /home/warner/_projects/cari4d_clean/run_demo_data_one_go.sh
```

The script uses four runtime contexts:

| Step | Runtime | Used for |
|---|---|---|
| Step 1 | conda env `sam3` | SAM3 text-prompted video masks |
| Step 2 | conda env `hy3d` | Hunyuan3D object reconstruction |
| Step 3 | conda env `cari4d` | Sapiens 2D keypoints |
| Step 4 | Docker container `cari4d` | CARI4D depth, NLF/SMPLH, scale, FoundationPose, CoCoNet, optimization |

Do not merge these into one environment. SAM3, Hunyuan3D, Sapiens/OpenMMLab, and the CARI4D Docker stack use different Python, PyTorch, CUDA, and compiled extension versions.

## 0. System tools

Required outside conda:

```bash
conda
docker
nvidia-container-toolkit
ffmpeg
/home/warner/tools/blender-3.6.17-linux-x64/blender
```

The script also needs:

```bash
export HF_TOKEN=<your_huggingface_token>
```

## 1. SAM3 environment

Used by:

```bash
conda activate sam3
python prep/run_sam3_masks.py ...
```

Current working versions on this machine:

| Package | Version |
|---|---|
| Python | 3.12.13 |
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| sam3 | 0.1.0 from `facebookresearch/sam3` commit `8e451d5eb43c817b64ae7577fb7b9ae223db88a9` |
| opencv-python | 4.10.0.84 |
| h5py | 3.16.0 |
| pycocotools | 2.0.11 |
| einops | 0.8.2 |
| imageio | 2.37.3 |
| psutil | 7.2.2 |
| huggingface_hub | 1.17.0 |

Install:

```bash
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/facebookresearch/sam3.git
cd sam3
git checkout 8e451d5eb43c817b64ae7577fb7b9ae223db88a9
pip install -e .
pip install -r /home/warner/_projects/cari4d_clean/requirements-run-demo-sam3.txt
huggingface-cli login --token "$HF_TOKEN"
```

## 2. Hunyuan3D environment

Used by:

```bash
conda activate hy3d
python prep/run_hy3d_recon.py ...
```

Current working versions on this machine:

| Package | Version |
|---|---|
| Python | 3.10.20 |
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| hy3dgen | 2.0.2 from `Tencent/Hunyuan3D-2` commit `f8db63096c8282cb27354314d896feba5ba6ff8a` |
| diffusers | 0.31.0 |
| transformers | 4.48.0 |
| opencv-python | 4.13.0.92 |
| h5py | 3.16.0 |
| trimesh | 4.12.2 |
| scipy | 1.15.3 |
| numpy | 2.2.6 |

Install:

```bash
conda create -n hy3d python=3.10 -y
conda activate hy3d
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/Tencent/Hunyuan3D-2.git@f8db63096c8282cb27354314d896feba5ba6ff8a
pip install -r /home/warner/_projects/cari4d_clean/requirements-run-demo-hy3d.txt
```

Blender is not a Python package. `run_demo_data_one_go.sh` expects:

```bash
/home/warner/tools/blender-3.6.17-linux-x64/blender
```

## 3. CARI4D conda environment for Sapiens

Used by:

```bash
conda activate cari4d
python prep/run_sapiens_pose.py ...
```

Current working versions on this machine:

| Package | Version |
|---|---|
| Python | 3.10.20 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| torchaudio | 2.6.0+cu124 |
| mmcv | 2.1.0 |
| mmengine | 0.10.7 |
| mmdet | 3.3.0 |
| mmpretrain | 1.2.0 |
| xtcocotools | 1.14.3 |
| json-tricks | 3.17.3 |
| munkres | 1.1.4 |
| opencv-python | 4.12.0.88 |
| h5py | 3.16.0 |
| numpy | 2.2.6 |

Install:

```bash
conda create -n cari4d python=3.10 -y
conda activate cari4d
conda install -c nvidia cuda-nvcc=12.4 cuda-cudart-dev=12.4 cuda-cudart=12.4 cuda-cccl=12.4 cuda-version=12.4 -y
conda install -c conda-forge cxx-compiler -y
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r /home/warner/_projects/cari4d_clean/requirements.txt --no-build-isolation
pip install -r /home/warner/_projects/cari4d_clean/requirements-run-demo-sapiens.txt
```

`mmcv` is installed as `mmcv==2.1.0`, not `mmcv-lite`, in the working env.

Sapiens assets:

```bash
git clone https://github.com/facebookresearch/sapiens.git /home/warner/_projects/cari4d_clean/sapiens
huggingface-cli login --token "$HF_TOKEN"
mkdir -p ~/sapiens_host/pose/checkpoints/sapiens_0.3b
huggingface-cli download noahcao/sapiens-pose-coco \
  sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth \
  --local-dir ~/sapiens_host/pose/checkpoints
```

Also place the COCO joint regressor at:

```bash
/home/warner/_projects/cari4d_clean/data/assets/J_regressor_coco.npy
```

## 4. CARI4D Docker runtime

Used by:

```bash
docker start cari4d
docker exec -e PYTHONPATH=/home/warner/_projects/cari4d_clean \
  -w /home/warner/_projects/cari4d_clean cari4d ...
```

The repo supports pulling the published image:

```bash
docker pull xiexh20/cari4d
docker tag xiexh20/cari4d cari4d
bash docker/run_container.sh
```

Or building from the local Dockerfile:

```bash
cd /home/warner/_projects/cari4d_clean/docker
docker build --network host -t cari4d .
cd ..
bash docker/run_container.sh
```

The local Dockerfile uses:

| Component | Version/source |
|---|---|
| Base image | `nvidia/cuda:12.1.0-devel-ubuntu20.04` |
| Conda env inside image | `my` |
| Python | 3.10 |
| torch | 2.4.1, CUDA 12.1 wheel |
| torchvision | 0.19.1, CUDA 12.1 wheel |
| torchaudio | 2.4.1, CUDA 12.1 wheel |
| numpy | 1.26.3 |
| pybind11 | v2.10.0 |
| Eigen | 3.4.0 |
| pytorch3d | `facebookresearch/pytorch3d@stable` |
| kaolin | `NVIDIAGameWorks/kaolin`, editable install |
| nvdiffrast | `NVlabs/nvdiffrast`, local install |
| chumpy | `mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17` |
| flash-attn | `<2.5` |
| ultralytics | 8.0.120 |

The container is started by `docker/run_container.sh` with GPU access, host networking, X11, `/home`, `/mnt`, and the repo mounted into the same absolute path.

## 5. Extra code, model files, and data required by Step 4

`run_demo_data_one_go.sh` checks these paths before Step 4:

```bash
/home/warner/_projects/cari4d_clean/unidepth
/home/warner/_projects/cari4d_clean/VolumetricSMPL
/home/warner/_projects/cari4d_clean/weights
/home/warner/_projects/cari4d_clean/experiments
/home/warner/_projects/cari4d_clean/data/assets
/home/warner/_projects/cari4d_clean/data/smpl
```

Install code/checkpoints following the repo README:

```bash
git clone https://github.com/lpiccinelli-eth/UniDepth.git
mv UniDepth/unidepth .
rm -rf UniDepth

git clone https://github.com/markomih/VolumetricSMPL.git
cd VolumetricSMPL
git apply ../scripts/volumetric_smplh.patch
find . -maxdepth 1 -type f -delete
mv VolumetricSMPL/*.py .
rm -r VolumetricSMPL
cd ..

mkdir -p weights
wget -O weights/nlf_l_multi_0.3.2.torchscript \
  https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript
```

Manual downloads:

| Path | Content |
|---|---|
| `weights/` | FoundationPose model weights |
| `experiments/cari4d-release/step031397.pth` | CoCoNet checkpoint |
| `data/smpl/smplh/SMPLH_female.pkl` | SMPL-H female model |
| `data/smpl/smplh/SMPLH_male.pkl` | SMPL-H male model |
| `data/smpl/kid_template.npy` | AGORA kid template |
| `data/assets/J_regressor_coco.npy` | COCO joint regressor |
| `data/cari4d-demo/` | CARI4D demo data |

## 6. Run

After all environments, container, checkpoints, and assets exist:

```bash
cd /home/warner/_projects/cari4d_clean
export HF_TOKEN=<your_huggingface_token>
bash run_demo_data_one_go.sh
```
