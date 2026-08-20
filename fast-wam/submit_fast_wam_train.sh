#!/usr/bin/env bash

#SBATCH --job-name=fastwam_train
#SBATCH --output=/projects/haitian003ssd/ICRA2027-baseline/fast-wam/logs/fastwam_train_%j.log
#SBATCH --error=/projects/haitian003ssd/ICRA2027-baseline/fast-wam/logs/fastwam_train_%j.err
#SBATCH --time=72:00:00
#SBATCH --gpus=pro6000:4
#SBATCH --cpus-per-task=16

# Train FastWAM. The preferred dataset is the preserved v2.1 copy; current
# LeRobot only supports v3.0, so the script automatically falls back to the
# converted dataset when the preferred directory cannot be imported.
set -euo pipefail

BASE_DIR=/projects/haitian003ssd/ICRA2027-baseline
LEROBOT_DIR=${BASE_DIR}/lerobot
FASTWAM_DIR=${BASE_DIR}/fast-wam
TRAIN_SCRIPT=${FASTWAM_DIR}/train_fast_wam.py
CONDA_ENV=${BASE_DIR}/.conda/lerobot-baselines
CONDA_PKGS_DIRS=${BASE_DIR}/.cache/conda/pkgs
HF_CACHE=${BASE_DIR}/.cache/hf
JOB_HOME=${HF_CACHE}/home

export CONDA_PKGS_DIRS
export HF_HOME=${HF_CACHE}
export HF_LEROBOT_HOME=${HF_CACHE}/lerobot
export PIP_CACHE_DIR=${HF_CACHE}/pip
export TORCH_HOME=${HF_CACHE}/torch
export UV_CACHE_DIR=${HF_CACHE}/uv
export XDG_CACHE_HOME=${HF_CACHE}/xdg
export HOME=${JOB_HOME}
export TMPDIR=${HF_CACHE}/tmp
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PATH=${CONDA_ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

mkdir -p \
  "${FASTWAM_DIR}/logs" \
  "${PIP_CACHE_DIR}" \
  "${TORCH_HOME}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}" \
  "${HF_LEROBOT_HOME}" \
  "${JOB_HOME}"

module load Miniforge3
eval "$(conda shell.bash hook)"
source activate "${CONDA_ENV}"
cd "${LEROBOT_DIR}"

# FastWAM is available in LeRobot >= 0.6.0. Set INSTALL_EXTRAS=1 only when
# intentionally refreshing the editable installation and its dependencies.
if [[ "${INSTALL_EXTRAS:-0}" == "1" ]]; then
  python -m pip install -e ".[fastwam]"
fi

export DATASET_REPO_ID=${DATASET_REPO_ID:-local/factory_conveyor_level2_seeded}
PRIMARY_DATASET_ROOT=${DATASET_ROOT:-/projects/haitian003ssd/Dataset/factory_conveyor_level2_seeded_old}
FALLBACK_DATASET_ROOT=${FALLBACK_DATASET_ROOT:-/projects/haitian003ssd/Dataset/factory_conveyor_level2_seeded}

dataset_importable() {
  local dataset_root=$1
  "${CONDA_ENV}/bin/python" - "${DATASET_REPO_ID}" "${dataset_root}" <<'PY'
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id, root = sys.argv[1:]
dataset = LeRobotDataset(repo_id, root=root, download_videos=False)
assert len(dataset) > 0, f"Dataset at {root} contains no frames"
PY
}

if dataset_importable "${PRIMARY_DATASET_ROOT}"; then
  export DATASET_ROOT=${PRIMARY_DATASET_ROOT}
else
  echo "Unable to import preferred dataset: ${PRIMARY_DATASET_ROOT}" >&2
  echo "Falling back to: ${FALLBACK_DATASET_ROOT}" >&2
  if ! dataset_importable "${FALLBACK_DATASET_ROOT}"; then
    echo "Neither FastWAM dataset could be imported." >&2
    exit 1
  fi
  export DATASET_ROOT=${FALLBACK_DATASET_ROOT}
fi

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in the job environment before submitting}"
export OUTPUT_DIR=${OUTPUT_DIR:-${FASTWAM_DIR}/train_outputs/fastwam_factory_conveyor_level2_${SLURM_JOB_ID:-local}}
export JOB_NAME=${JOB_NAME:-fastwam_factory_conveyor_level2}
export LEROBOT_TRAIN_BIN=${LEROBOT_TRAIN_BIN:-${CONDA_ENV}/bin/lerobot-train}
export ACCELERATE_BIN=${ACCELERATE_BIN:-${CONDA_ENV}/bin/accelerate}
export DDP_NUM_PROCESSES=${DDP_NUM_PROCESSES:-4}
export DDP_MIXED_PRECISION=${DDP_MIXED_PRECISION:-bf16}
export DISTRIBUTED_TYPE=${DISTRIBUTED_TYPE:-fsdp}
export DEVICE=${DEVICE:-cuda}
export BATCH_SIZE=${BATCH_SIZE:-8}
export STEPS=${STEPS:-80000}
export NUM_WORKERS=${NUM_WORKERS:-0}
export LOG_FREQ=${LOG_FREQ:-50}
export SAVE_FREQ=${SAVE_FREQ:-40000}
export EVAL_FREQ=${EVAL_FREQ:-0}
export SEED=${SEED:-1000}
export WANDB_ENABLE=${WANDB_ENABLE:-true}
export WANDB_PROJECT=${WANDB_PROJECT:-fastwam}
export MODEL_ID=${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}
export IMAGE_KEYS=${IMAGE_KEYS:-observation.images.overview,observation.images.front,observation.images.wrist}
export IMAGE_SIZE=${IMAGE_SIZE:-[224,672]}
export ACTION_HORIZON=${ACTION_HORIZON:-32}
export N_ACTION_STEPS=${N_ACTION_STEPS:-10}
export TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
export USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-true}
export VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}
export PUSH_TO_HUB=${PUSH_TO_HUB:-false}

mkdir -p "$(dirname "${OUTPUT_DIR}")"
echo "Training FastWAM"
echo "  dataset:      ${DATASET_ROOT}"
echo "  output_dir:   ${OUTPUT_DIR}"
echo "  model:        ${MODEL_ID}"
echo "  distributed:  ${DISTRIBUTED_TYPE} (${DDP_NUM_PROCESSES} processes)"
echo "  steps:        ${STEPS}"
echo "  batch_size:   ${BATCH_SIZE}"
echo "  image_size:   ${IMAGE_SIZE}"
echo "  video_backend:${VIDEO_BACKEND}"

python -u "${TRAIN_SCRIPT}" "--dataset.video_backend=${VIDEO_BACKEND}" "$@"
