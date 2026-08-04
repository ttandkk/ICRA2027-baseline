#!/bin/bash

# --- Slurm config ---
#SBATCH --job-name=smolvla_train
#SBATCH --output=/projects/hdd/ssd/ICLR2027/baseline/smolvla/logs/smolvla_train_%j.log
#SBATCH --error=/projects/hdd/ssd/ICLR2027/baseline/smolvla/logs/smolvla_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --gpus=pro6000:1
#SBATCH --cpus-per-task=8

set -euo pipefail

# --- Paths on SSD ---
BASE_DIR=/projects/hdd/ssd/ICLR2027/baseline
LEROBOT_DIR=${BASE_DIR}/lerobot
SMOLVLA_DIR=${BASE_DIR}/smolvla
TRAIN_SCRIPT=${SMOLVLA_DIR}/train_smolvla.py
CONDA_ENV=/projects/hdd/ssd/conda/envs/lerobot
CONDA_PKGS_DIRS=/projects/hdd/ssd/conda/pkgs
HF_CACHE=/projects/hdd/ssd/hf_cache
LIBERO_CACHE=${HF_CACHE}/libero
JOB_HOME=${HF_CACHE}/home

export CONDA_PKGS_DIRS
export HF_HOME=${HF_CACHE}
export HF_LEROBOT_HOME=${HF_CACHE}/lerobot
export PIP_CACHE_DIR=${HF_CACHE}/pip
export TORCH_HOME=${HF_CACHE}/torch
export UV_CACHE_DIR=${HF_CACHE}/uv
export XDG_CACHE_HOME=${HF_CACHE}/xdg
export HOME=${JOB_HOME}
export LIBERO_CONFIG_PATH=${LIBERO_CACHE}
export TMPDIR=${HF_CACHE}/tmp
export PYTHONNOUSERSITE=1
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PATH=${CONDA_ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

mkdir -p \
  "${SMOLVLA_DIR}/logs" \
  "${PIP_CACHE_DIR}" \
  "${TORCH_HOME}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}" \
  "${LIBERO_CACHE}" \
  "${LIBERO_CACHE}/datasets" \
  "${JOB_HOME}" \
  "${JOB_HOME}/.cache/libero/assets"

# --- Environment setup ---
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate "${CONDA_ENV}"

cd "${LEROBOT_DIR}"

LIBERO_PKG_DIR="${CONDA_ENV}/lib/python3.12/site-packages/libero/libero"
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  {
    printf "assets: %s\n" "${JOB_HOME}/.cache/libero/assets"
    printf "bddl_files: %s\n" "${LIBERO_PKG_DIR}/bddl_files"
    printf "benchmark_root: %s\n" "${LIBERO_PKG_DIR}"
    printf "datasets: %s\n" "${LIBERO_CACHE}/datasets"
    printf "init_states: %s\n" "${LIBERO_PKG_DIR}/init_files"
  } > "${LIBERO_CONFIG_PATH}/config.yaml"
fi

# Set INSTALL_EXTRAS=1 only when you intentionally want to refresh dependencies.
if [[ "${INSTALL_EXTRAS:-0}" == "1" ]]; then
  python -m pip install -e ".[training,smolvla]"
fi

# --- Training config ---
# Factory conveyor level-2 seeded dataset (already in LeRobot v3.0 format).
export DATASET_ROOT=${DATASET_ROOT:-/projects/hdd/ssd/ICLR2027/dataset/factory_conveyor_level2_seeded}
export DATASET_REPO_ID=${DATASET_REPO_ID:-local/factory_conveyor_level2_seeded}
export POLICY_PATH=${POLICY_PATH:-lerobot/smolvla_base}
export OUTPUT_DIR=${OUTPUT_DIR:-${SMOLVLA_DIR}/train_outputs/smolvla_factory_conveyor_level2_seeded_${SLURM_JOB_ID:-local}}
export JOB_NAME=${JOB_NAME:-smolvla_factory_conveyor_level2_seeded}
export LEROBOT_TRAIN_BIN=${LEROBOT_TRAIN_BIN:-${CONDA_ENV}/bin/lerobot-train}

export DEVICE=${DEVICE:-cuda}
export BATCH_SIZE=${BATCH_SIZE:-32}
export STEPS=${STEPS:-80000}
export NUM_WORKERS=${NUM_WORKERS:-4}
export LOG_FREQ=${LOG_FREQ:-50}
export SAVE_FREQ=${SAVE_FREQ:-40000}
export EVAL_FREQ=${EVAL_FREQ:-0}
export SEED=${SEED:-1000}
export WANDB_ENABLE=${WANDB_ENABLE:-true}
export WANDB_PROJECT=${WANDB_PROJECT:-smolvla}
export INFER_POLICY_FEATURES=${INFER_POLICY_FEATURES:-true}
export PUSH_TO_HUB=${PUSH_TO_HUB:-false}
export FULL_FINETUNE=${FULL_FINETUNE:-false}
export PEFT_LORA=${PEFT_LORA:-false}
export PEFT_R=${PEFT_R:-64}
export PEFT_ALPHA=${PEFT_ALPHA:-64}

# LeRobot v3 expects datasets in v3.0 format. Convert local v2.1 datasets once, in place.
# The converter keeps a backup next to the dataset as <dataset_name>_old.
CONVERT_DATASET=${CONVERT_DATASET:-auto}
DATASET_VERSION=$(sed -n "s/.*\"codebase_version\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "${DATASET_ROOT}/meta/info.json" | head -n 1)
if [[ "${DATASET_VERSION}" == "v2.1" ]]; then
  if [[ "${CONVERT_DATASET}" == "0" || "${CONVERT_DATASET}" == "false" ]]; then
    echo "Dataset is v2.1, but CONVERT_DATASET=${CONVERT_DATASET}; training will fail with current LeRobot." >&2
  else
    echo "Converting dataset from v2.1 to v3.0 at ${DATASET_ROOT}"
    python -m lerobot.scripts.convert_dataset_v21_to_v30 \
      --repo-id="${DATASET_REPO_ID}" \
      --root="${DATASET_ROOT}" \
      --push-to-hub=false
  fi
elif [[ "${DATASET_VERSION}" != "v3.0" ]]; then
  echo "Warning: dataset codebase_version is ${DATASET_VERSION:-unknown}; LeRobot expects v3.0." >&2
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "Training SmolVLA"
echo "  dataset:      ${DATASET_ROOT}"
echo "  policy:       ${POLICY_PATH}"
echo "  output_dir:   ${OUTPUT_DIR}"
echo "  steps:        ${STEPS}"
echo "  batch_size:   ${BATCH_SIZE}"
echo "  peft_lora:    ${PEFT_LORA}"
echo "  full_ft:      ${FULL_FINETUNE}"

python -u "${TRAIN_SCRIPT}" "$@"
