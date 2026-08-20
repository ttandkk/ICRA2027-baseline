#!/bin/bash

# --- Slurm config ---
#SBATCH --job-name=pi05_train
#SBATCH --output=/projects/haitian003ssd/ICRA2027-baseline/pi05/logs/pi05_train_%j.log
#SBATCH --error=/projects/haitian003ssd/ICRA2027-baseline/pi05/logs/pi05_train_%j.err
#SBATCH --time=72:00:00
#SBATCH --gpus=pro6000:1
#SBATCH --cpus-per-task=8

set -euo pipefail

# --- Paths on SSD ---
BASE_DIR=/projects/haitian003ssd/ICRA2027-baseline
LEROBOT_DIR=${BASE_DIR}/lerobot
PI05_DIR=${BASE_DIR}/pi05
TRAIN_SCRIPT=${PI05_DIR}/train_p05.py
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
  "${PI05_DIR}/logs" \
  "${PIP_CACHE_DIR}" \
  "${TORCH_HOME}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}" \
  "${HF_LEROBOT_HOME}" \
  "${JOB_HOME}"

# --- Environment setup ---
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate "${CONDA_ENV}"

cd "${LEROBOT_DIR}"

# Set INSTALL_EXTRAS=1 only when you intentionally want to refresh dependencies.
if [[ "${INSTALL_EXTRAS:-0}" == "1" ]]; then
  python -m pip install -e ".[training,pi]"
fi

# --- Training config ---
export DATASET_ROOT=${DATASET_ROOT:-/projects/haitian003ssd/Dataset/factory_conveyor_level2_seeded}
export DATASET_REPO_ID=${DATASET_REPO_ID:-local/factory_conveyor_level2_seeded}
export POLICY_PRETRAINED_PATH=${POLICY_PRETRAINED_PATH:-${PI05_DIR}/pretrained/pi05_base}
export OUTPUT_DIR=${OUTPUT_DIR:-${PI05_DIR}/train_outputs/pi05_level_level2_${SLURM_JOB_ID:-local}}
export JOB_NAME=${JOB_NAME:-pi05_level_level2}
export LEROBOT_TRAIN_BIN=${LEROBOT_TRAIN_BIN:-${CONDA_ENV}/bin/lerobot-train}

export DEVICE=${DEVICE:-cuda}
export BATCH_SIZE=${BATCH_SIZE:-32}
export STEPS=${STEPS:-80000}
export NUM_WORKERS=${NUM_WORKERS:-4}
export LOG_FREQ=${LOG_FREQ:-20}
export SAVE_FREQ=${SAVE_FREQ:-40000}
export EVAL_FREQ=${EVAL_FREQ:-0}
export SEED=${SEED:-1000}
export WANDB_ENABLE=${WANDB_ENABLE:-true}
export WANDB_PROJECT=${WANDB_PROJECT:-pi05}

export COMPILE_MODEL=${COMPILE_MODEL:-true}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
export DTYPE=${DTYPE:-bfloat16}
export FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER:-false}
export TRAIN_EXPERT_ONLY=${TRAIN_EXPERT_ONLY:-true}
export USE_RELATIVE_ACTIONS=${USE_RELATIVE_ACTIONS:-false}
if [[ -z "${RELATIVE_EXCLUDE_JOINTS:-}" ]]; then
  export RELATIVE_EXCLUDE_JOINTS="[\"gripper\"]"
else
  export RELATIVE_EXCLUDE_JOINTS
fi
export PUSH_TO_HUB=${PUSH_TO_HUB:-false}

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

# PI05 uses QUANTILES normalization for state/action, which requires q01/q99 stats.
# Converted v2.1 datasets may only have min/max/mean/std, so add quantiles locally once.
AUGMENT_QUANTILE_STATS=${AUGMENT_QUANTILE_STATS:-auto}
if [[ "${AUGMENT_QUANTILE_STATS}" != "0" && "${AUGMENT_QUANTILE_STATS}" != "false" ]]; then
  python "${PI05_DIR}/augment_pi05_quantile_stats.py" \
    --dataset-root="${DATASET_ROOT}"
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "Training PI05"
echo "  dataset:      ${DATASET_ROOT}"
echo "  pretrained:   ${POLICY_PRETRAINED_PATH}"
echo "  output_dir:   ${OUTPUT_DIR}"
echo "  steps:        ${STEPS}"
echo "  batch_size:   ${BATCH_SIZE}"
echo "  dtype:        ${DTYPE}"
echo "  grad_ckpt:    ${GRADIENT_CHECKPOINTING}"
echo "  compile:      ${COMPILE_MODEL}"

python -u "${TRAIN_SCRIPT}" "$@"
