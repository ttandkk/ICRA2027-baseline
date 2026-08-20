#!/usr/bin/env bash
#SBATCH --job-name=act_train
#SBATCH --output=/projects/haitian003ssd/ICRA2027-baseline/ACT/logs/act_train_%j.log
#SBATCH --error=/projects/haitian003ssd/ICRA2027-baseline/ACT/logs/act_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --gpus=pro6000:1
#SBATCH --cpus-per-task=8

# Train ACT from scratch with LeRobot. Override any exported variable or append
# native lerobot-train flags when submitting, e.g. `sbatch ... --policy.dim_model=256`.
set -euo pipefail

BASE_DIR=/projects/haitian003ssd/ICRA2027-baseline
LEROBOT_DIR=${BASE_DIR}/lerobot
ACT_DIR=${BASE_DIR}/ACT
CONDA_ENV=${BASE_DIR}/.conda/lerobot-baselines
HF_CACHE=${BASE_DIR}/.cache/hf

export HF_HOME=${HF_CACHE}
export HF_LEROBOT_HOME=${HF_CACHE}/lerobot
export PIP_CACHE_DIR=${HF_CACHE}/pip
export TORCH_HOME=${HF_CACHE}/torch
export XDG_CACHE_HOME=${HF_CACHE}/xdg
export TMPDIR=${HF_CACHE}/tmp
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PATH=${CONDA_ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

mkdir -p "${ACT_DIR}/logs" "${HF_HOME}" "${HF_LEROBOT_HOME}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
cd "${LEROBOT_DIR}"

export DATASET_ROOT=${DATASET_ROOT:-/projects/haitian003ssd/Dataset/factory_conveyor_level2_seeded}
export DATASET_REPO_ID=${DATASET_REPO_ID:-local/factory_conveyor_level2_seeded}
export OUTPUT_DIR=${OUTPUT_DIR:-${ACT_DIR}/train_outputs/act_factory_conveyor_level2_seeded_${SLURM_JOB_ID}}
export JOB_NAME=${JOB_NAME:-act_factory_conveyor_level2_seeded}
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
export WANDB_PROJECT=${WANDB_PROJECT:-act}
export CHUNK_SIZE=${CHUNK_SIZE:-100}
export N_ACTION_STEPS=${N_ACTION_STEPS:-100}
export GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}
export PUSH_TO_HUB=${PUSH_TO_HUB:-false}

if ! grep -q '"codebase_version"[[:space:]]*:[[:space:]]*"v3.0"' "${DATASET_ROOT}/meta/info.json"; then
  echo "Expected a LeRobot v3.0 dataset at ${DATASET_ROOT}; conversion is required before training." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"
echo "Training ACT: dataset=${DATASET_ROOT}, steps=${STEPS}, batch_size=${BATCH_SIZE}, chunk_size=${CHUNK_SIZE}"
python -u "${ACT_DIR}/train_act.py" "$@"
