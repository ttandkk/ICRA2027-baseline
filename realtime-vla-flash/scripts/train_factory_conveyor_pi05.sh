#!/usr/bin/env bash
#SBATCH --job-name=factory_pi05
#SBATCH --partition=cluster02
#SBATCH --output=/projects/haitian003ssd/ICRA2027-baseline/realtime-vla-flash/logs/factory_pi05_%j.log
#SBATCH --error=/projects/haitian003ssd/ICRA2027-baseline/realtime-vla-flash/logs/factory_pi05_%j.err
#SBATCH --time=48:00:00
#SBATCH --gpus=l40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
# Fine-tune the main π0.5 policy. Run normalization once before the first run.
set -euo pipefail

REPO_ROOT=${OPENPI_REPO_ROOT:-/projects/haitian003ssd/ICRA2027-baseline/realtime-vla-flash}
cd "${REPO_ROOT}"

RUN_NAME=${RUN_NAME:-factory_conveyor_pi05_${SLURM_JOB_ID:-local}}
STEPS=${STEPS:-80000}
BATCH_SIZE=${BATCH_SIZE:-32}
SAVE_INTERVAL=${SAVE_INTERVAL:-40000}
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
# The factory dataset is not stored in this repository. Set this to its mounted
# location when submitting, e.g. DATASET_ROOT=/path/to/factory_conveyor_level2_seeded.
DATASET_ROOT=${DATASET_ROOT:-/projects/haitian003ssd/Dataset/factory_conveyor_level2_seeded_old}

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "DATASET_ROOT does not exist or is not a directory: ${DATASET_ROOT}" >&2
  exit 1
fi

export FACTORY_CONVEYOR_DATASET_ROOT="${DATASET_ROOT}"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in the job environment before submitting}"
export WANDB_MODE=online

# Use a current W&B client because the lockfile version only accepts legacy API keys.
if [[ ! -f "${REPO_ROOT}/assets/pi05_factory_conveyor_lora/local/factory_conveyor_level2_seeded/norm_stats.json" ]]; then
  uv run --with "wandb>=0.22.0" "${REPO_ROOT}/scripts/compute_norm_stats.py" --config-name pi05_factory_conveyor_lora
fi

# `uv run` can rebuild the environment; apply the model-specific transformers patch to the package it will execute.
TRANSFORMERS_DIR=$(uv run --with "wandb>=0.22.0" python -c "import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)")
cp -r "${REPO_ROOT}/src/openpi/models_pytorch/transformers_replace/"* "${TRANSFORMERS_DIR}/"

uv run --with "wandb>=0.22.0" torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${REPO_ROOT}/scripts/train_pytorch.py" pi05_factory_conveyor_lora \
  --exp-name="${RUN_NAME}" \
  --num-train-steps="${STEPS}" \
  --batch-size="${BATCH_SIZE}" \
  --save-interval="${SAVE_INTERVAL}" \
  --overwrite
