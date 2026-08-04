#!/usr/bin/env bash
#SBATCH --job-name=factory_pi05
#SBATCH --partition=cluster02
#SBATCH --output=/projects/hdd/ssd/ICLR2027/baseline/realtime-vla-flash/logs/factory_pi05_%j.log
#SBATCH --error=/projects/hdd/ssd/ICLR2027/baseline/realtime-vla-flash/logs/factory_pi05_%j.err
#SBATCH --time=48:00:00
#SBATCH --gpus=6000ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# Fine-tune the main π0.5 policy. Run normalization once before the first run.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

RUN_NAME=${RUN_NAME:-factory_conveyor_pi05_${SLURM_JOB_ID:-local}}
STEPS=${STEPS:-80000}
BATCH_SIZE=${BATCH_SIZE:-32}
SAVE_INTERVAL=${SAVE_INTERVAL:-40000}

if [[ ! -f "assets/pi05_factory_conveyor_lora/local/factory_conveyor_level2_seeded_old/norm_stats.json" ]]; then
  uv run scripts/compute_norm_stats.py --config-name pi05_factory_conveyor_lora
fi

uv run scripts/train_pytorch.py pi05_factory_conveyor_lora \
  --exp-name="${RUN_NAME}" \
  --num-train-steps="${STEPS}" \
  --batch-size="${BATCH_SIZE}" \
  --save-interval="${SAVE_INTERVAL}" \
  --overwrite
