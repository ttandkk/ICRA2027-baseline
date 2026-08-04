#!/bin/bash
#SBATCH --job-name=factory_draft
#SBATCH --partition=cluster02
#SBATCH --output=logs/factory_draft_%j.log
#SBATCH --error=logs/factory_draft_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail
REPO=/projects/hdd/ssd/ICLR2027/baseline/realtime-vla-flash
cd "${REPO}"
export UV_CACHE_DIR=/projects/hdd/ssd
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

# Submit after the main job completes, for example:
# MAIN_CHECKPOINT=checkpoints/pi05_factory_conveyor_lora/factory_conveyor_pi05_<main_job>/<step> \
# sbatch --dependency=afterok:<main_job> jobs/train_factory_conveyor_draft.slurm
: "${MAIN_CHECKPOINT:?Export MAIN_CHECKPOINT before submitting this job.}"
export RUN_NAME="factory_conveyor_pi05"
bash scripts/train_factory_conveyor_draft.sh
