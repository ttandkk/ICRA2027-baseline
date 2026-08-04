#!/bin/bash
#SBATCH --job-name=factory_pi05
#SBATCH --partition=cluster02
#SBATCH --output=logs/factory_pi05_%j.log
#SBATCH --error=logs/factory_pi05_%j.err
#SBATCH --time=48:00:00
#SBATCH --gpus=l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail
REPO=/projects/hdd/ssd/ICLR2027/baseline/realtime-vla-flash
cd "${REPO}"
export UV_CACHE_DIR=/projects/hdd/ssd
export TOKENIZERS_PARALLELISM=false
export RUN_NAME="factory_conveyor_pi05_${SLURM_JOB_ID}"
source .venv/bin/activate
bash scripts/train_factory_conveyor_pi05.sh
