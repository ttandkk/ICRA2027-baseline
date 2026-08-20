#!/bin/bash
#SBATCH --job-name=factory_pi05
#SBATCH --partition=cluster02
#SBATCH --output=logs/factory_pi05_%j.log
#SBATCH --error=logs/factory_pi05_%j.err
#SBATCH --time=48:00:00
#SBATCH --gpus=l40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G

set -euo pipefail
REPO=${OPENPI_REPO_ROOT:-/projects/haitian003ssd/ICRA2027-baseline/realtime-vla-flash}
cd "${REPO}"
export UV_CACHE_DIR=/projects/haitian003ssd/.cache/uv
export TOKENIZERS_PARALLELISM=false
export RUN_NAME="factory_conveyor_pi05_${SLURM_JOB_ID}"
source "${REPO}/.venv/bin/activate"
bash "${REPO}/scripts/train_factory_conveyor_pi05.sh"
