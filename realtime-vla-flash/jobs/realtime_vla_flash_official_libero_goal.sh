#!/bin/bash

#SBATCH --job-name=flash_goal_eval
#SBATCH --partition=cluster02
#SBATCH --output=logs/flash_goal_eval_%j.log
#SBATCH --error=logs/flash_goal_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --gpus=6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

REPO=/projects/hdd/ssd/ICRA2027/baseline/realtime-vla-flash
BASE_ARTIFACT="$REPO/data/triton/pi0_libero_goal/base"
DRAFT_ARTIFACT="$REPO/data/triton/pi0_libero_goal/draft"
RESULT_DIR="$REPO/results/real_libero_goal"
PORT=8012

cd "$REPO"
module load git-lfs/3.6.1
export UV_CACHE_DIR=/projects/hdd/ssd
export HF_HOME="$REPO/data/huggingface"
export OPENPI_DATA_HOME="$REPO/data/checkpoints/openpi"
source .venv/bin/activate

mkdir -p "$RESULT_DIR" logs

echo "=== Start official Quick Start policy server ==="
python -u scripts/spec/spec_serve_policy.py \
    --host 127.0.0.1 \
    --port "$PORT" \
    --config pi0_libero \
    --base-triton-path "$BASE_ARTIFACT" \
    --draft-triton-path "$DRAFT_ARTIFACT" \
    --task-suite-name libero_goal \
    --backend triton \
    --hf-endpoint https://huggingface.co \
    > "logs/flash_goal_eval_server_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "server_pid=$SERVER_PID"
sleep 10
kill -0 "$SERVER_PID"

echo "=== Run official Quick Start LIBERO client: Goal task 0, one rollout ==="
MUJOCO_GL=egl \
PYTHONPATH="$REPO/third_party/libero:$REPO/packages/openpi-client" \
timeout 50m "$REPO/examples/libero/.venv/bin/python" -u scripts/spec/spec_client_libero.py \
    --host 127.0.0.1 \
    --port "$PORT" \
    --task-suite-name libero_goal \
    --task 0 \
    --num-trials-per-task 1 \
    --video-out-path "$RESULT_DIR" \
    --run-name "official_client_${SLURM_JOB_ID}"

echo "=== Official client summary ==="
cat "$RESULT_DIR/official_client_${SLURM_JOB_ID}/summary.json"
