#!/bin/bash

#SBATCH --job-name=flash_real_goal
#SBATCH --partition=cluster02
#SBATCH --output=logs/flash_real_goal_%j.log
#SBATCH --error=logs/flash_real_goal_%j.err
#SBATCH --time=02:00:00
#SBATCH --gpus=pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail

REPO=/projects/hdd/ssd/ICRA2027/baseline/realtime-vla-flash
JAX_CHECKPOINT="$REPO/data/checkpoints/openpi/openpi-assets/checkpoints/pi0_libero"
DRAFT_CHECKPOINT="$REPO/data/checkpoints/draft/draft_libero_goal.pt"
ARTIFACT_ROOT="$REPO/data/triton/pi0_libero_goal"
BASE_ARTIFACT="$ARTIFACT_ROOT/base"
DRAFT_ARTIFACT="$ARTIFACT_ROOT/draft"
RESULT_DIR="$REPO/results/real_libero_goal"
PORT=8011

cd "$REPO"
module load git-lfs/3.6.1
export UV_CACHE_DIR=/projects/hdd/ssd
export HF_HOME="$REPO/data/huggingface"
export OPENPI_DATA_HOME="$REPO/data/checkpoints/openpi"
export PYTHONPATH="$REPO"
source .venv/bin/activate

mkdir -p "$BASE_ARTIFACT" "$DRAFT_ARTIFACT" "$RESULT_DIR" logs

echo "=== Job and GPU ==="
echo "job_id=$SLURM_JOB_ID node=$(hostname)"
nvidia-smi

if [[ ! -f "$BASE_ARTIFACT/base_weights.pkl" ]]; then
    echo "=== Convert real pi0_libero base checkpoint ==="
    python -u scripts/spec/triton/convert_for_triton.py \
        --mode base \
        --jax-path "$JAX_CHECKPOINT" \
        --output "$BASE_ARTIFACT"
fi

if [[ ! -f "$DRAFT_ARTIFACT/draft_triton.pkl" ]]; then
    echo "=== Convert official LIBERO Goal draft checkpoint ==="
    python -u scripts/spec/triton/convert_for_triton.py \
        --mode draft \
        --draft-ckpt "$DRAFT_CHECKPOINT" \
        --output "$DRAFT_ARTIFACT"
fi

echo "=== Start real-weight LIBERO Goal FLASH server ==="
python -u scripts/spec/spec_serve_policy.py \
    --host 127.0.0.1 \
    --port "$PORT" \
    --config pi0_libero \
    --base-triton-path "$BASE_ARTIFACT" \
    --draft-triton-path "$DRAFT_ARTIFACT" \
    --task-suite-name libero_goal \
    --backend triton \
    --hf-endpoint https://huggingface.co \
    > "logs/flash_real_goal_server_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "server_pid=$SERVER_PID"
sleep 10
kill -0 "$SERVER_PID"

echo "=== Run real-weight FLASH inference benchmark ==="
timeout 60m python -u scripts/spec/real_libero_benchmark_client.py \
    --host 127.0.0.1 \
    --port "$PORT" \
    --runs 20 \
    --prompt "open the middle drawer of the cabinet" \
    --output "$RESULT_DIR/benchmark_${SLURM_JOB_ID}.json"

echo "=== Result ==="
cat "$RESULT_DIR/benchmark_${SLURM_JOB_ID}.json"
