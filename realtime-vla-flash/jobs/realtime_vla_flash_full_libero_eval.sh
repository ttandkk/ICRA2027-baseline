#!/bin/bash

#SBATCH --job-name=flash_full_libero
#SBATCH --partition=cluster02
#SBATCH --output=logs/flash_full_libero_%j.log
#SBATCH --error=logs/flash_full_libero_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --gpus=6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

REPO=/projects/hdd/ssd/ICRA2027/baseline/realtime-vla-flash
BASE_ARTIFACT="$REPO/data/triton/pi0_libero_goal/base"
DRAFT_CHECKPOINT_ROOT="$REPO/data/checkpoints/draft"
ARTIFACT_ROOT="$REPO/data/triton/full_libero"
RESULT_ROOT="$REPO/results/full_libero_${SLURM_JOB_ID}"
PORT=8013

cd "$REPO"
module load git-lfs/3.6.1
export UV_CACHE_DIR=/projects/hdd/ssd
export HF_HOME="$REPO/data/huggingface"
export OPENPI_DATA_HOME="$REPO/data/checkpoints/openpi"
export LIBERO_CONFIG_PATH="$REPO/data/libero_config"
source .venv/bin/activate

mkdir -p "$ARTIFACT_ROOT" "$RESULT_ROOT" logs

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}
trap cleanup_server EXIT

echo "=== Full LIBERO FLASH evaluation ==="
echo "job_id=$SLURM_JOB_ID node=$(hostname) result_root=$RESULT_ROOT"
nvidia-smi

for SUITE_SUFFIX in spatial object goal 10; do
    SUITE="libero_${SUITE_SUFFIX}"
    DRAFT_CHECKPOINT="$DRAFT_CHECKPOINT_ROOT/draft_${SUITE}.pt"
    DRAFT_ARTIFACT="$ARTIFACT_ROOT/$SUITE/draft"
    SERVER_LOG="logs/flash_full_libero_${SLURM_JOB_ID}_${SUITE}_server.log"

    echo "=== Prepare $SUITE draft artifact ==="
    mkdir -p "$DRAFT_ARTIFACT"
    if [[ ! -f "$DRAFT_ARTIFACT/draft_triton.pkl" ]]; then
        python -u scripts/spec/triton/convert_for_triton.py \
            --mode draft \
            --draft-ckpt "$DRAFT_CHECKPOINT" \
            --output "$DRAFT_ARTIFACT"
    fi

    echo "=== Start $SUITE policy server ==="
    python -u scripts/spec/spec_serve_policy.py \
        --host 127.0.0.1 \
        --port "$PORT" \
        --config pi0_libero \
        --base-triton-path "$BASE_ARTIFACT" \
        --draft-triton-path "$DRAFT_ARTIFACT" \
        --task-suite-name "$SUITE" \
        --backend triton \
        --hf-endpoint https://huggingface.co \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    sleep 10
    kill -0 "$SERVER_PID"

    echo "=== Evaluate $SUITE: 10 tasks x 50 rollouts ==="
    MUJOCO_GL=egl \
    PYTHONPATH="$REPO/third_party/libero:$REPO/packages/openpi-client" \
    "$REPO/examples/libero/.venv/bin/python" -u scripts/spec/spec_client_libero.py \
        --host 127.0.0.1 \
        --port "$PORT" \
        --task-suite-name "$SUITE" \
        --num-trials-per-task 50 \
        --video-out-path "$RESULT_ROOT" \
        --run-name "$SUITE"

    echo "=== Completed $SUITE ==="
    cleanup_server
done

echo "=== Full LIBERO evaluation completed ==="
find "$RESULT_ROOT" -name episode_log.json -print
