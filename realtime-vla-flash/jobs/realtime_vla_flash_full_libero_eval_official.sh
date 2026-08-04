#!/bin/bash

#SBATCH --job-name=flash_official
#SBATCH --partition=cluster02
#SBATCH --output=logs/flash_official_%j.log
#SBATCH --error=logs/flash_official_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --gpus=6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

REPO=/projects/hdd/ssd/ICRA2027/baseline/realtime-vla-flash
BASE_ARTIFACT="$REPO/data/triton/pi0_libero_goal/base"
DRAFT_ARTIFACT_ROOT="$REPO/data/triton/full_libero"
RESULT_ROOT="$REPO/results/full_libero_official_${SLURM_JOB_ID}"
PORT=8014

cd "$REPO"
module load git-lfs/3.6.1
export UV_CACHE_DIR=/projects/hdd/ssd
export HF_HOME="$REPO/data/huggingface"
export OPENPI_DATA_HOME="$REPO/data/checkpoints/openpi"
export LIBERO_CONFIG_PATH="$REPO/data/libero_config"
source .venv/bin/activate

mkdir -p "$RESULT_ROOT" logs

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}
trap cleanup_server EXIT

run_suite() {
    local suite="$1"
    local periodic_full="$2"
    local enable_gripper="$3"
    local draft_artifact="$DRAFT_ARTIFACT_ROOT/$suite/draft"
    local server_log="logs/flash_official_${SLURM_JOB_ID}_${suite}_server.log"

    if [[ ! -f "$draft_artifact/draft_triton.pkl" ]]; then
        echo "Missing draft Triton artifact: $draft_artifact/draft_triton.pkl" >&2
        exit 1
    fi

    echo "=== Start official-config server for $suite ==="
    local gripper_flags=()
    if [[ "$enable_gripper" == "true" ]]; then
        gripper_flags+=(--enable-gripper-verify --enable-gripper-post-verify)
    else
        gripper_flags+=(--no-enable-gripper-verify --no-enable-gripper-post-verify)
    fi

    python -u scripts/spec/spec_serve_policy.py \
        --host 127.0.0.1 \
        --port "$PORT" \
        --config pi0_libero \
        --base-triton-path "$BASE_ARTIFACT" \
        --draft-triton-path "$draft_artifact" \
        --task-suite-name "$suite" \
        --backend triton \
        --tau-radius 0.15 \
        --t-list 0.1 0.05 \
        --periodic-full-every-n-draft-rounds "$periodic_full" \
        "${gripper_flags[@]}" \
        --hf-endpoint https://huggingface.co \
        > "$server_log" 2>&1 &
    SERVER_PID=$!

    sleep 10
    kill -0 "$SERVER_PID"

    echo "=== Evaluate $suite with official-config parameters ==="
    MUJOCO_GL=egl \
    PYTHONPATH="$REPO/third_party/libero:$REPO/packages/openpi-client" \
    "$REPO/examples/libero/.venv/bin/python" -u scripts/spec/spec_client_libero.py \
        --host 127.0.0.1 \
        --port "$PORT" \
        --task-suite-name "$suite" \
        --num-trials-per-task 50 \
        --seed 7 \
        --video-out-path "$RESULT_ROOT" \
        --run-name "$suite"

    echo "=== Completed $suite ==="
    cleanup_server
}

echo "=== Official-config full LIBERO FLASH evaluation ==="
echo "job_id=$SLURM_JOB_ID node=$(hostname) result_root=$RESULT_ROOT"
nvidia-smi

run_suite libero_spatial 0 true
run_suite libero_object 0 false
run_suite libero_goal 4 true
run_suite libero_10 2 true

echo "=== Official-config evaluation completed ==="
find "$RESULT_ROOT" -name episode_log.json -print
