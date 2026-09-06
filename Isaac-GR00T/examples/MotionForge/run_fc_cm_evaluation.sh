#!/usr/bin/env bash
#SBATCH --job-name=gr00t_fc_cm_smoke
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/gr00t_fc_cm_smoke_%j.out
#SBATCH --error=logs/gr00t_fc_cm_smoke_%j.err

set -Eeuo pipefail

if [[ -n "${GROOT_SMOKE_SCRIPT_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd -- "${GROOT_SMOKE_SCRIPT_DIR}" && pwd -P)"
elif [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
GROOT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${GROOT_ROOT}/../.." && pwd -P)"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"

FC_RUNNER="${SCRIPT_DIR}/run_fc001_fc009_evaluation.sh"
CM_RUNNER="${SCRIPT_DIR}/run_cm000_cm009_evaluation.sh"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_ROOT}/.venv/bin/python}"
MOTIONFORGE_PYTHON="${MOTIONFORGE_PYTHON:-${WORKSPACE_ROOT}/isaacsim/python.sh}"
GROOT_BACKBONE_MODEL_PATH="${GROOT_BACKBONE_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/nvidia/Cosmos-Reason2-2B}"
GROOT_FFMPEG_PREFIX="${GROOT_FFMPEG_PREFIX:-${WORKSPACE_ROOT}/ICRA2027-baseline/.conda/lerobot-baselines}"
FC_MODEL_PATH="${FC_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/gr00t1.7-FC-80000}"
CM_MODEL_PATH="${CM_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/gr00t1.7-CM-80000}"

NUM_TRIALS="${GROOT_SMOKE_NUM_TRIALS:-50}"
ATTEMPTS_PER_WORKER="${GROOT_SMOKE_ATTEMPTS_PER_WORKER:-50}"
START_SEED="${GROOT_SMOKE_START_SEED:-0}"
ACCEL_MODE="${GROOT_SMOKE_ACCEL_MODE:-trt_full_pipeline}"
CUDA_DEVICE="${GROOT_SMOKE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_DIRECT_RUN="${ALLOW_DIRECT_RUN:-0}"
BATCH_ID="${GROOT_SMOKE_BATCH_ID:-groot_fc_cm_smoke_$(date +%Y%m%d_%H%M%S)}"
FC_RUN_ID="${BATCH_ID}_fc"
CM_RUN_ID="${BATCH_ID}_cm"

log() {
  printf '[GR00T-FC-CM-SMOKE] %s\n' "$*"
}

die() {
  printf '[GR00T-FC-CM-SMOKE] ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "required file not found: ${path}"
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || die "required directory not found: ${path}"
}

require_executable() {
  local path="$1"
  [[ -x "${path}" ]] || die "required executable not found or not executable: ${path}"
}

[[ "${NUM_TRIALS}" =~ ^[0-9]+$ ]] && ((10#${NUM_TRIALS} >= 1)) \
  || die "GROOT_SMOKE_NUM_TRIALS must be an integer >= 1, got ${NUM_TRIALS}"
[[ "${ATTEMPTS_PER_WORKER}" =~ ^[0-9]+$ ]] && ((10#${ATTEMPTS_PER_WORKER} >= 1)) \
  || die "GROOT_SMOKE_ATTEMPTS_PER_WORKER must be an integer >= 1, got ${ATTEMPTS_PER_WORKER}"
[[ "${START_SEED}" =~ ^[0-9]+$ ]] \
  || die "GROOT_SMOKE_START_SEED must be a non-negative integer, got ${START_SEED}"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ "${ALLOW_DIRECT_RUN}" == "0" || "${ALLOW_DIRECT_RUN}" == "1" ]] \
  || die "ALLOW_DIRECT_RUN must be 0 or 1"
case "${ACCEL_MODE}" in
  pytorch | torch_compile | trt_full_pipeline | trt_action_head | trt_dit_only)
    ;;
  *)
    die "GROOT_SMOKE_ACCEL_MODE must be pytorch, torch_compile, trt_full_pipeline, trt_action_head, or trt_dit_only; got ${ACCEL_MODE}"
    ;;
esac
[[ "${BATCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "GROOT_SMOKE_BATCH_ID may contain only letters, numbers, dot, underscore, and hyphen"

if [[ -z "${SLURM_JOB_ID:-}" && "${DRY_RUN}" != "1" && "${ALLOW_DIRECT_RUN}" != "1" ]]; then
  die "submit this launcher with sbatch; set ALLOW_DIRECT_RUN=1 only for an intentional direct GPU run"
fi

require_file "${FC_RUNNER}"
require_file "${CM_RUNNER}"
require_executable "${GROOT_PYTHON}"
require_executable "${MOTIONFORGE_PYTHON}"
require_dir "${MOTIONFORGE_ROOT}"
require_dir "${FC_MODEL_PATH}"
require_dir "${CM_MODEL_PATH}"
require_file "${FC_MODEL_PATH}/model.safetensors.index.json"
require_file "${CM_MODEL_PATH}/model.safetensors.index.json"
require_file "${GROOT_BACKBONE_MODEL_PATH}/model.safetensors"
require_file "${GROOT_BACKBONE_MODEL_PATH}/preprocessor_config.json"
require_file "${GROOT_BACKBONE_MODEL_PATH}/tokenizer_config.json"
require_file "${GROOT_FFMPEG_PREFIX}/lib/libavcodec.so.61"

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ ! -e "${SCRIPT_DIR}/outputs/${FC_RUN_ID}" ]] || die "FC result already exists: ${FC_RUN_ID}"
  [[ ! -e "${SCRIPT_DIR}/outputs/${CM_RUN_ID}" ]] || die "CM result already exists: ${CM_RUN_ID}"
fi

run_evaluation() {
  local label="$1"
  local runner="$2"
  local model_path="$3"
  local trial_variable="$4"
  local run_id_variable="$5"
  local cuda_variable="$6"
  local run_id="$7"
  local evaluation_status=0

  log "starting ${label} run_id=${run_id} tasks=10 trials_per_task=${NUM_TRIALS} attempts_per_worker=${ATTEMPTS_PER_WORKER}"
  if env \
      DRY_RUN="${DRY_RUN}" \
      MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT}" \
      MOTIONFORGE_PYTHON="${MOTIONFORGE_PYTHON}" \
      GROOT_PYTHON="${GROOT_PYTHON}" \
      GROOT_MODEL_PATH="${model_path}" \
      GROOT_BACKBONE_MODEL_PATH="${GROOT_BACKBONE_MODEL_PATH}" \
      GROOT_FFMPEG_PREFIX="${GROOT_FFMPEG_PREFIX}" \
      GROOT_ACCEL_MODE="${ACCEL_MODE}" \
      GROOT_EVAL_ATTEMPTS_PER_WORKER="${ATTEMPTS_PER_WORKER}" \
      GROOT_DEVICE="cuda:0" \
      "${trial_variable}=${NUM_TRIALS}" \
      "${run_id_variable}=${run_id}" \
      "${cuda_variable}=${CUDA_DEVICE}" \
      bash "${runner}"; then
    log "completed ${label} run_id=${run_id}"
    return 0
  else
    evaluation_status=$?
    log "failed ${label} run_id=${run_id} exit_code=${evaluation_status}"
    return "${evaluation_status}"
  fi
}

log "started_at=$(date --iso-8601=seconds) slurm_job_id=${SLURM_JOB_ID:-dry-run} batch_id=${BATCH_ID}"
log "workspace_root=${WORKSPACE_ROOT} motionforge_python=${MOTIONFORGE_PYTHON} accel_mode=${ACCEL_MODE} attempts_per_worker=${ATTEMPTS_PER_WORKER}"

overall_status=0
if ! run_evaluation \
    "FC000-FC009" "${FC_RUNNER}" "${FC_MODEL_PATH}" \
    "FC_EVAL_NUM_TRIALS" "FC_EVAL_RUN_ID" "FC_EVAL_CUDA_VISIBLE_DEVICES" "${FC_RUN_ID}"; then
  overall_status=1
fi
if ! run_evaluation \
    "CM000-CM009" "${CM_RUNNER}" "${CM_MODEL_PATH}" \
    "CM_EVAL_NUM_TRIALS" "CM_EVAL_RUN_ID" "CM_EVAL_CUDA_VISIBLE_DEVICES" "${CM_RUN_ID}"; then
  overall_status=1
fi

log "finished_at=$(date --iso-8601=seconds)"
log "fc_result=${SCRIPT_DIR}/outputs/${FC_RUN_ID}"
log "cm_result=${SCRIPT_DIR}/outputs/${CM_RUN_ID}"
exit "${overall_status}"
