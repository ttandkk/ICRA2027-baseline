#!/usr/bin/env bash

set -Eeuo pipefail

RECOVERY_GROUP="${FORMAL_HT_RECOVERY_GROUP:-}"
if [[ -n "${RECOVERY_GROUP}" ]]; then
  BASELINE_ROOT="${BASELINE_ROOT:?BASELINE_ROOT is required in worker mode}"
else
  BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
WORKSPACE_ROOT="$(cd -- "${BASELINE_ROOT}/.." && pwd -P)"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"
TRAINING_ROOT="${WORKSPACE_ROOT}/ICRA2027-baseline"
LEROBOT_ROOT="${BASELINE_ROOT}/lerobot"
LEROBOT_PYTHON="${BASELINE_ROOT}/.conda/lerobot-inference/bin/python"
HF_CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"
DP_HT_MODEL_PATH="${TRAINING_ROOT}/DP/train_outputs/diffusion_lerobot_ht_129916/checkpoints/080000/pretrained_model"
SMOLVLA_HT_MODEL_PATH="${TRAINING_ROOT}/smolvla/train_outputs/smolvla_lerobot_ht_129933/checkpoints/080000/pretrained_model"
DRY_RUN="${SUBMIT_DRY_RUN:-0}"
TIME_LIMIT="${FORMAL_HT_RECOVERY_TIME_LIMIT:-48:00:00}"

log() {
  printf '[FORMAL-50-HT-RECOVERY] %s\n' "$*"
}

die() {
  printf '[FORMAL-50-HT-RECOVERY] ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "required directory not found: $1"
}

require_executable() {
  [[ -x "$1" ]] || die "required executable not found: $1"
}

validate_common() {
  [[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] \
    || die "SUBMIT_DRY_RUN must be 0 or 1"
  [[ "${TIME_LIMIT}" =~ ^[0-9]+:[0-9]{2}:[0-9]{2}$ ]] \
    || die "FORMAL_HT_RECOVERY_TIME_LIMIT must use hours:minutes:seconds"
  require_file "${BASELINE_ROOT}/DP/run_ht000_ht009_evaluation.sh"
  require_file "${BASELINE_ROOT}/smolvla/run_ht000_ht009_evaluation.sh"
  require_file "${MOTIONFORGE_ROOT}/scripts/benchmark/run_env_server_trials.py"
  require_dir "${LEROBOT_ROOT}/src/lerobot"
  require_dir "${HF_CACHE_ROOT}"
  require_dir "${DP_HT_MODEL_PATH}"
  require_dir "${SMOLVLA_HT_MODEL_PATH}"
  require_executable "${LEROBOT_PYTHON}"
}

run_worker() {
  [[ -n "${SLURM_JOB_ID:-}" ]] \
    || die "worker mode requires a Slurm allocation"
  export MOTIONFORGE_ROOT
  export MOTIONFORGE_DEVICE=cpu
  export DRY_RUN

  case "${RECOVERY_GROUP}" in
    dp_ht)
      export DIFFUSION_PYTHON="${LEROBOT_PYTHON}"
      export DIFFUSION_MODEL_PATH="${DP_HT_MODEL_PATH}"
      export DIFFUSION_EVAL_TASKS="ht_000 ht_001 ht_002 ht_003 ht_004 ht_005 ht_006 ht_007 ht_008 ht_009"
      export DIFFUSION_EVAL_START_SEED=0
      export DIFFUSION_EVAL_NUM_TRIALS=50
      export DIFFUSION_EVAL_ATTEMPTS_PER_WORKER=50
      export DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES=0
      export DIFFUSION_EVAL_OBS_PORT=24396
      export DIFFUSION_EVAL_DIFFUSION_PORT=24398
      export DIFFUSION_EVAL_RUN_ID="formal50_ht_recovery_${SLURM_JOB_ID}_dp_ht"
      log "running DP HT000-HT009, 50 trials each, CPU Physics"
      exec bash "${BASELINE_ROOT}/DP/run_ht000_ht009_evaluation.sh"
      ;;
    smolvla_ht007)
      export SMOLVLA_SCRIPT_DIR="${BASELINE_ROOT}/smolvla"
      export LEROBOT_ROOT
      export SMOLVLA_PYTHON="${LEROBOT_PYTHON}"
      export SMOLVLA_HF_HOME="${HF_CACHE_ROOT}"
      export SMOLVLA_MODEL_PATH="${SMOLVLA_HT_MODEL_PATH}"
      export SMOLVLA_EVAL_TASKS=ht_007
      export SMOLVLA_EVAL_START_SEED=0
      export SMOLVLA_EVAL_NUM_TRIALS=50
      export SMOLVLA_EVAL_ATTEMPTS_PER_WORKER=50
      export SMOLVLA_EVAL_CUDA_VISIBLE_DEVICES=0
      export SMOLVLA_EVAL_OBS_PORT=25396
      export SMOLVLA_EVAL_ACT_PORT=25398
      export SMOLVLA_EVAL_RUN_ID="formal50_ht_recovery_${SLURM_JOB_ID}_smolvla_ht007"
      log "running SmolVLA HT007 only, 50 trials, CPU Physics"
      exec bash "${BASELINE_ROOT}/smolvla/run_ht000_ht009_evaluation.sh"
      ;;
    *)
      die "unsupported FORMAL_HT_RECOVERY_GROUP: ${RECOVERY_GROUP:-missing}"
      ;;
  esac
}

submit_recovery_jobs() {
  local group=""
  local job_name=""
  local script_path=""
  local submission=""
  local -a command=()

  [[ -z "${SLURM_JOB_ID:-}" ]] \
    || die "submit mode must run outside a Slurm allocation"
  command -v sbatch >/dev/null 2>&1 || die "sbatch was not found in PATH"
  script_path="$(readlink -f -- "${BASH_SOURCE[0]}")"
  require_file "${script_path}"

  for group in dp_ht smolvla_ht007; do
    case "${group}" in
      dp_ht) job_name="formal50-recover-dp-ht" ;;
      smolvla_ht007) job_name="formal50-recover-smol-ht007" ;;
    esac
    command=(
      sbatch
      --parsable
      "--job-name=${job_name}"
      --partition=cluster02
      --gres=gpu:rtx5090:1
      "--time=${TIME_LIMIT}"
      --cpus-per-task=4
      --mem=48G
      "--output=${BASELINE_ROOT}/slurm-${job_name}-%j.out"
      "--error=${BASELINE_ROOT}/slurm-${job_name}-%j.err"
      "--export=ALL,BASELINE_ROOT=${BASELINE_ROOT},MOTIONFORGE_ROOT=${MOTIONFORGE_ROOT},FORMAL_HT_RECOVERY_GROUP=${group},SUBMIT_DRY_RUN=0"
      "${script_path}"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[FORMAL-50-HT-RECOVERY] dry-run: '
      printf '%q ' "${command[@]}"
      printf '\n'
      continue
    fi
    submission="$("${command[@]}")"
    log "submitted group=${group} job_id=${submission%%;*}"
  done
}

validate_common
if [[ -n "${RECOVERY_GROUP}" ]]; then
  run_worker
else
  submit_recovery_jobs
fi
