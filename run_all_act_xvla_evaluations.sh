#!/usr/bin/env bash
#SBATCH --job-name=act_xvla_all_eval
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -uo pipefail

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ALLOW_DIRECT_RUN="${ALLOW_DIRECT_RUN:-0}"
FORMAL_NUM_TRIALS="${FORMAL_NUM_TRIALS:-50}"
FORMAL_ATTEMPTS_PER_WORKER="${FORMAL_ATTEMPTS_PER_WORKER:-1}"

MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${SCRIPT_DIR}/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-${SCRIPT_DIR}/.conda/lerobot-inference/bin/python}"
MOTIONFORGE_PYTHON="${MOTIONFORGE_PYTHON:-${WORKSPACE_ROOT}/isaacsim/python.sh}"
XVLA_HF_HOME="${XVLA_HF_HOME:-/home/mohanliu/.cache/huggingface}"

ACT_CM_MODEL_PATH="${ACT_CM_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/ACT-CM-80000}"
ACT_FC_MODEL_PATH="${ACT_FC_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/ACT-FC-80000}"
XVLA_CM_MODEL_PATH="${XVLA_CM_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/X-VLA-CM-80000}"
XVLA_FC_MODEL_PATH="${XVLA_FC_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/X-VLA-FC-80000/pretrained_model}"

log() {
  printf '[ACT-XVLA-ALL-EVAL] %s\n' "$*"
}

die() {
  printf '[ACT-XVLA-ALL-EVAL] ERROR: %s\n' "$*" >&2
  exit 1
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || die "required directory not found: ${path}"
}

require_executable() {
  local path="$1"
  [[ -x "${path}" ]] || die "required executable not found or not executable: ${path}"
}

[[ "${ALLOW_DIRECT_RUN}" == "0" || "${ALLOW_DIRECT_RUN}" == "1" ]] \
  || die "ALLOW_DIRECT_RUN must be 0 or 1"
[[ "${FORMAL_NUM_TRIALS}" =~ ^[0-9]+$ ]] && ((10#${FORMAL_NUM_TRIALS} >= 1)) \
  || die "FORMAL_NUM_TRIALS must be an integer >= 1, got ${FORMAL_NUM_TRIALS}"
[[ "${FORMAL_ATTEMPTS_PER_WORKER}" =~ ^[0-9]+$ ]] && ((10#${FORMAL_ATTEMPTS_PER_WORKER} >= 1)) \
  || die "FORMAL_ATTEMPTS_PER_WORKER must be an integer >= 1, got ${FORMAL_ATTEMPTS_PER_WORKER}"
if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_DIRECT_RUN}" != "1" ]]; then
  die "submit this launcher with: sbatch ${BASH_SOURCE[0]} (set ALLOW_DIRECT_RUN=1 only for an intentional direct run)"
fi

require_dir "${MOTIONFORGE_ROOT}"
require_dir "${LEROBOT_ROOT}/src/lerobot"
require_executable "${LEROBOT_PYTHON}"
require_executable "${MOTIONFORGE_PYTHON}"
require_dir "${XVLA_HF_HOME}"
require_dir "${ACT_CM_MODEL_PATH}"
require_dir "${ACT_FC_MODEL_PATH}"
require_dir "${XVLA_CM_MODEL_PATH}"
require_dir "${XVLA_FC_MODEL_PATH}"

EVALUATION_LABELS=(
  "ACT CM000-CM009"
  "ACT FC001-FC009"
  "X-VLA CM000-CM009"
  "X-VLA FC001-FC009"
)
EVALUATION_SCRIPTS=(
  "${SCRIPT_DIR}/ACT/run_cm000_cm009_evaluation.sh"
  "${SCRIPT_DIR}/ACT/run_fc001_fc009_evaluation.sh"
  "${SCRIPT_DIR}/X-VLA/run_cm000_cm009_evaluation.sh"
  "${SCRIPT_DIR}/X-VLA/run_fc001_fc009_evaluation.sh"
)
EVALUATION_DIR_VARIABLES=(
  "ACT_SCRIPT_DIR"
  "ACT_SCRIPT_DIR"
  "XVLA_SCRIPT_DIR"
  "XVLA_SCRIPT_DIR"
)

EVALUATION_MODEL_VARIABLES=(
  "ACT_MODEL_PATH"
  "ACT_MODEL_PATH"
  "XVLA_MODEL_PATH"
  "XVLA_MODEL_PATH"
)
EVALUATION_MODEL_PATHS=(
  "${ACT_CM_MODEL_PATH}"
  "${ACT_FC_MODEL_PATH}"
  "${XVLA_CM_MODEL_PATH}"
  "${XVLA_FC_MODEL_PATH}"
)
EVALUATION_PYTHON_VARIABLES=(
  "ACT_PYTHON"
  "ACT_PYTHON"
  "XVLA_PYTHON"
  "XVLA_PYTHON"
)
EVALUATION_HF_HOME_VARIABLES=(
  ""
  ""
  "XVLA_HF_HOME"
  "XVLA_HF_HOME"
)
EVALUATION_TRIAL_VARIABLES=(
  "ACT_EVAL_NUM_TRIALS"
  "ACT_EVAL_NUM_TRIALS"
  "XVLA_EVAL_NUM_TRIALS"
  "XVLA_EVAL_NUM_TRIALS"
)
EVALUATION_ATTEMPT_VARIABLES=(
  "ACT_EVAL_ATTEMPTS_PER_WORKER"
  "ACT_EVAL_ATTEMPTS_PER_WORKER"
  "XVLA_EVAL_ATTEMPTS_PER_WORKER"
  "XVLA_EVAL_ATTEMPTS_PER_WORKER"
)

for evaluation_script in "${EVALUATION_SCRIPTS[@]}"; do
  [[ -f "${evaluation_script}" ]] || die "evaluation script not found: ${evaluation_script}"
  [[ -r "${evaluation_script}" ]] || die "evaluation script is not readable: ${evaluation_script}"
done

trap 'exit 130' INT
trap 'exit 143' TERM

failed_evaluations=0
failed_labels=()

log "started_at=$(date --iso-8601=seconds)"
log "slurm_job_id=${SLURM_JOB_ID:-direct-run}"
log "evaluation_count=${#EVALUATION_SCRIPTS[@]}"
log "trials_per_task=${FORMAL_NUM_TRIALS}"
log "attempts_per_worker=${FORMAL_ATTEMPTS_PER_WORKER}"
log "motionforge_root=${MOTIONFORGE_ROOT}"
log "lerobot_root=${LEROBOT_ROOT}"
log "lerobot_python=${LEROBOT_PYTHON}"
log "motionforge_python=${MOTIONFORGE_PYTHON}"
log "xvla_hf_home=${XVLA_HF_HOME}"

for index in "${!EVALUATION_SCRIPTS[@]}"; do
  label="${EVALUATION_LABELS[index]}"
  evaluation_script="${EVALUATION_SCRIPTS[index]}"
  script_dir="$(dirname -- "${evaluation_script}")"
  dir_variable="${EVALUATION_DIR_VARIABLES[index]}"
  model_variable="${EVALUATION_MODEL_VARIABLES[index]}"
  model_path="${EVALUATION_MODEL_PATHS[index]}"
  python_variable="${EVALUATION_PYTHON_VARIABLES[index]}"
  hf_home_variable="${EVALUATION_HF_HOME_VARIABLES[index]}"
  trial_variable="${EVALUATION_TRIAL_VARIABLES[index]}"
  attempt_variable="${EVALUATION_ATTEMPT_VARIABLES[index]}"
  evaluation_environment=(
    "${dir_variable}=${script_dir}"
    "MOTIONFORGE_ROOT=${MOTIONFORGE_ROOT}"
    "LEROBOT_ROOT=${LEROBOT_ROOT}"
    "MOTIONFORGE_PYTHON=${MOTIONFORGE_PYTHON}"
    "${python_variable}=${LEROBOT_PYTHON}"
    "${model_variable}=${model_path}"
    "${trial_variable}=${FORMAL_NUM_TRIALS}"
    "${attempt_variable}=${FORMAL_ATTEMPTS_PER_WORKER}"
  )

  if [[ -n "${hf_home_variable}" ]]; then
    evaluation_environment+=("${hf_home_variable}=${XVLA_HF_HOME}")
  fi

  log "[$((index + 1))/${#EVALUATION_SCRIPTS[@]}] starting ${label} model=${model_path} trials=${FORMAL_NUM_TRIALS} attempts_per_worker=${FORMAL_ATTEMPTS_PER_WORKER}"
  if env "${evaluation_environment[@]}" bash "${evaluation_script}"; then
    log "[$((index + 1))/${#EVALUATION_SCRIPTS[@]}] completed ${label}"
  else
    status=$?
    ((failed_evaluations += 1))
    failed_labels+=("${label} (exit ${status})")
    log "[$((index + 1))/${#EVALUATION_SCRIPTS[@]}] failed ${label} with exit status ${status}"
  fi
done

log "finished_at=$(date --iso-8601=seconds)"
if ((failed_evaluations > 0)); then
  log "failed_evaluations=${failed_evaluations}"
  for failed_label in "${failed_labels[@]}"; do
    log "failure=${failed_label}"
  done
  exit 1
fi

log "all evaluations completed successfully"
