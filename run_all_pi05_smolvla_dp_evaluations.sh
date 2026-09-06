#!/usr/bin/env bash
#SBATCH --job-name=pi05_smolvla_dp_all_eval
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
FORMAL_ATTEMPTS_PER_WORKER="${FORMAL_ATTEMPTS_PER_WORKER:-50}"
START_EVALUATION_INDEX="${START_EVALUATION_INDEX:-1}"
JOB_PORT_SEED="${SLURM_JOB_ID:-0}"
EVALUATION_PORT_BASE="${EVALUATION_PORT_BASE:-}"

MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${SCRIPT_DIR}/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-${SCRIPT_DIR}/.conda/lerobot-inference/bin/python}"
MOTIONFORGE_PYTHON="${MOTIONFORGE_PYTHON:-${WORKSPACE_ROOT}/isaacsim/python.sh}"
SMOLVLA_HF_HOME="${SMOLVLA_HF_HOME:-/home/mohanliu/.cache/huggingface}"

PI05_CM_MODEL_PATH="${PI05_CM_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/pi05-CM-80000}"
PI05_FC_MODEL_PATH="${PI05_FC_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/pi05-FC-80000/pretrained_model}"
SMOLVLA_CM_MODEL_PATH="${SMOLVLA_CM_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/SmolVLA-CM-80000}"
SMOLVLA_FC_MODEL_PATH="${SMOLVLA_FC_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/SmolVLA-FC-80000/pretrained_model}"
DIFFUSION_CM_MODEL_PATH="${DIFFUSION_CM_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/DiffusionPolicy-CM-80000-3views}"
DIFFUSION_FC_MODEL_PATH="${DIFFUSION_FC_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/DiffusionPolicy-FC-80000-3views}"

log() {
  printf '[PI05-SMOLVLA-DP-ALL-EVAL] %s\n' "$*"
}

die() {
  printf '[PI05-SMOLVLA-DP-ALL-EVAL] ERROR: %s\n' "$*" >&2
  exit 1
}

[[ "${JOB_PORT_SEED}" =~ ^[0-9]+$ ]] || die "SLURM_JOB_ID must be numeric when used for automatic port allocation"
if [[ -z "${EVALUATION_PORT_BASE}" ]]; then
  EVALUATION_PORT_BASE=$((20000 + (10#${JOB_PORT_SEED} % 2000) * 20))
fi

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
require_dir "${SMOLVLA_HF_HOME}"
require_dir "${PI05_CM_MODEL_PATH}"
require_dir "${PI05_FC_MODEL_PATH}"
require_dir "${SMOLVLA_CM_MODEL_PATH}"
require_dir "${SMOLVLA_FC_MODEL_PATH}"
require_dir "${DIFFUSION_CM_MODEL_PATH}"
require_dir "${DIFFUSION_FC_MODEL_PATH}"

EVALUATION_LABELS=(
  "pi0.5 CM000-CM009"
  "pi0.5 FC000-FC009"
  "SmolVLA CM000-CM009"
  "SmolVLA FC000-FC009"
  "Diffusion Policy CM000-CM009"
  "Diffusion Policy FC000-FC009"
)
EVALUATION_SCRIPTS=(
  "${SCRIPT_DIR}/pi05/run_cm000_cm009_evaluation.sh"
  "${SCRIPT_DIR}/pi05/run_fc001_fc009_evaluation.sh"
  "${SCRIPT_DIR}/smolvla/run_cm000_cm009_evaluation.sh"
  "${SCRIPT_DIR}/smolvla/run_fc001_fc009_evaluation.sh"
  "${SCRIPT_DIR}/DP/run_cm000_cm009_evaluation.sh"
  "${SCRIPT_DIR}/DP/run_fc001_fc009_evaluation.sh"
)
EVALUATION_DIR_VARIABLES=(
  "PI05_SCRIPT_DIR"
  "PI05_SCRIPT_DIR"
  "SMOLVLA_SCRIPT_DIR"
  "SMOLVLA_SCRIPT_DIR"
  ""
  ""
)
EVALUATION_MODEL_VARIABLES=(
  "PI05_MODEL_PATH"
  "PI05_MODEL_PATH"
  "SMOLVLA_MODEL_PATH"
  "SMOLVLA_MODEL_PATH"
  "DIFFUSION_MODEL_PATH"
  "DIFFUSION_MODEL_PATH"
)
EVALUATION_MODEL_PATHS=(
  "${PI05_CM_MODEL_PATH}"
  "${PI05_FC_MODEL_PATH}"
  "${SMOLVLA_CM_MODEL_PATH}"
  "${SMOLVLA_FC_MODEL_PATH}"
  "${DIFFUSION_CM_MODEL_PATH}"
  "${DIFFUSION_FC_MODEL_PATH}"
)
EVALUATION_PYTHON_VARIABLES=(
  "PI05_PYTHON"
  "PI05_PYTHON"
  "SMOLVLA_PYTHON"
  "SMOLVLA_PYTHON"
  "DIFFUSION_PYTHON"
  "DIFFUSION_PYTHON"
)
EVALUATION_HF_HOME_VARIABLES=(
  ""
  ""
  "SMOLVLA_HF_HOME"
  "SMOLVLA_HF_HOME"
  ""
  ""
)
EVALUATION_TRIAL_VARIABLES=(
  "PI05_EVAL_NUM_TRIALS"
  "PI05_EVAL_NUM_TRIALS"
  "SMOLVLA_EVAL_NUM_TRIALS"
  "SMOLVLA_EVAL_NUM_TRIALS"
  "DIFFUSION_EVAL_NUM_TRIALS"
  "DIFFUSION_EVAL_NUM_TRIALS"
)
EVALUATION_OBS_PORT_VARIABLES=(
  "PI05_EVAL_OBS_PORT"
  "PI05_EVAL_OBS_PORT"
  "SMOLVLA_EVAL_OBS_PORT"
  "SMOLVLA_EVAL_OBS_PORT"
  "DIFFUSION_EVAL_OBS_PORT"
  "DIFFUSION_EVAL_OBS_PORT"
)
EVALUATION_ACTION_PORT_VARIABLES=(
  "PI05_EVAL_ACT_PORT"
  "PI05_EVAL_ACT_PORT"
  "SMOLVLA_EVAL_ACT_PORT"
  "SMOLVLA_EVAL_ACT_PORT"
  "DIFFUSION_EVAL_DIFFUSION_PORT"
  "DIFFUSION_EVAL_ACT_PORT"
)

[[ "${START_EVALUATION_INDEX}" =~ ^[0-9]+$ ]] && ((10#${START_EVALUATION_INDEX} >= 1)) && ((10#${START_EVALUATION_INDEX} <= ${#EVALUATION_SCRIPTS[@]})) || die "START_EVALUATION_INDEX must be between 1 and ${#EVALUATION_SCRIPTS[@]}, got ${START_EVALUATION_INDEX}"
[[ "${EVALUATION_PORT_BASE}" =~ ^[0-9]+$ ]] && ((10#${EVALUATION_PORT_BASE} >= 1024)) || die "EVALUATION_PORT_BASE must be an integer >= 1024, got ${EVALUATION_PORT_BASE}"
LAST_EVALUATION_PORT=$((10#${EVALUATION_PORT_BASE} + ${#EVALUATION_SCRIPTS[@]} * 2 - 1))
((LAST_EVALUATION_PORT <= 65535)) || die "evaluation port range must end at or below 65535, got ${LAST_EVALUATION_PORT}"

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
log "start_evaluation_index=${START_EVALUATION_INDEX}"
log "trials_per_task=${FORMAL_NUM_TRIALS}"
log "attempts_per_worker=${FORMAL_ATTEMPTS_PER_WORKER}"
log "evaluation_port_range=${EVALUATION_PORT_BASE}-${LAST_EVALUATION_PORT}"
log "motionforge_root=${MOTIONFORGE_ROOT}"
log "lerobot_root=${LEROBOT_ROOT}"
log "lerobot_python=${LEROBOT_PYTHON}"
log "motionforge_python=${MOTIONFORGE_PYTHON}"
log "smolvla_hf_home=${SMOLVLA_HF_HOME}"

for index in "${!EVALUATION_SCRIPTS[@]}"; do
  label="${EVALUATION_LABELS[index]}"
  if ((index + 1 < 10#${START_EVALUATION_INDEX})); then
    log "[$((index + 1))/${#EVALUATION_SCRIPTS[@]}] skipped ${label}"
    continue
  fi
  evaluation_script="${EVALUATION_SCRIPTS[index]}"
  script_dir="$(dirname -- "${evaluation_script}")"
  dir_variable="${EVALUATION_DIR_VARIABLES[index]}"
  model_variable="${EVALUATION_MODEL_VARIABLES[index]}"
  model_path="${EVALUATION_MODEL_PATHS[index]}"
  python_variable="${EVALUATION_PYTHON_VARIABLES[index]}"
  hf_home_variable="${EVALUATION_HF_HOME_VARIABLES[index]}"
  trial_variable="${EVALUATION_TRIAL_VARIABLES[index]}"
  obs_port_variable="${EVALUATION_OBS_PORT_VARIABLES[index]}"
  action_port_variable="${EVALUATION_ACTION_PORT_VARIABLES[index]}"
  obs_port=$((10#${EVALUATION_PORT_BASE} + index * 2))
  action_port=$((obs_port + 1))
  evaluation_environment=(
    "MOTIONFORGE_ROOT=${MOTIONFORGE_ROOT}"
    "LEROBOT_ROOT=${LEROBOT_ROOT}"
    "MOTIONFORGE_PYTHON=${MOTIONFORGE_PYTHON}"
    "${python_variable}=${LEROBOT_PYTHON}"
    "${model_variable}=${model_path}"
    "${trial_variable}=${FORMAL_NUM_TRIALS}"
    "MOTIONFORGE_ATTEMPTS_PER_WORKER=${FORMAL_ATTEMPTS_PER_WORKER}"
    "${obs_port_variable}=${obs_port}"
    "${action_port_variable}=${action_port}"
  )

  if [[ -n "${dir_variable}" ]]; then
    evaluation_environment+=("${dir_variable}=${script_dir}")
  fi
  if [[ -n "${hf_home_variable}" ]]; then
    evaluation_environment+=("${hf_home_variable}=${SMOLVLA_HF_HOME}")
  fi

  log "[$((index + 1))/${#EVALUATION_SCRIPTS[@]}] starting ${label} model=${model_path} trials=${FORMAL_NUM_TRIALS} attempts_per_worker=${FORMAL_ATTEMPTS_PER_WORKER} ports=${obs_port},${action_port}"
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
