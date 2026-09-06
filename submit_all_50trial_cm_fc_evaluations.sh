#!/usr/bin/env bash

set -Eeuo pipefail

readonly FORMAL_NUM_TRIALS="${FORMAL_NUM_TRIALS:-50}"
readonly FORMAL_ATTEMPTS_PER_WORKER="${FORMAL_ATTEMPTS_PER_WORKER:-50}"
readonly FORMAL_START_SEED="0"
readonly SUBMIT_DRY_RUN="${SUBMIT_DRY_RUN:-0}"
readonly FORMAL_VALIDATE_ONLY="${FORMAL_VALIDATE_ONLY:-0}"
readonly FORMAL_EVALUATION_GROUP="${FORMAL_EVALUATION_GROUP:-}"
readonly FORMAL_JOB_TIME_LIMIT="${FORMAL_JOB_TIME_LIMIT:-36:00:00}"

if [[ -n "${FORMAL_EVALUATION_GROUP}" ]]; then
  [[ -n "${BASELINE_ROOT:-}" ]] || {
    printf '[FORMAL-50-EVAL] ERROR: BASELINE_ROOT is required in worker mode\n' >&2
    exit 1
  }
else
  BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
readonly BASELINE_ROOT
readonly WORKSPACE_ROOT="$(cd -- "${BASELINE_ROOT}/.." && pwd -P)"
readonly MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"
readonly LEROBOT_ROOT="${BASELINE_ROOT}/lerobot"
readonly LEROBOT_PYTHON="${BASELINE_ROOT}/.conda/lerobot-inference/bin/python"
readonly MOTIONFORGE_PYTHON="${WORKSPACE_ROOT}/isaacsim/python.sh"
readonly HF_CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"

readonly ACT_CM_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/ACT-CM-80000"
readonly ACT_FC_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/ACT-FC-80000"
readonly PI05_CM_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/pi05-CM-80000"
readonly PI05_FC_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/pi05-FC-80000/pretrained_model"
readonly SMOLVLA_CM_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/SmolVLA-CM-80000"
readonly SMOLVLA_FC_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/SmolVLA-FC-80000/pretrained_model"
readonly DIFFUSION_CM_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/DiffusionPolicy-CM-80000-3views"
readonly DIFFUSION_FC_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/DiffusionPolicy-FC-80000-3views"
readonly XVLA_CM_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/X-VLA-CM-80000"
readonly XVLA_FC_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/X-VLA-FC-80000/pretrained_model"

readonly GROOT_ROOT="${BASELINE_ROOT}/Isaac-GR00T"
readonly GROOT_PYTHON="${GROOT_ROOT}/.venv/bin/python"
readonly GROOT_CM_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/gr00t1.7-CM-80000"
readonly GROOT_FC_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/gr00t1.7-FC-80000"
readonly GROOT_BACKBONE_MODEL_PATH="${WORKSPACE_ROOT}/ckpts/nvidia/Cosmos-Reason2-2B"
readonly GROOT_FFMPEG_PREFIX="${WORKSPACE_ROOT}/ICRA2027-baseline/.conda/lerobot-baselines"
readonly GROOT_CM_TRT_ENGINE_PATH="${GROOT_ROOT}/gr00t_trt_deployments/gr00t_trt_deployment_gr00t1.7-CM-80000/engines"
readonly GROOT_FC_TRT_ENGINE_PATH="${GROOT_ROOT}/gr00t_trt_deployments/gr00t_trt_deployment_gr00t1.7-FC-80000/engines"

log() {
  printf '[FORMAL-50-EVAL] %s\n' "$*"
}

die() {
  printf '[FORMAL-50-EVAL] ERROR: %s\n' "$*" >&2
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

[[ "${FORMAL_NUM_TRIALS}" == "50" ]] \
  || die "FORMAL_NUM_TRIALS is fixed at 50 for this launcher"
[[ "${FORMAL_ATTEMPTS_PER_WORKER}" == "50" ]] \
  || die "FORMAL_ATTEMPTS_PER_WORKER is fixed at 50 for this launcher"
[[ "${SUBMIT_DRY_RUN}" == "0" || "${SUBMIT_DRY_RUN}" == "1" ]] \
  || die "SUBMIT_DRY_RUN must be 0 or 1"
[[ "${FORMAL_VALIDATE_ONLY}" == "0" || "${FORMAL_VALIDATE_ONLY}" == "1" ]] \
  || die "FORMAL_VALIDATE_ONLY must be 0 or 1"
[[ "${FORMAL_JOB_TIME_LIMIT}" =~ ^[0-9]+:[0-9]{2}:[0-9]{2}$ ]] \
  || die "FORMAL_JOB_TIME_LIMIT must use hours:minutes:seconds, got ${FORMAL_JOB_TIME_LIMIT}"

BENCHMARK_CONFIGS=()
for task_number in {0..9}; do
  printf -v task_id 'cm_%03d' "${task_number}"
  BENCHMARK_CONFIGS+=("${MOTIONFORGE_ROOT}/configs/benchmarks/circular_motion/${task_id}_rgb_gr00t_zmq.yaml")
done
for task_number in {0..9}; do
  printf -v task_id 'fc_%03d' "${task_number}"
  BENCHMARK_CONFIGS+=("${MOTIONFORGE_ROOT}/configs/benchmarks/factory_conveyor/${task_id}_rgb_gr00t_zmq.yaml")
done
readonly -a BENCHMARK_CONFIGS

benchmark_hash() {
  sha256sum "${BENCHMARK_CONFIGS[@]}" | sha256sum | awk '{print $1}'
}

validate_benchmark_configs() {
  local config=""
  local timing_profile=""
  local max_steps=""

  ((${#BENCHMARK_CONFIGS[@]} == 20)) || die "expected 20 CM/FC benchmark configs"
  for config in "${BENCHMARK_CONFIGS[@]}"; do
    require_file "${config}"
    timing_profile="$(awk '/^[[:space:]]*timing_profile:[[:space:]]*/ {print $2; exit}' "${config}")"
    [[ "${timing_profile}" == "server_scheduled_120hz" ]] \
      || die "benchmark is not pinned to server_scheduled_120hz: ${config} (got ${timing_profile:-missing})"
    max_steps="$(awk '/^[[:space:]]*max_steps:[[:space:]]*[0-9]+[[:space:]]*$/ {print $2; exit}' "${config}")"
    [[ "${max_steps}" =~ ^[0-9]+$ ]] && ((10#${max_steps} >= 1)) \
      || die "benchmark max_steps is missing or invalid: ${config}"
  done
}

validate_trt_engine() {
  local engine_dir="$1"
  local engine_file=""

  require_dir "${engine_dir}"
  for engine_file in \
    export_metadata.json \
    state_encoder.engine \
    action_encoder.engine \
    dit_bf16.engine \
    action_decoder.engine \
    vit.engine \
    llm_bf16.engine \
    vl_self_attention.engine; do
    require_file "${engine_dir}/${engine_file}"
  done
  grep -Eq '"export_mode"[[:space:]]*:[[:space:]]*"full_pipeline"' \
    "${engine_dir}/export_metadata.json" \
    || die "TensorRT metadata is not a full_pipeline export: ${engine_dir}/export_metadata.json"
}

validate_prerequisites() {
  local runner=""
  local model_dir=""

  require_dir "${MOTIONFORGE_ROOT}"
  require_dir "${LEROBOT_ROOT}/src/lerobot"
  require_executable "${LEROBOT_PYTHON}"
  require_executable "${MOTIONFORGE_PYTHON}"
  require_executable "${GROOT_PYTHON}"
  require_dir "${HF_CACHE_ROOT}"
  require_dir "${GROOT_BACKBONE_MODEL_PATH}"
  require_file "${GROOT_FFMPEG_PREFIX}/lib/libavcodec.so.61"

  for runner in \
    "${BASELINE_ROOT}/ACT/run_cm000_cm009_evaluation.sh" \
    "${BASELINE_ROOT}/ACT/run_fc001_fc009_evaluation.sh" \
    "${BASELINE_ROOT}/DP/run_cm000_cm009_evaluation.sh" \
    "${BASELINE_ROOT}/DP/run_fc001_fc009_evaluation.sh" \
    "${BASELINE_ROOT}/pi05/run_cm000_cm009_evaluation.sh" \
    "${BASELINE_ROOT}/pi05/run_fc001_fc009_evaluation.sh" \
    "${BASELINE_ROOT}/smolvla/run_cm000_cm009_evaluation.sh" \
    "${BASELINE_ROOT}/smolvla/run_fc001_fc009_evaluation.sh" \
    "${BASELINE_ROOT}/X-VLA/run_cm000_cm009_evaluation.sh" \
    "${BASELINE_ROOT}/X-VLA/run_fc001_fc009_evaluation.sh" \
    "${GROOT_ROOT}/examples/MotionForge/run_cm000_cm009_evaluation.sh" \
    "${GROOT_ROOT}/examples/MotionForge/run_fc001_fc009_evaluation.sh"; do
    require_file "${runner}"
  done

  for model_dir in \
    "${ACT_CM_MODEL_PATH}" \
    "${ACT_FC_MODEL_PATH}" \
    "${PI05_CM_MODEL_PATH}" \
    "${PI05_FC_MODEL_PATH}" \
    "${SMOLVLA_CM_MODEL_PATH}" \
    "${SMOLVLA_FC_MODEL_PATH}" \
    "${DIFFUSION_CM_MODEL_PATH}" \
    "${DIFFUSION_FC_MODEL_PATH}" \
    "${XVLA_CM_MODEL_PATH}" \
    "${XVLA_FC_MODEL_PATH}" \
    "${GROOT_CM_MODEL_PATH}" \
    "${GROOT_FC_MODEL_PATH}"; do
    require_dir "${model_dir}"
  done

  validate_trt_engine "${GROOT_CM_TRT_ENGINE_PATH}"
  validate_trt_engine "${GROOT_FC_TRT_ENGINE_PATH}"
  validate_benchmark_configs
}

submit_jobs() {
  local script_path=""
  local config_hash=""
  local group=""
  local job_name=""
  local submission=""
  local job_id=""
  local status=0
  local failed_submissions=0
  local -a submitted_jobs=()
  local -a command=()

  [[ -z "${SLURM_JOB_ID:-}" ]] \
    || die "run this launcher from a login shell; it submits two worker jobs itself"
  command -v sbatch >/dev/null 2>&1 || die "sbatch was not found in PATH"

  script_path="$(readlink -f -- "${BASH_SOURCE[0]}")"
  require_file "${script_path}"
  config_hash="$(benchmark_hash)"

  log "validated 20 benchmark configs at 120Hz; config_sha256=${config_hash}"
  log "plan: two independent RTX 5090 jobs, 3 models per GPU, 50 trials per task"

  for group in act_pi05_groot dp_smolvla_xvla; do
    case "${group}" in
      act_pi05_groot)
        job_name="formal50-act-pi05-groot"
        ;;
      dp_smolvla_xvla)
        job_name="formal50-dp-smol-xvla"
        ;;
    esac

    command=(
      sbatch
      --parsable
      "--job-name=${job_name}"
      --partition=cluster02
      --gres=gpu:rtx5090:1
      "--time=${FORMAL_JOB_TIME_LIMIT}"
      --cpus-per-task=4
      --mem=48G
      "--output=${BASELINE_ROOT}/slurm-formal50-${group}-%j.out"
      "--error=${BASELINE_ROOT}/slurm-formal50-${group}-%j.err"
      "--export=ALL,BASELINE_ROOT=${BASELINE_ROOT},MOTIONFORGE_ROOT=${MOTIONFORGE_ROOT},FORMAL_EVALUATION_GROUP=${group},FORMAL_EXPECTED_CONFIG_SHA256=${config_hash},FORMAL_NUM_TRIALS=50,FORMAL_ATTEMPTS_PER_WORKER=50,SUBMIT_DRY_RUN=0,FORMAL_VALIDATE_ONLY=0"
      "${script_path}"
    )

    if [[ "${SUBMIT_DRY_RUN}" == "1" ]]; then
      printf '[FORMAL-50-EVAL] dry-run submit %s: ' "${group}"
      printf '%q ' "${command[@]}"
      printf '\n'
      continue
    fi

    if submission="$("${command[@]}")"; then
      job_id="${submission%%;*}"
      if [[ "${job_id}" =~ ^[0-9]+$ ]]; then
        submitted_jobs+=("${group}=${job_id}")
        log "submitted group=${group} job_id=${job_id}"
      else
        ((failed_submissions += 1))
        log "submission returned an unexpected job id for group=${group}: ${submission}"
      fi
    else
      status=$?
      ((failed_submissions += 1))
      log "submission failed for group=${group} with exit status ${status}"
    fi
  done

  if [[ "${SUBMIT_DRY_RUN}" == "1" ]]; then
    log "dry-run complete; no Slurm jobs were submitted"
    return 0
  fi

  for submission in "${submitted_jobs[@]}"; do
    log "submitted_job=${submission}"
  done
  ((failed_submissions == 0)) \
    || die "${failed_submissions} of 2 job submissions failed; already submitted jobs were left queued"
  log "both jobs submitted independently; neither job waits for the other"
}

FAILED_STAGES=0
FAILED_STAGE_LABELS=()
STAGE_INDEX=0
PORT_BASE=0
RUNNER_DRY_RUN=0
JOB_TAG=""

run_stage() {
  local label="$1"
  local runner="$2"
  local obs_port_variable="$3"
  local action_port_variable="$4"
  local obs_port=$((PORT_BASE + STAGE_INDEX * 2))
  local action_port=$((obs_port + 1))
  local status=0
  shift 4

  ((STAGE_INDEX += 1))
  log "[${STAGE_INDEX}/6] starting ${label}; ports=${obs_port},${action_port}; dry_run=${RUNNER_DRY_RUN}"

  if env \
    -u ACT_EVAL_TASKS \
    -u DIFFUSION_EVAL_TASKS \
    -u PI05_EVAL_TASKS \
    -u SMOLVLA_EVAL_TASKS \
    -u XVLA_EVAL_TASKS \
    -u CM_EVAL_TASKS \
    -u FC_EVAL_TASKS \
    -u ACT_EVAL_MOTION_LEVEL \
    -u DIFFUSION_EVAL_MOTION_LEVEL \
    -u PI05_EVAL_MOTION_LEVEL \
    -u SMOLVLA_EVAL_MOTION_LEVEL \
    -u XVLA_EVAL_MOTION_LEVEL \
    -u CM_EVAL_MOTION_LEVEL \
    -u FC_EVAL_MOTION_LEVEL \
    "DRY_RUN=${RUNNER_DRY_RUN}" \
    "MOTIONFORGE_ROOT=${MOTIONFORGE_ROOT}" \
    "MOTIONFORGE_PYTHON=${MOTIONFORGE_PYTHON}" \
    "LEROBOT_ROOT=${LEROBOT_ROOT}" \
    "MOTIONFORGE_ATTEMPTS_PER_WORKER=${FORMAL_ATTEMPTS_PER_WORKER}" \
    "${obs_port_variable}=${obs_port}" \
    "${action_port_variable}=${action_port}" \
    "$@" \
    bash "${runner}"; then
    log "[${STAGE_INDEX}/6] completed ${label}"
  else
    status=$?
    ((FAILED_STAGES += 1))
    FAILED_STAGE_LABELS+=("${label} (exit ${status})")
    log "[${STAGE_INDEX}/6] failed ${label} with exit status ${status}; continuing"
  fi
}

run_act_pi05_groot() {
  run_stage \
    "ACT CM000-CM009" \
    "${BASELINE_ROOT}/ACT/run_cm000_cm009_evaluation.sh" \
    ACT_EVAL_OBS_PORT ACT_EVAL_ACT_PORT \
    "ACT_SCRIPT_DIR=${BASELINE_ROOT}/ACT" \
    "ACT_PYTHON=${LEROBOT_PYTHON}" \
    "ACT_MODEL_PATH=${ACT_CM_MODEL_PATH}" \
    "ACT_EVAL_START_SEED=${FORMAL_START_SEED}" \
    "ACT_EVAL_NUM_TRIALS=${FORMAL_NUM_TRIALS}" \
    "ACT_EVAL_ATTEMPTS_PER_WORKER=${FORMAL_ATTEMPTS_PER_WORKER}" \
    "ACT_EVAL_RUN_ID=${JOB_TAG}_act_cm" \
    ACT_EVAL_USE_BENCHMARK_MAX_STEPS=1

  run_stage \
    "ACT FC000-FC009" \
    "${BASELINE_ROOT}/ACT/run_fc001_fc009_evaluation.sh" \
    ACT_EVAL_OBS_PORT ACT_EVAL_ACT_PORT \
    "ACT_SCRIPT_DIR=${BASELINE_ROOT}/ACT" \
    "ACT_PYTHON=${LEROBOT_PYTHON}" \
    "ACT_MODEL_PATH=${ACT_FC_MODEL_PATH}" \
    "ACT_EVAL_START_SEED=${FORMAL_START_SEED}" \
    "ACT_EVAL_NUM_TRIALS=${FORMAL_NUM_TRIALS}" \
    "ACT_EVAL_ATTEMPTS_PER_WORKER=${FORMAL_ATTEMPTS_PER_WORKER}" \
    "ACT_EVAL_RUN_ID=${JOB_TAG}_act_fc" \
    ACT_EVAL_USE_BENCHMARK_MAX_STEPS=1 \
    ACT_EVAL_INITIAL_POSITION_MODE=fixed

  run_stage \
    "Pi0.5 CM000-CM009" \
    "${BASELINE_ROOT}/pi05/run_cm000_cm009_evaluation.sh" \
    PI05_EVAL_OBS_PORT PI05_EVAL_ACT_PORT \
    "PI05_SCRIPT_DIR=${BASELINE_ROOT}/pi05" \
    "PI05_PYTHON=${LEROBOT_PYTHON}" \
    "PI05_MODEL_PATH=${PI05_CM_MODEL_PATH}" \
    "PI05_EVAL_START_SEED=${FORMAL_START_SEED}" \
    "PI05_EVAL_NUM_TRIALS=${FORMAL_NUM_TRIALS}" \
    "PI05_EVAL_RUN_ID=${JOB_TAG}_pi05_cm" \
    PI05_EVAL_USE_BENCHMARK_MAX_STEPS=1 \
    PI05_EVAL_INITIAL_POSITION_MODE=fixed

  run_stage \
    "Pi0.5 FC000-FC009" \
    "${BASELINE_ROOT}/pi05/run_fc001_fc009_evaluation.sh" \
    PI05_EVAL_OBS_PORT PI05_EVAL_ACT_PORT \
    "PI05_SCRIPT_DIR=${BASELINE_ROOT}/pi05" \
    "PI05_PYTHON=${LEROBOT_PYTHON}" \
    "PI05_MODEL_PATH=${PI05_FC_MODEL_PATH}" \
    "PI05_EVAL_START_SEED=${FORMAL_START_SEED}" \
    "PI05_EVAL_NUM_TRIALS=${FORMAL_NUM_TRIALS}" \
    "PI05_EVAL_RUN_ID=${JOB_TAG}_pi05_fc" \
    PI05_EVAL_USE_BENCHMARK_MAX_STEPS=1 \
    PI05_EVAL_INITIAL_POSITION_MODE=fixed

  run_stage \
    "GR00T CM000-CM009 (TensorRT full pipeline)" \
    "${GROOT_ROOT}/examples/MotionForge/run_cm000_cm009_evaluation.sh" \
    CM_EVAL_OBS_PORT CM_EVAL_ACT_PORT \
    "GROOT_PYTHON=${GROOT_PYTHON}" \
    "GROOT_MODEL_PATH=${GROOT_CM_MODEL_PATH}" \
    "GROOT_BACKBONE_MODEL_PATH=${GROOT_BACKBONE_MODEL_PATH}" \
    "GROOT_FFMPEG_PREFIX=${GROOT_FFMPEG_PREFIX}" \
    GROOT_ACCEL_MODE=trt_full_pipeline \
    "GROOT_TRT_ENGINE_PATH=${GROOT_CM_TRT_ENGINE_PATH}" \
    "GROOT_EVAL_ATTEMPTS_PER_WORKER=${FORMAL_ATTEMPTS_PER_WORKER}" \
    "CM_EVAL_START_SEED=${FORMAL_START_SEED}" \
    "CM_EVAL_NUM_TRIALS=${FORMAL_NUM_TRIALS}" \
    "CM_EVAL_RUN_ID=${JOB_TAG}_groot_cm"

  run_stage \
    "GR00T FC000-FC009 (TensorRT full pipeline)" \
    "${GROOT_ROOT}/examples/MotionForge/run_fc001_fc009_evaluation.sh" \
    FC_EVAL_OBS_PORT FC_EVAL_ACT_PORT \
    "GROOT_PYTHON=${GROOT_PYTHON}" \
    "GROOT_MODEL_PATH=${GROOT_FC_MODEL_PATH}" \
    "GROOT_BACKBONE_MODEL_PATH=${GROOT_BACKBONE_MODEL_PATH}" \
    "GROOT_FFMPEG_PREFIX=${GROOT_FFMPEG_PREFIX}" \
    GROOT_ACCEL_MODE=trt_full_pipeline \
    "GROOT_TRT_ENGINE_PATH=${GROOT_FC_TRT_ENGINE_PATH}" \
    "GROOT_EVAL_ATTEMPTS_PER_WORKER=${FORMAL_ATTEMPTS_PER_WORKER}" \
    "FC_EVAL_START_SEED=${FORMAL_START_SEED}" \
    "FC_EVAL_NUM_TRIALS=${FORMAL_NUM_TRIALS}" \
    "FC_EVAL_RUN_ID=${JOB_TAG}_groot_fc" \
    FC_EVAL_USE_BENCHMARK_MAX_STEPS=1 \
    FC_EVAL_INITIAL_POSITION_MODE=fixed
}

run_dp_smolvla_xvla() {
  run_stage \
    "Diffusion Policy CM000-CM009" \
    "${BASELINE_ROOT}/DP/run_cm000_cm009_evaluation.sh" \
    DIFFUSION_EVAL_OBS_PORT DIFFUSION_EVAL_DIFFUSION_PORT \
    "DIFFUSION_PYTHON=${LEROBOT_PYTHON}" \
    "DIFFUSION_MODEL_PATH=${DIFFUSION_CM_MODEL_PATH}" \
    "DIFFUSION_EVAL_START_SEED=${FORMAL_START_SEED}" \
    "DIFFUSION_EVAL_NUM_TRIALS=${FORMAL_NUM_TRIALS}" \
    "DIFFUSION_EVAL_RUN_ID=${JOB_TAG}_dp_cm"

  run_stage \
    "Diffusion Policy FC000-FC009" \
    "${BASELINE_ROOT}/DP/run_fc001_fc009_evaluation.sh" \
    DIFFUSION_EVAL_OBS_PORT DIFFUSION_EVAL_ACT_PORT \
    "DIFFUSION_PYTHON=${LEROBOT_PYTHON}" \
    "DIFFUSION_MODEL_PATH=${DIFFUSION_FC_MODEL_PATH}" \
    "DIFFUSION_EVAL_START_SEED=${FORMAL_START_SEED}" \
    "DIFFUSION_EVAL_NUM_TRIALS=${FORMAL_NUM_TRIALS}" \
    "DIFFUSION_EVAL_RUN_ID=${JOB_TAG}_dp_fc" \
    DIFFUSION_EVAL_USE_BENCHMARK_MAX_STEPS=1 \
    DIFFUSION_EVAL_INITIAL_POSITION_MODE=fixed

  run_stage \
    "SmolVLA CM000-CM009" \
    "${BASELINE_ROOT}/smolvla/run_cm000_cm009_evaluation.sh" \
    SMOLVLA_EVAL_OBS_PORT SMOLVLA_EVAL_ACT_PORT \
    "SMOLVLA_SCRIPT_DIR=${BASELINE_ROOT}/smolvla" \
    "SMOLVLA_PYTHON=${LEROBOT_PYTHON}" \
    "SMOLVLA_HF_HOME=${HF_CACHE_ROOT}" \
    "SMOLVLA_MODEL_PATH=${SMOLVLA_CM_MODEL_PATH}" \
    "SMOLVLA_EVAL_START_SEED=${FORMAL_START_SEED}" \
    "SMOLVLA_EVAL_NUM_TRIALS=${FORMAL_NUM_TRIALS}" \
    "SMOLVLA_EVAL_RUN_ID=${JOB_TAG}_smolvla_cm"

  run_stage \
    "SmolVLA FC000-FC009" \
    "${BASELINE_ROOT}/smolvla/run_fc001_fc009_evaluation.sh" \
    SMOLVLA_EVAL_OBS_PORT SMOLVLA_EVAL_ACT_PORT \
    "SMOLVLA_SCRIPT_DIR=${BASELINE_ROOT}/smolvla" \
    "SMOLVLA_PYTHON=${LEROBOT_PYTHON}" \
    "SMOLVLA_HF_HOME=${HF_CACHE_ROOT}" \
    "SMOLVLA_MODEL_PATH=${SMOLVLA_FC_MODEL_PATH}" \
    "SMOLVLA_EVAL_START_SEED=${FORMAL_START_SEED}" \
    "SMOLVLA_EVAL_NUM_TRIALS=${FORMAL_NUM_TRIALS}" \
    "SMOLVLA_EVAL_RUN_ID=${JOB_TAG}_smolvla_fc" \
    SMOLVLA_EVAL_USE_BENCHMARK_MAX_STEPS=1 \
    SMOLVLA_EVAL_INITIAL_POSITION_MODE=fixed

  run_stage \
    "X-VLA CM000-CM009" \
    "${BASELINE_ROOT}/X-VLA/run_cm000_cm009_evaluation.sh" \
    XVLA_EVAL_OBS_PORT XVLA_EVAL_ACT_PORT \
    "XVLA_SCRIPT_DIR=${BASELINE_ROOT}/X-VLA" \
    "XVLA_PYTHON=${LEROBOT_PYTHON}" \
    "XVLA_HF_HOME=${HF_CACHE_ROOT}" \
    "XVLA_MODEL_PATH=${XVLA_CM_MODEL_PATH}" \
    "XVLA_EVAL_START_SEED=${FORMAL_START_SEED}" \
    "XVLA_EVAL_NUM_TRIALS=${FORMAL_NUM_TRIALS}" \
    "XVLA_EVAL_ATTEMPTS_PER_WORKER=${FORMAL_ATTEMPTS_PER_WORKER}" \
    "XVLA_EVAL_RUN_ID=${JOB_TAG}_xvla_cm"

  run_stage \
    "X-VLA FC000-FC009" \
    "${BASELINE_ROOT}/X-VLA/run_fc001_fc009_evaluation.sh" \
    XVLA_EVAL_OBS_PORT XVLA_EVAL_ACT_PORT \
    "XVLA_SCRIPT_DIR=${BASELINE_ROOT}/X-VLA" \
    "XVLA_PYTHON=${LEROBOT_PYTHON}" \
    "XVLA_HF_HOME=${HF_CACHE_ROOT}" \
    "XVLA_MODEL_PATH=${XVLA_FC_MODEL_PATH}" \
    "XVLA_EVAL_START_SEED=${FORMAL_START_SEED}" \
    "XVLA_EVAL_NUM_TRIALS=${FORMAL_NUM_TRIALS}" \
    "XVLA_EVAL_ATTEMPTS_PER_WORKER=${FORMAL_ATTEMPTS_PER_WORKER}" \
    "XVLA_EVAL_RUN_ID=${JOB_TAG}_xvla_fc" \
    XVLA_EVAL_USE_BENCHMARK_MAX_STEPS=1 \
    XVLA_EVAL_INITIAL_POSITION_MODE=fixed
}

run_worker() {
  local expected_hash="${FORMAL_EXPECTED_CONFIG_SHA256:-}"
  local current_hash=""
  local job_number="${SLURM_JOB_ID:-}"

  case "${FORMAL_EVALUATION_GROUP}" in
    act_pi05_groot|dp_smolvla_xvla)
      ;;
    *)
      die "unknown FORMAL_EVALUATION_GROUP: ${FORMAL_EVALUATION_GROUP}"
      ;;
  esac
  [[ "${job_number}" =~ ^[0-9]+$ ]] \
    || die "worker mode requires a numeric SLURM_JOB_ID"
  [[ "${expected_hash}" =~ ^[0-9a-f]{64}$ ]] \
    || die "FORMAL_EXPECTED_CONFIG_SHA256 must be a lowercase SHA-256 digest"

  current_hash="$(benchmark_hash)"
  [[ "${current_hash}" == "${expected_hash}" ]] \
    || die "benchmark configs changed after submission: expected ${expected_hash}, got ${current_hash}"

  if [[ "${FORMAL_VALIDATE_ONLY}" == "1" ]]; then
    RUNNER_DRY_RUN=1
  fi
  case "${FORMAL_EVALUATION_GROUP}" in
    act_pi05_groot)
      PORT_BASE=$((20000 + (10#${job_number} % 1000) * 16))
      ;;
    dp_smolvla_xvla)
      PORT_BASE=$((40000 + (10#${job_number} % 1000) * 16))
      ;;
  esac
  JOB_TAG="formal50_${FORMAL_EVALUATION_GROUP}_${job_number}"

  log "started_at=$(date --iso-8601=seconds)"
  log "slurm_job_id=${job_number} group=${FORMAL_EVALUATION_GROUP}"
  log "trials_per_task=${FORMAL_NUM_TRIALS} attempts_per_worker=${FORMAL_ATTEMPTS_PER_WORKER} seeds=0-49"
  log "tasks_per_stage=10 stages=6 expected_trials=3000"
  log "timing_profile=server_scheduled_120hz max_steps=benchmark_config"
  log "config_sha256=${current_hash} port_range=${PORT_BASE}-$((PORT_BASE + 11))"

  case "${FORMAL_EVALUATION_GROUP}" in
    act_pi05_groot)
      run_act_pi05_groot
      ;;
    dp_smolvla_xvla)
      run_dp_smolvla_xvla
      ;;
  esac

  log "finished_at=$(date --iso-8601=seconds)"
  if ((FAILED_STAGES > 0)); then
    log "failed_stages=${FAILED_STAGES}"
    for failed_label in "${FAILED_STAGE_LABELS[@]}"; do
      log "failure=${failed_label}"
    done
    return 1
  fi
  log "all 6 evaluation stages completed successfully"
}

validate_prerequisites
if [[ -z "${FORMAL_EVALUATION_GROUP}" ]]; then
  submit_jobs
else
  run_worker
fi
