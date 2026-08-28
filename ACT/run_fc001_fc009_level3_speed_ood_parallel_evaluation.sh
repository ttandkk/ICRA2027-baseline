#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SINGLE_LANE_RUNNER="${SCRIPT_DIR}/run_fc001_fc009_evaluation.sh"
OUTPUT_ROOT="${SCRIPT_DIR}/output"

BATCH_ID="${ACT_LEVEL3_EVAL_BATCH_ID:-act_fc_level3_speed_ood_parallel2_$(date +%Y%m%d_%H%M%S)}"
NUM_TRIALS="${ACT_LEVEL3_EVAL_NUM_TRIALS:-50}"
START_SEED="${ACT_LEVEL3_EVAL_START_SEED:-0}"
GPU_LIST="${ACT_LEVEL3_EVAL_GPUS:-2 3}"
INITIAL_POSITION_MODE="${ACT_LEVEL3_EVAL_INITIAL_POSITION_MODE:-fixed}"
REQUIRE_IDLE_GPUS="${ACT_LEVEL3_EVAL_REQUIRE_IDLE_GPUS:-1}"
DRY_RUN="${DRY_RUN:-0}"

LANE_ONE_TASKS_VALUE="${ACT_LEVEL3_EVAL_LANE1_TASKS:-fc_001 fc_003 fc_006}"
LANE_TWO_TASKS_VALUE="${ACT_LEVEL3_EVAL_LANE2_TASKS:-fc_002 fc_004 fc_005 fc_007 fc_008 fc_009}"
LANE_ONE_OBS_PORT="${ACT_LEVEL3_EVAL_LANE1_OBS_PORT:-3596}"
LANE_ONE_ACT_PORT="${ACT_LEVEL3_EVAL_LANE1_ACT_PORT:-3598}"
LANE_TWO_OBS_PORT="${ACT_LEVEL3_EVAL_LANE2_OBS_PORT:-3696}"
LANE_TWO_ACT_PORT="${ACT_LEVEL3_EVAL_LANE2_ACT_PORT:-3698}"

read -r -a GPUS <<<"${GPU_LIST}"
read -r -a LANE_ONE_TASKS <<<"${LANE_ONE_TASKS_VALUE}"
read -r -a LANE_TWO_TASKS <<<"${LANE_TWO_TASKS_VALUE}"

LANE_PIDS=()

log() {
  printf '[ACT-FC-LEVEL3-SPEED-OOD] %s\n' "$*"
}

die() {
  printf '[ACT-FC-LEVEL3-SPEED-OOD] ERROR: %s\n' "$*" >&2
  exit 1
}

require_uint_at_least() {
  local name="$1"
  local value="$2"
  local minimum="$3"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || ((10#${value} < minimum)); then
    die "${name} must be an integer >= ${minimum}, got ${value}"
  fi
}

validate_task_partition() {
  local task=""
  local expected=""
  declare -A seen=()

  ((${#LANE_ONE_TASKS[@]} > 0)) || die "lane 1 must contain at least one task"
  ((${#LANE_TWO_TASKS[@]} > 0)) || die "lane 2 must contain at least one task"

  for task in "${LANE_ONE_TASKS[@]}" "${LANE_TWO_TASKS[@]}"; do
    [[ "${task}" =~ ^fc_00[1-9]$ ]] || die "invalid task id in lane partition: ${task}"
    [[ -z "${seen[${task}]:-}" ]] || die "task appears in more than one lane: ${task}"
    seen["${task}"]=1
  done
  for expected in fc_00{1..9}; do
    [[ -n "${seen[${expected}]:-}" ]] || die "lane partition is missing task: ${expected}"
  done
  ((${#seen[@]} == 9)) || die "lane partition must contain exactly fc_001 through fc_009"
}

validate_distinct_ports() {
  local port=""
  declare -A seen=()
  for port in \
    "${LANE_ONE_OBS_PORT}" \
    "${LANE_ONE_ACT_PORT}" \
    "${LANE_TWO_OBS_PORT}" \
    "${LANE_TWO_ACT_PORT}"; do
    require_uint_at_least "lane port" "${port}" 1
    ((10#${port} <= 65535)) || die "lane port must be <= 65535, got ${port}"
    [[ -z "${seen[${port}]:-}" ]] || die "lane ports must be distinct, duplicate: ${port}"
    seen["${port}"]=1
  done
}

port_is_listening() {
  local port="$1"
  ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$"
}

gpu_processes() {
  local gpu="$1"
  nvidia-smi \
    --id="${gpu}" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || return 1
}

validate_runtime_resources() {
  local gpu=""
  local processes=""
  local port=""

  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required for the GPU preflight"
  command -v ss >/dev/null 2>&1 || die "ss is required for the port preflight"

  for gpu in "${GPUS[@]}"; do
    nvidia-smi --id="${gpu}" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1 \
      || die "GPU ${gpu} is not available"
    if [[ "${REQUIRE_IDLE_GPUS}" == "1" ]]; then
      processes="$(gpu_processes "${gpu}")" || die "failed to inspect GPU ${gpu} processes"
      [[ -z "${processes}" ]] \
        || die "GPU ${gpu} is not idle; active compute processes: ${processes}"
    fi
  done

  for port in \
    "${LANE_ONE_OBS_PORT}" \
    "${LANE_ONE_ACT_PORT}" \
    "${LANE_TWO_OBS_PORT}" \
    "${LANE_TWO_ACT_PORT}"; do
    if port_is_listening "${port}"; then
      die "TCP port ${port} is already listening"
    fi
  done
}

validate_configuration() {
  [[ -x "${SINGLE_LANE_RUNNER}" ]] || die "single-lane runner is missing or not executable: ${SINGLE_LANE_RUNNER}"
  [[ -d "${OUTPUT_ROOT}" ]] || die "ACT output directory does not exist: ${OUTPUT_ROOT}"
  [[ "${BATCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "ACT_LEVEL3_EVAL_BATCH_ID may contain only letters, numbers, dot, underscore, and hyphen"
  require_uint_at_least "ACT_LEVEL3_EVAL_NUM_TRIALS" "${NUM_TRIALS}" 1
  require_uint_at_least "ACT_LEVEL3_EVAL_START_SEED" "${START_SEED}" 0
  [[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
  [[ "${REQUIRE_IDLE_GPUS}" == "0" || "${REQUIRE_IDLE_GPUS}" == "1" ]] \
    || die "ACT_LEVEL3_EVAL_REQUIRE_IDLE_GPUS must be 0 or 1"
  [[ "${INITIAL_POSITION_MODE}" == "fixed" || "${INITIAL_POSITION_MODE}" == "seeded" ]] \
    || die "ACT_LEVEL3_EVAL_INITIAL_POSITION_MODE must be fixed or seeded"
  ((${#GPUS[@]} == 2)) || die "ACT_LEVEL3_EVAL_GPUS must contain exactly two GPU indices"
  [[ "${GPUS[0]}" =~ ^[0-9]+$ && "${GPUS[1]}" =~ ^[0-9]+$ ]] \
    || die "ACT_LEVEL3_EVAL_GPUS must contain non-negative integer GPU indices"
  [[ "${GPUS[0]}" != "${GPUS[1]}" ]] || die "parallel lanes must use different GPUs"

  validate_task_partition
  validate_distinct_ports

  LANE_ONE_RUN_ID="${BATCH_ID}_lane1_gpu${GPUS[0]}"
  LANE_TWO_RUN_ID="${BATCH_ID}_lane2_gpu${GPUS[1]}"
  LANE_ONE_RESULT_DIR="${OUTPUT_ROOT}/${LANE_ONE_RUN_ID}"
  LANE_TWO_RESULT_DIR="${OUTPUT_ROOT}/${LANE_TWO_RUN_ID}"
  COMBINED_SUMMARY="${OUTPUT_ROOT}/${BATCH_ID}_parallel_summary.txt"

  if [[ "${DRY_RUN}" == "0" ]]; then
    [[ ! -e "${LANE_ONE_RESULT_DIR}" ]] || die "lane 1 result already exists: ${LANE_ONE_RESULT_DIR}"
    [[ ! -e "${LANE_TWO_RESULT_DIR}" ]] || die "lane 2 result already exists: ${LANE_TWO_RESULT_DIR}"
    [[ ! -e "${COMBINED_SUMMARY}" ]] || die "combined summary already exists: ${COMBINED_SUMMARY}"
    validate_runtime_resources
  fi
}

run_lane() {
  local lane_name="$1"
  local gpu="$2"
  local obs_port="$3"
  local act_port="$4"
  local run_id="$5"
  local tasks="$6"

  log "launching ${lane_name} gpu=${gpu} ports=${obs_port}/${act_port} tasks=${tasks}"
  exec env \
    DRY_RUN="${DRY_RUN}" \
    ACT_EVAL_TASKS="${tasks}" \
    ACT_EVAL_RUN_ID="${run_id}" \
    ACT_EVAL_NUM_TRIALS="${NUM_TRIALS}" \
    ACT_EVAL_START_SEED="${START_SEED}" \
    ACT_EVAL_CUDA_VISIBLE_DEVICES="${gpu}" \
    ACT_EVAL_OBS_PORT="${obs_port}" \
    ACT_EVAL_ACT_PORT="${act_port}" \
    ACT_EVAL_MOTION_LEVEL="level3" \
    ACT_EVAL_INITIAL_POSITION_MODE="${INITIAL_POSITION_MODE}" \
    ACT_EVAL_USE_BENCHMARK_MAX_STEPS="1" \
    MOTIONFORGE_DEVICE="cpu" \
    bash "${SINGLE_LANE_RUNNER}"
}

terminate_lanes() {
  local pid=""
  trap - INT TERM
  for pid in "${LANE_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${LANE_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

summary_value() {
  local summary_file="$1"
  local task_id="$2"
  local key="$3"
  awk -v section="[${task_id}]" -v key="${key}" '
    $0 == section { active = 1; next }
    active && /^\[/ { exit }
    active && index($0, key "=") == 1 {
      sub("^" key "=", "")
      print
      exit
    }
  ' "${summary_file}"
}

task_summary_file() {
  local task_id="$1"
  local candidate=""
  for candidate in "${LANE_ONE_TASKS[@]}"; do
    if [[ "${candidate}" == "${task_id}" ]]; then
      printf '%s\n' "${LANE_ONE_RESULT_DIR}/success_rates.txt"
      return 0
    fi
  done
  printf '%s\n' "${LANE_TWO_RESULT_DIR}/success_rates.txt"
}

write_combined_summary() {
  local lane_one_status="$1"
  local lane_two_status="$2"
  local task_id=""
  local task_file=""
  local status=""
  local trials=""
  local successes=""
  local failures=""
  local success_rate=""
  local pure_completed_tasks=0
  local pure_completed_trials=0
  local pure_successes=0
  local pure_failures=0
  local pure_success_rate="N/A"

  {
    printf 'ACT FC001-FC009 Level 3 speed OOD parallel evaluation\n'
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'batch_id=%s\n' "${BATCH_ID}"
    printf 'motion_level=level3\n'
    printf 'training_distribution=level2_seeded\n'
    printf 'initial_position_mode=%s\n' "${INITIAL_POSITION_MODE}"
    printf 'trials_per_task=%s\n' "${NUM_TRIALS}"
    printf 'seed_start=%s\n' "${START_SEED}"
    printf 'seed_end=%s\n' "$((START_SEED + NUM_TRIALS - 1))"
    printf 'max_steps_per_trial=benchmark_config\n'
    printf 'lane1_gpu=%s\n' "${GPUS[0]}"
    printf 'lane1_ports=%s/%s\n' "${LANE_ONE_OBS_PORT}" "${LANE_ONE_ACT_PORT}"
    printf 'lane1_tasks=%s\n' "${LANE_ONE_TASKS[*]}"
    printf 'lane1_exit_status=%s\n' "${lane_one_status}"
    printf 'lane1_summary=%s\n' "${LANE_ONE_RESULT_DIR}/success_rates.txt"
    printf 'lane2_gpu=%s\n' "${GPUS[1]}"
    printf 'lane2_ports=%s/%s\n' "${LANE_TWO_OBS_PORT}" "${LANE_TWO_ACT_PORT}"
    printf 'lane2_tasks=%s\n' "${LANE_TWO_TASKS[*]}"
    printf 'lane2_exit_status=%s\n' "${lane_two_status}"
    printf 'lane2_summary=%s\n' "${LANE_TWO_RESULT_DIR}/success_rates.txt"
    printf 'fc002_ood_axes=speed,initial_position\n'
    printf 'fc002_pure_speed_aggregate_included=false\n'

    for task_id in fc_00{1..9}; do
      task_file="$(task_summary_file "${task_id}")"
      status="N/A"
      trials="0"
      successes="0"
      failures="0"
      success_rate="N/A"
      if [[ -f "${task_file}" ]]; then
        status="$(summary_value "${task_file}" "${task_id}" status)"
        trials="$(summary_value "${task_file}" "${task_id}" trials)"
        successes="$(summary_value "${task_file}" "${task_id}" successes)"
        failures="$(summary_value "${task_file}" "${task_id}" failures)"
        success_rate="$(summary_value "${task_file}" "${task_id}" success_rate)"
        status="${status:-N/A}"
        trials="${trials:-0}"
        successes="${successes:-0}"
        failures="${failures:-0}"
        success_rate="${success_rate:-N/A}"
      fi
      printf '\n[%s]\n' "${task_id}"
      printf 'status=%s\n' "${status}"
      printf 'trials=%s\n' "${trials}"
      printf 'successes=%s\n' "${successes}"
      printf 'failures=%s\n' "${failures}"
      printf 'success_rate=%s\n' "${success_rate}"
      if [[ "${task_id}" == "fc_002" ]]; then
        printf 'ood_axes=speed,initial_position\n'
      else
        printf 'ood_axes=speed\n'
        if [[ "${status}" == "completed" && "${trials}" =~ ^[0-9]+$ && "${successes}" =~ ^[0-9]+$ && "${failures}" =~ ^[0-9]+$ ]]; then
          ((pure_completed_tasks += 1))
          pure_completed_trials=$((pure_completed_trials + trials))
          pure_successes=$((pure_successes + successes))
          pure_failures=$((pure_failures + failures))
        fi
      fi
    done

    if ((pure_completed_trials > 0)); then
      pure_success_rate="$(awk -v successes="${pure_successes}" -v trials="${pure_completed_trials}" 'BEGIN { printf "%.3f", successes / trials }')"
    fi
    printf '\n[pure_speed_overall]\n'
    printf 'expected_tasks=8\n'
    printf 'completed_tasks=%s\n' "${pure_completed_tasks}"
    printf 'expected_trials=%s\n' "$((8 * NUM_TRIALS))"
    printf 'completed_trials=%s\n' "${pure_completed_trials}"
    printf 'total_successes=%s\n' "${pure_successes}"
    printf 'total_failures=%s\n' "${pure_failures}"
    printf 'total_success_rate=%s\n' "${pure_success_rate}"
  } >"${COMBINED_SUMMARY}"
}

validate_configuration

trap 'terminate_lanes; exit 130' INT
trap 'terminate_lanes; exit 143' TERM

log "batch=${BATCH_ID} trials_per_task=${NUM_TRIALS} seeds=${START_SEED}-$((START_SEED + NUM_TRIALS - 1)) initial_position_mode=${INITIAL_POSITION_MODE}"
log "FC002 is tagged speed+initial_position OOD and excluded from pure-speed aggregate"

run_lane \
  lane1 \
  "${GPUS[0]}" \
  "${LANE_ONE_OBS_PORT}" \
  "${LANE_ONE_ACT_PORT}" \
  "${LANE_ONE_RUN_ID}" \
  "${LANE_ONE_TASKS[*]}" &
LANE_ONE_PID="$!"
LANE_PIDS+=("${LANE_ONE_PID}")

run_lane \
  lane2 \
  "${GPUS[1]}" \
  "${LANE_TWO_OBS_PORT}" \
  "${LANE_TWO_ACT_PORT}" \
  "${LANE_TWO_RUN_ID}" \
  "${LANE_TWO_TASKS[*]}" &
LANE_TWO_PID="$!"
LANE_PIDS+=("${LANE_TWO_PID}")

set +e
wait "${LANE_ONE_PID}"
LANE_ONE_STATUS="$?"
wait "${LANE_TWO_PID}"
LANE_TWO_STATUS="$?"
set -e
LANE_PIDS=()

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run completed lane1_status=${LANE_ONE_STATUS} lane2_status=${LANE_TWO_STATUS}; no result files were created"
else
  write_combined_summary "${LANE_ONE_STATUS}" "${LANE_TWO_STATUS}"
  log "lane1_results=${LANE_ONE_RESULT_DIR}"
  log "lane2_results=${LANE_TWO_RESULT_DIR}"
  log "combined_summary=${COMBINED_SUMMARY}"
fi

if ((LANE_ONE_STATUS != 0 || LANE_TWO_STATUS != 0)); then
  exit 1
fi
