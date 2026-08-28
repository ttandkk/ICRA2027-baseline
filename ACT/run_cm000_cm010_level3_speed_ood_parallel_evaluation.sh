#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
SINGLE_LANE_RUNNER="${SCRIPT_DIR}/run_cm000_cm010_evaluation.sh"
OUTPUT_ROOT="${ACT_CM_LEVEL3_EVAL_OUTPUT_ROOT:-${SCRIPT_DIR}/output/circular_motion/cm000_cm010/level3_speed_ood_fixed/control60_server_scheduled_v1}"
SOURCE_MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"

BATCH_ID="${ACT_CM_LEVEL3_EVAL_BATCH_ID:-act_cm_level3_speed_ood_parallel2_$(date +%Y%m%d_%H%M%S)}"
NUM_TRIALS="${ACT_CM_LEVEL3_EVAL_NUM_TRIALS:-50}"
START_SEED="${ACT_CM_LEVEL3_EVAL_START_SEED:-0}"
GPU_LIST="${ACT_CM_LEVEL3_EVAL_GPUS:-3}"
REQUIRE_IDLE_GPUS="${ACT_CM_LEVEL3_EVAL_REQUIRE_IDLE_GPUS:-1}"
DRY_RUN="${DRY_RUN:-0}"

# The split is balanced by the v2 sum of benchmark max_steps: 11220 vs. 11160.
LANE_ONE_TASKS_VALUE="${ACT_CM_LEVEL3_EVAL_LANE1_TASKS:-cm_000 cm_003 cm_005 cm_007 cm_010}"
LANE_TWO_TASKS_VALUE="${ACT_CM_LEVEL3_EVAL_LANE2_TASKS:-cm_001 cm_002 cm_004 cm_008 cm_009}"
LANE_ONE_OBS_PORT="${ACT_CM_LEVEL3_EVAL_LANE1_OBS_PORT:-3396}"
LANE_ONE_ACT_PORT="${ACT_CM_LEVEL3_EVAL_LANE1_ACT_PORT:-3398}"
LANE_TWO_OBS_PORT="${ACT_CM_LEVEL3_EVAL_LANE2_OBS_PORT:-3496}"
LANE_TWO_ACT_PORT="${ACT_CM_LEVEL3_EVAL_LANE2_ACT_PORT:-3498}"

ALL_TASKS=(
  cm_000
  cm_001
  cm_002
  cm_003
  cm_004
  cm_005
  cm_007
  cm_008
  cm_009
  cm_010
)

read -r -a GPUS <<<"${GPU_LIST}"
read -r -a LANE_ONE_TASKS <<<"${LANE_ONE_TASKS_VALUE}"
read -r -a LANE_TWO_TASKS <<<"${LANE_TWO_TASKS_VALUE}"

LANE_PIDS=()
TEMP_MOTIONFORGE_ROOT=""
LANE_ONE_GPU=""
LANE_TWO_GPU=""
LANE_EXECUTION_MODE=""

log() {
  printf '[ACT-CM-LEVEL3-SPEED-OOD] %s\n' "$*"
}

die() {
  printf '[ACT-CM-LEVEL3-SPEED-OOD] ERROR: %s\n' "$*" >&2
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
    [[ "${task}" =~ ^cm_(00[0-5]|00[7-9]|010)$ ]] \
      || die "invalid CM task id in lane partition: ${task}"
    [[ -z "${seen[${task}]:-}" ]] || die "task appears in more than one lane: ${task}"
    seen["${task}"]=1
  done
  for expected in "${ALL_TASKS[@]}"; do
    [[ -n "${seen[${expected}]:-}" ]] || die "lane partition is missing task: ${expected}"
  done
  ((${#seen[@]} == ${#ALL_TASKS[@]})) \
    || die "lane partition must contain exactly the 10 supported CM tasks"
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
  local gpu=""

  [[ -x "${SINGLE_LANE_RUNNER}" ]] \
    || die "single-lane runner is missing or not executable: ${SINGLE_LANE_RUNNER}"
  [[ -d "${SOURCE_MOTIONFORGE_ROOT}" ]] \
    || die "MotionForge root does not exist: ${SOURCE_MOTIONFORGE_ROOT}"
  [[ -d "${SOURCE_MOTIONFORGE_ROOT}/configs/benchmarks/circular_motion" ]] \
    || die "CM benchmark directory does not exist under ${SOURCE_MOTIONFORGE_ROOT}"
  awk '
    /^[[:space:]]*--initial_position_mode[[:space:]]*$/ {
      if (getline > 0 && $0 ~ /^[[:space:]]*fixed[[:space:]]*$/) {
        found = 1
      }
    }
    END { exit !found }
  ' "${SINGLE_LANE_RUNNER}" \
    || die "single-lane runner no longer guarantees fixed initial positions"
  command -v awk >/dev/null 2>&1 || die "awk is required"
  command -v grep >/dev/null 2>&1 || die "grep is required"
  command -v ln >/dev/null 2>&1 || die "ln is required"
  command -v mkdir >/dev/null 2>&1 || die "mkdir is required"
  command -v mktemp >/dev/null 2>&1 || die "mktemp is required"
  command -v rm >/dev/null 2>&1 || die "rm is required"

  [[ "${BATCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "ACT_CM_LEVEL3_EVAL_BATCH_ID may contain only letters, numbers, dot, underscore, and hyphen"
  require_uint_at_least "ACT_CM_LEVEL3_EVAL_NUM_TRIALS" "${NUM_TRIALS}" 1
  require_uint_at_least "ACT_CM_LEVEL3_EVAL_START_SEED" "${START_SEED}" 0
  [[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
  [[ "${REQUIRE_IDLE_GPUS}" == "0" || "${REQUIRE_IDLE_GPUS}" == "1" ]] \
    || die "ACT_CM_LEVEL3_EVAL_REQUIRE_IDLE_GPUS must be 0 or 1"
  ((${#GPUS[@]} == 1)) && [[ "${GPUS[0]}" == "3" ]] \
    || die "ACT_CM_LEVEL3_EVAL_GPUS must contain only GPU 3 for the CM v2 protocol"
  LANE_EXECUTION_MODE="sequential"
  LANE_ONE_GPU="3"
  LANE_TWO_GPU="3"

  validate_task_partition
  validate_distinct_ports

  LANE_ONE_RUN_ID="${BATCH_ID}_lane1_gpu${LANE_ONE_GPU}"
  LANE_TWO_RUN_ID="${BATCH_ID}_lane2_gpu${LANE_TWO_GPU}"
  LANE_ONE_RESULT_DIR="${OUTPUT_ROOT}/${LANE_ONE_RUN_ID}"
  LANE_TWO_RESULT_DIR="${OUTPUT_ROOT}/${LANE_TWO_RUN_ID}"
  COMBINED_SUMMARY="${OUTPUT_ROOT}/${BATCH_ID}_parallel_summary.txt"

  if [[ "${DRY_RUN}" == "0" ]]; then
    [[ ! -e "${LANE_ONE_RESULT_DIR}" ]] || die "lane 1 result already exists: ${LANE_ONE_RESULT_DIR}"
    [[ ! -e "${LANE_TWO_RESULT_DIR}" ]] || die "lane 2 result already exists: ${LANE_TWO_RESULT_DIR}"
    [[ ! -e "${COMBINED_SUMMARY}" ]] || die "combined summary already exists: ${COMBINED_SUMMARY}"
    validate_runtime_resources
    mkdir -p "${OUTPUT_ROOT}" || die "failed to create ACT output directory: ${OUTPUT_ROOT}"
  fi
}

cleanup_temp_motionforge_root() {
  local root="${TEMP_MOTIONFORGE_ROOT}"
  [[ -n "${root}" ]] || return 0
  TEMP_MOTIONFORGE_ROOT=""
  case "${root}" in
    /tmp/act_cm_level3.*)
      rm -rf -- "${root}"
      ;;
    *)
      log "refusing to remove unexpected temporary path: ${root}"
      return 1
      ;;
  esac
}

create_level3_motionforge_root() {
  local task=""
  local source_config=""
  local level3_config=""

  TEMP_MOTIONFORGE_ROOT="$(mktemp -d /tmp/act_cm_level3.XXXXXXXXXX)" \
    || die "failed to create temporary MotionForge root"
  mkdir -p "${TEMP_MOTIONFORGE_ROOT}/configs/benchmarks/circular_motion"
  ln -s "${SOURCE_MOTIONFORGE_ROOT}/scripts" "${TEMP_MOTIONFORGE_ROOT}/scripts"
  ln -s "${SOURCE_MOTIONFORGE_ROOT}/source" "${TEMP_MOTIONFORGE_ROOT}/source"

  for task in "${ALL_TASKS[@]}"; do
    source_config="${SOURCE_MOTIONFORGE_ROOT}/configs/benchmarks/circular_motion/${task}_rgb_gr00t_zmq.yaml"
    level3_config="${TEMP_MOTIONFORGE_ROOT}/configs/benchmarks/circular_motion/${task}_rgb_gr00t_zmq.yaml"
    [[ -f "${source_config}" ]] || die "benchmark config not found: ${source_config}"
    if ! awk '
      $0 == "  motion_level: level2" {
        print "  motion_level: level3"
        replacements += 1
        next
      }
      { print }
      END { if (replacements != 1) exit 42 }
    ' "${source_config}" >"${level3_config}"; then
      die "expected exactly one runtime motion_level: level2 entry in ${source_config}"
    fi
  done

  for task in "${ALL_TASKS[@]}"; do
    level3_config="${TEMP_MOTIONFORGE_ROOT}/configs/benchmarks/circular_motion/${task}_rgb_gr00t_zmq.yaml"
    [[ "$(awk '/^[[:space:]]*motion_level:/ { print $2; exit }' "${level3_config}")" == "level3" ]] \
      || die "temporary benchmark did not resolve level3 for ${task}"
  done
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
    MOTIONFORGE_ROOT="${TEMP_MOTIONFORGE_ROOT}" \
    MOTIONFORGE_DEVICE="cpu" \
    ACT_EVAL_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    ACT_EVAL_TASKS="${tasks}" \
    ACT_EVAL_RUN_ID="${run_id}" \
    ACT_EVAL_NUM_TRIALS="${NUM_TRIALS}" \
    ACT_EVAL_START_SEED="${START_SEED}" \
    ACT_EVAL_CUDA_VISIBLE_DEVICES="${gpu}" \
    ACT_EVAL_CLOCK_MODE="slowdown_scaled" \
    ACT_EVAL_OBS_PORT="${obs_port}" \
    ACT_EVAL_ACT_PORT="${act_port}" \
    ACT_EVAL_VIDEO_OUTCOME_SUFFIX="1" \
    bash "${SINGLE_LANE_RUNNER}"
}

run_lane_and_wait() {
  local pid=""
  local status=0

  run_lane "$@" &
  pid="$!"
  LANE_PIDS=("${pid}")
  set +e
  wait "${pid}"
  status="$?"
  set -e
  LANE_PIDS=()
  return "${status}"
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
  LANE_PIDS=()
}

handle_signal() {
  local status="$1"
  terminate_lanes
  exit "${status}"
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
  local completed_tasks=0
  local completed_trials=0
  local total_successes=0
  local total_failures=0
  local overall_status="incomplete"
  local total_success_rate="N/A"
  local partial_success_rate="N/A"

  {
    printf 'ACT CM000-CM010 Level 3 speed OOD parallel evaluation\n'
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'batch_id=%s\n' "${BATCH_ID}"
    printf 'source_benchmark_motion_level=level2\n'
    printf 'evaluation_motion_level=level3\n'
    printf 'initial_position_mode=fixed\n'
    printf 'ood_axis=speed\n'
    printf 'clock_mode=slowdown_scaled\n'
    printf 'timing_protocol=motionforge.slowdown_scaled.server.v1\n'
    printf 'control_hz=60\n'
    printf 'trials_per_task=%s\n' "${NUM_TRIALS}"
    printf 'seed_start=%s\n' "${START_SEED}"
    printf 'seed_end=%s\n' "$((START_SEED + NUM_TRIALS - 1))"
    printf 'max_steps_per_trial=benchmark_config\n'
    printf 'lane_execution_mode=%s\n' "${LANE_EXECUTION_MODE}"
    printf 'lane1_gpu=%s\n' "${LANE_ONE_GPU}"
    printf 'lane1_ports=%s/%s\n' "${LANE_ONE_OBS_PORT}" "${LANE_ONE_ACT_PORT}"
    printf 'lane1_tasks=%s\n' "${LANE_ONE_TASKS[*]}"
    printf 'lane1_exit_status=%s\n' "${lane_one_status}"
    printf 'lane1_summary=%s\n' "${LANE_ONE_RESULT_DIR}/success_rates.txt"
    printf 'lane2_gpu=%s\n' "${LANE_TWO_GPU}"
    printf 'lane2_ports=%s/%s\n' "${LANE_TWO_OBS_PORT}" "${LANE_TWO_ACT_PORT}"
    printf 'lane2_tasks=%s\n' "${LANE_TWO_TASKS[*]}"
    printf 'lane2_exit_status=%s\n' "${lane_two_status}"
    printf 'lane2_summary=%s\n' "${LANE_TWO_RESULT_DIR}/success_rates.txt"

    for task_id in "${ALL_TASKS[@]}"; do
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
      printf 'ood_axis=speed\n'
      if [[ "${status}" == "completed" && "${trials}" =~ ^[0-9]+$ && "${successes}" =~ ^[0-9]+$ && "${failures}" =~ ^[0-9]+$ ]]; then
        ((completed_tasks += 1))
        completed_trials=$((completed_trials + trials))
        total_successes=$((total_successes + successes))
        total_failures=$((total_failures + failures))
      fi
    done

    if ((completed_trials > 0)); then
      partial_success_rate="$(awk -v successes="${total_successes}" -v trials="${completed_trials}" 'BEGIN { printf "%.3f", successes / trials }')"
    fi
    if ((completed_tasks == ${#ALL_TASKS[@]} && lane_one_status == 0 && lane_two_status == 0)); then
      overall_status="completed"
      total_success_rate="${partial_success_rate}"
    fi
    printf '\n[overall]\n'
    printf 'status=%s\n' "${overall_status}"
    printf 'expected_tasks=%s\n' "${#ALL_TASKS[@]}"
    printf 'completed_tasks=%s\n' "${completed_tasks}"
    printf 'expected_trials=%s\n' "$((${#ALL_TASKS[@]} * NUM_TRIALS))"
    printf 'completed_trials=%s\n' "${completed_trials}"
    printf 'total_successes=%s\n' "${total_successes}"
    printf 'total_failures=%s\n' "${total_failures}"
    printf 'total_success_rate=%s\n' "${total_success_rate}"
    printf 'partial_success_rate=%s\n' "${partial_success_rate}"
  } >"${COMBINED_SUMMARY}"
}

validate_configuration
trap cleanup_temp_motionforge_root EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
create_level3_motionforge_root

log "batch=${BATCH_ID} motion_level=level3 trials_per_task=${NUM_TRIALS} seeds=${START_SEED}-$((START_SEED + NUM_TRIALS - 1)) initial_position_mode=fixed lane_execution_mode=${LANE_EXECUTION_MODE} gpu_list=${GPUS[*]}"
log "temporary level3 benchmarks=${TEMP_MOTIONFORGE_ROOT}/configs/benchmarks/circular_motion"

if [[ "${LANE_EXECUTION_MODE}" == "sequential" ]]; then
  if run_lane_and_wait \
    lane1 \
    "${LANE_ONE_GPU}" \
    "${LANE_ONE_OBS_PORT}" \
    "${LANE_ONE_ACT_PORT}" \
    "${LANE_ONE_RUN_ID}" \
    "${LANE_ONE_TASKS[*]}"; then
    LANE_ONE_STATUS=0
  else
    LANE_ONE_STATUS="$?"
  fi
  if run_lane_and_wait \
    lane2 \
    "${LANE_TWO_GPU}" \
    "${LANE_TWO_OBS_PORT}" \
    "${LANE_TWO_ACT_PORT}" \
    "${LANE_TWO_RUN_ID}" \
    "${LANE_TWO_TASKS[*]}"; then
    LANE_TWO_STATUS=0
  else
    LANE_TWO_STATUS="$?"
  fi
else
  run_lane \
    lane1 \
    "${LANE_ONE_GPU}" \
    "${LANE_ONE_OBS_PORT}" \
    "${LANE_ONE_ACT_PORT}" \
    "${LANE_ONE_RUN_ID}" \
    "${LANE_ONE_TASKS[*]}" &
  LANE_ONE_PID="$!"
  LANE_PIDS+=("${LANE_ONE_PID}")

  run_lane \
    lane2 \
    "${LANE_TWO_GPU}" \
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
fi

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
