#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DYNAMICVLA_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
BASELINE_ROOT="$(cd -- "${DYNAMICVLA_ROOT}/.." && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${BASELINE_ROOT}/.." && pwd -P)"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"

BRIDGE_CLIENT="${SCRIPT_DIR}/motionforge_dynamicvla_bridge_client.py"
TRIALS_SERVER="${MOTIONFORGE_ROOT}/scripts/benchmark/run_env_server_trials.py"
BENCHMARK_DIR="${MOTIONFORGE_ROOT}/configs/benchmarks/factory_conveyor"

DYNAMICVLA_MODEL_PATH="${DYNAMICVLA_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/MotionforgeGroup/DynamicVLA/FC-80000}"
DYNAMICVLA_PYTHON="${DYNAMICVLA_PYTHON:-${BASELINE_ROOT}/.conda/dynamicvla-train310/bin/python}"
DYNAMICVLA_DEVICE="${DYNAMICVLA_DEVICE:-cuda:0}"
DYNAMICVLA_ACTION_HZ="${DYNAMICVLA_ACTION_HZ:-30}"
DYNAMICVLA_MAX_INFERENCE_HZ="${DYNAMICVLA_MAX_INFERENCE_HZ:-30}"
DYNAMICVLA_SEND_HORIZON="${DYNAMICVLA_SEND_HORIZON:-16}"
DYNAMICVLA_EXECUTION_HORIZON="${DYNAMICVLA_EXECUTION_HORIZON:-8}"
DYNAMICVLA_PRINT_EVERY="${DYNAMICVLA_PRINT_EVERY:-10}"

MOTIONFORGE_CONDA_ENV="${MOTIONFORGE_CONDA_ENV:-motionforge}"
# Physics runs on CPU; AppLauncher rendering and DynamicVLA use the selected GPU.
MOTIONFORGE_DEVICE="${MOTIONFORGE_DEVICE:-cpu}"
DYNAMICVLA_EVAL_CUDA_VISIBLE_DEVICES="${DYNAMICVLA_EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-2}}"
DYNAMICVLA_HF_HOME="${DYNAMICVLA_HF_HOME:-${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}}"
CONDA_EXE="${MOTIONFORGE_CONDA_EXE:-${WORKSPACE_ROOT}/miniconda3/bin/conda}"
TIMEOUT_EXE="${MOTIONFORGE_TIMEOUT_EXE:-$(command -v timeout || true)}"

START_SEED="${DYNAMICVLA_EVAL_START_SEED:-0}"
NUM_TRIALS="${DYNAMICVLA_EVAL_NUM_TRIALS:-50}"
MAX_STEPS="${DYNAMICVLA_EVAL_MAX_STEPS:-1400}"
USE_BENCHMARK_MAX_STEPS="${DYNAMICVLA_EVAL_USE_BENCHMARK_MAX_STEPS:-1}"
MOTION_LEVEL="${DYNAMICVLA_EVAL_MOTION_LEVEL:-}"
INITIAL_POSITION_MODE="${DYNAMICVLA_EVAL_INITIAL_POSITION_MODE:-fixed}"
CLOCK_MODE="${DYNAMICVLA_EVAL_CLOCK_MODE:-wall_clock_strict}"
ACTION_ALIGNMENT="observation_aligned"
OBS_PORT="${DYNAMICVLA_EVAL_OBS_PORT:-3196}"
ACT_PORT="${DYNAMICVLA_EVAL_ACT_PORT:-3198}"
CLIENT_WARMUP_S="${DYNAMICVLA_EVAL_CLIENT_WARMUP_S:-5}"
TASK_TIMEOUT_S="${DYNAMICVLA_EVAL_TASK_TIMEOUT_S:-14400}"
BETWEEN_TASKS_S="${DYNAMICVLA_EVAL_BETWEEN_TASKS_S:-5}"

VIDEO_WIDTH="${DYNAMICVLA_EVAL_VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${DYNAMICVLA_EVAL_VIDEO_HEIGHT:-480}"
VIDEO_STRIDE="${DYNAMICVLA_EVAL_VIDEO_STRIDE:-1}"
VIDEO_OUTCOME_SUFFIX="${DYNAMICVLA_EVAL_VIDEO_OUTCOME_SUFFIX:-1}"

RESULT_ROOT="${SCRIPT_DIR}/output"
RUN_ID="${DYNAMICVLA_EVAL_RUN_ID:-dynamicvla_fc001_fc009_id${NUM_TRIALS}_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_ROOT}/${RUN_ID}"
SUMMARY_FILE="${RESULT_DIR}/success_rates.txt"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_TASK_IDS=(
  fc_001
  fc_002
  fc_003
  fc_004
  fc_005
  fc_006
  fc_007
  fc_008
  fc_009
)

if [[ -n "${DYNAMICVLA_EVAL_TASKS:-}" ]]; then
  read -r -a TASK_IDS <<<"${DYNAMICVLA_EVAL_TASKS}"
else
  TASK_IDS=("${DEFAULT_TASK_IDS[@]}")
fi

SERVER_PID=""
BRIDGE_PID=""
SERVER_COMMAND=()
BRIDGE_COMMAND=()

LAST_TRIALS=0
LAST_SUCCESSES=0
LAST_FAILURES=0
LAST_SUCCESS_RATE=""
LAST_RAW_SUMMARY=""

log() {
  printf '[DYNAMICVLA-FC001-FC009-EVAL] %s\n' "$*"
}

die() {
  printf '[DYNAMICVLA-FC001-FC009-EVAL] ERROR: %s\n' "$*" >&2
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

require_uint_at_least() {
  local name="$1"
  local value="$2"
  local minimum="$3"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || ((10#${value} < minimum)); then
    die "${name} must be an integer >= ${minimum}, got ${value}"
  fi
}

require_positive_number() {
  local name="$1"
  local value="$2"
  awk -v value="${value}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }' \
    || die "${name} must be a positive number, got ${value}"
}

validate_checkpoint() {
  require_dir "${DYNAMICVLA_MODEL_PATH}"
  require_file "${DYNAMICVLA_MODEL_PATH}/config.json"
  require_file "${DYNAMICVLA_MODEL_PATH}/model.safetensors"

  PYTHONDONTWRITEBYTECODE=1 "${DYNAMICVLA_PYTHON}" -c '
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

checkpoint = Path(sys.argv[1])
config = json.loads((checkpoint / "config.json").read_text())
expected_inputs = {
    "observation.state": [10],
    "observation.images.opst_cam": [3, 240, 320],
    "observation.images.wrist_cam": [3, 160, 160],
}
assert config.get("type") == "dynamicvla", config.get("type")
assert config.get("n_obs_steps") == 2, config.get("n_obs_steps")
assert config.get("chunk_size") == 20, config.get("chunk_size")
assert config.get("n_action_steps") == 20, config.get("n_action_steps")
assert config.get("use_delta_action") is True
assert config.get("enable_streaming") is False
assert config.get("delta_timestamps", {}).get("observation") == [-2, 0]
assert set(config.get("input_features", {})) == set(expected_inputs)
for key, shape in expected_inputs.items():
    assert config["input_features"][key].get("shape") == shape, key
assert config.get("output_features", {}).get("action", {}).get("shape") == [10]

stats = {
    "normalize_inputs.buffer_observation_state.mean": (10,),
    "normalize_inputs.buffer_observation_state.std": (10,),
    "normalize_targets.buffer_action.mean": (10,),
    "normalize_targets.buffer_action.std": (10,),
    "unnormalize_outputs.buffer_action.mean": (10,),
    "unnormalize_outputs.buffer_action.std": (10,),
}
with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as tensors:
    for key, expected_shape in stats.items():
        value = tensors.get_tensor(key)
        assert tuple(value.shape) == expected_shape, (key, tuple(value.shape))
        assert torch.isfinite(value).all(), key
' "${DYNAMICVLA_MODEL_PATH}" \
    || die "checkpoint does not match the trained DynamicVLA-FC contract"
}

validate_runtime_imports() {
  PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME="${DYNAMICVLA_HF_HOME}" \
    PYTHONPATH="${DYNAMICVLA_ROOT}:${MOTIONFORGE_ROOT}/source/motionforge${PYTHONPATH:+:${PYTHONPATH}}" \
    "${DYNAMICVLA_PYTHON}" -c '
from importlib.metadata import version

import easydict
import zmq
from motionforge.benchmark.client import BenchmarkClientBridge
from policies.dynamicvla.configuration_dynamicvla import DynamicVLAConfig
from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
from scripts.inference import get_vla_model

assert version("lerobot") == "0.3.3", version("lerobot")
' || die "DynamicVLA runtime imports failed; use its Python 3.10 / lerobot 0.3.3 environment"
}

validate_configuration() {
  local task_id=""
  local benchmark_config=""

  require_dir "${DYNAMICVLA_ROOT}"
  require_dir "${MOTIONFORGE_ROOT}"
  require_file "${BRIDGE_CLIENT}"
  require_file "${TRIALS_SERVER}"
  require_executable "${DYNAMICVLA_PYTHON}"
  require_executable "${CONDA_EXE}"
  require_executable "${TIMEOUT_EXE}"
  command -v awk >/dev/null 2>&1 || die "required executable not found: awk"
  command -v grep >/dev/null 2>&1 || die "required executable not found: grep"
  validate_checkpoint
  validate_runtime_imports

  ((${#TASK_IDS[@]} > 0)) || die "DYNAMICVLA_EVAL_TASKS must select at least one task"
  for task_id in "${TASK_IDS[@]}"; do
    [[ "${task_id}" =~ ^fc_00[1-9]$ ]] || die "invalid FC task id: ${task_id}"
    benchmark_config="${BENCHMARK_DIR}/${task_id}_rgb_gr00t_zmq.yaml"
    require_file "${benchmark_config}"
  done

  require_uint_at_least "DYNAMICVLA_EVAL_START_SEED" "${START_SEED}" 0
  require_uint_at_least "DYNAMICVLA_EVAL_NUM_TRIALS" "${NUM_TRIALS}" 1
  require_uint_at_least "DYNAMICVLA_EVAL_MAX_STEPS" "${MAX_STEPS}" 1
  require_uint_at_least "DYNAMICVLA_EVAL_OBS_PORT" "${OBS_PORT}" 1
  require_uint_at_least "DYNAMICVLA_EVAL_ACT_PORT" "${ACT_PORT}" 1
  require_uint_at_least "DYNAMICVLA_EVAL_CLIENT_WARMUP_S" "${CLIENT_WARMUP_S}" 0
  require_uint_at_least "DYNAMICVLA_EVAL_TASK_TIMEOUT_S" "${TASK_TIMEOUT_S}" 1
  require_uint_at_least "DYNAMICVLA_EVAL_BETWEEN_TASKS_S" "${BETWEEN_TASKS_S}" 0
  require_uint_at_least "DYNAMICVLA_PRINT_EVERY" "${DYNAMICVLA_PRINT_EVERY}" 0
  require_uint_at_least "DYNAMICVLA_SEND_HORIZON" "${DYNAMICVLA_SEND_HORIZON}" 1
  require_uint_at_least "DYNAMICVLA_EXECUTION_HORIZON" "${DYNAMICVLA_EXECUTION_HORIZON}" 1
  require_uint_at_least "DYNAMICVLA_EVAL_VIDEO_WIDTH" "${VIDEO_WIDTH}" 2
  require_uint_at_least "DYNAMICVLA_EVAL_VIDEO_HEIGHT" "${VIDEO_HEIGHT}" 2
  require_uint_at_least "DYNAMICVLA_EVAL_VIDEO_STRIDE" "${VIDEO_STRIDE}" 1
  require_positive_number "DYNAMICVLA_ACTION_HZ" "${DYNAMICVLA_ACTION_HZ}"
  require_positive_number "DYNAMICVLA_MAX_INFERENCE_HZ" "${DYNAMICVLA_MAX_INFERENCE_HZ}"

  ((10#${DYNAMICVLA_EXECUTION_HORIZON} <= 10#${DYNAMICVLA_SEND_HORIZON})) \
    || die "DYNAMICVLA_EXECUTION_HORIZON must be <= DYNAMICVLA_SEND_HORIZON"
  ((10#${DYNAMICVLA_SEND_HORIZON} <= 20)) \
    || die "DYNAMICVLA_SEND_HORIZON must be <= the checkpoint action horizon 20"
  ((10#${OBS_PORT} <= 65535)) || die "DYNAMICVLA_EVAL_OBS_PORT must be <= 65535"
  ((10#${ACT_PORT} <= 65535)) || die "DYNAMICVLA_EVAL_ACT_PORT must be <= 65535"
  [[ "${OBS_PORT}" != "${ACT_PORT}" ]] || die "observation and action ports must differ"
  [[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
  [[ "${USE_BENCHMARK_MAX_STEPS}" == "0" || "${USE_BENCHMARK_MAX_STEPS}" == "1" ]] \
    || die "DYNAMICVLA_EVAL_USE_BENCHMARK_MAX_STEPS must be 0 or 1"
  [[ -z "${MOTION_LEVEL}" || "${MOTION_LEVEL}" =~ ^level[123]$ ]] \
    || die "DYNAMICVLA_EVAL_MOTION_LEVEL must be empty, level1, level2, or level3"
  [[ "${INITIAL_POSITION_MODE}" == "fixed" || "${INITIAL_POSITION_MODE}" == "seeded" ]] \
    || die "DYNAMICVLA_EVAL_INITIAL_POSITION_MODE must be fixed or seeded"
  [[ "${CLOCK_MODE}" == "wall_clock_strict" ]] \
    || die "DYNAMICVLA_EVAL_CLOCK_MODE must remain wall_clock_strict for docs/protocol.md"
  [[ "${VIDEO_OUTCOME_SUFFIX}" == "0" || "${VIDEO_OUTCOME_SUFFIX}" == "1" ]] \
    || die "DYNAMICVLA_EVAL_VIDEO_OUTCOME_SUFFIX must be 0 or 1"
  [[ -n "${DYNAMICVLA_EVAL_CUDA_VISIBLE_DEVICES}" ]] \
    || die "DYNAMICVLA_EVAL_CUDA_VISIBLE_DEVICES must not be empty"
  [[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "DYNAMICVLA_EVAL_RUN_ID may contain only letters, numbers, dot, underscore, and hyphen"
}

terminate_process() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
  fi
  if [[ -n "${pid}" ]]; then
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup_processes() {
  terminate_process "${BRIDGE_PID}"
  terminate_process "${SERVER_PID}"
  BRIDGE_PID=""
  SERVER_PID=""
}

trap cleanup_processes EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

build_commands() {
  local benchmark_config="$1"
  local video_dir="$2"
  local video_name="$3"

  SERVER_COMMAND=(
    "${TIMEOUT_EXE}"
    --signal=TERM
    --kill-after=30s
    "${TASK_TIMEOUT_S}s"
    "${CONDA_EXE}"
    run
    --no-capture-output
    -n
    "${MOTIONFORGE_CONDA_ENV}"
    python
    "${TRIALS_SERVER}"
    --benchmark_config
    "${benchmark_config}"
    --seed
    "${START_SEED}"
    --num_trials
    "${NUM_TRIALS}"
    --initial_position_mode
    "${INITIAL_POSITION_MODE}"
    --clock_mode
    "${CLOCK_MODE}"
    --default_action_hz
    "${DYNAMICVLA_ACTION_HZ}"
    --max_inference_hz
    "${DYNAMICVLA_MAX_INFERENCE_HZ}"
    --device
    "${MOTIONFORGE_DEVICE}"
    --obs_port
    "${OBS_PORT}"
    --act_port
    "${ACT_PORT}"
    --client_warmup
    "${CLIENT_WARMUP_S}"
    --video_dir
    "${video_dir}"
    --video_name
    "${video_name}"
    --video_width
    "${VIDEO_WIDTH}"
    --video_height
    "${VIDEO_HEIGHT}"
    --video_stride
    "${VIDEO_STRIDE}"
  )
  if [[ "${VIDEO_OUTCOME_SUFFIX}" == "1" ]]; then
    SERVER_COMMAND+=(--video_outcome_suffix)
  fi
  if [[ "${USE_BENCHMARK_MAX_STEPS}" == "0" ]]; then
    SERVER_COMMAND+=(--max_steps "${MAX_STEPS}")
  fi
  if [[ -n "${MOTION_LEVEL}" ]]; then
    SERVER_COMMAND+=(--motion_level "${MOTION_LEVEL}")
  fi

  BRIDGE_COMMAND=(
    "${TIMEOUT_EXE}"
    --signal=TERM
    --kill-after=30s
    "${TASK_TIMEOUT_S}s"
    "${DYNAMICVLA_PYTHON}"
    "${BRIDGE_CLIENT}"
    --model-path
    "${DYNAMICVLA_MODEL_PATH}"
    --device
    "${DYNAMICVLA_DEVICE}"
    --motionforge-obs-port
    "${OBS_PORT}"
    --motionforge-act-port
    "${ACT_PORT}"
    --num-episodes
    "${NUM_TRIALS}"
    --action-hz
    "${DYNAMICVLA_ACTION_HZ}"
    --max-inference-hz
    "${DYNAMICVLA_MAX_INFERENCE_HZ}"
    --send-horizon
    "${DYNAMICVLA_SEND_HORIZON}"
    --execution-horizon
    "${DYNAMICVLA_EXECUTION_HORIZON}"
    --print-every
    "${DYNAMICVLA_PRINT_EVERY}"
  )
}

print_command() {
  local working_directory="$1"
  shift
  printf '  (cd %q && ' "${working_directory}"
  printf '%q ' "$@"
  printf ')\n'
}

print_bridge_command() {
  local pythonpath="${DYNAMICVLA_ROOT}:${MOTIONFORGE_ROOT}/source/motionforge${PYTHONPATH:+:${PYTHONPATH}}"
  printf '  (cd %q && CUDA_VISIBLE_DEVICES=%q PYTHONDONTWRITEBYTECODE=1 HF_HOME=%q PYTHONPATH=%q ' \
    "${DYNAMICVLA_ROOT}" "${DYNAMICVLA_EVAL_CUDA_VISIBLE_DEVICES}" "${DYNAMICVLA_HF_HOME}" "${pythonpath}"
  printf '%q ' "${BRIDGE_COMMAND[@]}"
  printf ')\n'
}

wait_for_server_and_bridge() {
  local server_pid="${SERVER_PID}"
  local bridge_pid="${BRIDGE_PID}"
  local completed_pid=""
  local first_status=0
  local second_status=0

  set +e
  wait -n -p completed_pid "${server_pid}" "${bridge_pid}"
  first_status="$?"
  set -e
  if ((first_status != 0)); then
    log "one process failed pid=${completed_pid:-unknown} status=${first_status}; stopping its peer"
    cleanup_processes
    return "${first_status}"
  fi

  if [[ "${completed_pid}" == "${server_pid}" ]]; then
    SERVER_PID=""
    set +e
    wait "${bridge_pid}"
    second_status="$?"
    set -e
    BRIDGE_PID=""
  else
    BRIDGE_PID=""
    set +e
    wait "${server_pid}"
    second_status="$?"
    set -e
    SERVER_PID=""
  fi
  if ((second_status != 0)); then
    log "peer process failed status=${second_status}"
    return "${second_status}"
  fi
  return 0
}

parse_task_summary() {
  local server_log="$1"
  local summary_line=""
  local pattern='trials=([0-9]+)[[:space:]]+successes=([0-9]+)[[:space:]]+failures=([0-9]+)[[:space:]]+success_rate=([0-9]+([.][0-9]+)?)'

  LAST_TRIALS=0
  LAST_SUCCESSES=0
  LAST_FAILURES=0
  LAST_SUCCESS_RATE=""
  LAST_RAW_SUMMARY=""
  summary_line="$(grep -F '[MOTIONFORGE-BENCH] trials_summary ' "${server_log}" | tail -n 1 || true)"
  [[ -n "${summary_line}" ]] || return 1
  [[ "${summary_line}" =~ ${pattern} ]] || return 1
  LAST_TRIALS="${BASH_REMATCH[1]}"
  LAST_SUCCESSES="${BASH_REMATCH[2]}"
  LAST_FAILURES="${BASH_REMATCH[3]}"
  LAST_SUCCESS_RATE="${BASH_REMATCH[4]}"
  LAST_RAW_SUMMARY="${summary_line}"
  ((10#${LAST_TRIALS} == 10#${NUM_TRIALS})) || return 1
  ((10#${LAST_SUCCESSES} + 10#${LAST_FAILURES} == 10#${LAST_TRIALS})) || return 1
}

count_videos() {
  local video_dir="$1"
  local videos=()
  shopt -s nullglob
  videos=("${video_dir}"/*.mp4)
  shopt -u nullglob
  printf '%s\n' "${#videos[@]}"
}

validate_video_outputs() {
  local task_id="$1"
  local video_dir="$2"
  local trial_number=0
  local base_path=""
  local success_path=""
  local failure_path=""
  for ((trial_number = 1; trial_number <= 10#${NUM_TRIALS}; trial_number++)); do
    if ((10#${NUM_TRIALS} == 1)); then
      base_path="${video_dir}/${task_id}_rollout.mp4"
    else
      printf -v base_path '%s/%s_rollout_trial_%03d.mp4' \
        "${video_dir}" "${task_id}" "${trial_number}"
    fi
    if [[ "${VIDEO_OUTCOME_SUFFIX}" == "1" ]]; then
      success_path="${base_path%.mp4}_success.mp4"
      failure_path="${base_path%.mp4}_failure.mp4"
      if [[ -s "${success_path}" ]]; then
        [[ ! -e "${failure_path}" ]] || return 1
      elif [[ -s "${failure_path}" ]]; then
        [[ ! -e "${success_path}" ]] || return 1
      else
        return 1
      fi
    else
      [[ -s "${base_path}" ]] || return 1
    fi
  done
}

append_task_result() {
  local task_id="$1"
  local status="$2"
  local benchmark_config="$3"
  local video_dir="$4"
  local reason="$5"
  local video_count=""
  video_count="$(count_videos "${video_dir}")"
  {
    printf '\n[%s]\n' "${task_id}"
    printf 'status=%s\n' "${status}"
    printf 'benchmark=%s\n' "${benchmark_config}"
    printf 'trials=%s\n' "${LAST_TRIALS:-N/A}"
    printf 'successes=%s\n' "${LAST_SUCCESSES:-N/A}"
    printf 'failures=%s\n' "${LAST_FAILURES:-N/A}"
    printf 'success_rate=%s\n' "${LAST_SUCCESS_RATE:-N/A}"
    printf 'video_dir=%s\n' "${video_dir}"
    printf 'video_count=%s\n' "${video_count}"
    printf 'reason=%s\n' "${reason}"
    printf 'raw_summary=%s\n' "${LAST_RAW_SUMMARY:-N/A}"
  } >>"${SUMMARY_FILE}"
}

run_task() {
  local task_id="$1"
  local benchmark_config="${BENCHMARK_DIR}/${task_id}_rgb_gr00t_zmq.yaml"
  local task_result_dir="${RESULT_DIR}/${task_id}"
  local server_log="${task_result_dir}/server.log"
  local client_log="${task_result_dir}/client.log"
  local video_dir="${task_result_dir}/videos"
  local process_status=0
  local max_steps_label="${MAX_STEPS}"

  if [[ "${USE_BENCHMARK_MAX_STEPS}" == "1" ]]; then
    max_steps_label="benchmark_config"
  fi
  LAST_TRIALS=0
  LAST_SUCCESSES=0
  LAST_FAILURES=0
  LAST_SUCCESS_RATE=""
  LAST_RAW_SUMMARY=""
  mkdir -p "${video_dir}" || return 1
  build_commands "${benchmark_config}" "${video_dir}" "${task_id}_rollout.mp4"
  log "starting task=${task_id} trials=${NUM_TRIALS} max_steps=${max_steps_label} motion_level=${MOTION_LEVEL:-benchmark_config} clock_mode=${CLOCK_MODE} action_alignment=${ACTION_ALIGNMENT}"

  (
    cd -- "${MOTIONFORGE_ROOT}"
    export CUDA_VISIBLE_DEVICES="${DYNAMICVLA_EVAL_CUDA_VISIBLE_DEVICES}"
    export OMNI_KIT_ACCEPT_EULA="YES"
    exec "${SERVER_COMMAND[@]}"
  ) >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  (
    cd -- "${DYNAMICVLA_ROOT}"
    export CUDA_VISIBLE_DEVICES="${DYNAMICVLA_EVAL_CUDA_VISIBLE_DEVICES}"
    export PYTHONDONTWRITEBYTECODE=1
    export HF_HOME="${DYNAMICVLA_HF_HOME}"
    export PYTHONPATH="${DYNAMICVLA_ROOT}:${MOTIONFORGE_ROOT}/source/motionforge${PYTHONPATH:+:${PYTHONPATH}}"
    exec "${BRIDGE_COMMAND[@]}"
  ) >"${client_log}" 2>&1 &
  BRIDGE_PID="$!"

  if wait_for_server_and_bridge; then
    process_status=0
  else
    process_status="$?"
  fi
  if ((process_status != 0)); then
    append_task_result "${task_id}" failed "${benchmark_config}" "${video_dir}" \
      "server/client process exit status ${process_status}"
    return "${process_status}"
  fi
  if ! parse_task_summary "${server_log}"; then
    append_task_result "${task_id}" failed "${benchmark_config}" "${video_dir}" \
      "server log has no valid ${NUM_TRIALS}-trial summary"
    return 1
  fi
  if ! validate_video_outputs "${task_id}" "${video_dir}"; then
    append_task_result "${task_id}" failed "${benchmark_config}" "${video_dir}" \
      "expected ${NUM_TRIALS} non-empty rollout videos with valid outcome suffixes"
    return 1
  fi
  append_task_result "${task_id}" completed "${benchmark_config}" "${video_dir}" "none"
  log "completed task=${task_id} successes=${LAST_SUCCESSES}/${LAST_TRIALS} success_rate=${LAST_SUCCESS_RATE}"
}

append_overall_summary() {
  local completed_tasks="$1"
  local failed_tasks="$2"
  local completed_trials="$3"
  local total_successes="$4"
  local total_failures="$5"
  local expected_tasks="${#TASK_IDS[@]}"
  local expected_trials=$((expected_tasks * NUM_TRIALS))
  local total_success_rate="N/A"
  local overall_status="incomplete"
  if ((completed_trials > 0)); then
    total_success_rate="$(awk -v successes="${total_successes}" -v trials="${completed_trials}" 'BEGIN { printf "%.3f", successes / trials }')"
  fi
  if ((completed_tasks == expected_tasks && failed_tasks == 0 && completed_trials == expected_trials)); then
    overall_status="completed"
  fi
  {
    printf '\n[overall]\n'
    printf 'status=%s\n' "${overall_status}"
    printf 'expected_tasks=%s\n' "${expected_tasks}"
    printf 'completed_tasks=%s\n' "${completed_tasks}"
    printf 'failed_tasks=%s\n' "${failed_tasks}"
    printf 'expected_trials=%s\n' "${expected_trials}"
    printf 'completed_trials=%s\n' "${completed_trials}"
    printf 'total_successes=%s\n' "${total_successes}"
    printf 'total_failures=%s\n' "${total_failures}"
    printf 'total_success_rate=%s\n' "${total_success_rate}"
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  } >>"${SUMMARY_FILE}"
}

validate_configuration

if [[ "${DRY_RUN}" == "1" ]]; then
  log "validated configuration; no process or result directory will be created"
  log "tasks=${#TASK_IDS[@]} trials_per_task=${NUM_TRIALS} physics_device=${MOTIONFORGE_DEVICE} gpu=${DYNAMICVLA_EVAL_CUDA_VISIBLE_DEVICES} action_hz=${DYNAMICVLA_ACTION_HZ} send_horizon=${DYNAMICVLA_SEND_HORIZON} execution_horizon=${DYNAMICVLA_EXECUTION_HORIZON} motion_level=${MOTION_LEVEL:-benchmark_config} max_steps=$([[ "${USE_BENCHMARK_MAX_STEPS}" == "1" ]] && printf benchmark_config || printf '%s' "${MAX_STEPS}") clock_mode=${CLOCK_MODE} action_alignment=${ACTION_ALIGNMENT} video_outcome_suffix=${VIDEO_OUTCOME_SUFFIX}"
  log "model=${DYNAMICVLA_MODEL_PATH} result_dir=${RESULT_DIR}"
  for task_id in "${TASK_IDS[@]}"; do
    benchmark_config="${BENCHMARK_DIR}/${task_id}_rgb_gr00t_zmq.yaml"
    video_dir="${RESULT_DIR}/${task_id}/videos"
    build_commands "${benchmark_config}" "${video_dir}" "${task_id}_rollout.mp4"
    log "dry-run task=${task_id} benchmark=${benchmark_config}"
    print_command "${MOTIONFORGE_ROOT}" "${SERVER_COMMAND[@]}"
    print_bridge_command
  done
  exit 0
fi

if [[ -e "${RESULT_DIR}" ]]; then
  die "result directory already exists; choose another DYNAMICVLA_EVAL_RUN_ID: ${RESULT_DIR}"
fi
mkdir -p "${RESULT_DIR}"
{
  printf 'DynamicVLA FC001-FC009 MotionForge evaluation\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'model=%s\n' "${DYNAMICVLA_MODEL_PATH}"
  printf 'device=%s\n' "${DYNAMICVLA_DEVICE}"
  printf 'motionforge_device=%s\n' "${MOTIONFORGE_DEVICE}"
  printf 'cuda_visible_devices=%s\n' "${DYNAMICVLA_EVAL_CUDA_VISIBLE_DEVICES}"
  printf 'tasks=%s\n' "${#TASK_IDS[@]}"
  printf 'trials_per_task=%s\n' "${NUM_TRIALS}"
  printf 'max_steps_per_trial=%s\n' "$([[ "${USE_BENCHMARK_MAX_STEPS}" == "1" ]] && printf benchmark_config || printf '%s' "${MAX_STEPS}")"
  printf 'motion_level=%s\n' "${MOTION_LEVEL:-benchmark_config}"
  printf 'seed_start=%s\n' "${START_SEED}"
  printf 'seed_end=%s\n' "$((START_SEED + NUM_TRIALS - 1))"
  printf 'initial_position_mode=%s\n' "${INITIAL_POSITION_MODE}"
  printf 'clock_mode=%s\n' "${CLOCK_MODE}"
  printf 'action_alignment=%s\n' "${ACTION_ALIGNMENT}"
  printf 'action_hz=%s\n' "${DYNAMICVLA_ACTION_HZ}"
  printf 'max_inference_hz=%s\n' "${DYNAMICVLA_MAX_INFERENCE_HZ}"
  printf 'action_horizon=20\n'
  printf 'send_horizon=%s\n' "${DYNAMICVLA_SEND_HORIZON}"
  printf 'execution_horizon=%s\n' "${DYNAMICVLA_EXECUTION_HORIZON}"
  printf 'streaming=false\n'
  printf 'video_enabled=true\n'
  printf 'video_outcome_suffix=%s\n' "${VIDEO_OUTCOME_SUFFIX}"
  printf 'video_width=%s\n' "${VIDEO_WIDTH}"
  printf 'video_height=%s\n' "${VIDEO_HEIGHT}"
  printf 'video_stride=%s\n' "${VIDEO_STRIDE}"
} >"${SUMMARY_FILE}"

overall_status=0
completed_tasks=0
failed_tasks=0
completed_trials=0
total_successes=0
total_failures=0

for task_id in "${TASK_IDS[@]}"; do
  if run_task "${task_id}"; then
    ((completed_tasks += 1))
    ((completed_trials += LAST_TRIALS))
    total_successes=$((total_successes + LAST_SUCCESSES))
    total_failures=$((total_failures + LAST_FAILURES))
  else
    overall_status=1
    ((failed_tasks += 1))
  fi
  cleanup_processes
  if [[ "${task_id}" != "${TASK_IDS[-1]}" ]] && ((10#${BETWEEN_TASKS_S} > 0)); then
    sleep "${BETWEEN_TASKS_S}"
  fi
done

append_overall_summary \
  "${completed_tasks}" \
  "${failed_tasks}" \
  "${completed_trials}" \
  "${total_successes}" \
  "${total_failures}"
log "results=${RESULT_DIR}"
log "summary=${SUMMARY_FILE}"
exit "${overall_status}"
