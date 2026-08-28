#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${WORKSPACE_ROOT}/lerobot}"
BRIDGE_PYTHONPATH="${MOTIONFORGE_ROOT}/source/motionforge:${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

BRIDGE_CLIENT="${SCRIPT_DIR}/motionforge_diffusion_policy_bridge_client.py"
TRIALS_SERVER="${MOTIONFORGE_ROOT}/scripts/benchmark/run_env_server_trials.py"
BENCHMARK_DIR="${MOTIONFORGE_ROOT}/configs/benchmarks/circular_motion"
RUNTIME_ASSET_DIR="${MOTIONFORGE_ROOT}/source/motionforge/motionforge/assets/runtime"

DIFFUSION_MODEL_PATH="${DIFFUSION_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/MotionforgeGroup/DiffusionPolicy/CM-80000-3views}"
DIFFUSION_PYTHON="${DIFFUSION_PYTHON:-${WORKSPACE_ROOT}/miniconda3/envs/lerobot/bin/python}"
DIFFUSION_DEVICE="${DIFFUSION_DEVICE:-cuda:0}"
DIFFUSION_POLICY_SEED="${DIFFUSION_POLICY_SEED:-0}"
DIFFUSION_NUM_INFERENCE_STEPS="${DIFFUSION_NUM_INFERENCE_STEPS:-20}"
DIFFUSION_ACTION_HZ="${DIFFUSION_ACTION_HZ:-30}"
DIFFUSION_MAX_INFERENCE_HZ="${DIFFUSION_MAX_INFERENCE_HZ:-30}"
DIFFUSION_SEND_HORIZON="${DIFFUSION_SEND_HORIZON:-16}"
DIFFUSION_EXECUTION_HORIZON="${DIFFUSION_EXECUTION_HORIZON:-8}"
DIFFUSION_ACTION_ALIGNMENT="${DIFFUSION_ACTION_ALIGNMENT:-observation_aligned}"
DIFFUSION_PRINT_EVERY="${DIFFUSION_PRINT_EVERY:-10}"

MOTIONFORGE_CONDA_ENV="${MOTIONFORGE_CONDA_ENV:-motionforge}"
# CM compound containers and articulated payloads require CPU PhysX. Rendering
# and Diffusion Policy inference still use the selected visible GPU.
MOTIONFORGE_DEVICE="${MOTIONFORGE_DEVICE:-cpu}"
DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES="${DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-3}}"
CONDA_EXE="${MOTIONFORGE_CONDA_EXE:-${WORKSPACE_ROOT}/miniconda3/bin/conda}"
TIMEOUT_EXE="${MOTIONFORGE_TIMEOUT_EXE:-$(command -v timeout || true)}"

START_SEED="${DIFFUSION_EVAL_START_SEED:-0}"
NUM_TRIALS="${DIFFUSION_EVAL_NUM_TRIALS:-50}"
CLOCK_MODE="${DIFFUSION_EVAL_CLOCK_MODE:-slowdown_scaled}"
TIMING_PROTOCOL="motionforge.slowdown_scaled.server.v1"
OBS_PORT="${DIFFUSION_EVAL_OBS_PORT:-3396}"
DIFFUSION_PORT="${DIFFUSION_EVAL_DIFFUSION_PORT:-3398}"
CLIENT_WARMUP_S="${DIFFUSION_EVAL_CLIENT_WARMUP_S:-5}"
TASK_TIMEOUT_S="${DIFFUSION_EVAL_TASK_TIMEOUT_S:-14400}"
BETWEEN_TASKS_S="${DIFFUSION_EVAL_BETWEEN_TASKS_S:-5}"

VIDEO_WIDTH="${DIFFUSION_EVAL_VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${DIFFUSION_EVAL_VIDEO_HEIGHT:-480}"
VIDEO_STRIDE="${DIFFUSION_EVAL_VIDEO_STRIDE:-1}"
VIDEO_OUTCOME_SUFFIX="${DIFFUSION_EVAL_VIDEO_OUTCOME_SUFFIX:-1}"

RESULT_ROOT="${DIFFUSION_EVAL_OUTPUT_ROOT:-${SCRIPT_DIR}/output/circular_motion/cm000_cm010/level2_fixed/control60_server_scheduled_v1}"
RUN_ID="${DIFFUSION_EVAL_RUN_ID:-diffusion_policy_cm000_cm010_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_ROOT}/${RUN_ID}"
SUMMARY_FILE="${RESULT_DIR}/success_rates.txt"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_TASK_IDS=(
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

if [[ -n "${DIFFUSION_EVAL_TASKS:-}" ]]; then
  read -r -a TASK_IDS <<<"${DIFFUSION_EVAL_TASKS}"
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
  printf '[DIFFUSION-CM000-CM010-EVAL] %s\n' "$*"
}

die() {
  printf '[DIFFUSION-CM000-CM010-EVAL] ERROR: %s\n' "$*" >&2
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

benchmark_max_steps() {
  local benchmark_config="$1"
  local configured=""
  configured="$(awk '/^[[:space:]]*max_steps:[[:space:]]*[0-9]+[[:space:]]*$/ { print $2; exit }' "${benchmark_config}")"
  [[ -n "${configured}" ]] || die "runtime.max_steps not found in benchmark: ${benchmark_config}"
  printf '%s\n' "${configured}"
}

validate_checkpoint() {
  require_dir "${DIFFUSION_MODEL_PATH}"
  require_file "${DIFFUSION_MODEL_PATH}/config.json"
  require_file "${DIFFUSION_MODEL_PATH}/train_config.json"
  require_file "${DIFFUSION_MODEL_PATH}/model.safetensors"
  require_file "${DIFFUSION_MODEL_PATH}/policy_preprocessor.json"
  require_file "${DIFFUSION_MODEL_PATH}/policy_postprocessor.json"
  require_file "${DIFFUSION_MODEL_PATH}/policy_preprocessor_step_3_normalizer_processor.safetensors"
  require_file "${DIFFUSION_MODEL_PATH}/policy_postprocessor_step_0_unnormalizer_processor.safetensors"

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${BRIDGE_PYTHONPATH}" \
    "${DIFFUSION_PYTHON}" -c '
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])
import motionforge_diffusion_policy_bridge_client as client

checkpoint = Path(sys.argv[1])
client.validate_checkpoint_files(checkpoint)
config = client.load_diffusion_config(checkpoint, "cpu")
client.validate_diffusion_contract(checkpoint, config)
train_config = json.loads((checkpoint / "train_config.json").read_text(encoding="utf-8"))
assert train_config.get("dataset", {}).get("repo_id") == "local/pi05_merged_v30"
' "${DIFFUSION_MODEL_PATH}" "${SCRIPT_DIR}" \
    || die "checkpoint does not match the trained three-view DiffusionPolicy-CM contract"
}

validate_runtime_imports() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${BRIDGE_PYTHONPATH}" \
    "${DIFFUSION_PYTHON}" -c '
import diffusers
import zmq
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig
' || die "Diffusion Policy runtime imports failed; verify LeRobot, diffusers, and pyzmq"
}

validate_bridge_contract() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${BRIDGE_PYTHONPATH}" \
    "${DIFFUSION_PYTHON}" -c '
import sys

import numpy as np
from motionforge.benchmark.protocol import SERVER_SCHEDULED_TIMING_PROTOCOL

sys.path.insert(0, sys.argv[1])
import motionforge_diffusion_policy_bridge_client as client

assert SERVER_SCHEDULED_TIMING_PROTOCOL == "motionforge.slowdown_scaled.server.v1"
packet = client.action_packet(
    actions=np.zeros((16, 10), dtype=np.float32),
    message={
        "protocol_version": client.PROTOCOL_VERSION,
        "index": 12,
        "request_id": 3,
        "request_kind": "plan",
        "timing_protocol": SERVER_SCHEDULED_TIMING_PROTOCOL,
    },
    action_horizon=32,
    execution_horizon=8,
    inference_duration_s=0.1,
    action_hz=30.0,
    max_inference_hz=30.0,
    observation_horizon=2,
    diffusion_horizon=64,
    num_inference_steps=40,
    policy_seed=0,
)
assert packet["protocol_version"] == client.PROTOCOL_VERSION
assert packet["observation_index"] == 12
assert packet["request_id"] == 3
assert packet["action_alignment"] == "observation_aligned"
assert packet["action_hz"] == 30.0
assert packet["max_inference_hz"] == 30.0
assert packet["metadata"]["send_horizon"] == 16
assert packet["metadata"]["execution_horizon"] == 8
' "${SCRIPT_DIR}" || die "Diffusion Policy bridge does not match the CM server-scheduled contract"
}

validate_runtime_assets() {
  require_file "${RUNTIME_ASSET_DIR}/environments/circular_motion/kitchen_rotary_room_01/derived/kitchen_rotary_room_01_visual.usda"
  require_file "${RUNTIME_ASSET_DIR}/furniture/kitchen_islands/geniesim_table_03/derived/kitchen_island_table_03_visual.usda"
  require_file "${RUNTIME_ASSET_DIR}/objects/xuanyu/turntables/turntable_visual_smoked_glass_steel_v1/derived/turntable_visual_smoked_glass_steel_v1.usda"
  require_file "${RUNTIME_ASSET_DIR}/mechanisms/mug_trees/kelcode_modular_mug_tree_01/derived/mug_tree_4_hook_articulation.usda"
  require_file "${RUNTIME_ASSET_DIR}/mechanisms/lab/rotating_test_tube_rack_01/lab_rotating_test_tube_rack_opaque_01.usda"
  require_file "${RUNTIME_ASSET_DIR}/mechanisms/lab/rotating_test_tube_rack_01/lab_test_tube_18x105_opaque_blue_01.usda"
  require_file "${RUNTIME_ASSET_DIR}/mechanisms/lab/rotating_test_tube_rack_01/lab_test_tube_18x105_opaque_yellow_01.usda"
  require_file "${RUNTIME_ASSET_DIR}/mechanisms/control_panels/rotating_three_button_panel_01/derived/rotating_three_button_panel_01.usda"
  require_file "${RUNTIME_ASSET_DIR}/containers/baskets/wicker_basket_01/wicker_basket_01_rigid.usda"
  require_file "${RUNTIME_ASSET_DIR}/objects/puzzles/rubik_cube_01/derived/graspable.usda"
  require_file "${RUNTIME_ASSET_DIR}/objects/graspable_pool/google_scanned_objects/fruit_snack_package/derived/graspable.usda"
  require_file "${RUNTIME_ASSET_DIR}/objects/graspable_pool/google_scanned_objects/medicine_bottle/derived/graspable.usda"
  require_file "${RUNTIME_ASSET_DIR}/objects/graspable_pool/google_scanned_objects/classic_blue_mug/derived/graspable.usda"
  require_file "${RUNTIME_ASSET_DIR}/objects/graspable_pool/polyhaven/rubber_duck/derived/graspable.usda"
}

validate_configuration() {
  local task_id=""
  local benchmark_config=""
  local task_max_steps=""

  require_dir "${MOTIONFORGE_ROOT}"
  require_dir "${LEROBOT_ROOT}/src/lerobot"
  require_file "${BRIDGE_CLIENT}"
  require_file "${TRIALS_SERVER}"
  require_dir "${BENCHMARK_DIR}"
  require_executable "${DIFFUSION_PYTHON}"
  require_executable "${CONDA_EXE}"
  require_executable "${TIMEOUT_EXE}"
  command -v awk >/dev/null 2>&1 || die "required executable not found: awk"
  command -v grep >/dev/null 2>&1 || die "required executable not found: grep"
  validate_checkpoint
  validate_runtime_imports
  validate_bridge_contract
  validate_runtime_assets

  [[ "${MOTIONFORGE_DEVICE}" == "cpu" ]] || die \
    "MOTIONFORGE_DEVICE must be cpu for CM evaluation; GPU PhysX makes rotating compound/articulated payloads sink"
  [[ "${CLOCK_MODE}" == "slowdown_scaled" ]] || die \
    "DIFFUSION_EVAL_CLOCK_MODE must be slowdown_scaled for CM v2 evaluation; got ${CLOCK_MODE}"
  [[ "${DIFFUSION_ACTION_ALIGNMENT}" == "observation_aligned" ]] || die \
    "DIFFUSION_ACTION_ALIGNMENT must be observation_aligned for realtime CM evaluation; got ${DIFFUSION_ACTION_ALIGNMENT}"

  ((${#TASK_IDS[@]} > 0)) || die "DIFFUSION_EVAL_TASKS must select at least one task"
  for task_id in "${TASK_IDS[@]}"; do
    [[ "${task_id}" =~ ^cm_(00[0-5]|00[7-9]|010)$ ]] || die "invalid CM v2 task id: ${task_id}"
    benchmark_config="${BENCHMARK_DIR}/${task_id}_rgb_gr00t_zmq.yaml"
    require_file "${benchmark_config}"
    grep -Fqx "benchmark_id: ${task_id}_rgb_gr00t_zmq_v2" "${benchmark_config}" \
      || die "benchmark_id is not v2 in ${benchmark_config}"
    grep -Fqx "  clock_mode: slowdown_scaled" "${benchmark_config}" \
      || die "clock_mode is not slowdown_scaled in ${benchmark_config}"
    grep -Fqx "  timing_protocol: ${TIMING_PROTOCOL}" "${benchmark_config}" \
      || die "timing_protocol is not ${TIMING_PROTOCOL} in ${benchmark_config}"
    grep -Eq '^[[:space:]]*physics_hz:[[:space:]]*240[[:space:]]*$' "${benchmark_config}" \
      || die "physics_hz is not 240 in ${benchmark_config}"
    grep -Eq '^[[:space:]]*control_hz:[[:space:]]*60[[:space:]]*$' "${benchmark_config}" \
      || die "control_hz is not 60 in ${benchmark_config}"
    grep -Eq 'rgb_hz:[[:space:]]*30([,}]|[[:space:]]*$)' "${benchmark_config}" \
      || die "rgb_hz is not 30 in ${benchmark_config}"
    grep -Eq 'state_hz:[[:space:]]*60([,}]|[[:space:]]*$)' "${benchmark_config}" \
      || die "state_hz is not 60 in ${benchmark_config}"
    grep -Eq '^[[:space:]]*default_action_hz:[[:space:]]*30[[:space:]]*$' "${benchmark_config}" \
      || die "default_action_hz is not 30 in ${benchmark_config}"
    grep -Eq '^[[:space:]]*required_action_hz:[[:space:]]*30[[:space:]]*$' "${benchmark_config}" \
      || die "required_action_hz is not 30 in ${benchmark_config}"
    grep -Eq '^[[:space:]]*max_source_actions:[[:space:]]*16[[:space:]]*$' "${benchmark_config}" \
      || die "max_source_actions is not 16 in ${benchmark_config}"
    grep -Eq '^[[:space:]]*max_execution_control_steps:[[:space:]]*16[[:space:]]*$' "${benchmark_config}" \
      || die "max_execution_control_steps is not 16 in ${benchmark_config}"
    grep -Eq '^[[:space:]]*max_inference_hz:[[:space:]]*30[[:space:]]*$' "${benchmark_config}" \
      || die "max_inference_hz is not 30 in ${benchmark_config}"
    task_max_steps="$(benchmark_max_steps "${benchmark_config}")"
    require_uint_at_least "${task_id} max_steps" "${task_max_steps}" 1
  done

  require_uint_at_least "DIFFUSION_EVAL_START_SEED" "${START_SEED}" 0
  require_uint_at_least "DIFFUSION_EVAL_NUM_TRIALS" "${NUM_TRIALS}" 1
  require_uint_at_least "DIFFUSION_EVAL_OBS_PORT" "${OBS_PORT}" 1
  require_uint_at_least "DIFFUSION_EVAL_DIFFUSION_PORT" "${DIFFUSION_PORT}" 1
  require_uint_at_least "DIFFUSION_EVAL_CLIENT_WARMUP_S" "${CLIENT_WARMUP_S}" 0
  require_uint_at_least "DIFFUSION_EVAL_TASK_TIMEOUT_S" "${TASK_TIMEOUT_S}" 1
  require_uint_at_least "DIFFUSION_EVAL_BETWEEN_TASKS_S" "${BETWEEN_TASKS_S}" 0
  require_uint_at_least "DIFFUSION_POLICY_SEED" "${DIFFUSION_POLICY_SEED}" 0
  require_uint_at_least "DIFFUSION_NUM_INFERENCE_STEPS" "${DIFFUSION_NUM_INFERENCE_STEPS}" 1
  require_uint_at_least "DIFFUSION_PRINT_EVERY" "${DIFFUSION_PRINT_EVERY}" 0
  require_uint_at_least "DIFFUSION_SEND_HORIZON" "${DIFFUSION_SEND_HORIZON}" 1
  require_uint_at_least "DIFFUSION_EXECUTION_HORIZON" "${DIFFUSION_EXECUTION_HORIZON}" 1
  require_uint_at_least "DIFFUSION_EVAL_VIDEO_WIDTH" "${VIDEO_WIDTH}" 2
  require_uint_at_least "DIFFUSION_EVAL_VIDEO_HEIGHT" "${VIDEO_HEIGHT}" 2
  require_uint_at_least "DIFFUSION_EVAL_VIDEO_STRIDE" "${VIDEO_STRIDE}" 1
  require_positive_number "DIFFUSION_ACTION_HZ" "${DIFFUSION_ACTION_HZ}"
  require_positive_number "DIFFUSION_MAX_INFERENCE_HZ" "${DIFFUSION_MAX_INFERENCE_HZ}"

  awk -v value="${DIFFUSION_ACTION_HZ}" 'BEGIN { exit !(value == 30) }' \
    || die "DIFFUSION_ACTION_HZ must be 30 for the CM v2 protocol"
  awk -v value="${DIFFUSION_MAX_INFERENCE_HZ}" 'BEGIN { exit !(value == 30) }' \
    || die "DIFFUSION_MAX_INFERENCE_HZ must be 30 for the CM v2 protocol"
  ((10#${DIFFUSION_NUM_INFERENCE_STEPS} <= 100)) \
    || die "DIFFUSION_NUM_INFERENCE_STEPS must be <= 100"
  ((10#${DIFFUSION_SEND_HORIZON} == 16)) \
    || die "DIFFUSION_SEND_HORIZON must be 16 for the CM v2 protocol"
  ((10#${DIFFUSION_EXECUTION_HORIZON} == 8)) \
    || die "DIFFUSION_EXECUTION_HORIZON must be 8 for the CM v2 protocol"
  ((10#${OBS_PORT} <= 65535)) || die "DIFFUSION_EVAL_OBS_PORT must be <= 65535"
  ((10#${DIFFUSION_PORT} <= 65535)) || die "DIFFUSION_EVAL_DIFFUSION_PORT must be <= 65535"
  [[ "${OBS_PORT}" != "${DIFFUSION_PORT}" ]] || die "observation and action ports must differ"
  [[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
  [[ "${VIDEO_OUTCOME_SUFFIX}" == "0" || "${VIDEO_OUTCOME_SUFFIX}" == "1" ]] \
    || die "DIFFUSION_EVAL_VIDEO_OUTCOME_SUFFIX must be 0 or 1"
  require_uint_at_least \
    "DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES" "${DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES}" 0
  [[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "DIFFUSION_EVAL_RUN_ID may contain only letters, numbers, dot, underscore, and hyphen"
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
  local task_max_steps="$2"
  local video_dir="$3"
  local video_name="$4"

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
    --max_steps
    "${task_max_steps}"
    --initial_position_mode
    fixed
    --device
    "${MOTIONFORGE_DEVICE}"
    --obs_port
    "${OBS_PORT}"
    --act_port
    "${DIFFUSION_PORT}"
    --client_warmup
    "${CLIENT_WARMUP_S}"
    --clock_mode
    "${CLOCK_MODE}"
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

  BRIDGE_COMMAND=(
    "${TIMEOUT_EXE}"
    --signal=TERM
    --kill-after=30s
    "${TASK_TIMEOUT_S}s"
    "${DIFFUSION_PYTHON}"
    "${BRIDGE_CLIENT}"
    --model-path
    "${DIFFUSION_MODEL_PATH}"
    --device
    "${DIFFUSION_DEVICE}"
    --policy-seed
    "${DIFFUSION_POLICY_SEED}"
    --num-inference-steps
    "${DIFFUSION_NUM_INFERENCE_STEPS}"
    --motionforge-obs-port
    "${OBS_PORT}"
    --motionforge-act-port
    "${DIFFUSION_PORT}"
    --num-episodes
    "${NUM_TRIALS}"
    --action-hz
    "${DIFFUSION_ACTION_HZ}"
    --max-inference-hz
    "${DIFFUSION_MAX_INFERENCE_HZ}"
    --send-horizon
    "${DIFFUSION_SEND_HORIZON}"
    --execution-horizon
    "${DIFFUSION_EXECUTION_HORIZON}"
    --print-every
    "${DIFFUSION_PRINT_EVERY}"
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
  printf '  (cd %q && CUDA_VISIBLE_DEVICES=%q PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=%q ' \
    "${LEROBOT_ROOT}" "${DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES}" "${BRIDGE_PYTHONPATH}"
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
  return 0
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
  local video_path=""
  local success_path=""
  local failure_path=""

  for ((trial_number = 1; trial_number <= 10#${NUM_TRIALS}; trial_number++)); do
    if ((10#${NUM_TRIALS} == 1)); then
      video_path="${video_dir}/${task_id}_rollout.mp4"
    else
      printf -v video_path '%s/%s_rollout_trial_%03d.mp4' "${video_dir}" "${task_id}" "${trial_number}"
    fi
    if [[ "${VIDEO_OUTCOME_SUFFIX}" == "1" ]]; then
      success_path="${video_path%.mp4}_success.mp4"
      failure_path="${video_path%.mp4}_failure.mp4"
      if [[ -s "${success_path}" && ! -e "${failure_path}" ]]; then
        continue
      fi
      if [[ -s "${failure_path}" && ! -e "${success_path}" ]]; then
        continue
      fi
      return 1
    fi
    [[ -s "${video_path}" ]] || return 1
  done
  return 0
}

append_task_result() {
  local task_id="$1"
  local status="$2"
  local benchmark_config="$3"
  local task_max_steps="$4"
  local video_dir="$5"
  local reason="$6"
  local video_count=""
  video_count="$(count_videos "${video_dir}")"
  {
    printf '\n[%s]\n' "${task_id}"
    printf 'status=%s\n' "${status}"
    printf 'benchmark=%s\n' "${benchmark_config}"
    printf 'max_steps=%s\n' "${task_max_steps}"
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
  local task_max_steps=""
  local task_result_dir="${RESULT_DIR}/${task_id}"
  local server_log="${task_result_dir}/server.log"
  local client_log="${task_result_dir}/client.log"
  local video_dir="${task_result_dir}/videos"
  local process_status=0

  task_max_steps="$(benchmark_max_steps "${benchmark_config}")"
  LAST_TRIALS=0
  LAST_SUCCESSES=0
  LAST_FAILURES=0
  LAST_SUCCESS_RATE=""
  LAST_RAW_SUMMARY=""
  mkdir -p "${video_dir}" || return 1
  build_commands "${benchmark_config}" "${task_max_steps}" "${video_dir}" "${task_id}_rollout.mp4"
  log "starting task=${task_id} trials=${NUM_TRIALS} max_steps=${task_max_steps} seeds=${START_SEED}-$((START_SEED + NUM_TRIALS - 1))"
  log "server_log=${server_log}"
  log "client_log=${client_log}"

  (
    cd -- "${MOTIONFORGE_ROOT}"
    export CUDA_VISIBLE_DEVICES="${DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES}"
    export OMNI_KIT_ACCEPT_EULA="YES"
    exec "${SERVER_COMMAND[@]}"
  ) >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  (
    cd -- "${LEROBOT_ROOT}"
    export CUDA_VISIBLE_DEVICES="${DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES}"
    export PYTHONDONTWRITEBYTECODE=1
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export PYTHONPATH="${BRIDGE_PYTHONPATH}"
    exec "${BRIDGE_COMMAND[@]}"
  ) >"${client_log}" 2>&1 &
  BRIDGE_PID="$!"

  if wait_for_server_and_bridge; then
    process_status=0
  else
    process_status="$?"
  fi
  if ((process_status != 0)); then
    append_task_result "${task_id}" failed "${benchmark_config}" "${task_max_steps}" "${video_dir}" \
      "server/client process exit status ${process_status}"
    return "${process_status}"
  fi
  if ! parse_task_summary "${server_log}"; then
    append_task_result "${task_id}" failed "${benchmark_config}" "${task_max_steps}" "${video_dir}" \
      "server log has no valid ${NUM_TRIALS}-trial summary"
    return 1
  fi
  if ! validate_video_outputs "${task_id}" "${video_dir}"; then
    append_task_result "${task_id}" failed "${benchmark_config}" "${task_max_steps}" "${video_dir}" \
      "expected ${NUM_TRIALS} non-empty rollout videos with unambiguous outcome suffixes"
    return 1
  fi
  append_task_result "${task_id}" completed "${benchmark_config}" "${task_max_steps}" "${video_dir}" "none"
  log "completed task=${task_id} successes=${LAST_SUCCESSES}/${LAST_TRIALS} success_rate=${LAST_SUCCESS_RATE}"
  return 0
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
  local partial_success_rate="N/A"
  local overall_status="incomplete"

  if ((completed_trials > 0)); then
    partial_success_rate="$(awk -v successes="${total_successes}" -v trials="${completed_trials}" 'BEGIN { printf "%.3f", successes / trials }')"
  fi
  if ((completed_tasks == expected_tasks && failed_tasks == 0 && completed_trials == expected_trials)); then
    overall_status="completed"
    total_success_rate="${partial_success_rate}"
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
    printf 'partial_success_rate=%s\n' "${partial_success_rate}"
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  } >>"${SUMMARY_FILE}"
}

validate_configuration

if [[ "${DRY_RUN}" == "1" ]]; then
  log "validated configuration; no process or result directory will be created"
  log "tasks=${#TASK_IDS[@]} trials_per_task=${NUM_TRIALS} max_steps=benchmark_config clock_mode=${CLOCK_MODE} timing_protocol=${TIMING_PROTOCOL} action_alignment=${DIFFUSION_ACTION_ALIGNMENT} physics_device=${MOTIONFORGE_DEVICE} num_inference_steps=${DIFFUSION_NUM_INFERENCE_STEPS} send_horizon=${DIFFUSION_SEND_HORIZON} execution_horizon=${DIFFUSION_EXECUTION_HORIZON} server_max_source_actions=16 server_max_execution_control_steps=16 video_outcome_suffix=${VIDEO_OUTCOME_SUFFIX} expected_trials=$((${#TASK_IDS[@]} * NUM_TRIALS))"
  log "model=${DIFFUSION_MODEL_PATH} result_dir=${RESULT_DIR} policy_seed=${DIFFUSION_POLICY_SEED}"
  for task_id in "${TASK_IDS[@]}"; do
    benchmark_config="${BENCHMARK_DIR}/${task_id}_rgb_gr00t_zmq.yaml"
    task_max_steps="$(benchmark_max_steps "${benchmark_config}")"
    video_dir="${RESULT_DIR}/${task_id}/videos"
    build_commands "${benchmark_config}" "${task_max_steps}" "${video_dir}" "${task_id}_rollout.mp4"
    log "dry-run task=${task_id} max_steps=${task_max_steps} benchmark=${benchmark_config}"
    print_command "${MOTIONFORGE_ROOT}" "${SERVER_COMMAND[@]}"
    print_bridge_command
  done
  exit 0
fi

if [[ -e "${RESULT_DIR}" ]]; then
  die "result directory already exists; choose another DIFFUSION_EVAL_RUN_ID: ${RESULT_DIR}"
fi
mkdir -p "${RESULT_DIR}"
{
  printf 'Diffusion Policy CM000-CM010 MotionForge evaluation\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'model=%s\n' "${DIFFUSION_MODEL_PATH}"
  printf 'device=%s\n' "${DIFFUSION_DEVICE}"
  printf 'motionforge_physics_device=%s\n' "${MOTIONFORGE_DEVICE}"
  printf 'cuda_visible_devices=%s\n' "${DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES}"
  printf 'policy_seed=%s\n' "${DIFFUSION_POLICY_SEED}"
  printf 'num_inference_steps=%s\n' "${DIFFUSION_NUM_INFERENCE_STEPS}"
  printf 'tasks=%s\n' "${#TASK_IDS[@]}"
  printf 'task_ids=%s\n' "${TASK_IDS[*]}"
  printf 'trials_per_task=%s\n' "${NUM_TRIALS}"
  printf 'max_steps_source=benchmark_config\n'
  printf 'clock_mode=%s\n' "${CLOCK_MODE}"
  printf 'timing_protocol=%s\n' "${TIMING_PROTOCOL}"
  printf 'action_alignment=%s\n' "${DIFFUSION_ACTION_ALIGNMENT}"
  printf 'seed_start=%s\n' "${START_SEED}"
  printf 'seed_end=%s\n' "$((START_SEED + NUM_TRIALS - 1))"
  printf 'initial_position_mode=fixed\n'
  printf 'observation_horizon=2\n'
  printf 'diffusion_horizon=64\n'
  printf 'action_horizon=32\n'
  printf 'action_hz=%s\n' "${DIFFUSION_ACTION_HZ}"
  printf 'max_inference_hz=%s\n' "${DIFFUSION_MAX_INFERENCE_HZ}"
  printf 'send_horizon=%s\n' "${DIFFUSION_SEND_HORIZON}"
  printf 'execution_horizon=%s\n' "${DIFFUSION_EXECUTION_HORIZON}"
  printf 'max_source_actions=16\n'
  printf 'max_execution_control_steps=16\n'
  printf 'video_enabled=true\n'
  printf 'video_width=%s\n' "${VIDEO_WIDTH}"
  printf 'video_height=%s\n' "${VIDEO_HEIGHT}"
  printf 'video_stride=%s\n' "${VIDEO_STRIDE}"
  printf 'video_outcome_suffix=%s\n' "${VIDEO_OUTCOME_SUFFIX}"
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
