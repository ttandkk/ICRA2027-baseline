#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${WORKSPACE_ROOT}/lerobot}"
BRIDGE_PYTHONPATH="${MOTIONFORGE_ROOT}/source/motionforge:${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

BRIDGE_CLIENT="${SCRIPT_DIR}/motionforge_xvla_bridge_client.py"
TRIALS_SERVER="${MOTIONFORGE_ROOT}/scripts/benchmark/run_env_server_trials.py"
BENCHMARK_DIR="${MOTIONFORGE_ROOT}/configs/benchmarks/circular_motion"
RUNTIME_ASSET_DIR="${MOTIONFORGE_ROOT}/source/motionforge/motionforge/assets/runtime"

XVLA_MODEL_PATH="${XVLA_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/MotionforgeGroup/X-VLA/CM-80000}"
XVLA_PYTHON="${XVLA_PYTHON:-${WORKSPACE_ROOT}/miniconda3/envs/lerobot/bin/python}"
XVLA_DEVICE="${XVLA_DEVICE:-cuda:0}"
XVLA_ACTION_HZ="${XVLA_ACTION_HZ:-30}"
XVLA_ACTION_ALIGNMENT="${XVLA_ACTION_ALIGNMENT:-observation_aligned}"
XVLA_PRINT_EVERY="${XVLA_PRINT_EVERY:-10}"

MOTIONFORGE_CONDA_ENV="${MOTIONFORGE_CONDA_ENV:-motionforge}"
# CM compound containers and articulated payloads require CPU PhysX. Rendering
# and X-VLA inference still use the selected visible GPU.
MOTIONFORGE_DEVICE="${MOTIONFORGE_DEVICE:-cpu}"
XVLA_EVAL_CUDA_VISIBLE_DEVICES="${XVLA_EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-3}}"
XVLA_HF_HOME="${XVLA_HF_HOME:-${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}}"
CONDA_EXE="${MOTIONFORGE_CONDA_EXE:-${WORKSPACE_ROOT}/miniconda3/bin/conda}"
TIMEOUT_EXE="${MOTIONFORGE_TIMEOUT_EXE:-$(command -v timeout || true)}"

START_SEED="${XVLA_EVAL_START_SEED:-0}"
NUM_TRIALS="${XVLA_EVAL_NUM_TRIALS:-10}"
CLOCK_MODE="${XVLA_EVAL_CLOCK_MODE:-slowdown_scaled}"
TIMING_PROTOCOL="motionforge.slowdown_scaled.server.v1"
OBS_PORT="${XVLA_EVAL_OBS_PORT:-3396}"
ACT_PORT="${XVLA_EVAL_ACT_PORT:-3398}"
CLIENT_WARMUP_S="${XVLA_EVAL_CLIENT_WARMUP_S:-5}"
TASK_TIMEOUT_S="${XVLA_EVAL_TASK_TIMEOUT_S:-14400}"
BETWEEN_TASKS_S="${XVLA_EVAL_BETWEEN_TASKS_S:-5}"

VIDEO_WIDTH="${XVLA_EVAL_VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${XVLA_EVAL_VIDEO_HEIGHT:-480}"
VIDEO_STRIDE="${XVLA_EVAL_VIDEO_STRIDE:-1}"
VIDEO_OUTCOME_SUFFIX="${XVLA_EVAL_VIDEO_OUTCOME_SUFFIX:-1}"

RESULT_ROOT="${XVLA_EVAL_OUTPUT_ROOT:-${SCRIPT_DIR}/output/circular_motion/cm000_cm010/level2_fixed/control60_server_scheduled_v1}"
RUN_ID="${XVLA_EVAL_RUN_ID:-xvla_cm000_cm010_$(date +%Y%m%d_%H%M%S)}"
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

if [[ -n "${XVLA_EVAL_TASKS:-}" ]]; then
  read -r -a TASK_IDS <<<"${XVLA_EVAL_TASKS}"
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
  printf '[XVLA-CM000-CM010-EVAL] %s\n' "$*"
}

die() {
  printf '[XVLA-CM000-CM010-EVAL] ERROR: %s\n' "$*" >&2
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
  require_dir "${XVLA_MODEL_PATH}"
  require_file "${XVLA_MODEL_PATH}/config.json"
  require_file "${XVLA_MODEL_PATH}/train_config.json"
  require_file "${XVLA_MODEL_PATH}/model.safetensors"
  require_file "${XVLA_MODEL_PATH}/policy_preprocessor.json"
  require_file "${XVLA_MODEL_PATH}/policy_postprocessor.json"
  require_file "${XVLA_MODEL_PATH}/policy_preprocessor_step_7_normalizer_processor.safetensors"
  require_file "${XVLA_MODEL_PATH}/policy_postprocessor_step_0_unnormalizer_processor.safetensors"

  PYTHONDONTWRITEBYTECODE=1 "${XVLA_PYTHON}" -c '
import json
import sys
from pathlib import Path
from safetensors import safe_open

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
train = json.loads((root / "train_config.json").read_text(encoding="utf-8"))
preprocessor = json.loads((root / "policy_preprocessor.json").read_text(encoding="utf-8"))
postprocessor = json.loads((root / "policy_postprocessor.json").read_text(encoding="utf-8"))

expected_images = {
    "observation.images.image": [3, 256, 256],
    "observation.images.image2": [3, 256, 256],
    "observation.images.image3": [3, 224, 224],
}
expected_rename_map = {
    "observation.images.front": "observation.images.image",
    "observation.images.overview": "observation.images.image2",
    "observation.images.wrist": "observation.images.image3",
}
assert config.get("type") == "xvla"
assert config.get("chunk_size") == 30
assert config.get("n_action_steps") == 30
assert config.get("max_state_dim") == 20
assert config.get("max_action_dim") == 20
assert config.get("action_mode") == "auto"
assert config.get("tokenizer_name") == "facebook/bart-large"
assert config.get("input_features", {}).get("observation.state", {}).get("shape") == [8]
assert config.get("output_features", {}).get("action", {}).get("shape") == [10]
for key, shape in expected_images.items():
    assert config.get("input_features", {}).get(key, {}).get("shape") == shape, key

assert train.get("dataset", {}).get("repo_id") == "local/lerobot_cm_pi05_v30"
assert train.get("steps") == 80000
train_policy = train.get("policy", {})
for key in (
    "type", "chunk_size", "n_action_steps", "max_state_dim", "max_action_dim",
    "action_mode", "tokenizer_name", "input_features", "output_features",
):
    assert train_policy.get(key) == config.get(key), key

rename_steps = [
    step for step in preprocessor.get("steps", [])
    if step.get("registry_name") == "rename_observations_processor"
]
assert len(rename_steps) == 1
assert rename_steps[0].get("config", {}).get("rename_map") == expected_rename_map
normalizer_steps = [
    step for step in preprocessor.get("steps", [])
    if step.get("registry_name") == "normalizer_processor"
]
assert len(normalizer_steps) == 1
assert normalizer_steps[0].get("state_file") == "policy_preprocessor_step_7_normalizer_processor.safetensors"
unnormalizer_steps = [
    step for step in postprocessor.get("steps", [])
    if step.get("registry_name") == "unnormalizer_processor"
]
assert len(unnormalizer_steps) == 1
assert unnormalizer_steps[0].get("state_file") == "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

with safe_open(
    root / "policy_preprocessor_step_7_normalizer_processor.safetensors",
    framework="pt",
    device="cpu",
) as stats:
    assert tuple(stats.get_tensor("observation.state.mean").shape) == (10,)
    assert tuple(stats.get_tensor("action.mean").shape) == (10,)
with safe_open(
    root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    framework="pt",
    device="cpu",
) as stats:
    assert tuple(stats.get_tensor("action.mean").shape) == (10,)
' "${XVLA_MODEL_PATH}" || die "checkpoint does not match the trained X-VLA-CM contract"
}

validate_runtime_imports() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${BRIDGE_PYTHONPATH}" \
    "${XVLA_PYTHON}" -c '
import zmq
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
' || die "X-VLA runtime imports failed; verify the LeRobot environment and pyzmq"
}

validate_bridge_contract() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${BRIDGE_PYTHONPATH}" \
    "${XVLA_PYTHON}" -c '
import inspect
import sys

sys.path.insert(0, sys.argv[1])
import motionforge_xvla_bridge_client as client
from motionforge.benchmark.client import BenchmarkClientBridge, ClientBridgeConfig
from motionforge.benchmark.protocol import SERVER_SCHEDULED_TIMING_PROTOCOL

assert SERVER_SCHEDULED_TIMING_PROTOCOL == "motionforge.slowdown_scaled.server.v1"
assert list(inspect.signature(client.XVLAInference.predict).parameters) == ["self", "observation"]
assert "action_hz" in client.XVLAInference.__dataclass_fields__
assert not hasattr(client, "require_dt_scale")
assert not hasattr(client, "scaled_inference_timing")
assert not hasattr(client, "MotionForgeTransport")
assert BenchmarkClientBridge is not None
assert ClientBridgeConfig(num_episodes=1).legacy_send_horizon is None
' "${SCRIPT_DIR}" || die "X-VLA bridge does not match the CM server-scheduled contract"
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
  require_dir "${XVLA_HF_HOME}"
  require_dir "${BENCHMARK_DIR}"
  require_file "${BRIDGE_CLIENT}"
  require_file "${TRIALS_SERVER}"
  require_executable "${XVLA_PYTHON}"
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
    "XVLA_EVAL_CLOCK_MODE must be slowdown_scaled for CM v2 evaluation; got ${CLOCK_MODE}"
  [[ "${XVLA_ACTION_ALIGNMENT}" == "observation_aligned" ]] || die \
    "XVLA_ACTION_ALIGNMENT must be observation_aligned for realtime CM evaluation; got ${XVLA_ACTION_ALIGNMENT}"

  ((${#TASK_IDS[@]} > 0)) || die "XVLA_EVAL_TASKS must select at least one task"
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

  require_uint_at_least "XVLA_EVAL_START_SEED" "${START_SEED}" 0
  require_uint_at_least "XVLA_EVAL_NUM_TRIALS" "${NUM_TRIALS}" 1
  require_uint_at_least "XVLA_EVAL_OBS_PORT" "${OBS_PORT}" 1
  require_uint_at_least "XVLA_EVAL_ACT_PORT" "${ACT_PORT}" 1
  require_uint_at_least "XVLA_EVAL_CLIENT_WARMUP_S" "${CLIENT_WARMUP_S}" 0
  require_uint_at_least "XVLA_EVAL_TASK_TIMEOUT_S" "${TASK_TIMEOUT_S}" 1
  require_uint_at_least "XVLA_EVAL_BETWEEN_TASKS_S" "${BETWEEN_TASKS_S}" 0
  require_uint_at_least "XVLA_PRINT_EVERY" "${XVLA_PRINT_EVERY}" 0
  require_uint_at_least "XVLA_EVAL_VIDEO_WIDTH" "${VIDEO_WIDTH}" 2
  require_uint_at_least "XVLA_EVAL_VIDEO_HEIGHT" "${VIDEO_HEIGHT}" 2
  require_uint_at_least "XVLA_EVAL_VIDEO_STRIDE" "${VIDEO_STRIDE}" 1
  require_positive_number "XVLA_ACTION_HZ" "${XVLA_ACTION_HZ}"

  awk -v value="${XVLA_ACTION_HZ}" 'BEGIN { exit !(value == 30) }' \
    || die "XVLA_ACTION_HZ must be 30 for the CM v2 protocol"
  ((10#${OBS_PORT} <= 65535)) || die "XVLA_EVAL_OBS_PORT must be <= 65535"
  ((10#${ACT_PORT} <= 65535)) || die "XVLA_EVAL_ACT_PORT must be <= 65535"
  [[ "${OBS_PORT}" != "${ACT_PORT}" ]] || die "observation and action ports must differ"
  [[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
  [[ "${VIDEO_OUTCOME_SUFFIX}" == "0" || "${VIDEO_OUTCOME_SUFFIX}" == "1" ]] \
    || die "XVLA_EVAL_VIDEO_OUTCOME_SUFFIX must be 0 or 1"
  require_uint_at_least \
    "XVLA_EVAL_CUDA_VISIBLE_DEVICES" "${XVLA_EVAL_CUDA_VISIBLE_DEVICES}" 0
  [[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "XVLA_EVAL_RUN_ID may contain only letters, numbers, dot, underscore, and hyphen"
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
    "${TIMEOUT_EXE}" --signal=TERM --kill-after=30s "${TASK_TIMEOUT_S}s"
    "${CONDA_EXE}" run --no-capture-output -n "${MOTIONFORGE_CONDA_ENV}"
    python "${TRIALS_SERVER}"
    --benchmark_config "${benchmark_config}"
    --seed "${START_SEED}"
    --num_trials "${NUM_TRIALS}"
    --max_steps "${task_max_steps}"
    --initial_position_mode fixed
    --device "${MOTIONFORGE_DEVICE}"
    --obs_port "${OBS_PORT}"
    --act_port "${ACT_PORT}"
    --client_warmup "${CLIENT_WARMUP_S}"
    --clock_mode "${CLOCK_MODE}"
    --video_dir "${video_dir}"
    --video_name "${video_name}"
    --video_width "${VIDEO_WIDTH}"
    --video_height "${VIDEO_HEIGHT}"
    --video_stride "${VIDEO_STRIDE}"
  )
  if [[ "${VIDEO_OUTCOME_SUFFIX}" == "1" ]]; then
    SERVER_COMMAND+=(--video_outcome_suffix)
  fi

  BRIDGE_COMMAND=(
    "${TIMEOUT_EXE}" --signal=TERM --kill-after=30s "${TASK_TIMEOUT_S}s"
    "${XVLA_PYTHON}" "${BRIDGE_CLIENT}"
    --model-path "${XVLA_MODEL_PATH}"
    --device "${XVLA_DEVICE}"
    --motionforge-obs-port "${OBS_PORT}"
    --motionforge-act-port "${ACT_PORT}"
    --num-episodes "${NUM_TRIALS}"
    --action-hz "${XVLA_ACTION_HZ}"
    --print-every "${XVLA_PRINT_EVERY}"
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
  printf '  (cd %q && CUDA_VISIBLE_DEVICES=%q HF_HOME=%q PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=%q ' \
    "${LEROBOT_ROOT}" "${XVLA_EVAL_CUDA_VISIBLE_DEVICES}" "${XVLA_HF_HOME}" "${BRIDGE_PYTHONPATH}"
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
    export CUDA_VISIBLE_DEVICES="${XVLA_EVAL_CUDA_VISIBLE_DEVICES}"
    export OMNI_KIT_ACCEPT_EULA="YES"
    exec "${SERVER_COMMAND[@]}"
  ) >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  (
    cd -- "${LEROBOT_ROOT}"
    export CUDA_VISIBLE_DEVICES="${XVLA_EVAL_CUDA_VISIBLE_DEVICES}"
    export HF_HOME="${XVLA_HF_HOME}"
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
  log "tasks=${#TASK_IDS[@]} trials_per_task=${NUM_TRIALS} max_steps=benchmark_config clock_mode=${CLOCK_MODE} timing_protocol=${TIMING_PROTOCOL} action_alignment=${XVLA_ACTION_ALIGNMENT} physics_device=${MOTIONFORGE_DEVICE} server_max_source_actions=16 server_max_execution_control_steps=16 video_outcome_suffix=${VIDEO_OUTCOME_SUFFIX} expected_trials=$((${#TASK_IDS[@]} * NUM_TRIALS))"
  log "model=${XVLA_MODEL_PATH} result_dir=${RESULT_DIR}"
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
  die "result directory already exists; choose another XVLA_EVAL_RUN_ID: ${RESULT_DIR}"
fi
mkdir -p "${RESULT_DIR}"
{
  printf 'X-VLA CM000-CM010 MotionForge evaluation\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'model=%s\n' "${XVLA_MODEL_PATH}"
  printf 'device=%s\n' "${XVLA_DEVICE}"
  printf 'motionforge_physics_device=%s\n' "${MOTIONFORGE_DEVICE}"
  printf 'cuda_visible_devices=%s\n' "${XVLA_EVAL_CUDA_VISIBLE_DEVICES}"
  printf 'hf_home=%s\n' "${XVLA_HF_HOME}"
  printf 'tasks=%s\n' "${#TASK_IDS[@]}"
  printf 'task_ids=%s\n' "${TASK_IDS[*]}"
  printf 'trials_per_task=%s\n' "${NUM_TRIALS}"
  printf 'max_steps_source=benchmark_config\n'
  printf 'clock_mode=%s\n' "${CLOCK_MODE}"
  printf 'timing_protocol=%s\n' "${TIMING_PROTOCOL}"
  printf 'action_alignment=%s\n' "${XVLA_ACTION_ALIGNMENT}"
  printf 'seed_start=%s\n' "${START_SEED}"
  printf 'seed_end=%s\n' "$((START_SEED + NUM_TRIALS - 1))"
  printf 'initial_position_mode=fixed\n'
  printf 'action_hz=%s\n' "${XVLA_ACTION_HZ}"
  printf 'max_inference_hz_source=benchmark_config\n'
  printf 'action_horizon=30\n'
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
