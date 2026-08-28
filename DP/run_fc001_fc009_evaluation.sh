#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${WORKSPACE_ROOT}/lerobot}"

BRIDGE_CLIENT="${SCRIPT_DIR}/motionforge_diffusion_policy_bridge_client.py"
TRIALS_SERVER="${MOTIONFORGE_ROOT}/scripts/benchmark/run_env_server_trials.py"
BENCHMARK_DIR="${MOTIONFORGE_ROOT}/configs/benchmarks/factory_conveyor"

DIFFUSION_MODEL_PATH="${DIFFUSION_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/MotionforgeGroup/DiffusionPolicy/FC-80000-3views}"
DIFFUSION_PYTHON="${DIFFUSION_PYTHON:-${WORKSPACE_ROOT}/miniconda3/envs/lerobot/bin/python}"
DIFFUSION_DEVICE="${DIFFUSION_DEVICE:-cuda:0}"
DIFFUSION_POLICY_SEED="${DIFFUSION_POLICY_SEED:-0}"
DIFFUSION_NUM_INFERENCE_STEPS="${DIFFUSION_NUM_INFERENCE_STEPS:-40}"
DIFFUSION_ACTION_HZ="${DIFFUSION_ACTION_HZ:-30}"
DIFFUSION_MAX_INFERENCE_HZ="${DIFFUSION_MAX_INFERENCE_HZ:-30}"
DIFFUSION_SEND_HORIZON="${DIFFUSION_SEND_HORIZON:-16}"
DIFFUSION_EXECUTION_HORIZON="${DIFFUSION_EXECUTION_HORIZON:-8}"
DIFFUSION_PRINT_EVERY="${DIFFUSION_PRINT_EVERY:-10}"

MOTIONFORGE_CONDA_ENV="${MOTIONFORGE_CONDA_ENV:-motionforge}"
# Keep physics on CPU while AppLauncher renders on the selected visible GPU.
MOTIONFORGE_DEVICE="${MOTIONFORGE_DEVICE:-cpu}"
DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES="${DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-2}}"
CONDA_EXE="${MOTIONFORGE_CONDA_EXE:-${WORKSPACE_ROOT}/miniconda3/bin/conda}"
TIMEOUT_EXE="${MOTIONFORGE_TIMEOUT_EXE:-$(command -v timeout || true)}"

START_SEED="${DIFFUSION_EVAL_START_SEED:-0}"
NUM_TRIALS="${DIFFUSION_EVAL_NUM_TRIALS:-50}"
# Keep an explicit override available, but use each benchmark YAML by default.
MAX_STEPS="${DIFFUSION_EVAL_MAX_STEPS:-1400}"
USE_BENCHMARK_MAX_STEPS="${DIFFUSION_EVAL_USE_BENCHMARK_MAX_STEPS:-1}"
MOTION_LEVEL="${DIFFUSION_EVAL_MOTION_LEVEL:-}"
INITIAL_POSITION_MODE="${DIFFUSION_EVAL_INITIAL_POSITION_MODE:-fixed}"
OBS_PORT="${DIFFUSION_EVAL_OBS_PORT:-3196}"
ACT_PORT="${DIFFUSION_EVAL_ACT_PORT:-3198}"
CLIENT_WARMUP_S="${DIFFUSION_EVAL_CLIENT_WARMUP_S:-5}"
TASK_TIMEOUT_S="${DIFFUSION_EVAL_TASK_TIMEOUT_S:-14400}"
BETWEEN_TASKS_S="${DIFFUSION_EVAL_BETWEEN_TASKS_S:-5}"

VIDEO_WIDTH="${DIFFUSION_EVAL_VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${DIFFUSION_EVAL_VIDEO_HEIGHT:-480}"
VIDEO_STRIDE="${DIFFUSION_EVAL_VIDEO_STRIDE:-1}"

RESULT_ROOT="${SCRIPT_DIR}/output"
RUN_ID="${DIFFUSION_EVAL_RUN_ID:-diffusion_policy_fc001_fc009_$(date +%Y%m%d_%H%M%S)}"
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
  printf '[DIFFUSION-FC001-FC009-EVAL] %s\n' "$*"
}

die() {
  printf '[DIFFUSION-FC001-FC009-EVAL] ERROR: %s\n' "$*" >&2
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
  require_dir "${DIFFUSION_MODEL_PATH}"
  require_file "${DIFFUSION_MODEL_PATH}/config.json"
  require_file "${DIFFUSION_MODEL_PATH}/train_config.json"
  require_file "${DIFFUSION_MODEL_PATH}/model.safetensors"
  require_file "${DIFFUSION_MODEL_PATH}/policy_preprocessor.json"
  require_file "${DIFFUSION_MODEL_PATH}/policy_postprocessor.json"
  require_file "${DIFFUSION_MODEL_PATH}/policy_preprocessor_step_3_normalizer_processor.safetensors"
  require_file "${DIFFUSION_MODEL_PATH}/policy_postprocessor_step_0_unnormalizer_processor.safetensors"

  PYTHONDONTWRITEBYTECODE=1 "${DIFFUSION_PYTHON}" -c '
import json
import sys
from pathlib import Path

from safetensors import safe_open

checkpoint = Path(sys.argv[1])
config = json.loads((checkpoint / "config.json").read_text())
train_config = json.loads((checkpoint / "train_config.json").read_text())
preprocessor = json.loads((checkpoint / "policy_preprocessor.json").read_text())
expected_inputs = {
    "observation.state": [10],
    "observation.images.overview": [3, 240, 320],
    "observation.images.front": [3, 240, 320],
    "observation.images.wrist": [3, 240, 320],
}
expected_image_transforms = {
    "enable": True,
    "max_num_transforms": 1,
    "random_order": False,
    "tfs": {
        "resize": {
            "weight": 1.0,
            "type": "Resize",
            "kwargs": {"size": [240, 320], "antialias": True},
        }
    },
}
assert config.get("type") == "diffusion", config.get("type")
assert train_config.get("policy") == config
assert train_config.get("dataset", {}).get("image_transforms") == expected_image_transforms
assert config.get("n_obs_steps") == 2, config.get("n_obs_steps")
assert config.get("horizon") == 64, config.get("horizon")
assert config.get("n_action_steps") == 32, config.get("n_action_steps")
assert config.get("num_train_timesteps") == 100, config.get("num_train_timesteps")
assert config.get("num_inference_steps") is None, config.get("num_inference_steps")
assert config.get("noise_scheduler_type") == "DDPM", config.get("noise_scheduler_type")
assert config.get("use_separate_rgb_encoder_per_camera") is True
assert config.get("resize_shape") is None
assert config.get("crop_shape") is None
assert config.get("output_features", {}).get("action", {}).get("shape") == [10]
assert set(config.get("input_features", {})) == set(expected_inputs)
for key, shape in expected_inputs.items():
    assert config.get("input_features", {}).get(key, {}).get("shape") == shape, key
steps = preprocessor.get("steps", [])
assert [step.get("registry_name") for step in steps] == [
    "rename_observations_processor",
    "to_batch_processor",
    "device_processor",
    "normalizer_processor",
]
assert steps[0].get("config", {}).get("rename_map") == {}
stats_path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
with safe_open(stats_path, framework="pt", device="cpu") as stats:
    assert tuple(stats.get_tensor("observation.state.min").shape) == (10,)
    assert tuple(stats.get_tensor("action.min").shape) == (10,)
    assert tuple(stats.get_tensor("observation.images.overview.mean").shape) == (3, 1, 1)
    assert tuple(stats.get_tensor("observation.images.front.mean").shape) == (3, 1, 1)
    assert tuple(stats.get_tensor("observation.images.wrist.mean").shape) == (3, 1, 1)
' "${DIFFUSION_MODEL_PATH}" || die "checkpoint does not match the trained three-view DiffusionPolicy-FC contract"
}

validate_runtime_imports() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${DIFFUSION_PYTHON}" -c '
import diffusers
import zmq
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig
' || die "Diffusion Policy runtime imports failed; verify LeRobot, diffusers, and pyzmq"
}

validate_configuration() {
  local task_id=""
  local benchmark_config=""

  require_dir "${MOTIONFORGE_ROOT}"
  require_dir "${LEROBOT_ROOT}/src/lerobot"
  require_file "${BRIDGE_CLIENT}"
  require_file "${TRIALS_SERVER}"
  require_executable "${DIFFUSION_PYTHON}"
  require_executable "${CONDA_EXE}"
  require_executable "${TIMEOUT_EXE}"
  command -v awk >/dev/null 2>&1 || die "required executable not found: awk"
  command -v grep >/dev/null 2>&1 || die "required executable not found: grep"
  validate_checkpoint
  validate_runtime_imports

  ((${#TASK_IDS[@]} > 0)) || die "DIFFUSION_EVAL_TASKS must select at least one task"
  for task_id in "${TASK_IDS[@]}"; do
    [[ "${task_id}" =~ ^fc_00[1-9]$ ]] || die "invalid FC task id: ${task_id}"
    benchmark_config="${BENCHMARK_DIR}/${task_id}_rgb_gr00t_zmq.yaml"
    require_file "${benchmark_config}"
  done

  require_uint_at_least "DIFFUSION_EVAL_START_SEED" "${START_SEED}" 0
  require_uint_at_least "DIFFUSION_EVAL_NUM_TRIALS" "${NUM_TRIALS}" 1
  require_uint_at_least "DIFFUSION_EVAL_MAX_STEPS" "${MAX_STEPS}" 1
  require_uint_at_least "DIFFUSION_EVAL_OBS_PORT" "${OBS_PORT}" 1
  require_uint_at_least "DIFFUSION_EVAL_ACT_PORT" "${ACT_PORT}" 1
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

  ((10#${DIFFUSION_EXECUTION_HORIZON} <= 10#${DIFFUSION_SEND_HORIZON})) \
    || die "DIFFUSION_EXECUTION_HORIZON must be <= DIFFUSION_SEND_HORIZON"
  ((10#${DIFFUSION_SEND_HORIZON} <= 32)) \
    || die "DIFFUSION_SEND_HORIZON must be <= the checkpoint action horizon 32"
  ((10#${DIFFUSION_NUM_INFERENCE_STEPS} <= 100)) \
    || die "DIFFUSION_NUM_INFERENCE_STEPS must be <= the checkpoint training timestep count 100"
  ((10#${OBS_PORT} <= 65535)) || die "DIFFUSION_EVAL_OBS_PORT must be <= 65535"
  ((10#${ACT_PORT} <= 65535)) || die "DIFFUSION_EVAL_ACT_PORT must be <= 65535"
  [[ "${OBS_PORT}" != "${ACT_PORT}" ]] || die "observation and action ports must differ"
  [[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
  [[ "${USE_BENCHMARK_MAX_STEPS}" == "0" || "${USE_BENCHMARK_MAX_STEPS}" == "1" ]] \
    || die "DIFFUSION_EVAL_USE_BENCHMARK_MAX_STEPS must be 0 or 1"
  [[ -z "${MOTION_LEVEL}" || "${MOTION_LEVEL}" =~ ^level[123]$ ]] \
    || die "DIFFUSION_EVAL_MOTION_LEVEL must be empty, level1, level2, or level3"
  [[ "${INITIAL_POSITION_MODE}" == "fixed" || "${INITIAL_POSITION_MODE}" == "seeded" ]] \
    || die "DIFFUSION_EVAL_INITIAL_POSITION_MODE must be fixed or seeded"
  [[ -n "${DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES}" ]] \
    || die "DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES must not be empty"
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
    --video_outcome_suffix
    --video_width
    "${VIDEO_WIDTH}"
    --video_height
    "${VIDEO_HEIGHT}"
    --video_stride
    "${VIDEO_STRIDE}"
  )

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
    "${ACT_PORT}"
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
  printf '  (cd %q && CUDA_VISIBLE_DEVICES=%q PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=%q ' \
    "${LEROBOT_ROOT}" "${DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES}" "${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
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
  local base_video_path=""
  local success_video_path=""
  local failure_video_path=""
  for ((trial_number = 1; trial_number <= 10#${NUM_TRIALS}; trial_number++)); do
    if ((10#${NUM_TRIALS} == 1)); then
      base_video_path="${video_dir}/${task_id}_rollout.mp4"
    else
      printf -v base_video_path '%s/%s_rollout_trial_%03d.mp4' \
        "${video_dir}" "${task_id}" "${trial_number}"
    fi
    success_video_path="${base_video_path%.mp4}_success.mp4"
    failure_video_path="${base_video_path%.mp4}_failure.mp4"
    if [[ -s "${success_video_path}" ]]; then
      [[ ! -e "${failure_video_path}" ]] || return 1
    elif [[ -s "${failure_video_path}" ]]; then
      [[ ! -e "${success_video_path}" ]] || return 1
    else
      return 1
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
  log "starting task=${task_id} trials=${NUM_TRIALS} max_steps=${max_steps_label} motion_level=${MOTION_LEVEL:-benchmark_config} initial_position_mode=${INITIAL_POSITION_MODE} seeds=${START_SEED}-$((START_SEED + NUM_TRIALS - 1))"

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
    export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
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
      "expected ${NUM_TRIALS} non-empty rollout videos with success/failure suffixes"
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
  if [[ "${USE_BENCHMARK_MAX_STEPS}" == "1" ]]; then
    max_steps_label="benchmark_config"
  else
    max_steps_label="${MAX_STEPS}"
  fi
  log "tasks=${#TASK_IDS[@]} trials_per_task=${NUM_TRIALS} max_steps=${max_steps_label} motion_level=${MOTION_LEVEL:-benchmark_config} initial_position_mode=${INITIAL_POSITION_MODE} send_horizon=${DIFFUSION_SEND_HORIZON} execution_horizon=${DIFFUSION_EXECUTION_HORIZON} num_inference_steps=${DIFFUSION_NUM_INFERENCE_STEPS}"
  log "model=${DIFFUSION_MODEL_PATH} result_dir=${RESULT_DIR} policy_seed=${DIFFUSION_POLICY_SEED}"
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
  die "result directory already exists; choose another DIFFUSION_EVAL_RUN_ID: ${RESULT_DIR}"
fi
mkdir -p "${RESULT_DIR}"
{
  printf 'Diffusion Policy FC001-FC009 MotionForge evaluation\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'model=%s\n' "${DIFFUSION_MODEL_PATH}"
  printf 'device=%s\n' "${DIFFUSION_DEVICE}"
  printf 'motionforge_device=%s\n' "${MOTIONFORGE_DEVICE}"
  printf 'cuda_visible_devices=%s\n' "${DIFFUSION_EVAL_CUDA_VISIBLE_DEVICES}"
  printf 'policy_seed=%s\n' "${DIFFUSION_POLICY_SEED}"
  printf 'num_inference_steps=%s\n' "${DIFFUSION_NUM_INFERENCE_STEPS}"
  printf 'tasks=%s\n' "${#TASK_IDS[@]}"
  printf 'trials_per_task=%s\n' "${NUM_TRIALS}"
  if [[ "${USE_BENCHMARK_MAX_STEPS}" == "1" ]]; then
    printf 'max_steps_per_trial=benchmark_config\n'
  else
    printf 'max_steps_per_trial=%s\n' "${MAX_STEPS}"
  fi
  printf 'motion_level=%s\n' "${MOTION_LEVEL:-benchmark_config}"
  printf 'seed_start=%s\n' "${START_SEED}"
  printf 'seed_end=%s\n' "$((START_SEED + NUM_TRIALS - 1))"
  printf 'initial_position_mode=%s\n' "${INITIAL_POSITION_MODE}"
  printf 'action_hz=%s\n' "${DIFFUSION_ACTION_HZ}"
  printf 'max_inference_hz=%s\n' "${DIFFUSION_MAX_INFERENCE_HZ}"
  printf 'observation_horizon=2\n'
  printf 'diffusion_horizon=64\n'
  printf 'action_horizon=32\n'
  printf 'send_horizon=%s\n' "${DIFFUSION_SEND_HORIZON}"
  printf 'execution_horizon=%s\n' "${DIFFUSION_EXECUTION_HORIZON}"
  printf 'video_enabled=true\n'
  printf 'video_outcome_suffix=true\n'
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
