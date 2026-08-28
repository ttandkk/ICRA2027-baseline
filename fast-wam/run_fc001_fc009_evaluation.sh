#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${WORKSPACE_ROOT}/lerobot}"

BRIDGE_CLIENT="${SCRIPT_DIR}/motionforge_fastwam_bridge_client.py"
TRIALS_SERVER="${MOTIONFORGE_ROOT}/scripts/benchmark/run_env_server_trials.py"
BENCHMARK_DIR="${MOTIONFORGE_ROOT}/configs/benchmarks/factory_conveyor"

FASTWAM_MODEL_PATH="${FASTWAM_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/MotionforgeGroup/Fast-WAM/FC-40000}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-${WORKSPACE_ROOT}/miniconda3/envs/lerobot/bin/python}"
FASTWAM_DEVICE="${FASTWAM_DEVICE:-cuda:0}"
FASTWAM_ACTION_HZ="${FASTWAM_ACTION_HZ:-30}"
FASTWAM_MAX_INFERENCE_HZ="${FASTWAM_MAX_INFERENCE_HZ:-30}"
FASTWAM_SEND_HORIZON="${FASTWAM_SEND_HORIZON:-16}"
FASTWAM_EXECUTION_HORIZON="${FASTWAM_EXECUTION_HORIZON:-8}"
FASTWAM_PRINT_EVERY="${FASTWAM_PRINT_EVERY:-10}"

MOTIONFORGE_CONDA_ENV="${MOTIONFORGE_CONDA_ENV:-motionforge}"
# Protocol default: CPU physics, with rendering and policy inference on the visible GPU.
MOTIONFORGE_DEVICE="${MOTIONFORGE_DEVICE:-cpu}"
FASTWAM_EVAL_CUDA_VISIBLE_DEVICES="${FASTWAM_EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-2}}"
FASTWAM_HF_HOME="${FASTWAM_HF_HOME:-${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}}"
CONDA_EXE="${MOTIONFORGE_CONDA_EXE:-${WORKSPACE_ROOT}/miniconda3/bin/conda}"
TIMEOUT_EXE="${MOTIONFORGE_TIMEOUT_EXE:-$(command -v timeout || true)}"

START_SEED="${FASTWAM_EVAL_START_SEED:-0}"
NUM_TRIALS="${FASTWAM_EVAL_NUM_TRIALS:-50}"
# Keep an explicit override available, but use each benchmark YAML by default.
MAX_STEPS="${FASTWAM_EVAL_MAX_STEPS:-1400}"
USE_BENCHMARK_MAX_STEPS="${FASTWAM_EVAL_USE_BENCHMARK_MAX_STEPS:-1}"
MOTION_LEVEL="${FASTWAM_EVAL_MOTION_LEVEL:-}"
INITIAL_POSITION_MODE="${FASTWAM_EVAL_INITIAL_POSITION_MODE:-fixed}"
OBS_PORT="${FASTWAM_EVAL_OBS_PORT:-3196}"
ACT_PORT="${FASTWAM_EVAL_ACT_PORT:-3198}"
CLIENT_WARMUP_S="${FASTWAM_EVAL_CLIENT_WARMUP_S:-5}"
TASK_TIMEOUT_S="${FASTWAM_EVAL_TASK_TIMEOUT_S:-14400}"
BETWEEN_TASKS_S="${FASTWAM_EVAL_BETWEEN_TASKS_S:-5}"
PERSISTENT_BRIDGE="${FASTWAM_EVAL_PERSISTENT_BRIDGE:-0}"
BRIDGE_READY_TIMEOUT_S="${FASTWAM_EVAL_BRIDGE_READY_TIMEOUT_S:-900}"

VIDEO_WIDTH="${FASTWAM_EVAL_VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${FASTWAM_EVAL_VIDEO_HEIGHT:-480}"
VIDEO_STRIDE="${FASTWAM_EVAL_VIDEO_STRIDE:-1}"

RESULT_ROOT="${SCRIPT_DIR}/output"
RUN_ID="${FASTWAM_EVAL_RUN_ID:-fastwam_fc001_fc009_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_ROOT}/${RUN_ID}"
SUMMARY_FILE="${RESULT_DIR}/success_rates.txt"
BRIDGE_LOG="${RESULT_DIR}/bridge.log"
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

if [[ -n "${FASTWAM_EVAL_TASKS:-}" ]]; then
  read -r -a TASK_IDS <<<"${FASTWAM_EVAL_TASKS}"
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
PERSISTENT_BRIDGE_FAILED=0

log() {
  printf '[FASTWAM-FC001-FC009-EVAL] %s\n' "$*"
}

die() {
  printf '[FASTWAM-FC001-FC009-EVAL] ERROR: %s\n' "$*" >&2
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
  require_dir "${FASTWAM_MODEL_PATH}"
  require_file "${FASTWAM_MODEL_PATH}/config.json"
  require_file "${FASTWAM_MODEL_PATH}/train_config.json"
  require_file "${FASTWAM_MODEL_PATH}/model.safetensors"
  require_file "${FASTWAM_MODEL_PATH}/policy_preprocessor.json"
  require_file "${FASTWAM_MODEL_PATH}/policy_postprocessor.json"
  require_file "${FASTWAM_MODEL_PATH}/policy_preprocessor_step_3_normalizer_processor.safetensors"
  require_file "${FASTWAM_MODEL_PATH}/policy_postprocessor_step_0_unnormalizer_processor.safetensors"

  PYTHONDONTWRITEBYTECODE=1 "${FASTWAM_PYTHON}" -c '
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

checkpoint = Path(sys.argv[1])
config = json.loads((checkpoint / "config.json").read_text())
train_policy = json.loads((checkpoint / "train_config.json").read_text()).get("policy")
assert isinstance(train_policy, dict)
expected_saved = dict(train_policy)
expected_saved["pretrained_path"] = None
assert config == expected_saved
assert config.get("type") == "fastwam"
assert config.get("action_dim") == 10
assert config.get("proprio_dim") == 10
assert config.get("action_horizon") == 32
assert config.get("n_action_steps") == 10
assert config.get("num_video_frames") == 33
assert config.get("action_video_freq_ratio") == 4
assert config.get("image_size") == [224, 672]
assert config.get("num_inference_steps") == 10
assert config.get("inference_seed") == 42
assert config.get("rand_device") == "cpu"
assert config.get("torch_dtype") == "bfloat16"
assert config.get("model_id") == "Wan-AI/Wan2.2-TI2V-5B"
assert config.get("tokenizer_model_id") == "google/umt5-xxl"
assert config.get("text_encoder_model_id") == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
assert config.get("load_text_encoder") is True
assert config.get("toggle_action_dimensions") == []
assert config.get("video_dit_config", {}).get("video_attention_mask_mode") == "first_frame_causal"
expected_inputs = {
    "observation.images.front": [3, 224, 224],
    "observation.images.overview": [3, 224, 224],
    "observation.images.wrist": [3, 224, 224],
    "observation.state": [10],
}
assert {
    key: value.get("shape") for key, value in config.get("input_features", {}).items()
} == expected_inputs
assert config.get("output_features", {}).get("action", {}).get("shape") == [10]

preprocessor = json.loads((checkpoint / "policy_preprocessor.json").read_text())
postprocessor = json.loads((checkpoint / "policy_postprocessor.json").read_text())
assert [step.get("registry_name") for step in preprocessor.get("steps", [])] == [
    "rename_observations_processor",
    "to_batch_processor",
    "device_processor",
    "normalizer_processor",
]
assert [step.get("registry_name") for step in postprocessor.get("steps", [])] == [
    "unnormalizer_processor",
    "device_processor",
]
assert preprocessor["steps"][0].get("config", {}).get("rename_map") == {}
expected_norm_map = {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
assert preprocessor["steps"][-1].get("config", {}).get("norm_map") == expected_norm_map
assert postprocessor["steps"][0].get("config", {}).get("norm_map") == expected_norm_map

pre_path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
post_path = checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
with safe_open(pre_path, framework="pt", device="cpu") as pre:
    for key in ("observation.state.mean", "observation.state.std", "action.mean", "action.std"):
        tensor = pre.get_tensor(key)
        assert tuple(tensor.shape) == (10,)
        assert bool(torch.isfinite(tensor).all())
    pre_action = {
        key: pre.get_tensor(key)
        for key in pre.keys()
        if key.startswith("action.")
    }
with safe_open(post_path, framework="pt", device="cpu") as post:
    assert set(post.keys()) == set(pre_action)
    for key, expected in pre_action.items():
        assert torch.equal(post.get_tensor(key), expected)
with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
    assert tuple(weights.get_slice("model.proprio_encoder.weight").get_shape()) == (4096, 10)
    assert tuple(weights.get_slice("model.mot.mixtures.action.head.weight").get_shape()) == (10, 1024)
' "${FASTWAM_MODEL_PATH}" || die "checkpoint does not match the trained Fast-WAM-FC contract"
}

validate_runtime_imports() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${FASTWAM_PYTHON}" -c '
import zmq
from lerobot.configs import PreTrainedConfig
from lerobot.policies import FastWAMConfig, make_pre_post_processors
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
' || die "FastWAM runtime imports failed; verify the LeRobot fastwam extra and pyzmq"
}

validate_configuration() {
  local task_id=""
  local benchmark_config=""

  require_dir "${MOTIONFORGE_ROOT}"
  require_dir "${LEROBOT_ROOT}/src/lerobot"
  require_dir "${FASTWAM_HF_HOME}"
  require_file "${BRIDGE_CLIENT}"
  require_file "${TRIALS_SERVER}"
  require_executable "${FASTWAM_PYTHON}"
  require_executable "${CONDA_EXE}"
  require_executable "${TIMEOUT_EXE}"
  command -v awk >/dev/null 2>&1 || die "required executable not found: awk"
  command -v grep >/dev/null 2>&1 || die "required executable not found: grep"
  validate_checkpoint
  validate_runtime_imports

  ((${#TASK_IDS[@]} > 0)) || die "FASTWAM_EVAL_TASKS must select at least one task"
  for task_id in "${TASK_IDS[@]}"; do
    [[ "${task_id}" =~ ^fc_00[1-9]$ ]] || die "invalid FC task id: ${task_id}"
    benchmark_config="${BENCHMARK_DIR}/${task_id}_rgb_gr00t_zmq.yaml"
    require_file "${benchmark_config}"
  done

  require_uint_at_least "FASTWAM_EVAL_START_SEED" "${START_SEED}" 0
  require_uint_at_least "FASTWAM_EVAL_NUM_TRIALS" "${NUM_TRIALS}" 1
  require_uint_at_least "FASTWAM_EVAL_MAX_STEPS" "${MAX_STEPS}" 1
  require_uint_at_least "FASTWAM_EVAL_OBS_PORT" "${OBS_PORT}" 1
  require_uint_at_least "FASTWAM_EVAL_ACT_PORT" "${ACT_PORT}" 1
  require_uint_at_least "FASTWAM_EVAL_CLIENT_WARMUP_S" "${CLIENT_WARMUP_S}" 0
  require_uint_at_least "FASTWAM_EVAL_TASK_TIMEOUT_S" "${TASK_TIMEOUT_S}" 1
  require_uint_at_least "FASTWAM_EVAL_BETWEEN_TASKS_S" "${BETWEEN_TASKS_S}" 0
  require_uint_at_least "FASTWAM_EVAL_BRIDGE_READY_TIMEOUT_S" "${BRIDGE_READY_TIMEOUT_S}" 1
  require_uint_at_least "FASTWAM_PRINT_EVERY" "${FASTWAM_PRINT_EVERY}" 0
  require_uint_at_least "FASTWAM_SEND_HORIZON" "${FASTWAM_SEND_HORIZON}" 1
  require_uint_at_least "FASTWAM_EXECUTION_HORIZON" "${FASTWAM_EXECUTION_HORIZON}" 1
  require_uint_at_least "FASTWAM_EVAL_VIDEO_WIDTH" "${VIDEO_WIDTH}" 2
  require_uint_at_least "FASTWAM_EVAL_VIDEO_HEIGHT" "${VIDEO_HEIGHT}" 2
  require_uint_at_least "FASTWAM_EVAL_VIDEO_STRIDE" "${VIDEO_STRIDE}" 1
  require_positive_number "FASTWAM_ACTION_HZ" "${FASTWAM_ACTION_HZ}"
  require_positive_number "FASTWAM_MAX_INFERENCE_HZ" "${FASTWAM_MAX_INFERENCE_HZ}"

  ((10#${FASTWAM_EXECUTION_HORIZON} <= 10#${FASTWAM_SEND_HORIZON})) \
    || die "FASTWAM_EXECUTION_HORIZON must be <= FASTWAM_SEND_HORIZON"
  ((10#${FASTWAM_SEND_HORIZON} <= 32)) \
    || die "FASTWAM_SEND_HORIZON must be <= the checkpoint action horizon 32"
  ((10#${OBS_PORT} <= 65535)) || die "FASTWAM_EVAL_OBS_PORT must be <= 65535"
  ((10#${ACT_PORT} <= 65535)) || die "FASTWAM_EVAL_ACT_PORT must be <= 65535"
  [[ "${OBS_PORT}" != "${ACT_PORT}" ]] || die "observation and action ports must differ"
  [[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
  [[ "${PERSISTENT_BRIDGE}" == "0" || "${PERSISTENT_BRIDGE}" == "1" ]] \
    || die "FASTWAM_EVAL_PERSISTENT_BRIDGE must be 0 or 1"
  [[ "${USE_BENCHMARK_MAX_STEPS}" == "0" || "${USE_BENCHMARK_MAX_STEPS}" == "1" ]] \
    || die "FASTWAM_EVAL_USE_BENCHMARK_MAX_STEPS must be 0 or 1"
  [[ -z "${MOTION_LEVEL}" || "${MOTION_LEVEL}" =~ ^level[123]$ ]] \
    || die "FASTWAM_EVAL_MOTION_LEVEL must be empty, level1, level2, or level3"
  [[ "${INITIAL_POSITION_MODE}" == "fixed" || "${INITIAL_POSITION_MODE}" == "seeded" ]] \
    || die "FASTWAM_EVAL_INITIAL_POSITION_MODE must be fixed or seeded"
  [[ -n "${FASTWAM_EVAL_CUDA_VISIBLE_DEVICES}" ]] \
    || die "FASTWAM_EVAL_CUDA_VISIBLE_DEVICES must not be empty"
  [[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "FASTWAM_EVAL_RUN_ID may contain only letters, numbers, dot, underscore, and hyphen"
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

build_server_command() {
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
}

build_bridge_command() {
  local num_episodes="${NUM_TRIALS}"
  if [[ "${PERSISTENT_BRIDGE}" == "1" ]]; then
    num_episodes=0
  fi
  BRIDGE_COMMAND=()
  if [[ "${PERSISTENT_BRIDGE}" == "0" ]]; then
    BRIDGE_COMMAND+=(
      "${TIMEOUT_EXE}"
      --signal=TERM
      --kill-after=30s
      "${TASK_TIMEOUT_S}s"
    )
  fi
  BRIDGE_COMMAND+=(
    "${FASTWAM_PYTHON}"
    "${BRIDGE_CLIENT}"
    --model-path
    "${FASTWAM_MODEL_PATH}"
    --device
    "${FASTWAM_DEVICE}"
    --motionforge-obs-port
    "${OBS_PORT}"
    --motionforge-act-port
    "${ACT_PORT}"
    --num-episodes
    "${num_episodes}"
    --action-hz
    "${FASTWAM_ACTION_HZ}"
    --max-inference-hz
    "${FASTWAM_MAX_INFERENCE_HZ}"
    --send-horizon
    "${FASTWAM_SEND_HORIZON}"
    --execution-horizon
    "${FASTWAM_EXECUTION_HORIZON}"
    --print-every
    "${FASTWAM_PRINT_EVERY}"
  )
}

start_bridge() {
  local bridge_log="$1"
  (
    cd -- "${LEROBOT_ROOT}"
    export CUDA_VISIBLE_DEVICES="${FASTWAM_EVAL_CUDA_VISIBLE_DEVICES}"
    export HF_HOME="${FASTWAM_HF_HOME}"
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    exec "${BRIDGE_COMMAND[@]}"
  ) >"${bridge_log}" 2>&1 &
  BRIDGE_PID="$!"
}

wait_for_bridge_ready() {
  local bridge_log="$1"
  local timeout_seconds=$((10#${BRIDGE_READY_TIMEOUT_S}))
  local deadline=$((SECONDS + timeout_seconds))
  local bridge_status=0

  while ((SECONDS < deadline)); do
    if grep -Fq '[MOTIONFORGE-FASTWAM] listening ' "${bridge_log}"; then
      log "persistent bridge ready pid=${BRIDGE_PID} log=${bridge_log}"
      return 0
    fi
    if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
      set +e
      wait "${BRIDGE_PID}"
      bridge_status="$?"
      set -e
      BRIDGE_PID=""
      ((bridge_status != 0)) || bridge_status=1
      log "persistent bridge exited before readiness status=${bridge_status} log=${bridge_log}"
      return "${bridge_status}"
    fi
    sleep 1
  done
  log "persistent bridge readiness timed out after ${BRIDGE_READY_TIMEOUT_S}s log=${bridge_log}"
  return 124
}

print_command() {
  local working_directory="$1"
  shift
  printf '  (cd %q && ' "${working_directory}"
  printf '%q ' "$@"
  printf ')\n'
}

print_bridge_command() {
  printf '  (cd %q && CUDA_VISIBLE_DEVICES=%q HF_HOME=%q PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=%q ' \
    "${LEROBOT_ROOT}" "${FASTWAM_EVAL_CUDA_VISIBLE_DEVICES}" "${FASTWAM_HF_HOME}" "${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
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

wait_for_server_with_persistent_bridge() {
  local server_pid="${SERVER_PID}"
  local bridge_pid="${BRIDGE_PID}"
  local completed_pid=""
  local process_status=0

  set +e
  wait -n -p completed_pid "${server_pid}" "${bridge_pid}"
  process_status="$?"
  set -e
  if [[ "${completed_pid}" == "${server_pid}" ]]; then
    SERVER_PID=""
    return "${process_status}"
  fi

  BRIDGE_PID=""
  PERSISTENT_BRIDGE_FAILED=1
  ((process_status != 0)) || process_status=1
  log "persistent bridge exited unexpectedly status=${process_status}; stopping task server"
  terminate_process "${SERVER_PID}"
  SERVER_PID=""
  return "${process_status}"
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
  [[ -n "${summary_line}" && "${summary_line}" =~ ${pattern} ]] || return 1
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

trial_video_outcome() {
  local server_log="$1"
  local trial_number="$2"
  local result_line=""
  local pattern="trial=${trial_number}/${NUM_TRIALS}[[:space:]]+seed=[0-9]+[[:space:]]+success=(True|False)[[:space:]]"

  result_line="$(grep -E "\\[MOTIONFORGE-BENCH\\] trial_result trial=${trial_number}/${NUM_TRIALS} .* success=(True|False) " "${server_log}" | tail -n 1 || true)"
  [[ -n "${result_line}" && "${result_line}" =~ ${pattern} ]] || return 1
  if [[ "${BASH_REMATCH[1]}" == "True" ]]; then
    printf 'success\n'
  else
    printf 'failure\n'
  fi
}

rename_video_outputs() {
  local task_id="$1"
  local video_dir="$2"
  local server_log="$3"
  local trial_number=0
  local source_path=""
  local destination_path=""
  local outcome=""
  for ((trial_number = 1; trial_number <= 10#${NUM_TRIALS}; trial_number++)); do
    outcome="$(trial_video_outcome "${server_log}" "${trial_number}")" || return 1
    if ((10#${NUM_TRIALS} == 1)); then
      source_path="${video_dir}/${task_id}_rollout.mp4"
      destination_path="${video_dir}/${task_id}_rollout_${outcome}.mp4"
    else
      printf -v source_path '%s/%s_rollout_trial_%03d.mp4' \
        "${video_dir}" "${task_id}" "${trial_number}"
      printf -v destination_path '%s/%s_rollout_trial_%03d_%s.mp4' \
        "${video_dir}" "${task_id}" "${trial_number}" "${outcome}"
    fi
    [[ -s "${source_path}" && ! -e "${destination_path}" ]] || return 1
    mv -- "${source_path}" "${destination_path}" || return 1
  done
}

validate_video_outputs() {
  local task_id="$1"
  local video_dir="$2"
  local server_log="$3"
  local trial_number=0
  local video_path=""
  local outcome=""
  for ((trial_number = 1; trial_number <= 10#${NUM_TRIALS}; trial_number++)); do
    outcome="$(trial_video_outcome "${server_log}" "${trial_number}")" || return 1
    if ((10#${NUM_TRIALS} == 1)); then
      video_path="${video_dir}/${task_id}_rollout_${outcome}.mp4"
    else
      printf -v video_path '%s/%s_rollout_trial_%03d_%s.mp4' \
        "${video_dir}" "${task_id}" "${trial_number}" "${outcome}"
    fi
    [[ -s "${video_path}" ]] || return 1
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
  build_server_command "${benchmark_config}" "${video_dir}" "${task_id}_rollout.mp4"
  log "starting task=${task_id} trials=${NUM_TRIALS} max_steps=${max_steps_label} motion_level=${MOTION_LEVEL:-benchmark_config} initial_position_mode=${INITIAL_POSITION_MODE} seeds=${START_SEED}-$((START_SEED + NUM_TRIALS - 1))"

  (
    cd -- "${MOTIONFORGE_ROOT}"
    export CUDA_VISIBLE_DEVICES="${FASTWAM_EVAL_CUDA_VISIBLE_DEVICES}"
    export OMNI_KIT_ACCEPT_EULA="YES"
    exec "${SERVER_COMMAND[@]}"
  ) >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  if [[ "${PERSISTENT_BRIDGE}" == "1" ]]; then
    if wait_for_server_with_persistent_bridge; then
      process_status=0
    else
      process_status="$?"
    fi
  else
    build_bridge_command
    start_bridge "${client_log}"
    if wait_for_server_and_bridge; then
      process_status=0
    else
      process_status="$?"
    fi
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
  if ! rename_video_outputs "${task_id}" "${video_dir}" "${server_log}"; then
    append_task_result "${task_id}" failed "${benchmark_config}" "${video_dir}" \
      "could not match and rename every rollout video with its trial outcome"
    return 1
  fi
  if ! validate_video_outputs "${task_id}" "${video_dir}" "${server_log}"; then
    append_task_result "${task_id}" failed "${benchmark_config}" "${video_dir}" \
      "expected ${NUM_TRIALS} non-empty outcome-labeled rollout videos"
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
  log "tasks=${#TASK_IDS[@]} trials_per_task=${NUM_TRIALS} max_steps=${max_steps_label} motion_level=${MOTION_LEVEL:-benchmark_config} initial_position_mode=${INITIAL_POSITION_MODE} send_horizon=${FASTWAM_SEND_HORIZON} execution_horizon=${FASTWAM_EXECUTION_HORIZON}"
  log "model=${FASTWAM_MODEL_PATH} hf_home=${FASTWAM_HF_HOME} persistent_bridge=${PERSISTENT_BRIDGE} result_dir=${RESULT_DIR}"
  if [[ "${PERSISTENT_BRIDGE}" == "1" ]]; then
    build_bridge_command
    log "dry-run persistent bridge log=${BRIDGE_LOG} ready_timeout_s=${BRIDGE_READY_TIMEOUT_S}"
    print_bridge_command
  fi
  for task_id in "${TASK_IDS[@]}"; do
    benchmark_config="${BENCHMARK_DIR}/${task_id}_rgb_gr00t_zmq.yaml"
    video_dir="${RESULT_DIR}/${task_id}/videos"
    build_server_command "${benchmark_config}" "${video_dir}" "${task_id}_rollout.mp4"
    log "dry-run task=${task_id} benchmark=${benchmark_config}"
    print_command "${MOTIONFORGE_ROOT}" "${SERVER_COMMAND[@]}"
    if [[ "${PERSISTENT_BRIDGE}" == "0" ]]; then
      build_bridge_command
      print_bridge_command
    fi
  done
  exit 0
fi

if [[ -e "${RESULT_DIR}" ]]; then
  die "result directory already exists; choose another FASTWAM_EVAL_RUN_ID: ${RESULT_DIR}"
fi
mkdir -p "${RESULT_DIR}"
{
  printf 'FastWAM FC001-FC009 MotionForge evaluation\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'model=%s\n' "${FASTWAM_MODEL_PATH}"
  printf 'device=%s\n' "${FASTWAM_DEVICE}"
  printf 'motionforge_device=%s\n' "${MOTIONFORGE_DEVICE}"
  printf 'cuda_visible_devices=%s\n' "${FASTWAM_EVAL_CUDA_VISIBLE_DEVICES}"
  printf 'hf_home=%s\n' "${FASTWAM_HF_HOME}"
  printf 'persistent_bridge=%s\n' "${PERSISTENT_BRIDGE}"
  if [[ "${PERSISTENT_BRIDGE}" == "1" ]]; then
    printf 'bridge_log=%s\n' "${BRIDGE_LOG}"
    printf 'bridge_ready_timeout_s=%s\n' "${BRIDGE_READY_TIMEOUT_S}"
  else
    printf 'bridge_log=per_task_client.log\n'
  fi
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
  printf 'action_hz=%s\n' "${FASTWAM_ACTION_HZ}"
  printf 'max_inference_hz=%s\n' "${FASTWAM_MAX_INFERENCE_HZ}"
  printf 'action_horizon=32\n'
  printf 'trained_action_steps=10\n'
  printf 'send_horizon=%s\n' "${FASTWAM_SEND_HORIZON}"
  printf 'execution_horizon=%s\n' "${FASTWAM_EXECUTION_HORIZON}"
  printf 'video_enabled=true\n'
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

if [[ "${PERSISTENT_BRIDGE}" == "1" ]]; then
  build_bridge_command
  log "starting persistent bridge log=${BRIDGE_LOG} ready_timeout_s=${BRIDGE_READY_TIMEOUT_S}"
  start_bridge "${BRIDGE_LOG}"
  if ! wait_for_bridge_ready "${BRIDGE_LOG}"; then
    die "persistent bridge did not become ready; inspect ${BRIDGE_LOG}"
  fi
fi

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
  if [[ "${PERSISTENT_BRIDGE}" == "1" ]]; then
    terminate_process "${SERVER_PID}"
    SERVER_PID=""
    if [[ "${PERSISTENT_BRIDGE_FAILED}" == "1" ]]; then
      log "aborting remaining tasks because the persistent bridge is unavailable"
      break
    fi
  else
    cleanup_processes
  fi
  if [[ "${task_id}" != "${TASK_IDS[-1]}" ]] && ((10#${BETWEEN_TASKS_S} > 0)); then
    sleep "${BETWEEN_TASKS_S}"
  fi
done

if [[ "${PERSISTENT_BRIDGE}" == "1" ]]; then
  terminate_process "${BRIDGE_PID}"
  BRIDGE_PID=""
fi

append_overall_summary \
  "${completed_tasks}" \
  "${failed_tasks}" \
  "${completed_trials}" \
  "${total_successes}" \
  "${total_failures}"
log "results=${RESULT_DIR}"
log "summary=${SUMMARY_FILE}"
exit "${overall_status}"
