#!/usr/bin/env bash
#SBATCH --job-name=groot_hri_eval
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/groot_hri_eval_%j.out
#SBATCH --error=logs/groot_hri_eval_%j.err

set -Eeuo pipefail

if [[ -n "${GROOT_SCRIPT_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd -- "${GROOT_SCRIPT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
GROOT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BASELINE_ROOT="$(cd -- "${GROOT_ROOT}/.." && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${BASELINE_ROOT}/.." && pwd -P)"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${BASELINE_ROOT}/lerobot}"
BRIDGE_PYTHONPATH="${MOTIONFORGE_ROOT}/source/motionforge:${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

BRIDGE_CLIENT="${SCRIPT_DIR}/motionforge_groot_bridge_client.py"
TRIALS_SERVER="${MOTIONFORGE_ROOT}/scripts/benchmark/run_env_server_trials.py"
BENCHMARK_DIR="${MOTIONFORGE_ROOT}/configs/benchmarks/human_robot_interaction"

GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/gr00t1.7-HRI-80000}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_ROOT}/.venv/bin/python}"
GROOT_ACCEL_MODE="${GROOT_ACCEL_MODE:-trt_full_pipeline}"
GROOT_EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-NEW_EMBODIMENT}"
GROOT_BACKBONE_MODEL_PATH="${GROOT_BACKBONE_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/nvidia/Cosmos-Reason2-2B}"
GROOT_FFMPEG_PREFIX="${GROOT_FFMPEG_PREFIX:-${WORKSPACE_ROOT}/ICRA2027-baseline/.conda/lerobot-baselines}"
GROOT_TRT_ENGINE_PATH="${GROOT_TRT_ENGINE_PATH:-${GROOT_ROOT}/gr00t_trt_deployments/gr00t_trt_deployment_gr00t1.7-HRI-80000/engines}"
GROOT_DEVICE="${GROOT_DEVICE:-cuda:0}"
GROOT_PRINT_EVERY="${GROOT_PRINT_EVERY:-10}"

MOTIONFORGE_PYTHON="${MOTIONFORGE_PYTHON:-${WORKSPACE_ROOT}/isaacsim/python.sh}"
MOTIONFORGE_DEFAULT_DEVICE="${MOTIONFORGE_DEFAULT_DEVICE:-cpu}"
MOTIONFORGE_HRI006_DEVICE="${MOTIONFORGE_HRI006_DEVICE:-cuda:0}"
GROOT_EVAL_CUDA_VISIBLE_DEVICES="${GROOT_EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
TIMEOUT_EXE="${MOTIONFORGE_TIMEOUT_EXE:-$(command -v timeout || true)}"

START_SEED="${GROOT_EVAL_START_SEED:-0}"
NUM_TRIALS="${GROOT_EVAL_NUM_TRIALS:-50}"
ATTEMPTS_PER_WORKER="${GROOT_EVAL_ATTEMPTS_PER_WORKER:-50}"
HRI006_ATTEMPTS_PER_WORKER="${GROOT_EVAL_HRI006_ATTEMPTS_PER_WORKER:-1}"
WORKER_START_TIMEOUT_S="${GROOT_EVAL_WORKER_START_TIMEOUT_S:-1200}"
ATTEMPT_TIMEOUT_S="${GROOT_EVAL_ATTEMPT_TIMEOUT_S:-900}"
TASK_TIMEOUT_S="${GROOT_EVAL_TASK_TIMEOUT_S:-14400}"
BETWEEN_TASKS_S="${GROOT_EVAL_BETWEEN_TASKS_S:-5}"
OBS_PORT="${GROOT_EVAL_OBS_PORT:-3496}"
GROOT_PORT="${GROOT_EVAL_ACT_PORT:-3498}"

VIDEO_WIDTH="${GROOT_EVAL_VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${GROOT_EVAL_VIDEO_HEIGHT:-480}"
VIDEO_STRIDE="${GROOT_EVAL_VIDEO_STRIDE:-1}"
VIDEO_OUTCOME_SUFFIX="${GROOT_EVAL_VIDEO_OUTCOME_SUFFIX:-1}"

RESULT_ROOT="${GROOT_EVAL_OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/human_robot_interaction/hri000_hri009/level2_fixed/server_scheduled}"
RUN_ID="${GROOT_EVAL_RUN_ID:-groot_hri_eval_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_ROOT}/${RUN_ID}"
SUMMARY_FILE="${RESULT_DIR}/success_rates.txt"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_TASK_IDS=(
  hri_000
  hri_001
  hri_002
  hri_003
  hri_004
  hri_005
  hri_006
  hri_007
  hri_008
  hri_009
)

if [[ -n "${GROOT_EVAL_TASKS:-}" ]]; then
  read -r -a TASK_IDS <<<"${GROOT_EVAL_TASKS}"
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

log() {
  printf '[GROOT-HRI000-HRI009-EVAL] %s\n' "$*"
}

die() {
  printf '[GROOT-HRI000-HRI009-EVAL] ERROR: %s\n' "$*" >&2
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

benchmark_max_steps() {
  local benchmark_config="$1"
  local configured=""
  configured="$(awk '/^[[:space:]]*max_steps:[[:space:]]*[0-9]+[[:space:]]*$/ { print $2; exit }' "${benchmark_config}")"
  [[ -n "${configured}" ]] || die "runtime.max_steps not found in benchmark: ${benchmark_config}"
  printf '%s\n' "${configured}"
}

task_physics_device() {
  local task_id="$1"
  if [[ "${task_id}" == "hri_006" ]]; then
    printf '%s\n' "${MOTIONFORGE_HRI006_DEVICE}"
  else
    printf '%s\n' "${MOTIONFORGE_DEFAULT_DEVICE}"
  fi
}

task_attempts_per_worker() {
  local task_id="$1"
  if [[ "${task_id}" == "hri_006" ]]; then
    printf '%s\n' "${HRI006_ATTEMPTS_PER_WORKER}"
  else
    printf '%s\n' "${ATTEMPTS_PER_WORKER}"
  fi
}

validate_checkpoint() {
  require_dir "${GROOT_MODEL_PATH}"
  require_file "${GROOT_MODEL_PATH}/config.json"
  require_file "${GROOT_MODEL_PATH}/model.safetensors.index.json"
  require_file "${GROOT_MODEL_PATH}/processor_config.json"
  require_file "${GROOT_MODEL_PATH}/statistics.json"
  require_file "${GROOT_MODEL_PATH}/trainer_state.json"
  require_file "${GROOT_BACKBONE_MODEL_PATH}/config.json"
  require_file "${GROOT_BACKBONE_MODEL_PATH}/model.safetensors"
  require_file "${GROOT_BACKBONE_MODEL_PATH}/tokenizer_config.json"
  require_dir "${GROOT_FFMPEG_PREFIX}/lib"
  PYTHONOPTIMIZE= PYTHONDONTWRITEBYTECODE=1 "${GROOT_PYTHON}" -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
read = lambda name: json.loads((root / name).read_text())
assert read("config.json")["model_type"] == "Gr00tN1d7"
assert read("trainer_state.json")["global_step"] == 80000
assert "lerobot_hri" in (root / "experiment_cfg/config.yaml").read_text()
mod = read("processor_config.json")["processor_kwargs"]["modality_configs"]["new_embodiment"]
assert mod["video"]["modality_keys"] == ["overview", "front", "wrist"]
assert mod["state"]["modality_keys"] == ["joint_position", "gripper_width"]
assert mod["action"]["modality_keys"] == ["eef_pose_rot6d", "gripper"]
assert mod["action"]["delta_indices"] == list(range(16))
assert all(c["rep"] == "ABSOLUTE" for c in mod["action"]["action_configs"])
for shard in set(read("model.safetensors.index.json")["weight_map"].values()):
    assert (root / shard).stat().st_size > 0, shard
' "${GROOT_MODEL_PATH}" || die "checkpoint does not match the trained GR00T-HRI contract"
  case "${GROOT_ACCEL_MODE}" in
    pytorch | torch_compile) ;;
    trt_full_pipeline)
      local engine=""
      for engine in state_encoder.engine action_encoder.engine dit_bf16.engine action_decoder.engine vit.engine llm_bf16.engine vl_self_attention.engine export_metadata.json; do
        require_file "${GROOT_TRT_ENGINE_PATH}/${engine}"
      done
      ;;
    *) die "GROOT_ACCEL_MODE must be pytorch, torch_compile, or trt_full_pipeline" ;;
  esac
}

validate_bridge_contract() {
  PYTHONOPTIMIZE= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${BRIDGE_PYTHONPATH}" \
    "${GROOT_PYTHON}" -c '
import inspect
import sys

sys.path.insert(0, sys.argv[1])
import motionforge_groot_bridge_client as client
from motionforge.benchmark.client import BenchmarkClientBridge
from motionforge.benchmark.protocol import PROTOCOL

assert PROTOCOL == "motionforge.server_scheduled"
assert list(inspect.signature(client.GR00TInference.reset).parameters) == ["self", "reset"]
assert list(inspect.signature(client.GR00TInference.predict).parameters) == ["self", "observation"]
assert BenchmarkClientBridge is not None
' "${SCRIPT_DIR}" || die "GROOT bridge does not match the server-scheduled contract"
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
  require_executable "${GROOT_PYTHON}"
  require_executable "${MOTIONFORGE_PYTHON}"
  require_executable "${TIMEOUT_EXE}"
  command -v awk >/dev/null 2>&1 || die "required executable not found: awk"
  command -v grep >/dev/null 2>&1 || die "required executable not found: grep"
  validate_checkpoint
  validate_bridge_contract

  [[ "${MOTIONFORGE_DEFAULT_DEVICE}" == "cpu" ]] || die \
    "MOTIONFORGE_DEFAULT_DEVICE must be cpu for rigid-body HRI tasks"
  [[ "${MOTIONFORGE_HRI006_DEVICE}" == "cuda:0" ]] || die \
    "MOTIONFORGE_HRI006_DEVICE must be cuda:0 because particle cloth requires GPU PhysX"

  ((${#TASK_IDS[@]} > 0)) || die "GROOT_EVAL_TASKS must select at least one task"
  for task_id in "${TASK_IDS[@]}"; do
    [[ "${task_id}" =~ ^hri_00[0-9]$ ]] || die "invalid HRI task id: ${task_id}"
    benchmark_config="${BENCHMARK_DIR}/${task_id}_rgb_gr00t_zmq.yaml"
    require_file "${benchmark_config}"
    task_max_steps="$(benchmark_max_steps "${benchmark_config}")"
    require_uint_at_least "${task_id} max_steps" "${task_max_steps}" 1
  done

  require_uint_at_least "GROOT_EVAL_START_SEED" "${START_SEED}" 0
  require_uint_at_least "GROOT_EVAL_NUM_TRIALS" "${NUM_TRIALS}" 1
  require_uint_at_least "GROOT_EVAL_ATTEMPTS_PER_WORKER" "${ATTEMPTS_PER_WORKER}" 1
  require_uint_at_least "GROOT_EVAL_HRI006_ATTEMPTS_PER_WORKER" "${HRI006_ATTEMPTS_PER_WORKER}" 1
  [[ "${HRI006_ATTEMPTS_PER_WORKER}" == "1" ]] || die \
    "GROOT_EVAL_HRI006_ATTEMPTS_PER_WORKER must be 1 because particle cloth is not resettable"
  require_uint_at_least "GROOT_EVAL_WORKER_START_TIMEOUT_S" "${WORKER_START_TIMEOUT_S}" 1
  require_uint_at_least "GROOT_EVAL_ATTEMPT_TIMEOUT_S" "${ATTEMPT_TIMEOUT_S}" 1
  require_uint_at_least "GROOT_EVAL_TASK_TIMEOUT_S" "${TASK_TIMEOUT_S}" 1
  require_uint_at_least "GROOT_EVAL_BETWEEN_TASKS_S" "${BETWEEN_TASKS_S}" 0
  require_uint_at_least "GROOT_EVAL_OBS_PORT" "${OBS_PORT}" 1
  require_uint_at_least "GROOT_EVAL_ACT_PORT" "${GROOT_PORT}" 1
  require_uint_at_least "GROOT_PRINT_EVERY" "${GROOT_PRINT_EVERY}" 0
  require_uint_at_least "GROOT_EVAL_VIDEO_WIDTH" "${VIDEO_WIDTH}" 2
  require_uint_at_least "GROOT_EVAL_VIDEO_HEIGHT" "${VIDEO_HEIGHT}" 2
  require_uint_at_least "GROOT_EVAL_VIDEO_STRIDE" "${VIDEO_STRIDE}" 1

  ((10#${OBS_PORT} <= 65535)) || die "GROOT_EVAL_OBS_PORT must be <= 65535"
  ((10#${GROOT_PORT} <= 65535)) || die "GROOT_EVAL_ACT_PORT must be <= 65535"
  [[ "${OBS_PORT}" != "${GROOT_PORT}" ]] || die "observation and action ports must differ"
  [[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
  [[ "${VIDEO_OUTCOME_SUFFIX}" == "1" ]] \
    || die "GROOT_EVAL_VIDEO_OUTCOME_SUFFIX must be 1 for result validation"
  [[ "${GROOT_EVAL_CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]] \
    || die "GROOT_EVAL_CUDA_VISIBLE_DEVICES must select one CUDA device index"
  [[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "GROOT_EVAL_RUN_ID may contain only letters, numbers, dot, underscore, and hyphen"
}

terminate_process() {
  local pid="$1"
  local attempts=0
  local process_state=""
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    kill -TERM -- "${pid}" 2>/dev/null || true
    while ((attempts < 150)); do
      process_state="$(awk '{ print $3 }' "/proc/${pid}/stat" 2>/dev/null || true)"
      if [[ -z "${process_state}" || "${process_state}" == "Z" || "${process_state}" == "X" ]]; then
        break
      fi
      sleep 0.2
      ((attempts += 1))
    done
    process_state="$(awk '{ print $3 }' "/proc/${pid}/stat" 2>/dev/null || true)"
    if [[ -n "${process_state}" && "${process_state}" != "Z" && "${process_state}" != "X" ]]; then
      log "process group did not stop after 30s; sending KILL pid=${pid}"
      kill -KILL -- "-${pid}" 2>/dev/null || true
      kill -KILL -- "${pid}" 2>/dev/null || true
    fi
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
  local task_id="$5"
  local physics_device=""
  local worker_attempts=""

  physics_device="$(task_physics_device "${task_id}")"
  worker_attempts="$(task_attempts_per_worker "${task_id}")"
  SERVER_COMMAND=(
    "${TIMEOUT_EXE}"
    --signal=TERM
    --kill-after=30s
    "${TASK_TIMEOUT_S}s"
    "${MOTIONFORGE_PYTHON}"
    "${TRIALS_SERVER}"
    --benchmark_config
    "${benchmark_config}"
    --seed
    "${START_SEED}"
    --num_trials
    "${NUM_TRIALS}"
    --attempts_per_worker
    "${worker_attempts}"
    --worker_start_timeout_s
    "${WORKER_START_TIMEOUT_S}"
    --attempt_timeout_s
    "${ATTEMPT_TIMEOUT_S}"
    --max_steps
    "${task_max_steps}"
    --initial_position_mode
    fixed
    --visual_asset_variance_scope
    fixed
    --device
    "${physics_device}"
    --obs_port
    "${OBS_PORT}"
    --act_port
    "${GROOT_PORT}"
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
    "${GROOT_PYTHON}"
    "${BRIDGE_CLIENT}"
    --motionforge-obs-port
    "${OBS_PORT}"
    --motionforge-act-port
    "${GROOT_PORT}"
    --groot-model-path
    "${GROOT_MODEL_PATH}"
    --groot-embodiment-tag
    "${GROOT_EMBODIMENT_TAG}"
    --groot-device
    "${GROOT_DEVICE}"
    --groot-accel-mode
    "${GROOT_ACCEL_MODE}"
    --groot-observation-format
    flat
    --num-episodes
    "${NUM_TRIALS}"
    --print-every
    "${GROOT_PRINT_EVERY}"
    --no-groot-strict
    --use-sim-policy-wrapper
  )

  if [[ "${GROOT_ACCEL_MODE}" == trt_* ]]; then
    BRIDGE_COMMAND+=(--groot-trt-engine-path "${GROOT_TRT_ENGINE_PATH}")
  fi
}

print_command() {
  local working_directory="$1"
  shift
  printf '  (cd %q && ' "${working_directory}"
  printf '%q ' "$@"
  printf ')\n'
}

print_bridge_command() {
  printf '  (cd %q && CUDA_VISIBLE_DEVICES=%q GR00T_BACKBONE_MODEL_PATH=%q LD_LIBRARY_PATH=%q PYTHONOPTIMIZE= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=%q ' \
    "${GROOT_ROOT}" "${GROOT_EVAL_CUDA_VISIBLE_DEVICES}" "${GROOT_BACKBONE_MODEL_PATH}" \
    "${GROOT_FFMPEG_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "${GROOT_ROOT}:${BRIDGE_PYTHONPATH}"
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
  summary_line="$(grep -F '[MOTIONFORGE-BENCH] trials_summary ' "${server_log}" | tail -n 1 || true)"
  [[ -n "${summary_line}" ]] || return 1
  [[ "${summary_line}" =~ ${pattern} ]] || return 1
  LAST_TRIALS="${BASH_REMATCH[1]}"
  LAST_SUCCESSES="${BASH_REMATCH[2]}"
  LAST_FAILURES="${BASH_REMATCH[3]}"
  LAST_SUCCESS_RATE="${BASH_REMATCH[4]}"
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
  local video_stem=""
  local candidate_path=""
  local retry_paths=()
  local scored_paths=()

  for ((trial_number = 1; trial_number <= 10#${NUM_TRIALS}; trial_number++)); do
    if ((10#${NUM_TRIALS} == 1)); then
      video_stem="${video_dir}/${task_id}_rollout"
    else
      printf -v video_stem '%s/%s_rollout_trial_%03d' \
        "${video_dir}" "${task_id}" "${trial_number}"
    fi
    scored_paths=()
    for candidate_path in \
      "${video_stem}_success.mp4" \
      "${video_stem}_failure.mp4"; do
      [[ -e "${candidate_path}" ]] && scored_paths+=("${candidate_path}")
    done
    shopt -s nullglob
    retry_paths=(
      "${video_stem}"_retry_*_success.mp4
      "${video_stem}"_retry_*_failure.mp4
    )
    shopt -u nullglob
    scored_paths+=("${retry_paths[@]}")
    ((${#scored_paths[@]} == 1)) || return 1
    [[ -s "${scored_paths[0]}" ]] || return 1
  done
}

append_task_result() {
  local task_id="$1"
  local status="$2"
  local benchmark_config="$3"
  local task_max_steps="$4"
  local physics_device="$5"
  local worker_attempts="$6"
  local video_dir="$7"
  local reason="$8"
  local video_count=""
  video_count="$(count_videos "${video_dir}")"
  {
    printf '\n[%s]\n' "${task_id}"
    printf 'status=%s\n' "${status}"
    printf 'benchmark=%s\n' "${benchmark_config}"
    printf 'max_steps=%s\n' "${task_max_steps}"
    printf 'physics_device=%s\n' "${physics_device}"
    printf 'attempts_per_worker=%s\n' "${worker_attempts}"
    printf 'trials=%s\n' "${LAST_TRIALS:-N/A}"
    printf 'successes=%s\n' "${LAST_SUCCESSES:-N/A}"
    printf 'failures=%s\n' "${LAST_FAILURES:-N/A}"
    printf 'success_rate=%s\n' "${LAST_SUCCESS_RATE:-N/A}"
    printf 'video_dir=%s\n' "${video_dir}"
    printf 'video_count=%s\n' "${video_count}"
    printf 'reason=%s\n' "${reason}"
  } >>"${SUMMARY_FILE}"
}

run_task() {
  local task_id="$1"
  local benchmark_config="${BENCHMARK_DIR}/${task_id}_rgb_gr00t_zmq.yaml"
  local task_max_steps=""
  local physics_device=""
  local worker_attempts=""
  local task_result_dir="${RESULT_DIR}/${task_id}"
  local server_log="${task_result_dir}/server.log"
  local client_log="${task_result_dir}/client.log"
  local video_dir="${task_result_dir}/videos"
  local process_status=0

  task_max_steps="$(benchmark_max_steps "${benchmark_config}")"
  physics_device="$(task_physics_device "${task_id}")"
  worker_attempts="$(task_attempts_per_worker "${task_id}")"
  LAST_TRIALS=0
  LAST_SUCCESSES=0
  LAST_FAILURES=0
  LAST_SUCCESS_RATE=""
  mkdir -p "${video_dir}" || return 1
  build_commands "${benchmark_config}" "${task_max_steps}" "${video_dir}" "${task_id}_rollout.mp4" "${task_id}"
  log "starting task=${task_id} trials=${NUM_TRIALS} attempts_per_worker=${worker_attempts} max_steps=${task_max_steps} physics_device=${physics_device} seeds=${START_SEED}-$((START_SEED + NUM_TRIALS - 1))"
  log "server_log=${server_log}"
  log "client_log=${client_log}"

  (
    cd -- "${MOTIONFORGE_ROOT}"
    export CUDA_VISIBLE_DEVICES="${GROOT_EVAL_CUDA_VISIBLE_DEVICES}"
    export OMNI_KIT_ACCEPT_EULA="YES"
    export MOTIONFORGE_LIGHTING_MODE=fixed
    unset MOTIONFORGE_LIGHTING_PROFILE MOTIONFORGE_LIGHTING_BRIGHTNESS
    unset MOTIONFORGE_LIGHTING_DIRECTION MOTIONFORGE_LIGHTING_COLOR MOTIONFORGE_LIGHTING_SEED
    exec "${SERVER_COMMAND[@]}"
  ) >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  (
    cd -- "${GROOT_ROOT}"
    export CUDA_VISIBLE_DEVICES="${GROOT_EVAL_CUDA_VISIBLE_DEVICES}"
    export PYTHONOPTIMIZE=""
    export PYTHONDONTWRITEBYTECODE=1
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export PYTHONPATH="${GROOT_ROOT}:${BRIDGE_PYTHONPATH}"
    export GR00T_BACKBONE_MODEL_PATH="${GROOT_BACKBONE_MODEL_PATH}"
    export LD_LIBRARY_PATH="${GROOT_FFMPEG_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    exec "${BRIDGE_COMMAND[@]}"
  ) >"${client_log}" 2>&1 &
  BRIDGE_PID="$!"

  if wait_for_server_and_bridge; then
    process_status=0
  else
    process_status="$?"
  fi
  if ((process_status != 0)); then
    append_task_result "${task_id}" failed "${benchmark_config}" "${task_max_steps}" \
      "${physics_device}" "${worker_attempts}" "${video_dir}" \
      "server/client process exit status ${process_status}"
    return "${process_status}"
  fi
  if ! parse_task_summary "${server_log}"; then
    append_task_result "${task_id}" failed "${benchmark_config}" "${task_max_steps}" \
      "${physics_device}" "${worker_attempts}" "${video_dir}" \
      "server log has no valid ${NUM_TRIALS}-trial summary"
    return 1
  fi
  if ! validate_video_outputs "${task_id}" "${video_dir}"; then
    append_task_result "${task_id}" failed "${benchmark_config}" "${task_max_steps}" \
      "${physics_device}" "${worker_attempts}" "${video_dir}" \
      "expected ${NUM_TRIALS} non-empty rollout videos with unambiguous outcome suffixes"
    return 1
  fi
  append_task_result "${task_id}" completed "${benchmark_config}" "${task_max_steps}" \
    "${physics_device}" "${worker_attempts}" "${video_dir}" none
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
  log "validated configuration; no server/client or result directory will be created"
  log "tasks=${#TASK_IDS[@]} trials_per_task=${NUM_TRIALS} expected_trials=$((${#TASK_IDS[@]} * NUM_TRIALS))"
  log "default_attempts_per_worker=${ATTEMPTS_PER_WORKER} hri006_attempts_per_worker=${HRI006_ATTEMPTS_PER_WORKER}"
  log "worker_start_timeout_s=${WORKER_START_TIMEOUT_S} attempt_timeout_s=${ATTEMPT_TIMEOUT_S} task_timeout_s=${TASK_TIMEOUT_S}"
  log "model=${GROOT_MODEL_PATH} result_dir=${RESULT_DIR}"
  for task_id in "${TASK_IDS[@]}"; do
    benchmark_config="${BENCHMARK_DIR}/${task_id}_rgb_gr00t_zmq.yaml"
    task_max_steps="$(benchmark_max_steps "${benchmark_config}")"
    physics_device="$(task_physics_device "${task_id}")"
    worker_attempts="$(task_attempts_per_worker "${task_id}")"
    video_dir="${RESULT_DIR}/${task_id}/videos"
    build_commands "${benchmark_config}" "${task_max_steps}" "${video_dir}" "${task_id}_rollout.mp4" "${task_id}"
    log "dry-run task=${task_id} max_steps=${task_max_steps} physics_device=${physics_device} attempts_per_worker=${worker_attempts} benchmark=${benchmark_config}"
    print_command "${MOTIONFORGE_ROOT}" "${SERVER_COMMAND[@]}"
    print_bridge_command
  done
  exit 0
fi

if [[ -e "${RESULT_DIR}" ]]; then
  die "result directory already exists; choose another GROOT_EVAL_RUN_ID: ${RESULT_DIR}"
fi
mkdir -p "${RESULT_DIR}"
{
  printf 'GROOT HRI000-HRI009 MotionForge evaluation\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'model=%s\n' "${GROOT_MODEL_PATH}"
  printf 'accel_mode=%s\n' "${GROOT_ACCEL_MODE}"
  printf 'trt_engine_path=%s\n' "${GROOT_TRT_ENGINE_PATH}"
  printf 'groot_device=%s\n' "${GROOT_DEVICE}"
  printf 'default_physics_device=%s\n' "${MOTIONFORGE_DEFAULT_DEVICE}"
  printf 'hri006_physics_device=%s\n' "${MOTIONFORGE_HRI006_DEVICE}"
  printf 'cuda_visible_devices=%s\n' "${GROOT_EVAL_CUDA_VISIBLE_DEVICES}"
  printf 'tasks=%s\n' "${#TASK_IDS[@]}"
  printf 'task_ids=%s\n' "${TASK_IDS[*]}"
  printf 'trials_per_task=%s\n' "${NUM_TRIALS}"
  printf 'default_attempts_per_worker=%s\n' "${ATTEMPTS_PER_WORKER}"
  printf 'hri006_attempts_per_worker=%s\n' "${HRI006_ATTEMPTS_PER_WORKER}"
  printf 'worker_start_timeout_s=%s\n' "${WORKER_START_TIMEOUT_S}"
  printf 'attempt_timeout_s=%s\n' "${ATTEMPT_TIMEOUT_S}"
  printf 'task_timeout_s=%s\n' "${TASK_TIMEOUT_S}"
  printf 'max_steps_source=benchmark_config\n'
  printf 'timing_source=benchmark_config\n'
  printf 'seed_start=%s\n' "${START_SEED}"
  printf 'seed_end=%s\n' "$((START_SEED + NUM_TRIALS - 1))"
  printf 'initial_position_mode=fixed\n'
  printf 'visual_asset_variance_scope=fixed\n'
  printf 'n_obs_steps=1\n'
  printf 'model_action_horizon=16\n'
  printf 'wire_action_horizon=server_required_16\n'
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
