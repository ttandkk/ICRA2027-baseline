#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GROOT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${GROOT_ROOT}/../.." && pwd -P)"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"

BRIDGE_CLIENT="${GROOT_ROOT}/examples/MotionForge/motionforge_groot_bridge_client.py"
TRIALS_SERVER="${MOTIONFORGE_ROOT}/scripts/benchmark/run_env_server_trials.py"
BENCHMARK_DIR="${MOTIONFORGE_ROOT}/configs/benchmarks/circular_motion"
RUNTIME_ASSET_DIR="${MOTIONFORGE_ROOT}/source/motionforge/motionforge/assets/runtime"

GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/MotionforgeGroup/gr00t1.7/CM-80000}"
GROOT_TRT_ENGINE_PATH="${GROOT_TRT_ENGINE_PATH:-${GROOT_ROOT}/gr00t_trt_deployments/gr00t_trt_deployment_gr00t1.7-CM-80000_square/engines}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_ROOT}/.venv/bin/python}"
GROOT_FFMPEG_PREFIX="${GROOT_FFMPEG_PREFIX:-${WORKSPACE_ROOT}/.cache/gr00t-ffmpeg}"
GROOT_ACCEL_MODE="${GROOT_ACCEL_MODE:-trt_full_pipeline}"
GROOT_DEVICE="${GROOT_DEVICE:-cuda:0}"
GROOT_EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-NEW_EMBODIMENT}"
GROOT_BACKBONE_MODEL_PATH="${GROOT_BACKBONE_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/nvidia/Cosmos-Reason2-2B}"

MOTIONFORGE_CONDA_ENV="${MOTIONFORGE_CONDA_ENV:-motionforge}"
# CM compound containers and articulated payloads require CPU PhysX to match
# the data-generation environment. Rendering and GR00T/TensorRT still use the
# GPU selected by CM_EVAL_CUDA_VISIBLE_DEVICES.
MOTIONFORGE_DEVICE="${MOTIONFORGE_DEVICE:-cpu}"
CM_EVAL_CUDA_VISIBLE_DEVICES="${CM_EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-2}}"
CONDA_EXE="${MOTIONFORGE_CONDA_EXE:-${WORKSPACE_ROOT}/miniconda3/bin/conda}"
TIMEOUT_EXE="${MOTIONFORGE_TIMEOUT_EXE:-$(command -v timeout || true)}"

START_SEED="${CM_EVAL_START_SEED:-0}"
NUM_TRIALS="${CM_EVAL_NUM_TRIALS:-10}"
CM_EVAL_CLOCK_MODE="${CM_EVAL_CLOCK_MODE:-wall_clock_strict}"
CM_EVAL_ACTION_ALIGNMENT="${CM_EVAL_ACTION_ALIGNMENT:-observation_aligned}"
OBS_PORT="${CM_EVAL_OBS_PORT:-3396}"
ACT_PORT="${CM_EVAL_ACT_PORT:-3398}"
CLIENT_WARMUP_S="${CM_EVAL_CLIENT_WARMUP_S:-5}"
TASK_TIMEOUT_S="${CM_EVAL_TASK_TIMEOUT_S:-14400}"
BETWEEN_TASKS_S="${CM_EVAL_BETWEEN_TASKS_S:-5}"

GROOT_EXECUTION_HORIZON="${GROOT_EXECUTION_HORIZON:-8}"
GROOT_ACTION_HZ="${GROOT_ACTION_HZ:-30}"
GROOT_MAX_INFERENCE_HZ="${GROOT_MAX_INFERENCE_HZ:-30}"
GROOT_PRINT_EVERY="${GROOT_PRINT_EVERY:-10}"

VIDEO_WIDTH="${CM_EVAL_VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${CM_EVAL_VIDEO_HEIGHT:-480}"
VIDEO_STRIDE="${CM_EVAL_VIDEO_STRIDE:-1}"

RESULT_ROOT="${SCRIPT_DIR}/outputs"
RUN_ID="${CM_EVAL_RUN_ID:-cm000_cm010_$(date +%Y%m%d_%H%M%S)}"
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

if [[ -n "${CM_EVAL_TASKS:-}" ]]; then
  read -r -a TASK_IDS <<<"${CM_EVAL_TASKS}"
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
  printf '[CM000-CM010-EVAL] %s\n' "$*"
}

die() {
  printf '[CM000-CM010-EVAL] ERROR: %s\n' "$*" >&2
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

resolve_backbone_model_path() {
  if [[ -n "${GROOT_BACKBONE_MODEL_PATH}" ]]; then
    require_dir "${GROOT_BACKBONE_MODEL_PATH}"
    GROOT_BACKBONE_MODEL_PATH="$(cd -- "${GROOT_BACKBONE_MODEL_PATH}" && pwd -P)"
    return
  fi

  local hub_cache=""
  if [[ -n "${HF_HUB_CACHE:-}" ]]; then
    hub_cache="${HF_HUB_CACHE}"
  elif [[ -n "${HF_HOME:-}" ]]; then
    hub_cache="${HF_HOME}/hub"
  elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
    hub_cache="${XDG_CACHE_HOME}/huggingface/hub"
  else
    [[ -n "${HOME:-}" ]] || die "HOME is unset; set GROOT_BACKBONE_MODEL_PATH explicitly"
    hub_cache="${HOME}/.cache/huggingface/hub"
  fi

  local cache_repo="${hub_cache}/models--nvidia--Cosmos-Reason2-2B"
  local main_ref="${cache_repo}/refs/main"
  require_file "${main_ref}"

  local revision=""
  revision="$(<"${main_ref}")"
  [[ "${revision}" =~ ^[0-9a-fA-F]{40}$ ]] || die "invalid Cosmos backbone revision in ${main_ref}"

  local snapshot_dir="${cache_repo}/snapshots/${revision}"
  require_dir "${snapshot_dir}"
  GROOT_BACKBONE_MODEL_PATH="$(cd -- "${snapshot_dir}" && pwd -P)"
}

benchmark_max_steps() {
  local benchmark_config="$1"
  local configured=""

  configured="$(awk '/^[[:space:]]*max_steps:[[:space:]]*[0-9]+[[:space:]]*$/ { print $2; exit }' "${benchmark_config}")"
  [[ -n "${configured}" ]] || die "runtime.max_steps not found in benchmark: ${benchmark_config}"
  printf '%s\n' "${configured}"
}

validate_configuration() {
  local task_id=""
  local benchmark_config=""
  local task_max_steps=""

  resolve_backbone_model_path
  require_dir "${MOTIONFORGE_ROOT}"
  require_file "${BRIDGE_CLIENT}"
  require_file "${TRIALS_SERVER}"
  require_dir "${GROOT_MODEL_PATH}"
  require_file "${GROOT_MODEL_PATH}/model.safetensors.index.json"
  require_file "${GROOT_BACKBONE_MODEL_PATH}/config.json"
  require_file "${GROOT_BACKBONE_MODEL_PATH}/model.safetensors"
  require_file "${GROOT_BACKBONE_MODEL_PATH}/preprocessor_config.json"
  require_file "${GROOT_BACKBONE_MODEL_PATH}/tokenizer_config.json"

  # Canonical fixed-mode CM assets plus all task-specific interaction assets.
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

  require_executable "${GROOT_PYTHON}"
  require_dir "${GROOT_FFMPEG_PREFIX}/lib"
  require_executable "${CONDA_EXE}"
  require_executable "${TIMEOUT_EXE}"
  command -v awk >/dev/null 2>&1 || die "required executable not found: awk"

  [[ "${MOTIONFORGE_DEVICE}" == "cpu" ]] || die \
    "MOTIONFORGE_DEVICE must be cpu for CM evaluation; GPU PhysX makes rotating compound/articulated payloads sink"

  ((${#TASK_IDS[@]} > 0)) || die "CM_EVAL_TASKS must select at least one task"
  for task_id in "${TASK_IDS[@]}"; do
    [[ "${task_id}" =~ ^cm_(00[0-9]|010)$ ]] || die "invalid CM task id: ${task_id}"
    benchmark_config="${BENCHMARK_DIR}/${task_id}_rgb_gr00t_zmq.yaml"
    require_file "${benchmark_config}"
    task_max_steps="$(benchmark_max_steps "${benchmark_config}")"
    require_uint_at_least "${task_id} max_steps" "${task_max_steps}" 1
  done

  case "${GROOT_ACCEL_MODE}" in
    pytorch | torch_compile)
      ;;
    trt_full_pipeline | trt_action_head | trt_dit_only)
      require_dir "${GROOT_TRT_ENGINE_PATH}"
      require_file "${GROOT_TRT_ENGINE_PATH}/export_metadata.json"
      require_file "${GROOT_TRT_ENGINE_PATH}/state_encoder.engine"
      require_file "${GROOT_TRT_ENGINE_PATH}/action_encoder.engine"
      require_file "${GROOT_TRT_ENGINE_PATH}/dit_bf16.engine"
      require_file "${GROOT_TRT_ENGINE_PATH}/action_decoder.engine"
      if [[ "${GROOT_ACCEL_MODE}" == "trt_full_pipeline" ]]; then
        require_file "${GROOT_TRT_ENGINE_PATH}/vit.engine"
        require_file "${GROOT_TRT_ENGINE_PATH}/llm_bf16.engine"
        require_file "${GROOT_TRT_ENGINE_PATH}/vl_self_attention.engine"
      fi
      ;;
    *)
      die "GROOT_ACCEL_MODE must be pytorch, torch_compile, trt_full_pipeline, trt_action_head, or trt_dit_only; got ${GROOT_ACCEL_MODE}"
      ;;
  esac

  require_uint_at_least "CM_EVAL_START_SEED" "${START_SEED}" 0
  require_uint_at_least "CM_EVAL_NUM_TRIALS" "${NUM_TRIALS}" 1
  [[ "${CM_EVAL_CLOCK_MODE}" == "wall_clock_strict" ]] || die \
    "CM_EVAL_CLOCK_MODE must be wall_clock_strict for realtime CM evaluation; got ${CM_EVAL_CLOCK_MODE}"
  [[ "${CM_EVAL_ACTION_ALIGNMENT}" == "observation_aligned" ]] || die \
    "CM_EVAL_ACTION_ALIGNMENT must be observation_aligned for realtime CM evaluation; got ${CM_EVAL_ACTION_ALIGNMENT}"
  require_uint_at_least "CM_EVAL_OBS_PORT" "${OBS_PORT}" 1
  require_uint_at_least "CM_EVAL_ACT_PORT" "${ACT_PORT}" 1
  require_uint_at_least "CM_EVAL_CLIENT_WARMUP_S" "${CLIENT_WARMUP_S}" 0
  require_uint_at_least "CM_EVAL_TASK_TIMEOUT_S" "${TASK_TIMEOUT_S}" 1
  require_uint_at_least "CM_EVAL_BETWEEN_TASKS_S" "${BETWEEN_TASKS_S}" 0
  require_uint_at_least "GROOT_EXECUTION_HORIZON" "${GROOT_EXECUTION_HORIZON}" 1
  require_uint_at_least "GROOT_ACTION_HZ" "${GROOT_ACTION_HZ}" 1
  require_uint_at_least "GROOT_MAX_INFERENCE_HZ" "${GROOT_MAX_INFERENCE_HZ}" 1
  require_uint_at_least "GROOT_PRINT_EVERY" "${GROOT_PRINT_EVERY}" 0
  require_uint_at_least "CM_EVAL_VIDEO_WIDTH" "${VIDEO_WIDTH}" 2
  require_uint_at_least "CM_EVAL_VIDEO_HEIGHT" "${VIDEO_HEIGHT}" 2
  require_uint_at_least "CM_EVAL_VIDEO_STRIDE" "${VIDEO_STRIDE}" 1

  ((10#${OBS_PORT} <= 65535)) || die "CM_EVAL_OBS_PORT must be <= 65535"
  ((10#${ACT_PORT} <= 65535)) || die "CM_EVAL_ACT_PORT must be <= 65535"
  [[ "${OBS_PORT}" != "${ACT_PORT}" ]] || die "observation and action ports must differ"
  [[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
  [[ -n "${CM_EVAL_CUDA_VISIBLE_DEVICES}" ]] || die "CM_EVAL_CUDA_VISIBLE_DEVICES must not be empty"
  [[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "CM_EVAL_RUN_ID may contain only letters, numbers, dot, underscore, and hyphen"
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
    "${ACT_PORT}"
    --client_warmup
    "${CLIENT_WARMUP_S}"
    --clock_mode
    "${CM_EVAL_CLOCK_MODE}"
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
    "${ACT_PORT}"
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
    --execution-horizon
    "${GROOT_EXECUTION_HORIZON}"
    --action-hz
    "${GROOT_ACTION_HZ}"
    --action-alignment
    "${CM_EVAL_ACTION_ALIGNMENT}"
    --max-inference-hz
    "${GROOT_MAX_INFERENCE_HZ}"
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
  printf '  (cd %q && CUDA_VISIBLE_DEVICES=%q GR00T_BACKBONE_MODEL_PATH=%q LD_LIBRARY_PATH=%q HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ' \
    "${GROOT_ROOT}" "${CM_EVAL_CUDA_VISIBLE_DEVICES}" "${GROOT_BACKBONE_MODEL_PATH}" "${GROOT_FFMPEG_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
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
  local summary_pattern='trials=([0-9]+)[[:space:]]+successes=([0-9]+)[[:space:]]+failures=([0-9]+)[[:space:]]+success_rate=([0-9]+([.][0-9]+)?)'

  LAST_TRIALS=0
  LAST_SUCCESSES=0
  LAST_FAILURES=0
  LAST_SUCCESS_RATE=""
  LAST_RAW_SUMMARY=""

  summary_line="$(grep -F '[MOTIONFORGE-BENCH] trials_summary ' "${server_log}" | tail -n 1 || true)"
  [[ -n "${summary_line}" ]] || return 1
  [[ "${summary_line}" =~ ${summary_pattern} ]] || return 1

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

  for ((trial_number = 1; trial_number <= 10#${NUM_TRIALS}; trial_number++)); do
    if ((10#${NUM_TRIALS} == 1)); then
      video_path="${video_dir}/${task_id}_rollout.mp4"
    else
      printf -v video_path '%s/%s_rollout_trial_%03d.mp4' "${video_dir}" "${task_id}" "${trial_number}"
    fi
    [[ -s "${video_path}" ]] || return 1
  done
  return 0
}

append_failed_task() {
  local task_id="$1"
  local benchmark_config="$2"
  local task_max_steps="$3"
  local video_dir="$4"
  local reason="$5"
  local video_count=""
  local trials="N/A"
  local successes="N/A"
  local failures="N/A"
  local success_rate="N/A"
  video_count="$(count_videos "${video_dir}")"
  if [[ -n "${LAST_SUCCESS_RATE}" ]]; then
    trials="${LAST_TRIALS}"
    successes="${LAST_SUCCESSES}"
    failures="${LAST_FAILURES}"
    success_rate="${LAST_SUCCESS_RATE}"
  fi
  {
    printf '\n[%s]\n' "${task_id}"
    printf 'status=failed\n'
    printf 'benchmark=%s\n' "${benchmark_config}"
    printf 'max_steps=%s\n' "${task_max_steps}"
    printf 'trials=%s\n' "${trials}"
    printf 'successes=%s\n' "${successes}"
    printf 'failures=%s\n' "${failures}"
    printf 'success_rate=%s\n' "${success_rate}"
    printf 'video_dir=%s\n' "${video_dir}"
    printf 'video_count=%s\n' "${video_count}"
    printf 'reason=%s\n' "${reason}"
  } >>"${SUMMARY_FILE}"
}

append_completed_task() {
  local task_id="$1"
  local benchmark_config="$2"
  local task_max_steps="$3"
  local video_dir="$4"
  local video_count=""
  video_count="$(count_videos "${video_dir}")"
  {
    printf '\n[%s]\n' "${task_id}"
    printf 'status=completed\n'
    printf 'benchmark=%s\n' "${benchmark_config}"
    printf 'max_steps=%s\n' "${task_max_steps}"
    printf 'trials=%s\n' "${LAST_TRIALS}"
    printf 'successes=%s\n' "${LAST_SUCCESSES}"
    printf 'failures=%s\n' "${LAST_FAILURES}"
    printf 'success_rate=%s\n' "${LAST_SUCCESS_RATE}"
    printf 'video_dir=%s\n' "${video_dir}"
    printf 'video_count=%s\n' "${video_count}"
    printf 'raw_summary=%s\n' "${LAST_RAW_SUMMARY}"
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
  local video_name="${task_id}_rollout.mp4"
  local process_status=0

  task_max_steps="$(benchmark_max_steps "${benchmark_config}")"
  LAST_TRIALS=0
  LAST_SUCCESSES=0
  LAST_FAILURES=0
  LAST_SUCCESS_RATE=""
  LAST_RAW_SUMMARY=""

  mkdir -p "${video_dir}" || return 1
  build_commands "${benchmark_config}" "${task_max_steps}" "${video_dir}" "${video_name}"

  log "starting task=${task_id} trials=${NUM_TRIALS} max_steps=${task_max_steps} seeds=${START_SEED}-$((START_SEED + NUM_TRIALS - 1))"
  log "server_log=${server_log}"
  log "client_log=${client_log}"

  (
    cd -- "${MOTIONFORGE_ROOT}"
    export CUDA_VISIBLE_DEVICES="${CM_EVAL_CUDA_VISIBLE_DEVICES}"
    export OMNI_KIT_ACCEPT_EULA="YES"
    exec "${SERVER_COMMAND[@]}"
  ) >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  (
    cd -- "${GROOT_ROOT}"
    export CUDA_VISIBLE_DEVICES="${CM_EVAL_CUDA_VISIBLE_DEVICES}"
    export GR00T_BACKBONE_MODEL_PATH="${GROOT_BACKBONE_MODEL_PATH}"
    export LD_LIBRARY_PATH="${GROOT_FFMPEG_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    exec "${BRIDGE_COMMAND[@]}"
  ) >"${client_log}" 2>&1 &
  BRIDGE_PID="$!"

  if wait_for_server_and_bridge; then
    process_status=0
  else
    process_status="$?"
  fi
  if ((process_status != 0)); then
    append_failed_task "${task_id}" "${benchmark_config}" "${task_max_steps}" "${video_dir}" "server/client process exit status ${process_status}"
    return "${process_status}"
  fi

  if ! parse_task_summary "${server_log}"; then
    append_failed_task "${task_id}" "${benchmark_config}" "${task_max_steps}" "${video_dir}" "server log has no valid ${NUM_TRIALS}-trial summary"
    return 1
  fi

  if ! validate_video_outputs "${task_id}" "${video_dir}"; then
    append_failed_task "${task_id}" "${benchmark_config}" "${task_max_steps}" "${video_dir}" "expected ${NUM_TRIALS} non-empty rollout videos"
    return 1
  fi

  append_completed_task "${task_id}" "${benchmark_config}" "${task_max_steps}" "${video_dir}"
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
  log "tasks=${#TASK_IDS[@]} trials_per_task=${NUM_TRIALS} max_steps=benchmark_config clock_mode=${CM_EVAL_CLOCK_MODE} action_alignment=${CM_EVAL_ACTION_ALIGNMENT} physics_device=${MOTIONFORGE_DEVICE} expected_trials=$((${#TASK_IDS[@]} * NUM_TRIALS))"
  log "result_dir=${RESULT_DIR}"
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
  die "result directory already exists; choose another CM_EVAL_RUN_ID: ${RESULT_DIR}"
fi
mkdir -p "${RESULT_DIR}"
{
  printf 'CM000-CM010 MotionForge evaluation\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'model=%s\n' "${GROOT_MODEL_PATH}"
  printf 'accel_mode=%s\n' "${GROOT_ACCEL_MODE}"
  printf 'trt_engine_path=%s\n' "${GROOT_TRT_ENGINE_PATH}"
  printf 'backbone_model_path=%s\n' "${GROOT_BACKBONE_MODEL_PATH}"
  printf 'cuda_visible_devices=%s\n' "${CM_EVAL_CUDA_VISIBLE_DEVICES}"
  printf 'motionforge_physics_device=%s\n' "${MOTIONFORGE_DEVICE}"
  printf 'tasks=%s\n' "${#TASK_IDS[@]}"
  printf 'task_ids=%s\n' "${TASK_IDS[*]}"
  printf 'trials_per_task=%s\n' "${NUM_TRIALS}"
  printf 'max_steps_source=benchmark_config\n'
  printf 'clock_mode=%s\n' "${CM_EVAL_CLOCK_MODE}"
  printf 'action_alignment=%s\n' "${CM_EVAL_ACTION_ALIGNMENT}"
  printf 'seed_start=%s\n' "${START_SEED}"
  printf 'seed_end=%s\n' "$((START_SEED + NUM_TRIALS - 1))"
  printf 'initial_position_mode=fixed\n'
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
