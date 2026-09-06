#!/usr/bin/env bash
# DRY_RUN=1 validates all twelve commands without submitting or creating results.
set -Eeuo pipefail

die() { printf '[CM000-LIGHT-SMOKE] ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[CM000-LIGHT-SMOKE] %s\n' "$*"; }

WORKER="${CM_LIGHT_SMOKE_WORKER-0}"
DRY_RUN="${DRY_RUN-0}"
[[ "${WORKER}" == 0 || "${WORKER}" == 1 ]] || die "CM_LIGHT_SMOKE_WORKER must be 0 or 1"
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || die "DRY_RUN must be 0 or 1"
if [[ "${WORKER}" == 1 ]]; then
  [[ -n "${SLURM_JOB_ID:-}" && -n "${CM_LIGHT_SMOKE_BASELINE_ROOT:-}" ]] \
    || die "worker mode requires a Slurm allocation and baseline root"
fi
BASELINE_ROOT="$(cd -- "${CM_LIGHT_SMOKE_BASELINE_ROOT:-$(dirname -- "${BASH_SOURCE[0]}")}" && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${BASELINE_ROOT}/.." && pwd -P)"
SCRIPT_PATH="${BASELINE_ROOT}/submit_cm000_lighting_ood_smoke.sh"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"
MOTIONFORGE_PYTHON="${MOTIONFORGE_PYTHON:-${WORKSPACE_ROOT}/isaacsim/python.sh}"
LEROBOT_PYTHON="${BASELINE_ROOT}/.conda/lerobot-inference/bin/python"
HF_CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"
RUN_ID="${CM_LIGHT_SMOKE_RUN_ID:-cm000_light_ood_$(date -u +%Y%m%d_%H%M%S)}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ && "${RUN_ID}" != . && "${RUN_ID}" != .. ]] \
  || die "CM_LIGHT_SMOKE_RUN_ID must be a non-traversing directory name"
OUTPUT_ROOT="${BASELINE_ROOT}/results/smoke/cm000_lighting_ood"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"

MODELS=(dp act xvla groot pi05 smolvla)
MODEL_DIRS=(DP ACT X-VLA Isaac-GR00T/examples/MotionForge pi05 smolvla)
EVAL_PREFIXES=(DIFFUSION ACT XVLA CM PI05 SMOLVLA)
CLIENT_PREFIXES=(DIFFUSION ACT XVLA GROOT PI05 SMOLVLA)
CHECKPOINTS=(DiffusionPolicy-CM-80000-3views ACT-CM-80000 X-VLA-CM-80000 gr00t1.7-CM-80000 pi05-CM-80000 SmolVLA-CM-80000)
SEEDS=(0 10)

export PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Keep login-node preflight light; Slurm workers receive four CPU cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

input_hash() {
  local dir
  {
    sha256sum "${SCRIPT_PATH}" \
      "${MOTIONFORGE_ROOT}/configs/benchmark_ood/lighting.yaml" \
      "${MOTIONFORGE_ROOT}/configs/benchmarks/timing_profiles.yaml" \
      "${MOTIONFORGE_ROOT}/configs/benchmarks/circular_motion/cm_000_rgb_gr00t_zmq.yaml" \
      "${MOTIONFORGE_ROOT}/configs/scenarios/circular_motion/cm_000_hang_mug_on_rotating_mug_tree_rgb_franka.yaml"
    for dir in "${MODEL_DIRS[@]}"; do
      sha256sum "${BASELINE_ROOT}/${dir}/run_cm000_cm009_evaluation.sh"
    done
  } | sha256sum | awk '{print $1}'
}

run_trial() {
  local index="$1" seed="$2" dry_run="$3"
  local prefix="${EVAL_PREFIXES[index]}" client="${CLIENT_PREFIXES[index]}"
  local model_dir="${BASELINE_ROOT}/${MODEL_DIRS[index]}"
  local python="${LEROBOT_PYTHON}"
  local obs_port=$((48000 + index * 10)) action_port=$((48002 + index * 10))
  local -a extra=()
  if [[ "${MODELS[index]}" == groot ]]; then
    python="${BASELINE_ROOT}/Isaac-GR00T/.venv/bin/python"
    extra+=(
      GROOT_ACCEL_MODE=trt_full_pipeline
      "GROOT_BACKBONE_MODEL_PATH=${WORKSPACE_ROOT}/ckpts/nvidia/Cosmos-Reason2-2B"
      "GROOT_TRT_ENGINE_PATH=${BASELINE_ROOT}/Isaac-GR00T/gr00t_trt_deployments/gr00t_trt_deployment_gr00t1.7-CM-80000/engines"
      "GROOT_FFMPEG_PREFIX=${WORKSPACE_ROOT}/ICRA2027-baseline/.conda/lerobot-baselines"
    )
  fi
  env -u MOTIONFORGE_ROOT_SEED -u MOTIONFORGE_TARGET_OBJECT_SEED \
    -u MOTIONFORGE_VISUAL_ASSET_SEED -u MOTIONFORGE_BACKGROUND_PROFILE \
    -u MOTIONFORGE_BACKGROUND_ASSET_ID \
    "DRY_RUN=${dry_run}" \
    "MOTIONFORGE_ROOT=${MOTIONFORGE_ROOT}" "MOTIONFORGE_PYTHON=${MOTIONFORGE_PYTHON}" \
    "LEROBOT_ROOT=${BASELINE_ROOT}/lerobot" \
    "PYTHONPATH=${MOTIONFORGE_ROOT}/source/motionforge" \
    "HF_HOME=${HF_CACHE_ROOT}" "XVLA_HF_HOME=${HF_CACHE_ROOT}" "SMOLVLA_HF_HOME=${HF_CACHE_ROOT}" \
    MOTIONFORGE_DEVICE=cpu MOTIONFORGE_ATTEMPTS_PER_WORKER=1 \
    MOTIONFORGE_TARGET_OBJECT_MODE=fixed MOTIONFORGE_VISUAL_ASSET_VARIANCE_SCOPE=fixed \
    DIFFUSION_NUM_INFERENCE_STEPS=20 PI05_TOKENIZER_PATH=google/paligemma-3b-pt-224 \
    "${client}_SCRIPT_DIR=${model_dir}" "${client}_PYTHON=${python}" \
    "${client}_MODEL_PATH=${WORKSPACE_ROOT}/ckpts/${CHECKPOINTS[index]}" "${client}_DEVICE=cuda:0" \
    "${prefix}_EVAL_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    "${prefix}_EVAL_TASKS=cm_000" "${prefix}_EVAL_NUM_TRIALS=1" \
    "${prefix}_EVAL_START_SEED=${seed}" "${prefix}_EVAL_OOD_LIGHTING=1" \
    "${client}_EVAL_ATTEMPTS_PER_WORKER=1" \
    "${prefix}_EVAL_USE_BENCHMARK_MAX_STEPS=1" "${prefix}_EVAL_MOTION_LEVEL=" \
    "${prefix}_EVAL_INITIAL_POSITION_MODE=fixed" "${prefix}_EVAL_VIDEO_OUTCOME_SUFFIX=1" \
    "${prefix}_EVAL_TASK_TIMEOUT_S=1800" "${prefix}_EVAL_BETWEEN_TASKS_S=0" \
    "${prefix}_EVAL_OBS_PORT=${obs_port}" "${prefix}_EVAL_ACT_PORT=${action_port}" \
    "DIFFUSION_EVAL_DIFFUSION_PORT=${action_port}" \
    "${prefix}_EVAL_OUTPUT_ROOT=${RUN_DIR}/${MODELS[index]}" "${prefix}_EVAL_RUN_ID=seed_${seed}" \
    "${extra[@]}" bash "${model_dir}/run_cm000_cm009_evaluation.sh"
}

preflight() {
  local index="$1" seed output
  for seed in "${SEEDS[@]}"; do
    log "preflight model=${MODELS[index]} seed=${seed}"
    if ! output="$(run_trial "${index}" "${seed}" 1 2>&1)"; then
      printf '%s\n' "${output}" >&2
      die "preflight failed: ${MODELS[index]} seed ${seed}"
    fi
    [[ "${output}" == *--ood_lighting* && "${output}" == *cm_000_rgb_gr00t_zmq.yaml* ]] \
      || die "dry-run command is missing lighting OOD or CM000"
    log "preflight passed model=${MODELS[index]} seed=${seed}"
  done
}

run_model() (
  local index="$1" seed trial_status status=0
  preflight "${index}"
  [[ -d "${RUN_DIR}" ]] || die "submission directory is missing"
  MODEL_RESULT_DIR="${RUN_DIR}/${MODELS[index]}"
  mkdir -- "${MODEL_RESULT_DIR}" || die "model results already exist; refusing to overwrite"
  printf 'seed\texit_code\tsummary\n' >"${MODEL_RESULT_DIR}/run_status.tsv" \
    || die "cannot initialize model status file"
  for seed in "${SEEDS[@]}"; do
    log "starting model=${MODELS[index]} seed=${seed} job=${SLURM_JOB_ID}"
    if run_trial "${index}" "${seed}" 0 >"${MODEL_RESULT_DIR}/seed_${seed}.log" 2>&1; then
      trial_status=0
    else
      trial_status=$?
      status=1
    fi
    printf '%s\t%s\t%s\n' "${seed}" "${trial_status}" "seed_${seed}/success_rates.txt" \
      >>"${MODEL_RESULT_DIR}/run_status.tsv" || die "cannot record trial status"
    log "finished model=${MODELS[index]} seed=${seed} exit_code=${trial_status}"
  done
  exit "${status}"
)

if [[ "${WORKER}" == 1 ]]; then
  [[ -n "${CM_LIGHT_SMOKE_INPUT_SHA256:-}" ]] || die "missing submission fingerprint"
  [[ "$(input_hash)" == "${CM_LIGHT_SMOKE_INPUT_SHA256}" ]] || die "runner or configuration changed since submission"
  status=0
  # One job runs all six models serially. Model failures remain isolated.
  for index in "${!MODELS[@]}"; do
    if run_model "${index}"; then
      log "model completed: ${MODELS[index]}"
    else
      status=1
      log "model failed: ${MODELS[index]}; continuing with the remaining models"
    fi
  done
  exit "${status}"
fi

[[ -z "${SLURM_JOB_ID:-}" ]] || die "submit from a login shell, not an existing allocation"
[[ ! -e "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] || die "run directory already exists: ${RUN_DIR}"
for index in "${!MODELS[@]}"; do
  preflight "${index}"
done
INPUT_SHA256="$(input_hash)"
if [[ "${DRY_RUN}" == 1 ]]; then
  log "dry-run passed: six models x CM000 x seeds 0,10 = 12 trials; no jobs or results created"
  exit 0
fi

command -v sbatch >/dev/null || die "sbatch is unavailable"
mkdir -p -- "${OUTPUT_ROOT}"
mkdir -- "${RUN_DIR}" || die "run directory was created concurrently"
{
  printf 'task=cm_000\nseeds=0,10\nexpected_trials=12\nood_lighting=1\nphysics=cpu\n'
  printf 'app_reuse_across_seeds=false\nmodels_per_worker=6\nmax_concurrent_models=1\ninput_sha256=%s\n' "${INPUT_SHA256}"
  printf 'baseline_head=%s\nmotionforge_head=%s\n' \
    "$(git -C "${BASELINE_ROOT}" rev-parse HEAD)" "$(git -C "${MOTIONFORGE_ROOT}" rev-parse HEAD)"
  for index in "${!MODELS[@]}"; do
    printf '%s_checkpoint=%s\n' "${MODELS[index]}" "${WORKSPACE_ROOT}/ckpts/${CHECKPOINTS[index]}"
  done
} >"${RUN_DIR}/provenance.txt"

submission="$(sbatch --parsable --job-name=cm000_light_ood_smoke \
  --partition=cluster02 --gres=gpu:rtx5090:1 --cpus-per-task=4 --mem=48G --time=02:00:00 \
  "--chdir=${BASELINE_ROOT}" \
  "--output=${RUN_DIR}/slurm_%j.out" "--error=${RUN_DIR}/slurm_%j.err" \
  "--export=ALL,CM_LIGHT_SMOKE_WORKER=1,CM_LIGHT_SMOKE_BASELINE_ROOT=${BASELINE_ROOT},CM_LIGHT_SMOKE_RUN_ID=${RUN_ID},CM_LIGHT_SMOKE_INPUT_SHA256=${INPUT_SHA256},MOTIONFORGE_ROOT=${MOTIONFORGE_ROOT},MOTIONFORGE_PYTHON=${MOTIONFORGE_PYTHON},DRY_RUN=0" \
  "${SCRIPT_PATH}")" || die "Slurm submission failed; inspect the run directory before retrying"
printf '%s\n' "${submission}" >"${RUN_DIR}/submission.txt"
job_id="${submission%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || die "unexpected Slurm response; check squeue before any retry"
log "submitted job=${job_id} models=6 max_concurrent=1 results=${RUN_DIR}"
for index in "${!MODELS[@]}"; do
  log "job=${job_id} model=${MODELS[index]} seeds=0,10"
done
