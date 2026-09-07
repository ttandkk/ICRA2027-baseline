#!/usr/bin/env bash
#SBATCH --job-name=cm000_speed_ood
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# DRY_RUN=1 validates six models x one persistent 50-trial speed run without submitting or creating results.
set -Eeuo pipefail

die() { printf '[CM000-SPEED-OOD] ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[CM000-SPEED-OOD] %s\n' "$*"; }

WORKER="${CM_SPEED_OOD_WORKER-0}"
DRY_RUN="${DRY_RUN-0}"
[[ "${WORKER}" == 0 || "${WORKER}" == 1 || "${WORKER}" == 2 ]] || die "CM_SPEED_OOD_WORKER must be 0, 1, or 2"
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || die "DRY_RUN must be 0 or 1"
if [[ "${WORKER}" != 0 ]]; then
  [[ -n "${SLURM_JOB_ID:-}" && -n "${CM_SPEED_OOD_BASELINE_ROOT:-}" ]] \
    || die "worker mode requires a Slurm allocation and baseline root"
fi
BASELINE_ROOT="$(cd -- "${CM_SPEED_OOD_BASELINE_ROOT:-$(dirname -- "${BASH_SOURCE[0]}")}" && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${BASELINE_ROOT}/.." && pwd -P)"
SCRIPT_PATH="${BASELINE_ROOT}/submit_cm000_lighting_ood_smoke.sh"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"
MOTIONFORGE_PYTHON="${MOTIONFORGE_PYTHON:-${WORKSPACE_ROOT}/.conda/motionforge/bin/python}"
CM000_SCENARIO="${MOTIONFORGE_ROOT}/configs/scenarios/circular_motion/cm_000_hang_mug_on_rotating_mug_tree_rgb_franka.yaml"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${WORKSPACE_ROOT}/ckpt/cm}"
REFERENCE_BASELINE_ROOT="${REFERENCE_BASELINE_ROOT:-${WORKSPACE_ROOT}/ICRA2027-baseline}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-${REFERENCE_BASELINE_ROOT}/.conda/lerobot-baselines/bin/python}"
HF_CACHE_ROOT="${CM_SPEED_OOD_HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
RUN_ID="${CM_SPEED_OOD_RUN_ID:-cm000_speed_ood_$(date -u +%Y%m%d_%H%M%S)}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ && "${RUN_ID}" != . && "${RUN_ID}" != .. ]] \
  || die "CM_SPEED_OOD_RUN_ID must be a non-traversing directory name"
OUTPUT_ROOT="${BASELINE_ROOT}/results/ood/cm000_speed"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"

MODELS=(dp act xvla groot pi05 smolvla)
MODEL_DIRS=(DP ACT X-VLA Isaac-GR00T/examples/MotionForge pi05 smolvla)
EVAL_PREFIXES=(DIFFUSION ACT XVLA CM PI05 SMOLVLA)
CLIENT_PREFIXES=(DIFFUSION ACT XVLA GROOT PI05 SMOLVLA)
CHECKPOINTS=(DiffusionPolicy-CM-80000-3views ACT-CM-80000 X-VLA-CM-80000 gr00t1.7-CM-80000 pi05-CM-80000 SmolVLA-CM-80000)

export PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Keep login-node preflight light; each of the two Slurm steps receives four CPU cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

model_checkpoint_path() {
  local index="$1"
  local path="${CHECKPOINT_ROOT}/${CHECKPOINTS[index]}"
  if [[ "${MODELS[index]}" == smolvla ]]; then
    path+=/pretrained_model
  fi
  printf "%s\n" "${path}"
}

input_hash() {
  local dir
  {
    sha256sum "${SCRIPT_PATH}" \
      "${MOTIONFORGE_ROOT}/configs/benchmarks/timing_profiles.yaml" \
      "${MOTIONFORGE_ROOT}/configs/benchmarks/circular_motion/cm_000_rgb_gr00t_zmq.yaml" \
      "${MOTIONFORGE_ROOT}/configs/scenarios/circular_motion/cm_000_hang_mug_on_rotating_mug_tree_rgb_franka.yaml"
    for dir in "${MODEL_DIRS[@]}"; do
      sha256sum "${BASELINE_ROOT}/${dir}/run_cm000_cm009_evaluation.sh"
    done
  } | sha256sum | awk '{print $1}'
}

run_trial() {
  local index="$1" dry_run="$2"
  local prefix="${EVAL_PREFIXES[index]}" client="${CLIENT_PREFIXES[index]}"
  local model_dir="${BASELINE_ROOT}/${MODEL_DIRS[index]}"
  local checkpoint_path
  checkpoint_path="$(model_checkpoint_path "${index}")"
  local python="${LEROBOT_PYTHON}"
  local obs_port=$((48000 + index * 10)) action_port=$((48002 + index * 10))
  local -a extra=()
  if [[ "${MODELS[index]}" == groot ]]; then
    python="${GROOT_PYTHON:-${REFERENCE_BASELINE_ROOT}/Isaac-GR00T/.venv/bin/python}"
    extra+=(
      GROOT_ACCEL_MODE=pytorch
      "GROOT_BACKBONE_MODEL_PATH=${REFERENCE_BASELINE_ROOT}/.cache/.hf-cache/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
      "GROOT_FFMPEG_PREFIX=${WORKSPACE_ROOT}/ICRA2027-baseline/.conda/lerobot-baselines"
    )
  fi
  /usr/bin/env -u MOTIONFORGE_ROOT_SEED -u MOTIONFORGE_TARGET_OBJECT_SEED \
    -u MOTIONFORGE_VISUAL_ASSET_SEED -u MOTIONFORGE_BACKGROUND_PROFILE \
    -u MOTIONFORGE_BACKGROUND_ASSET_ID -u MOTIONFORGE_MOTION_LEVEL \
    "DRY_RUN=${dry_run}" \
    "MOTIONFORGE_ROOT=${MOTIONFORGE_ROOT}" "MOTIONFORGE_PYTHON=${MOTIONFORGE_PYTHON}" \
    "LEROBOT_ROOT=${BASELINE_ROOT}/lerobot" \
    "PYTHONPATH=${MOTIONFORGE_ROOT}/source/motionforge" \
    "HF_HOME=${HF_CACHE_ROOT}" "XVLA_HF_HOME=${HF_CACHE_ROOT}" "SMOLVLA_HF_HOME=${HF_CACHE_ROOT}" \
    MOTIONFORGE_DEVICE=cpu MOTIONFORGE_ATTEMPTS_PER_WORKER=50 \
    MOTIONFORGE_TARGET_OBJECT_MODE=fixed MOTIONFORGE_VISUAL_ASSET_VARIANCE_SCOPE=fixed \
    DIFFUSION_NUM_INFERENCE_STEPS=20 PI05_TOKENIZER_PATH=google/paligemma-3b-pt-224 \
    "${client}_SCRIPT_DIR=${model_dir}" "${client}_PYTHON=${python}" \
    "${client}_MODEL_PATH=${checkpoint_path}" "${client}_DEVICE=cuda:0" \
    "${prefix}_EVAL_CUDA_VISIBLE_DEVICES=0" \
    "${prefix}_EVAL_TASKS=cm_000" "${prefix}_EVAL_NUM_TRIALS=50" \
    "${prefix}_EVAL_START_SEED=0" "${prefix}_EVAL_OOD_LIGHTING=0" \
    "${prefix}_EVAL_OOD_SPEED=1" "${prefix}_EVAL_MOTION_LEVEL=" \
    "${client}_EVAL_ATTEMPTS_PER_WORKER=50" \
    "${prefix}_EVAL_USE_BENCHMARK_MAX_STEPS=1" \
    "${prefix}_EVAL_INITIAL_POSITION_MODE=fixed" "${prefix}_EVAL_VIDEO_OUTCOME_SUFFIX=1" \
    "${prefix}_EVAL_TASK_TIMEOUT_S=14400" "${prefix}_EVAL_BETWEEN_TASKS_S=0" \
    "${prefix}_EVAL_OBS_PORT=${obs_port}" "${prefix}_EVAL_ACT_PORT=${action_port}" \
    "DIFFUSION_EVAL_DIFFUSION_PORT=${action_port}" \
    "${prefix}_EVAL_OUTPUT_ROOT=${RUN_DIR}/${MODELS[index]}" "${prefix}_EVAL_RUN_ID=all_seeds" \
    "${extra[@]}" bash "${model_dir}/run_cm000_cm009_evaluation.sh"
}

preflight() {
  local index="$1" output
  log "preflight model=${MODELS[index]} seeds=0-49"
  if ! output="$(run_trial "${index}" 1 2>&1)"; then
    printf "%s\n" "${output}" >&2
    die "preflight failed: ${MODELS[index]}"
  fi
  [[ "${output}" == *--ood_speed* \
    && "${output}" == *cm_000_rgb_gr00t_zmq.yaml* \
    && "${output}" != *--motion_level* \
    && "${output}" != *--ood_lighting* ]] \
    || die "dry-run command is missing --ood_speed or contains a conflicting OOD option"
  log "preflight passed model=${MODELS[index]} seeds=0-49"
}

run_model() (
  local index="$1" trial_status
  preflight "${index}"
  [[ -d "${RUN_DIR}" ]] || die "submission directory is missing"
  MODEL_RESULT_DIR="${RUN_DIR}/${MODELS[index]}"
  mkdir -- "${MODEL_RESULT_DIR}" || die "model results already exist; refusing to overwrite"
  log "starting model=${MODELS[index]} seeds=0-49 job=${SLURM_JOB_ID}"
  if run_trial "${index}" 0 >"${MODEL_RESULT_DIR}/all_seeds.log" 2>&1; then
    trial_status=0
  else
    trial_status=$?
  fi
  printf "run\texit_code\tsummary\nall_seeds\t%s\tall_seeds/success_rates.txt\n" \
    "${trial_status}" >"${MODEL_RESULT_DIR}/run_status.tsv" \
    || die "cannot record trial status"
  log "finished model=${MODELS[index]} seeds=0-49 exit_code=${trial_status}"
  exit "${trial_status}"
)

validate_step_gpu() {
  local gpu_identity
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die "Slurm step did not expose a GPU"
  [[ "${CUDA_VISIBLE_DEVICES}" != *,* ]] || die "each Slurm step must see exactly one GPU"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable in Slurm step"
  gpu_identity="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES}")"
  [[ -n "${gpu_identity}" && "${gpu_identity}" != *$'\n'* ]] || die "could not resolve exactly one GPU UUID/name"
  log "step=${CM_SPEED_OOD_STEP_INDEX} slurm_step=${SLURM_STEP_ID:-unknown} visible_device=${CUDA_VISIBLE_DEVICES} gpu=${gpu_identity}"
  log "model and Isaac renderer inherit this step-local CUDA_VISIBLE_DEVICES; both address it as cuda:0"
}

if [[ "${WORKER}" == 2 ]]; then
  [[ -n "${CM_SPEED_OOD_INPUT_SHA256:-}" ]] || die "missing submission fingerprint"
  [[ "$(input_hash)" == "${CM_SPEED_OOD_INPUT_SHA256}" ]] || die "runner or configuration changed since submission"
  [[ "${CM_SPEED_OOD_STEP_INDEX:-}" == 0 || "${CM_SPEED_OOD_STEP_INDEX:-}" == 1 ]] \
    || die "step worker requires CM_SPEED_OOD_STEP_INDEX=0 or 1"
  validate_step_gpu
  status=0
  for index in "${!MODELS[@]}"; do
    ((index % 2 == CM_SPEED_OOD_STEP_INDEX)) || continue
    if run_model "${index}"; then
      log "model completed: ${MODELS[index]}"
    else
      status=1
      log "model failed: ${MODELS[index]}; continuing with this step's remaining models"
    fi
  done
  exit "${status}"
fi

if [[ "${WORKER}" == 1 ]]; then
  [[ -n "${CM_SPEED_OOD_INPUT_SHA256:-}" ]] || die "missing submission fingerprint"
  [[ "$(input_hash)" == "${CM_SPEED_OOD_INPUT_SHA256}" ]] || die "runner or configuration changed since submission"
  command -v srun >/dev/null || die "srun is unavailable inside allocation"
  step_pids=()
  for step_index in 0 1; do
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:rtx5090:1 \
      --cpus-per-task=4 --mem=48G \
      "--chdir=${BASELINE_ROOT}" \
      "--output=${RUN_DIR}/step_${step_index}_%j_%s.out" \
      "--error=${RUN_DIR}/step_${step_index}_%j_%s.err" \
      /usr/bin/env CM_SPEED_OOD_WORKER=2 CM_SPEED_OOD_STEP_INDEX="${step_index}" \
        CM_SPEED_OOD_BASELINE_ROOT="${BASELINE_ROOT}" CM_SPEED_OOD_RUN_ID="${RUN_ID}" \
        CM_SPEED_OOD_INPUT_SHA256="${CM_SPEED_OOD_INPUT_SHA256}" \
        MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT}" MOTIONFORGE_PYTHON="${MOTIONFORGE_PYTHON}" \
        DRY_RUN=0 bash "${SCRIPT_PATH}" &
    step_pids+=("$!")
  done
  status=0
  for step_pid in "${step_pids[@]}"; do
    if ! wait "${step_pid}"; then
      status=1
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
  log "dry-run passed: six models x CM000 x 50 seeds = 300 trials; no jobs or results created"
  exit 0
fi

command -v sbatch >/dev/null || die "sbatch is unavailable"
mkdir -p -- "${OUTPUT_ROOT}"
mkdir -- "${RUN_DIR}" || die "run directory was created concurrently"
{
  printf 'task=cm_000\nseeds=0-49\nexpected_trials=300\nood_lighting=0\nspeed_ood=1\nphysics=cpu\n'
  printf 'seed_0_24_motion_level=level1\nseed_25_49_motion_level=level3_or_level1_fallback\n'
  printf 'app_reuse_across_speed_levels=true\nattempts_per_worker=50\nmotion_level_batches=1\nfailure_policy=record_cleanup_continue_next_task\nslurm_steps=2\nmodels_per_step=3\nmax_concurrent_models=2\ninput_sha256=%s\n' "${INPUT_SHA256}"
  printf 'baseline_head=%s\nmotionforge_head=%s\n' \
    "$(git -C "${BASELINE_ROOT}" rev-parse HEAD)" "$(git -C "${MOTIONFORGE_ROOT}" rev-parse HEAD)"
  for index in "${!MODELS[@]}"; do
    printf '%s_checkpoint=%s\n' "${MODELS[index]}" "$(model_checkpoint_path "${index}")"
  done
} >"${RUN_DIR}/provenance.txt"

submission="$(sbatch --parsable --job-name=cm000_speed_ood \
  --partition=cluster02 --gres=gpu:rtx5090:2 --cpus-per-task=8 --mem=96G --time=72:00:00 \
  "--chdir=${BASELINE_ROOT}" \
  "--output=${RUN_DIR}/slurm_%j.out" "--error=${RUN_DIR}/slurm_%j.err" \
  "--export=ALL,CM_SPEED_OOD_WORKER=1,CM_SPEED_OOD_BASELINE_ROOT=${BASELINE_ROOT},CM_SPEED_OOD_RUN_ID=${RUN_ID},CM_SPEED_OOD_INPUT_SHA256=${INPUT_SHA256},MOTIONFORGE_ROOT=${MOTIONFORGE_ROOT},MOTIONFORGE_PYTHON=${MOTIONFORGE_PYTHON},DRY_RUN=0" \
  "${SCRIPT_PATH}")" || die "Slurm submission failed; inspect the run directory before retrying"
printf '%s\n' "${submission}" >"${RUN_DIR}/submission.txt"
job_id="${submission%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || die "unexpected Slurm response; check squeue before any retry"
log "submitted job=${job_id} models=6 seeds=0-49 speed_ood=1 steps=2 max_concurrent=2 results=${RUN_DIR}"
for index in "${!MODELS[@]}"; do
  log "job=${job_id} model=${MODELS[index]} seeds=0-24:level1,25-49:level3-or-level1-fallback"
done
