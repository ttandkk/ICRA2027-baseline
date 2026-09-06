#!/usr/bin/env bash
#SBATCH --job-name=act_lighting_ood_smoke
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# Submit with ACT_SCRIPT_DIR set to this directory; DRY_RUN=1 writes no results.
set -Eeuo pipefail

die() { printf '[ACT-LIGHTING-OOD] ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[ACT-LIGHTING-OOD] %s\n' "$*"; }

if [[ -n "${SLURM_JOB_ID:-}" && -z "${ACT_SCRIPT_DIR:-}" ]]; then
  die "set ACT_SCRIPT_DIR when submitting this script through Slurm"
fi
SCRIPT_DIR="$(cd -- "${ACT_SCRIPT_DIR:-$(dirname -- "${BASH_SOURCE[0]}")}" && pwd -P)"
BASELINE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${BASELINE_ROOT}/.." && pwd -P)"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"
ACT_PYTHON="${ACT_PYTHON:-${BASELINE_ROOT}/.conda/lerobot-inference/bin/python}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${BASELINE_ROOT}/lerobot}"
MOTIONFORGE_PYTHON="${MOTIONFORGE_PYTHON:-${WORKSPACE_ROOT}/isaacsim/python.sh}"
DRY_RUN="${DRY_RUN-0}"
RUN_ID="${ACT_OOD_SMOKE_RUN_ID:-act_lighting_ood_$(date -u +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${ACT_OOD_SMOKE_OUTPUT_ROOT:-${SCRIPT_DIR}/output/lighting_ood_smoke}"
RUN_DIR="${RESULT_ROOT}/${RUN_ID}"

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ && "${RUN_ID}" != "." && "${RUN_ID}" != ".." ]] \
  || die "ACT_OOD_SMOKE_RUN_ID must be a non-traversing directory name"
if [[ "${DRY_RUN}" == "0" && -z "${SLURM_JOB_ID:-}" ]]; then
  die "run the smoke inside a Slurm GPU allocation, or use DRY_RUN=1"
fi

FAMILIES=(fc ht hri)
SEEDS=(0 10)
RUNNERS=(run_fc001_fc009_evaluation.sh run_ht000_ht009_evaluation.sh run_hri000_hri009_evaluation.sh)
MODELS=(
  "${ACT_FC_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/ACT-FC-80000}"
  "${ACT_HT_MODEL_PATH:-${WORKSPACE_ROOT}/ICRA2027-baseline/ACT/train_outputs/act_lerobot_ht_129864/checkpoints/080000/pretrained_model}"
  "${ACT_HRI_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/ACT-HRI-80000}"
)

export PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${MOTIONFORGE_ROOT}/source/motionforge${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

run_family() {
  local index="$1" seed="$2" dry_run="$3"
  # Separate single-trial runs preserve the existing contiguous-seed server API.
  # All ten tasks are selected by the family runner, regardless of inherited filters.
  env -u ACT_EVAL_TASKS -u ACT_EVAL_MOTION_LEVEL \
    "DRY_RUN=${dry_run}" \
    "ACT_SCRIPT_DIR=${SCRIPT_DIR}" \
    "MOTIONFORGE_ROOT=${MOTIONFORGE_ROOT}" \
    "MOTIONFORGE_PYTHON=${MOTIONFORGE_PYTHON}" \
    "ACT_PYTHON=${ACT_PYTHON}" "LEROBOT_ROOT=${LEROBOT_ROOT}" \
    "ACT_MODEL_PATH=${MODELS[index]}" \
    "ACT_EVAL_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    ACT_DEVICE=cuda:0 MOTIONFORGE_DEVICE=cpu \
    MOTIONFORGE_DEFAULT_DEVICE=cpu MOTIONFORGE_HRI006_DEVICE=cuda:0 \
    ACT_EVAL_OOD_LIGHTING=1 ACT_EVAL_NUM_TRIALS=1 \
    "ACT_EVAL_START_SEED=${seed}" \
    ACT_EVAL_ATTEMPTS_PER_WORKER=1 ACT_EVAL_HRI006_ATTEMPTS_PER_WORKER=1 \
    ACT_EVAL_USE_BENCHMARK_MAX_STEPS=1 ACT_EVAL_INITIAL_POSITION_MODE=fixed \
    ACT_EVAL_VIDEO_OUTCOME_SUFFIX=1 ACT_EVAL_BETWEEN_TASKS_S=5 \
    ACT_EVAL_TASK_TIMEOUT_S=1800 \
    ACT_EVAL_OBS_PORT=3596 ACT_EVAL_ACT_PORT=3598 \
    "ACT_EVAL_OUTPUT_ROOT=${RUN_DIR}/${FAMILIES[index]}" \
    "ACT_EVAL_RUN_ID=seed_${seed}" \
    bash "${SCRIPT_DIR}/${RUNNERS[index]}"
}

# Validate every family/seed before starting any simulation or creating results.
for index in "${!FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_family "${index}" "${seed}" 1
  done
done
if [[ "${DRY_RUN}" == "1" ]]; then
  log "preflight passed: 30 tasks x seeds 0,10 = 60 trials; no jobs or results created"
  exit 0
fi

[[ ! -e "${RUN_DIR}" ]] || die "result directory already exists: ${RUN_DIR}"
mkdir -p -- "${RUN_DIR}"
{
  printf 'job_id=%s\nseeds=0,10\nexpected_tasks=30\nexpected_trials=60\n' "${SLURM_JOB_ID}"
  printf 'ood_lighting=1\nphysics=cpu_except_hri006_cuda:0\n'
  printf 'app_reuse_across_seeds=false\nstarted_at=%s\n' "$(date -u --iso-8601=seconds)"
  for index in "${!FAMILIES[@]}"; do
    printf '%s_model=%s\n' "${FAMILIES[index]}" "${MODELS[index]}"
  done
  sha256sum "${MOTIONFORGE_ROOT}/configs/benchmark_ood/lighting.yaml"
} >"${RUN_DIR}/provenance.txt"
printf 'family\tseed\texit_code\tsummary\n' >"${RUN_DIR}/run_status.tsv"

overall_status=0
for index in "${!FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    label="${FAMILIES[index]}_seed_${seed}"
    log "starting ${label}; log=${RUN_DIR}/${label}.log"
    if run_family "${index}" "${seed}" 0 >"${RUN_DIR}/${label}.log" 2>&1; then
      status=0
    else
      status=$?
      overall_status=1
    fi
    printf '%s\t%s\t%s\t%s\n' "${FAMILIES[index]}" "${seed}" "${status}" \
      "${FAMILIES[index]}/seed_${seed}/success_rates.txt" >>"${RUN_DIR}/run_status.tsv"
    log "finished ${label}; exit_code=${status}"
  done
done
log "finished all stages; exit_code=${overall_status}; results=${RUN_DIR}"
exit "${overall_status}"
