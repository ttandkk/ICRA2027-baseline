#!/bin/bash

#SBATCH --job-name=gr00t_fc003_fcs002_mixed_ft_full
#SBATCH --output=/projects/haitian003ssd/ICRA2027-baseline/Isaac-GR00T/logs/gr00t_fc003_fcs002_mixed_ft_full_%j.log
#SBATCH --error=/projects/haitian003ssd/ICRA2027-baseline/Isaac-GR00T/logs/gr00t_fc003_fcs002_mixed_ft_full_%j.err
#SBATCH --time=72:00:00
#SBATCH --gpus=pro6000:1

set -euo pipefail

# -----------------------------
# Cluster / environment
# -----------------------------
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.8.0}"
BASE_DIR="${BASE_DIR:-/projects/haitian003ssd/ICRA2027-baseline}"
GROOT_DIR="${GROOT_DIR:-${BASE_DIR}/Isaac-GR00T}"
CACHE_ROOT="${CACHE_ROOT:-${BASE_DIR}/.cache}"
JOB_ID="${SLURM_JOB_ID:-local}"
MASTER_PORT="${MASTER_PORT:-29500}"

# -----------------------------
# Data / model paths
# -----------------------------
MODEL_PATH="${MODEL_PATH:-${GROOT_DIR}/pretrained/GR00T-N1.7-3B}"
FACTORY_DATASET_PATH="${FACTORY_DATASET_PATH:-/projects/haitian003ssd/Dataset/factory_conveyor_level2_seeded_old}"
FCS002_DATASET_PATH="${FCS002_DATASET_PATH:-/projects/haitian003ssd/Dataset/fcs_002_two_static_objects_to_container_slots}"
# Defaults are calibrated for static FCS002 : dynamic Factory sampling near 4:1.
# Factory has 131,039 effective steps and FCS002 has 11,830; one Factory
# occurrence and forty-four FCS002 occurrences yield FCS002:Factory = 3.972:1
# (about 79.89% static and 20.11% dynamic samples).
FACTORY_CONVEYOR_REPEATS="${FACTORY_CONVEYOR_REPEATS:-1}"
FCS002_REPEATS="${FCS002_REPEATS:-44}"
PATH_SEPARATOR=":"
DATASET_PATH="${DATASET_PATH:-}"
if [[ -z "${DATASET_PATH}" ]]; then
  DATASET_PATH=""
  for ((i = 0; i < FACTORY_CONVEYOR_REPEATS; i++)); do
    DATASET_PATH+="${DATASET_PATH:+${PATH_SEPARATOR}}${FACTORY_DATASET_PATH}"
  done
  for ((i = 0; i < FCS002_REPEATS; i++)); do
    DATASET_PATH+="${DATASET_PATH:+${PATH_SEPARATOR}}${FCS002_DATASET_PATH}"
  done
fi
if [[ -z "${DATASET_PATH}" ]]; then
  echo "ERROR: At least one dataset repeat count must be positive." >&2
  exit 2
fi
MODALITY_CONFIG="${MODALITY_CONFIG:-${GROOT_DIR}/examples/fc003_panda_config.py}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
RUN_NAME="${RUN_NAME:-fc003_fcs002_mixed_finetune_full_${JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${GROOT_DIR}/outputs/${RUN_NAME}}"

# -----------------------------
# Training scale
# -----------------------------
NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-80000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
OPTIM="${OPTIM:-adamw_torch}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

# -----------------------------
# Runtime / memory behavior
# -----------------------------
DEEPSPEED_STAGE=2
USE_DDP="${USE_DDP:-0}"
USE_FSDP="${USE_FSDP:-0}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-100000000}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
TF32="${TF32:-1}"
FP16="${FP16:-0}"
BF16="${BF16:-1}"

# -----------------------------
# Checkpointing
# -----------------------------
SAVE_STEPS="${SAVE_STEPS:-40000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-0}"
SKIP_WEIGHT_LOADING="${SKIP_WEIGHT_LOADING:-0}"

# -----------------------------
# Model tuning flags
# Defaults match GR00T finetune defaults:
#   frozen LLM/vision backbone, train projector + diffusion/action head.
# -----------------------------
TUNE_LLM="${TUNE_LLM:-0}"
TUNE_VISUAL="${TUNE_VISUAL:-0}"
TUNE_PROJECTOR="${TUNE_PROJECTOR:-1}"
TUNE_DIFFUSION_MODEL="${TUNE_DIFFUSION_MODEL:-1}"
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.2}"

# -----------------------------
# Data loading / sharding
# -----------------------------
VIDEO_BACKEND="${VIDEO_BACKEND:-opencv}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"

# -----------------------------
# Augmentation
# Leave RANDOM_ROTATION_ANGLE empty to disable.
# COLOR_JITTER_* are passed as brightness/contrast/saturation/hue.
# EXTRA_AUGMENTATION_CONFIG can be a JSON string.
# -----------------------------
RANDOM_ROTATION_ANGLE="${RANDOM_ROTATION_ANGLE:-}"
COLOR_JITTER_BRIGHTNESS="${COLOR_JITTER_BRIGHTNESS:-0.3}"
COLOR_JITTER_CONTRAST="${COLOR_JITTER_CONTRAST:-0.4}"
COLOR_JITTER_SATURATION="${COLOR_JITTER_SATURATION:-0.5}"
COLOR_JITTER_HUE="${COLOR_JITTER_HUE:-0.08}"
EXTRA_AUGMENTATION_CONFIG="${EXTRA_AUGMENTATION_CONFIG:-}"

# -----------------------------
# W&B
# -----------------------------
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-finetune-gr00t-n1d7}"

# -----------------------------
# Free-form extra args for launch_finetune.py.
# To use it, edit the array below, e.g.:
#   EXTRA_ARGS=(--random-rotation-angle 5)
# Hidden training defaults surfaced here on purpose:
#   - DEEPSPEED_STAGE controls which ZeRO config is selected on multi-GPU runs.
#   - GRADIENT_CHECKPOINTING trades compute for activation memory.
#   - TF32 / BF16 / FP16 control trainer precision behavior.
# -----------------------------
EXTRA_ARGS=()

module load "${CUDA_MODULE}"

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in the job environment before submitting}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in the job environment before submitting}"
export UV_CACHE_DIR="${CACHE_ROOT}/.uv-cache"
export UV_LINK_MODE=copy
export HF_HUB_CACHE="${CACHE_ROOT}/.hf-cache/hub"
export TORCH_HOME="${CACHE_ROOT}/.hf-cache/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/.hf-cache/xdg"
export TMPDIR="${CACHE_ROOT}/.hf-cache/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export NO_ALBUMENTATIONS_UPDATE=1
export NCCL_DEBUG=INFO
export GROOT_SKIP_DEEPSPEED_MODEL_BROADCAST=0
unset NCCL_CUMEM_ENABLE NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL NCCL_IB_DISABLE NCCL_DMABUF_ENABLE NCCL_PROTO
export PATH="${CACHE_ROOT}/.toolenv-git-lfs/bin:${PATH}"

mkdir -p "${GROOT_DIR}/logs" "${OUTPUT_DIR}" "${UV_CACHE_DIR}" \
  "${HF_HUB_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"

cd "${GROOT_DIR}"

# GR00T's LeRobot episode loader currently requires the v2 metadata layout.
IFS="${PATH_SEPARATOR}" read -r -a DATASET_PATHS <<< "${DATASET_PATH}"
for dataset_path in "${DATASET_PATHS[@]}"; do
  if [[ ! -f "${dataset_path}/meta/episodes.jsonl" ]]; then
  echo "ERROR: Dataset must provide LeRobot v2 metadata: ${dataset_path}/meta/episodes.jsonl" >&2
  echo "The v3 parquet metadata layout is not supported by this GR00T loader." >&2
    exit 2
  fi
done

echo "Job: ${JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: ${GROOT_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Modality config: ${MODALITY_CONFIG}"
echo "Run name: ${RUN_NAME}"
echo "CUDA module: ${CUDA_MODULE}"
echo "CUDA visibility before run:"
nvidia-smi || true

uv run --no-sync python - <<'PYIN'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(i, props.name, round(props.total_memory / 1024**3, 1), "GiB")
PYIN

is_true() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "True" ]]
}

add_true_flag() {
  local name="$1"
  local value="$2"
  if is_true "${value}"; then
    LAUNCH_CMD+=("--${name}")
  fi
}

add_default_true_flag() {
  local name="$1"
  local value="$2"
  if ! is_true "${value}"; then
    LAUNCH_CMD+=("--no-${name}")
  fi
}

LAUNCH_CMD=(
  gr00t/experiment/launch_finetune.py
  --base-model-path "${MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --embodiment-tag "${EMBODIMENT_TAG}"
  --modality-config-path "${MODALITY_CONFIG}"
  --num-gpus "${NUM_GPUS}"
  --output-dir "${OUTPUT_DIR}"
  --experiment-name "${RUN_NAME}"
  --wandb-project "${WANDB_PROJECT}"
  --max-steps "${MAX_STEPS}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-ratio "${WARMUP_RATIO}"
  --optim "${OPTIM}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --logging-steps "${LOGGING_STEPS}"
  --deepspeed-stage 2
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
  --video-backend "${VIDEO_BACKEND}"
  --shard-size "${SHARD_SIZE}"
  --num-shards-per-epoch "${NUM_SHARDS_PER_EPOCH}"
  --episode-sampling-rate "${EPISODE_SAMPLING_RATE}"
  --state-dropout-prob "${STATE_DROPOUT_PROB}"
  --color-jitter-params
  brightness "${COLOR_JITTER_BRIGHTNESS}"
  contrast "${COLOR_JITTER_CONTRAST}"
  saturation "${COLOR_JITTER_SATURATION}"
  hue "${COLOR_JITTER_HUE}"
)

add_true_flag use-wandb "${USE_WANDB}"
add_true_flag tune-llm "${TUNE_LLM}"
add_true_flag tune-visual "${TUNE_VISUAL}"
add_default_true_flag tune-projector "${TUNE_PROJECTOR}"
add_default_true_flag tune-diffusion-model "${TUNE_DIFFUSION_MODEL}"
add_true_flag gradient-checkpointing "${GRADIENT_CHECKPOINTING}"
add_default_true_flag tf32 "${TF32}"
add_true_flag fp16 "${FP16}"
add_default_true_flag bf16 "${BF16}"
add_true_flag save-only-model "${SAVE_ONLY_MODEL}"
add_true_flag resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}"
add_true_flag skip-weight-loading "${SKIP_WEIGHT_LOADING}"

if [[ "" == "1" ]]; then
  LAUNCH_CMD+=(--use-ddp)
fi

if [[ "${USE_FSDP}" == "1" ]]; then
  LAUNCH_CMD+=(--use-fsdp --fsdp-min-num-params "${FSDP_MIN_NUM_PARAMS}")
fi

if [[ -n "${RANDOM_ROTATION_ANGLE}" ]]; then
  LAUNCH_CMD+=(--random-rotation-angle "${RANDOM_ROTATION_ANGLE}")
fi

if [[ -n "${EXTRA_AUGMENTATION_CONFIG}" ]]; then
  LAUNCH_CMD+=(--extra-augmentation-config "${EXTRA_AUGMENTATION_CONFIG}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  LAUNCH_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Launch command:"
printf '  %q' "${LAUNCH_CMD[@]}"
echo

if [[ "${NUM_GPUS}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  uv run --no-sync python "${LAUNCH_CMD[@]}"
else
  uv run --no-sync torchrun --nproc-per-node="${NUM_GPUS}" --master-port="${MASTER_PORT}" "${LAUNCH_CMD[@]}"
fi

echo "FC003 + FCS002 mixed full finetune finished."
find "${OUTPUT_DIR}" -maxdepth 3 -type f -print
