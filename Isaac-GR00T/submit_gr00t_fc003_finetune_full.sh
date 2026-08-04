#!/bin/bash

#SBATCH --job-name=gr00t_level2_ft_full
#SBATCH --output=logs/gr00t_level2_ft_full_%j.log
#SBATCH --error=logs/gr00t_level2_ft_full_%j.err
#SBATCH --time=72:00:00
#SBATCH --gpus=pro6000:2

set -euo pipefail

# -----------------------------
# Cluster / environment
# -----------------------------
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.8.0}"
BASE_DIR="${BASE_DIR:-/projects/hdd/ssd/ICLR2027/baseline}"
GROOT_DIR="${GROOT_DIR:-${BASE_DIR}/Isaac-GR00T}"
CACHE_ROOT="${CACHE_ROOT:-/projects/hdd/ssd}"
JOB_ID="${SLURM_JOB_ID:-local}"
MASTER_PORT="${MASTER_PORT:-29500}"

# -----------------------------
# Data / model paths
# -----------------------------
MODEL_PATH="${MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/projects/hdd/ssd/ICLR2027/dataset/factory_conveyor_level2_seeded}"
MODALITY_CONFIG="${MODALITY_CONFIG:-${GROOT_DIR}/examples/fc003_panda_config.py}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
RUN_NAME="${RUN_NAME:-level_level2_finetune_full_${JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${GROOT_DIR}/outputs/${RUN_NAME}}"

# -----------------------------
# Training scale
# -----------------------------
NUM_GPUS="${NUM_GPUS:-2}"
MAX_STEPS="${MAX_STEPS:-80000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
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
DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-2}"
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
#   EXTRA_ARGS=(--random_rotation_angle 5)
# Hidden training defaults surfaced here on purpose:
#   - DEEPSPEED_STAGE controls which ZeRO config is selected on multi-GPU runs.
#   - GRADIENT_CHECKPOINTING trades compute for activation memory.
#   - TF32 / BF16 / FP16 control trainer precision behavior.
# -----------------------------
EXTRA_ARGS=()

module load "${CUDA_MODULE}"

export UV_CACHE_DIR="${CACHE_ROOT}/.uv-cache"
export UV_LINK_MODE=copy
export HF_HUB_CACHE="${CACHE_ROOT}/.hf-cache/hub"
export TORCH_HOME="${CACHE_ROOT}/.hf-cache/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/.hf-cache/xdg"
export TMPDIR="${CACHE_ROOT}/.hf-cache/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export NO_ALBUMENTATIONS_UPDATE=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export PATH="${CACHE_ROOT}/.toolenv-git-lfs/bin:${PATH}"

mkdir -p "${GROOT_DIR}/logs" "${OUTPUT_DIR}" "${UV_CACHE_DIR}" \
  "${HF_HUB_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"

cd "${GROOT_DIR}"

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

uv run python - <<'PYIN'
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
  --base_model_path "${MODEL_PATH}"
  --dataset_path "${DATASET_PATH}"
  --embodiment_tag "${EMBODIMENT_TAG}"
  --modality_config_path "${MODALITY_CONFIG}"
  --num_gpus "${NUM_GPUS}"
  --output_dir "${OUTPUT_DIR}"
  --experiment_name "${RUN_NAME}"
  --wandb_project "${WANDB_PROJECT}"
  --max_steps "${MAX_STEPS}"
  --global_batch_size "${GLOBAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --optim "${OPTIM}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --logging_steps "${LOGGING_STEPS}"
  --deepspeed_stage "${DEEPSPEED_STAGE}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --video_backend "${VIDEO_BACKEND}"
  --shard_size "${SHARD_SIZE}"
  --num_shards_per_epoch "${NUM_SHARDS_PER_EPOCH}"
  --episode_sampling_rate "${EPISODE_SAMPLING_RATE}"
  --state_dropout_prob "${STATE_DROPOUT_PROB}"
  --color_jitter_params
  brightness "${COLOR_JITTER_BRIGHTNESS}"
  contrast "${COLOR_JITTER_CONTRAST}"
  saturation "${COLOR_JITTER_SATURATION}"
  hue "${COLOR_JITTER_HUE}"
)

add_true_flag use_wandb "${USE_WANDB}"
add_true_flag tune_llm "${TUNE_LLM}"
add_true_flag tune_visual "${TUNE_VISUAL}"
add_default_true_flag tune_projector "${TUNE_PROJECTOR}"
add_default_true_flag tune_diffusion_model "${TUNE_DIFFUSION_MODEL}"
add_true_flag gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
add_default_true_flag tf32 "${TF32}"
add_true_flag fp16 "${FP16}"
add_default_true_flag bf16 "${BF16}"
add_true_flag save_only_model "${SAVE_ONLY_MODEL}"
add_true_flag resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"
add_true_flag skip_weight_loading "${SKIP_WEIGHT_LOADING}"

if [[ -n "${RANDOM_ROTATION_ANGLE}" ]]; then
  LAUNCH_CMD+=(--random_rotation_angle "${RANDOM_ROTATION_ANGLE}")
fi

if [[ -n "${EXTRA_AUGMENTATION_CONFIG}" ]]; then
  LAUNCH_CMD+=(--extra_augmentation_config "${EXTRA_AUGMENTATION_CONFIG}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  LAUNCH_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Launch command:"
printf '  %q' "${LAUNCH_CMD[@]}"
echo

if [[ "${NUM_GPUS}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  uv run python "${LAUNCH_CMD[@]}"
else
  uv run torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" "${LAUNCH_CMD[@]}"
fi

echo "Level level2 full finetune finished."
find "${OUTPUT_DIR}" -maxdepth 3 -type f -print
