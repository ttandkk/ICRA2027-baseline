#!/usr/bin/env bash
#SBATCH --job-name=vla_adapter_fc
#SBATCH --output=/projects/hdd/ssd/ICLR2027/baseline/VLA-Adapter/logs/vla_adapter_fc_%j.log
#SBATCH --error=/projects/hdd/ssd/ICLR2027/baseline/VLA-Adapter/logs/vla_adapter_fc_%j.err
#SBATCH --time=48:00:00
#SBATCH --gpus=6000ada:1
#SBATCH --cpus-per-task=8

# Train VLA-Adapter on the local factory-conveyor LeRobot v3 dataset.
# Full-parameter fine-tuning uses activation checkpointing and gradient clipping.
# The effective batch size remains 32 (16 x 2 gradient accumulation).
set -euo pipefail

PROJECT_ROOT="/projects/hdd/ssd/ICLR2027/baseline/VLA-Adapter"
ENV_PATH="${PROJECT_ROOT}/.conda/vla-adapter-train"
WANDB_PROJECT="${WANDB_PROJECT:-vla-adapter-factory-conveyor}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_ARGS=(--wandb_project "${WANDB_PROJECT}" --wandb_mode online)
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi

mkdir -p "${PROJECT_ROOT}/logs"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
cd "${PROJECT_ROOT}"

torchrun --standalone --nnodes=1 --nproc-per-node=1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir /projects/hdd/ssd/ICLR2027/dataset \
  --dataset_name factory_conveyor_level2_seeded \
  --dataset_format lerobot_v3 \
  --use_minivlm True \
  --use_proprio True \
  --use_lora False \
  --use_fz True \
  --enable_gradient_checkpointing True \
  --max_grad_norm 1.0 \
  --max_steps 80000 \
  --save_freq 40000 \
  --batch_size 16 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-5 \
  "${WANDB_ARGS[@]}"
