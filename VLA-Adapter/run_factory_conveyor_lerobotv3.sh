#!/usr/bin/env bash
# Fine-tune VLA-Adapter on the local factory-conveyor LeRobot v3 dataset.
set -euo pipefail

cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/projects/hdd/ssd/ICLR2027/dataset}"
DATASET_NAME="${DATASET_NAME:-factory_conveyor_level2_seeded}"
VLM_PATH="${VLM_PATH:-pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}"
CONFIG_PATH="${CONFIG_PATH:-pretrained_models/configs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
GPU_COUNT="${GPU_COUNT:-1}"
# The LeRobot v3 tasks.parquet already supplies task text.  Set this only to
# override it with a custom JSON mapping such as {"0": "place ...", "1": "..."}.
TASK_INSTRUCTIONS_PATH="${TASK_INSTRUCTIONS_PATH:-}"
EXTRA_DATA_ARGS=()
if [[ -n "${TASK_INSTRUCTIONS_PATH}" ]]; then
  EXTRA_DATA_ARGS=(--lerobot_task_instructions_path "${TASK_INSTRUCTIONS_PATH}")
fi

torchrun --standalone --nnodes 1 --nproc-per-node "${GPU_COUNT}" vla-scripts/finetune.py \
  --vlm_path "${VLM_PATH}" \
  --config_file_path "${CONFIG_PATH}" \
  --data_root_dir "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_format lerobot_v3 \
  --run_root_dir "${OUTPUT_ROOT}" \
  --lerobot_image_key observation.images.front \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug False \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --num_steps_before_decay 100000 \
  --max_steps 100005 \
  --save_freq 5000 \
  --merge_lora_during_training True \
  --use_pro_version True \
  --wandb_project "${DATASET_NAME}" \
  --run_id_note "VLA-Adapter-${DATASET_NAME}" \
  "${EXTRA_DATA_ARGS[@]}"
