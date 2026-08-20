#!/usr/bin/env bash
#SBATCH --job-name=factory_draft
#SBATCH --partition=cluster02
#SBATCH --output=/projects/haitian003ssd/ICRA2027-baseline/realtime-vla-flash/logs/factory_draft_%j.log
#SBATCH --error=/projects/haitian003ssd/ICRA2027-baseline/realtime-vla-flash/logs/factory_draft_%j.err
#SBATCH --time=48:00:00
#SBATCH --gpus=l40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
# Train FLASH's draft head after the π0.5 checkpoint has been produced.
set -euo pipefail

REPO_ROOT=${OPENPI_REPO_ROOT:-/projects/haitian003ssd/ICRA2027-baseline/realtime-vla-flash}
cd "${REPO_ROOT}"

MAIN_CHECKPOINT=${MAIN_CHECKPOINT:-/projects/_ssd/haitian003ssd/ICRA2027-baseline/realtime-vla-flash/checkpoints/pi05_factory_conveyor_lora/factory_conveyor_pi05_lora_ddp4_80000/80000}
DATASET_ROOT=${DATASET_ROOT:-/projects/haitian003ssd/Dataset/factory_conveyor_level2_seeded_old}
RUN_NAME=${RUN_NAME:-factory_conveyor_pi05_lora_ddp4_80000_draft}
CACHE_ROOT=${CACHE_ROOT:-${REPO_ROOT}/outputs/factory_conveyor_flash_cache}
DRAFT_OUTPUT=${DRAFT_OUTPUT:-outputs/${RUN_NAME}/draft_model.pt}
CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-32}
DRAFT_BATCH_SIZE=${DRAFT_BATCH_SIZE:-256}
DRAFT_EPOCHS=${DRAFT_EPOCHS:-100}
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-2}}

if [[ ! -f "${MAIN_CHECKPOINT}/model.safetensors" ]]; then
  echo "MAIN_CHECKPOINT must contain model.safetensors: ${MAIN_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "DATASET_ROOT does not exist or is not a directory: ${DATASET_ROOT}" >&2
  exit 1
fi
mkdir -p "${CACHE_ROOT}" "$(dirname "${DRAFT_OUTPUT}")"

# Match the model-specific Transformers patch used by the main π0.5 training job.
TRANSFORMERS_DIR=$(uv run python -c "import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)")
cp -r "${REPO_ROOT}/src/openpi/models_pytorch/transformers_replace/"* "${TRANSFORMERS_DIR}/"

# The cache is supervised by the fixed main policy. Its 10-D target/action chunk
# exactly matches the MotionForge action space and π0.5 action horizon.
uv run torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" scripts/spec/enc_cache.py \
  --config=pi05_factory_conveyor_lora \
  --checkpoint-dir="${MAIN_CHECKPOINT}" \
  --dataset-root="${DATASET_ROOT}" \
  --dataset-format=factory_conveyor \
  --chunk-m=10 \
  --out-dim=10 \
  --max-exec-steps=10 \
  --cache-dir="${CACHE_ROOT}" \
  --batch-size="${CACHE_BATCH_SIZE}" \
  --video-backend=pyav \
  --overwrite

CACHE_RUN_DIR=$(find "${CACHE_ROOT}/pi05_factory_conveyor_lora" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
if [[ -z "${CACHE_RUN_DIR}" || ! -f "${CACHE_RUN_DIR}/manifest.json" ]]; then
  echo "FLASH cache creation did not produce a manifest under ${CACHE_ROOT}" >&2
  exit 1
fi

uv run torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" scripts/spec/spec_draft_train.py \
  --cache-run-dir="${CACHE_RUN_DIR}" \
  --out-ckpt="${DRAFT_OUTPUT}" \
  --batch-size="${DRAFT_BATCH_SIZE}" \
  --epochs="${DRAFT_EPOCHS}" \
  --num-workers=4 \
  --eval-interval-epochs=1 \
  --eval-exec-steps=10
