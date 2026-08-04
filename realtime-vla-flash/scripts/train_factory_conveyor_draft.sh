#!/usr/bin/env bash
# Train FLASH's draft head after the π0.5 checkpoint has been produced.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

: "${MAIN_CHECKPOINT:?Set MAIN_CHECKPOINT to the final pi05 PyTorch checkpoint directory.}"
RUN_NAME=${RUN_NAME:-factory_conveyor_pi05_lora}
CACHE_ROOT=${CACHE_ROOT:-outputs/factory_conveyor_flash_cache}
DRAFT_OUTPUT=${DRAFT_OUTPUT:-outputs/${RUN_NAME}/draft_model.pt}
CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-32}
DRAFT_BATCH_SIZE=${DRAFT_BATCH_SIZE:-256}
DRAFT_EPOCHS=${DRAFT_EPOCHS:-100}

# The cache is supervised by the fixed main policy. Its 10-D target/action chunk
# exactly matches the MotionForge action space and π0.5 action horizon.
uv run scripts/spec/enc_cache.py \
  --config=pi05_factory_conveyor_lora \
  --checkpoint-dir="${MAIN_CHECKPOINT}" \
  --dataset-root=/projects/hdd/ssd/ICLR2027/dataset/factory_conveyor_level2_seeded_old \
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

uv run scripts/spec/spec_draft_train.py \
  --cache-run-dir="${CACHE_RUN_DIR}" \
  --out-ckpt="${DRAFT_OUTPUT}" \
  --batch-size="${DRAFT_BATCH_SIZE}" \
  --epochs="${DRAFT_EPOCHS}" \
  --num-workers=4 \
  --eval-interval-epochs=1 \
  --eval-exec-steps=10
