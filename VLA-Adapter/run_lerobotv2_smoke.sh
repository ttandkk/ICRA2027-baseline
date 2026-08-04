#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATASET_ROOT="${1:-fc_003_pick_moving_object_to_static_container}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
IMAGE_KEY="${IMAGE_KEY:-observation.images.front}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

conda run -n starVLA python smoke_lerobotv2.py \
  --dataset-root "$DATASET_ROOT" \
  --num-samples "$NUM_SAMPLES" \
  --image-key "$IMAGE_KEY" \
  --image-size "$IMAGE_SIZE"
