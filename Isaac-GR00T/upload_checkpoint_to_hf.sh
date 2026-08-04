#!/usr/bin/env bash

set -euo pipefail

# Upload a checkpoint directory to the Hugging Face Hub.
# Usage:
#   bash upload_checkpoint_to_hf.sh <repo_id> [source_dir] [path_in_repo]
#
# Examples:
#   bash upload_checkpoint_to_hf.sh yourname/gr00t-checkpoint-80000
#   bash upload_checkpoint_to_hf.sh yourname/gr00t-checkpoint-80000 \
#     /projects/hdd/ssd/ICLR2027/baseline/Isaac-GR00T/outputs/level_level2_finetune_full_87682/level_level2_finetune_full_87682/checkpoint-80000 \
#     checkpoint-80000

BASE_DIR="${BASE_DIR:-/projects/hdd/ssd/ICLR2027/baseline}"
GROOT_DIR="${GROOT_DIR:-${BASE_DIR}/Isaac-GR00T}"

REPO_ID="${1:-${REPO_ID:-}}"
SOURCE_DIR="${2:-${SOURCE_DIR:-${GROOT_DIR}/outputs/level_level2_finetune_full_87682/level_level2_finetune_full_87682/checkpoint-80000}}"
PATH_IN_REPO="${3:-${PATH_IN_REPO:-checkpoint-80000}}"

REPO_TYPE="${REPO_TYPE:-model}"
PRIVATE_REPO="${PRIVATE_REPO:-1}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload checkpoint ${PATH_IN_REPO}}"

if [[ -z "${REPO_ID}" ]]; then
  echo "error: missing repo id" >&2
  echo "usage: bash $0 <repo_id> [source_dir] [path_in_repo]" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "error: source directory not found: ${SOURCE_DIR}" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-/projects/hdd/ssd/.hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TMPDIR="${TMPDIR:-/projects/hdd/ssd/.hf-cache/tmp}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TMPDIR}"

cd "${GROOT_DIR}"

uv run python - "${REPO_ID}" "${SOURCE_DIR}" "${PATH_IN_REPO}" "${REPO_TYPE}" "${PRIVATE_REPO}" "${COMMIT_MESSAGE}" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id, source_dir, path_in_repo, repo_type, private_repo, commit_message = sys.argv[1:7]
private = private_repo.lower() in {"1", "true", "yes", "on"}

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

print(f"Uploading {source_dir} -> {repo_id}/{path_in_repo}")
api.upload_folder(
    folder_path=source_dir,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo=path_in_repo,
    commit_message=commit_message,
    ignore_patterns=["*.tmp", "*.log", ".DS_Store", "__pycache__/*", "*.pyc"],
)
print("Upload complete.")
PY
