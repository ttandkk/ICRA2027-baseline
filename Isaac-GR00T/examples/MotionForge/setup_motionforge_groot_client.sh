#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GROOT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${GROOT_ROOT}/../.." && pwd -P)"
MOTIONFORGE_ROOT="${MOTIONFORGE_ROOT:-${WORKSPACE_ROOT}/MotionForge}"

UV_EXE="${UV_EXE:-}"
if [[ -z "${UV_EXE}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    UV_EXE="$(command -v uv)"
  else
    UV_EXE="${WORKSPACE_ROOT}/miniconda3/envs/lerobot/bin/uv"
  fi
fi

CONDA_EXE="${MOTIONFORGE_CONDA_EXE:-${WORKSPACE_ROOT}/miniconda3/bin/conda}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_ROOT}/.cache/gr00t-uv}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${WORKSPACE_ROOT}/.cache/gr00t-python}"
FFMPEG_PREFIX="${GROOT_FFMPEG_PREFIX:-${WORKSPACE_ROOT}/.cache/gr00t-ffmpeg}"
BACKBONE_DIR="${GROOT_BACKBONE_MODEL_PATH:-${WORKSPACE_ROOT}/ckpts/nvidia/Cosmos-Reason2-2B}"
HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}"
ASSET_SYNC_SCRIPT="${MOTIONFORGE_ROOT}/scripts/assets/pull_runtime_assets.py"

[[ -x "${UV_EXE}" ]] || { echo "uv not found: ${UV_EXE}" >&2; exit 1; }
[[ -x "${CONDA_EXE}" ]] || { echo "conda not found: ${CONDA_EXE}" >&2; exit 1; }

mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${HF_HOME}" "$(dirname -- "${BACKBONE_DIR}")"

echo "[SETUP] Installing the locked x86_64 GR00T environment..."
env UV_CACHE_DIR="${UV_CACHE_DIR}" UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR}" \
  "${UV_EXE}" sync \
  --project "${GROOT_ROOT}" \
  --python 3.12 \
  --python-platform x86_64-unknown-linux-gnu \
  --frozen \
  --no-install-package torchcodec

# The repository carries the aarch64 torchcodec wheel through Git LFS.  Installing
# the published x86_64 wheel separately avoids requiring Git LFS on this host.
if ! env UV_CACHE_DIR="${UV_CACHE_DIR}" \
  "${UV_EXE}" pip install \
  --offline \
  --python "${GROOT_ROOT}/.venv/bin/python" \
  torchcodec==0.8.0; then
  env UV_CACHE_DIR="${UV_CACHE_DIR}" \
    "${UV_EXE}" pip install \
    --python "${GROOT_ROOT}/.venv/bin/python" \
    torchcodec==0.8.0
fi

if [[ ! -f "${FFMPEG_PREFIX}/lib/libavcodec.so.61" ]]; then
  echo "[SETUP] Installing an isolated FFmpeg 7 runtime for torchcodec..."
  "${CONDA_EXE}" create -y -p "${FFMPEG_PREFIX}" \
    -c conda-forge --override-channels ffmpeg=7
fi

if [[ ! -f "${BACKBONE_DIR}/model.safetensors" ]]; then
  [[ -s "${HF_TOKEN_PATH}" ]] || {
    echo "Hugging Face token not found at ${HF_TOKEN_PATH}. Run 'hf auth login' first." >&2
    exit 1
  }
  echo "[SETUP] Downloading nvidia/Cosmos-Reason2-2B..."
  env HF_HOME="${HF_HOME}" HF_TOKEN_PATH="${HF_TOKEN_PATH}" HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    "${GROOT_ROOT}/.venv/bin/hf" download \
    nvidia/Cosmos-Reason2-2B \
    --local-dir "${BACKBONE_DIR}"
fi

missing_asset_patterns=()
if [[ ! -f "${MOTIONFORGE_ROOT}/source/motionforge/motionforge/assets/runtime/containers/boxes/box06/box06.usd" ]]; then
  missing_asset_patterns+=("containers/boxes/box06/**")
fi
if [[ ! -f "${MOTIONFORGE_ROOT}/source/motionforge/motionforge/assets/runtime/containers/trays/tray05/tray05.usd" ]]; then
  missing_asset_patterns+=("containers/trays/tray05/**")
fi
if ((${#missing_asset_patterns[@]} > 0)); then
  [[ -f "${ASSET_SYNC_SCRIPT}" ]] || {
    echo "MotionForge asset sync script not found: ${ASSET_SYNC_SCRIPT}" >&2
    exit 1
  }
  echo "[SETUP] Downloading required MotionForge container assets..."
  env HF_HOME="${HF_HOME}" HF_TOKEN_PATH="${HF_TOKEN_PATH}" HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    "${GROOT_ROOT}/.venv/bin/python" "${ASSET_SYNC_SCRIPT}" \
    --allow-patterns "${missing_asset_patterns[@]}"
fi

echo "[SETUP] Verifying client imports..."
env LD_LIBRARY_PATH="${FFMPEG_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${GROOT_ROOT}/.venv/bin/python" - <<'PY'
import gr00t
import torch
import torchcodec
import transformers
import zmq

print(f"GR00T: {gr00t.__file__}")
print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}; CUDA available: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}; TorchCodec: {torchcodec.__version__}; PyZMQ: {zmq.__version__}")
PY

echo "[SETUP] Client environment is ready."
