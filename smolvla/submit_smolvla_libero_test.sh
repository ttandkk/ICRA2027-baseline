#!/bin/bash

# --- Slurm config ---
#SBATCH --job-name=smolvla_libero_test
#SBATCH --output=logs/smolvla_libero_%j.log
#SBATCH --error=logs/smolvla_libero_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=6000ada:1

set -euo pipefail

# --- Paths on SSD ---
BASE_DIR=/projects/hdd/ssd/ICLR2027/baseline
LEROBOT_DIR=${BASE_DIR}/lerobot
CONDA_ENV=/projects/hdd/ssd/conda/envs/lerobot
CONDA_PKGS_DIRS=/projects/hdd/ssd/conda/pkgs
HF_CACHE=/projects/hdd/ssd/hf_cache
LIBERO_CACHE=${HF_CACHE}/libero
JOB_HOME=${HF_CACHE}/home

export CONDA_PKGS_DIRS
export HF_HOME=${HF_CACHE}
export PIP_CACHE_DIR=${HF_CACHE}/pip
export TORCH_HOME=${HF_CACHE}/torch
export UV_CACHE_DIR=${HF_CACHE}/uv
export XDG_CACHE_HOME=${HF_CACHE}/xdg
export HOME=${JOB_HOME}
export LIBERO_CONFIG_PATH=${LIBERO_CACHE}
export TMPDIR=${HF_CACHE}/tmp
export PYTHONNOUSERSITE=1
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PATH=${CONDA_ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

mkdir -p logs "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${LIBERO_CACHE}" "${JOB_HOME}" "${LIBERO_CACHE}/datasets" "${JOB_HOME}/.cache/libero/assets"

# --- Environment setup ---
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate "${CONDA_ENV}"

cd "${LEROBOT_DIR}"

LIBERO_PKG_DIR="${CONDA_ENV}/lib/python3.12/site-packages/libero/libero"
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  {
    printf "assets: %s\n" "${JOB_HOME}/.cache/libero/assets"
    printf "bddl_files: %s\n" "${LIBERO_PKG_DIR}/bddl_files"
    printf "benchmark_root: %s\n" "${LIBERO_PKG_DIR}"
    printf "datasets: %s\n" "${LIBERO_CACHE}/datasets"
    printf "init_states: %s\n" "${LIBERO_PKG_DIR}/init_files"
  } > "${LIBERO_CONFIG_PATH}/config.yaml"
fi

# Install the extras needed by the official SmolVLA and LIBERO docs.
# Set INSTALL_EXTRAS=1 only when you intentionally want to refresh dependencies.
if [[ "${INSTALL_EXTRAS:-0}" == "1" ]]; then
  python -m pip install -e ".[smolvla,libero]"
fi

# --- Evaluation config ---
# POLICY_PATH should point to a LIBERO-compatible SmolVLA checkpoint for real results.
# The default is only a quick wiring/cache smoke test.
POLICY_PATH=${POLICY_PATH:-lerobot/smolvla_base}
TASKS=${TASKS:-libero_spatial}
TASK_IDS=${TASK_IDS:-"[0]"}
N_EPISODES=${N_EPISODES:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_PARALLEL_TASKS=${MAX_PARALLEL_TASKS:-1}
CONTROL_MODE=${CONTROL_MODE:-relative}
RENAME_MAP=${RENAME_MAP:-'{}'}
POLICY_INPUT_FEATURES=${POLICY_INPUT_FEATURES:-null}
POLICY_OUTPUT_FEATURES=${POLICY_OUTPUT_FEATURES:-null}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/smolvla/eval_outputs/smolvla_libero_${SLURM_JOB_ID:-local}}

mkdir -p "${OUTPUT_DIR}"

cmd=(
  lerobot-eval
  --policy.path="${POLICY_PATH}"
  --env.type=libero
  --env.task="${TASKS}"
  --env.control_mode="${CONTROL_MODE}"
  --eval.batch_size="${BATCH_SIZE}"
  --eval.n_episodes="${N_EPISODES}"
  --env.max_parallel_tasks="${MAX_PARALLEL_TASKS}"
  --output_dir="${OUTPUT_DIR}"
  --rename_map="${RENAME_MAP}"
  --policy.input_features="${POLICY_INPUT_FEATURES}"
  --policy.output_features="${POLICY_OUTPUT_FEATURES}"
)

if [[ -n "${TASK_IDS}" ]]; then
  cmd+=(--env.task_ids="${TASK_IDS}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
