#!/bin/bash

# --- Slurm config ---
#SBATCH --job-name=smolvla_dummy_bench
#SBATCH --output=logs/smolvla_dummy_bench_%j.log
#SBATCH --error=logs/smolvla_dummy_bench_%j.err
#SBATCH --time=04:00:00
#SBATCH --gpus=6000ada:1

set -euo pipefail

BASE_DIR=/projects/hdd/ssd/ICLR2027/baseline
CONDA_ENV=/projects/hdd/ssd/conda/envs/lerobot
CONDA_PKGS_DIRS=/projects/hdd/ssd/conda/pkgs
HF_CACHE=/projects/hdd/ssd/hf_cache

export CONDA_PKGS_DIRS
export HF_HOME=${HF_CACHE}
export HF_LEROBOT_HOME=${HF_CACHE}/lerobot
export PIP_CACHE_DIR=${HF_CACHE}/pip
export TORCH_HOME=${HF_CACHE}/torch
export UV_CACHE_DIR=${HF_CACHE}/uv
export XDG_CACHE_HOME=${HF_CACHE}/xdg
export TMPDIR=${HF_CACHE}/tmp
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PATH=${CONDA_ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

mkdir -p logs "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}"

module load Miniforge3
eval "$(conda shell.bash hook)"
source activate "${CONDA_ENV}"

cd "${BASE_DIR}"

POLICY_PATH=${POLICY_PATH:-lerobot/smolvla_base}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-1}
WARMUP_STEPS=${WARMUP_STEPS:-5}
BENCH_STEPS=${BENCH_STEPS:-50}
MODE=${MODE:-chunk}
NUM_DENOISE_STEPS=${NUM_DENOISE_STEPS:-}
MAX_CAMERAS=${MAX_CAMERAS:-}
TASK=${TASK:-"pick up the object and place it in the target area"}
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/smolvla/benchmark_outputs/smolvla_dummy_${MODE}_${SLURM_JOB_ID:-local}.json}

extra_args=()
if [[ -n "${NUM_DENOISE_STEPS}" ]]; then
  extra_args+=(--num-denoise-steps="${NUM_DENOISE_STEPS}")
fi
if [[ -n "${MAX_CAMERAS}" ]]; then
  extra_args+=(--max-cameras="${MAX_CAMERAS}")
fi

python -u smolvla/benchmark_smolvla_dummy.py \
  --policy-path="${POLICY_PATH}" \
  --device="${DEVICE}" \
  --batch-size="${BATCH_SIZE}" \
  --warmup-steps="${WARMUP_STEPS}" \
  --bench-steps="${BENCH_STEPS}" \
  --mode="${MODE}" \
  --task="${TASK}" \
  --output-json="${OUTPUT_JSON}" \
  "${extra_args[@]}"
