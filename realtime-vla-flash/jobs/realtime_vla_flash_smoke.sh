#!/bin/bash

#SBATCH --job-name=flash_smoke
#SBATCH --partition=cluster02
#SBATCH --output=logs/flash_smoke_%j.log
#SBATCH --error=logs/flash_smoke_%j.err
#SBATCH --time=00:15:00
#SBATCH --gpus=6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

REPO=/projects/hdd/ssd/ICRA2027/baseline/realtime-vla-flash
cd "$REPO"

module load git-lfs/3.6.1
export UV_CACHE_DIR=/projects/hdd/ssd
source .venv/bin/activate

echo "=== Slurm ==="
echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

echo "=== GPU ==="
nvidia-smi

echo "=== Python environment ==="
python -c "import sys, torch, triton, transformers, jax, openpi; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count()); print('gpu', torch.cuda.get_device_name(0)); print('triton', triton.__version__); print('transformers', transformers.__version__); print('jax', jax.__version__, 'devices', jax.devices()); print('openpi', openpi.__file__)"

echo "=== CUDA tensor smoke test ==="
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; torch.cuda.synchronize(); print('matmul_ok', y.shape, y.dtype, 'allocated_mb', round(torch.cuda.memory_allocated()/1024**2, 2))"

echo "=== FLASH converter self-test ==="
python scripts/spec/triton/convert_for_triton.py --self-test

echo "=== FLASH core tests ==="
PYTHONPATH=. pytest -q scripts/spec/test/spec_verify_test.py

echo "FLASH smoke test passed"
