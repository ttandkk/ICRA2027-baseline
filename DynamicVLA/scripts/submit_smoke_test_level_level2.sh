#!/bin/bash

# --- Slurm 配置参数 ---
#SBATCH --job-name=dynamicvla_smoke
#SBATCH --output=logs/dynamicvla_smoke_%j.log
#SBATCH --error=logs/dynamicvla_smoke_%j.err
#SBATCH --time=02:00:00
#SBATCH --gpus=6000ada:1

# --- 环境准备 ---
module load Miniforge3
eval "$(conda shell.bash hook)"

# 激活训练环境
conda activate /projects/hdd/ssd/ICLR2027/baseline/.conda/dynamicvla-train310

# 进入项目目录
cd /projects/hdd/ssd/ICLR2027/baseline/DynamicVLA

# 确保日志目录存在
mkdir -p logs

# --- 执行任务 ---
python scripts/smoke_test_level_level2.py --load-checkpoint
