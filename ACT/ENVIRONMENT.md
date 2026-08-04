# ACT environment

This launcher calls the local `../lerobot` checkout. Use a dedicated environment, shared only with the ACT/DP/SmolVLA/pi0.5 experiments.

```bash
conda create -n lerobot-baselines python=3.12 -y
conda activate lerobot-baselines
pip install --upgrade pip
pip install -r requirements-repro.txt
pip install -e ../lerobot
```

Validated run: Python 3.12, CUDA 13.0 runtime, `torch==2.11.0`, `torchvision==0.26.0`, and `lerobot==0.5.2`. An NVIDIA driver compatible with CUDA 13 is required. The dataset must be LeRobot v3 (`meta/info.json` has `codebase_version: v3.0`); set `DATASET_ROOT` before `bash submit_train_act.sh`.
