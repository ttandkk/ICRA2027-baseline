# Diffusion Policy environment

DP uses the same isolated LeRobot stack as ACT, with an additional Diffusers dependency.

```bash
conda create -n diffusion-policy-baseline python=3.12 -y
conda activate diffusion-policy-baseline
pip install --upgrade pip
pip install -r requirements-repro.txt
pip install -e ../lerobot
```

Validated with CUDA 13.0 runtime, `torch==2.11.0`, `torchvision==0.26.0`, and `lerobot==0.5.2`. Use an NVIDIA driver compatible with CUDA 13. The launcher requires a LeRobot v3 dataset; set `DATASET_ROOT` before `bash submit_train_diffusion_policy.sh`.
