# Baseline collection

This directory contains the baseline implementations used in this project. Do **not** install all projects into one Python environment.

| Project | Python | Accelerator stack | Setup guide |
| --- | --- | --- | --- |
| ACT | 3.12 | PyTorch 2.11 / CUDA 13.0 | [ACT/ENVIRONMENT.md](ACT/ENVIRONMENT.md) |
| DP | 3.12 | PyTorch 2.11 / CUDA 13.0 | [DP/ENVIRONMENT.md](DP/ENVIRONMENT.md) |
| DynamicVLA | 3.10 | PyTorch 2.7.1; separate Isaac Lab environment for simulation | [DynamicVLA/ENVIRONMENT.md](DynamicVLA/ENVIRONMENT.md) |
| Isaac-GR00T | 3.10 | CUDA 12.8 dGPU | [Isaac-GR00T/ENVIRONMENT.md](Isaac-GR00T/ENVIRONMENT.md) |
| LeRobot | 3.12 | PyTorch 2.11 / CUDA 13.0 | [lerobot/ENVIRONMENT.md](lerobot/ENVIRONMENT.md) |
| pi0.5 | 3.12 | PyTorch 2.11 / CUDA 13.0 | [pi05/ENVIRONMENT.md](pi05/ENVIRONMENT.md) |
| Realtime VLA Flash | 3.11 | project-managed `uv` environment | [realtime-vla-flash/ENVIRONMENT.md](realtime-vla-flash/ENVIRONMENT.md) |
| SmolVLA | 3.12 | PyTorch 2.11 / CUDA 13.0 | [smolvla/ENVIRONMENT.md](smolvla/ENVIRONMENT.md) |
| VLA-Adapter | 3.10.16 | PyTorch 2.2 / CUDA 12.2 | [VLA-Adapter/ENVIRONMENT.md](VLA-Adapter/ENVIRONMENT.md) |

The `*-repro` files record versions used by the local training runs. Dataset paths, checkpoints, WandB logs, Conda environments, and other generated artifacts are deliberately excluded from Git; obtain those separately and set paths in the launch scripts.
