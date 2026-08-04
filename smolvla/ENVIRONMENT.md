# SmolVLA environment

```bash
conda create -n smolvla-baseline python=3.12 -y
conda activate smolvla-baseline
pip install --upgrade pip
pip install -r requirements-repro.txt
```

Validated run: CUDA 13.0 runtime, `torch==2.11.0`, `torchvision==0.26.0`, `lerobot==0.5.2`, and `transformers==5.5.4`. Use an NVIDIA driver compatible with CUDA 13. Dataset and checkpoint files are excluded from the Git release; configure their locations in your run command.
