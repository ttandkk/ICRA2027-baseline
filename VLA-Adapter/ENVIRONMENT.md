# VLA-Adapter environment

```bash
conda create -n vla-adapter python=3.10.16 -y
conda activate vla-adapter
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
pip install -e .
pip install packaging ninja
pip install 'flash-attn==2.5.5' --no-build-isolation
```

The validated setup uses CUDA 12.2 and the matching FlashAttention wheel (`flash_attn-2.5.5+cu122torch2.2...cp310...whl`) when available. For LIBERO evaluation, also install its checkout and `experiments/robot/libero/libero_requirements.txt` as described in the project README. Linux OpenGL/EGL development libraries are required for simulation.
