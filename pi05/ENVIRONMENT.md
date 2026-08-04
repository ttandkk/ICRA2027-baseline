# pi0.5 environment

```bash
conda create -n pi05-baseline python=3.12 -y
conda activate pi05-baseline
pip install --upgrade pip
pip install -r requirements-repro.txt
```

This is the top-level package set recorded by the pi0.5 run: CUDA 13.0 runtime, PyTorch 2.11, and LeRobot 0.5.2. Supply a compatible NVIDIA driver and configure dataset/output paths in the run command. Data and outputs are intentionally not versioned in Git.
