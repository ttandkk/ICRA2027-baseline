# LeRobot environment used by the local baselines

The local ACT and DP launchers were run with Python 3.12, CUDA 13.0 runtime, PyTorch 2.11, and LeRobot 0.5.2. To recreate that base environment:

```bash
conda create -n lerobot-baselines python=3.12 -y
conda activate lerobot-baselines
pip install torch==2.11.0 torchvision==0.26.0 lerobot==0.5.2
pip install -e .
```

Install policy extras only in the environment that needs them, for example `pip install '.[diffusion]'` for Diffusion Policy. Hardware-specific camera, robot, and simulator requirements are documented upstream and should not be installed unless needed.
