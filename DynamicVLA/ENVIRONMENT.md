# DynamicVLA environments

DynamicVLA needs two separate Python 3.10 environments. Do not install Isaac Lab into the training environment.

## Training and inference

Use Python 3.10 and PyTorch 2.7.1 (choose the official PyTorch wheel matching your CUDA driver), then install repository requirements:

```bash
conda create -n dynamicvla-train python=3.10 -y
conda activate dynamicvla-train
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
conda install -c conda-forge 'ffmpeg<8'
```

## Simulation

Create another Python 3.10 environment and install Isaac Sim 4.5.0 plus Isaac Lab 2.2.1 using the [Isaac Lab guide](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/index.html), then run `pip install shapely pyzmq h5py`.

Use the training environment for training/inference and the Isaac Lab environment only for `simulations/`.
