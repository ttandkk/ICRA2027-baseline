# Isaac-GR00T environment

The supported dGPU configuration is Linux, Python 3.10, and CUDA 12.8. Install `git-lfs`, `ffmpeg`, and [`uv`](https://docs.astral.sh/uv/), then use the locked project environment:

```bash
sudo apt-get install -y git-lfs ffmpeg
git lfs install
uv sync --python 3.10
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

Jetson Orin requires CUDA 12.6/Python 3.10; Jetson Thor and DGX Spark require CUDA 13.0/Python 3.12. Run the platform-specific installation scripts under `scripts/deployment/` for those targets. Simulation benchmarks have their own setup scripts and should remain separate from the main GR00T environment.
