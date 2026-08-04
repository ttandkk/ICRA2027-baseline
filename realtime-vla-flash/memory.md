# Project Memory

## Repository And Environments

- Repository root: `/projects/hdd/ssd/ICRA2027/baseline/realtime-vla-flash`
- Main server environment: `.venv` using Python 3.11.15.
- LIBERO client environment: `examples/libero/.venv` using Python 3.8.20.
- Always use `UV_CACHE_DIR=/projects/hdd/ssd` to avoid filling Home.
- Load `git-lfs/3.6.1` before Git/LFS operations.
- The main environment contains PyTorch 2.7.1+cu126, Triton 3.3.1, JAX 0.5.3, and the required patched Transformers 4.53.2.

Typical server setup:

```bash
cd /projects/hdd/ssd/ICRA2027/baseline/realtime-vla-flash
module load git-lfs/3.6.1
export UV_CACHE_DIR=/projects/hdd/ssd
source .venv/bin/activate
```

## Important Paths

- OpenPI JAX base checkpoint:
  `data/checkpoints/openpi/openpi-assets/checkpoints/pi0_libero`
- Official draft checkpoints:
  `data/checkpoints/draft/draft_libero_{spatial,object,goal,10}.pt`
- Converted shared Triton base:
  `data/triton/pi0_libero_goal/base/base_weights.pkl`
- Converted Goal draft:
  `data/triton/pi0_libero_goal/draft/draft_triton.pkl`
- Project-specific LIBERO config:
  `data/libero_config/config.yaml`

Use the project-specific LIBERO config instead of `~/.libero/config.yaml`. The global config points to an old SimVLA path.

```bash
export LIBERO_CONFIG_PATH=/projects/hdd/ssd/ICRA2027/baseline/realtime-vla-flash/data/libero_config
```

## Quick Start Details

The official Quick Start works, but its checkpoint paths are placeholders:

- The official Dexmal Hugging Face repository provides the four draft checkpoints.
- The base checkpoint must be downloaded from `gs://openpi-assets/checkpoints/pi0_libero`.
- `scripts/spec/pi0_benchmark.py` uses random micro weights. It measures kernels but is not a real model/LIBERO benchmark.

Real Triton conversion:

```bash
uv run scripts/spec/triton/convert_for_triton.py \
  --mode base \
  --jax-path data/checkpoints/openpi/openpi-assets/checkpoints/pi0_libero \
  --output data/triton/pi0_libero_goal/base

uv run scripts/spec/triton/convert_for_triton.py \
  --mode draft \
  --draft-ckpt data/checkpoints/draft/draft_libero_goal.pt \
  --output data/triton/pi0_libero_goal/draft
```

## Verified Results

Real LIBERO Goal task 0 was successfully completed:

- Task: `open the middle drawer of the cabinet`
- Success: 1/1
- Stable FLASH `sample_actions` latency after Triton compilation:
  mean `7.471 ms`, median `7.469 ms`, range `7.450-7.504 ms`
- Stable client roundtrip mean: `10.018 ms`
- Stable stage means: encoder `3.993 ms`, draft `0.938 ms`, verify `2.051 ms`

Result paths:

- `results/real_libero_goal/official_client_79062/episode_log.json`
- `results/real_libero_goal/official_client_79062/episodes/task00_ep000_open_the_middle_drawer_of_the_cabinet_success/`

The first full and draft calls are slow because Triton compiles kernels. Exclude these calls when reporting steady-state latency.

## Cluster And Evaluation

- Slurm partition: `cluster02`
- `6000ada` gives approximately 46 GB GPU memory and 64 GB host memory.
- Cluster policy may override requested CPU and host memory.
- Complete four-suite evaluation script:
  `jobs/realtime_vla_flash_full_libero_eval.sh`
- Complete evaluation covers:
  `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`;
  10 tasks per suite, 50 rollouts per task, 2000 episodes total.
- Current full evaluation job at the time this file was written:
  Job `79066`, results in `results/full_libero_79066`.

Useful monitoring:

```bash
squeue -j 79066
tail -f logs/flash_full_libero_79066.log
```

Each suite continuously writes an `episode_log.json`, so progress and partial success rates remain available while the job runs.

## Known Issues

- The old Conda environment `/projects/hdd/conda_envs/libero` is not usable for this repository's client: it lacks the installed LIBERO package and has OpenGL library issues.
- EGL cleanup can print `libGLU.so.0` or `EGL_NOT_INITIALIZED` errors after an episode. These appeared after a successful rollout and did not invalidate the result.
- A Slurm job can be marked failed by a bad final `cat` command even when the rollout itself succeeded. Check `episode_log.json` and generated videos before treating the evaluation as failed.
