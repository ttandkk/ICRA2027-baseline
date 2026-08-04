# Local Modifications

This file records local changes made after deploying `hzxie/DynamicVLA` into this workspace.

## Scope

Repository root:
`/projects/hdd/ssd/ICLR2027/baseline/DynamicVLA`

Record date:
`2026-07-06`

## Summary

Local code/config changes relative to upstream:
- `core/train.py`
- `core/test.py`
- `utils/datasets.py`
- `utils/distributed.py`
- `utils/instruction_generator.py`
- `configs/dynamicvla_level_level2.yaml` (new)
- `scripts/smoke_test_level_level2.py` (new)
- `scripts/submit_train_level_level2.sh` (new)
- `scripts/submit_smoke_test_level_level2.sh` (new)

Local non-code additions:
- `data/` contains user-provided local datasets and extracted files
- `pretrained/dynamic-vla-DOM/` contains the downloaded pretrained checkpoint from Hugging Face

## Detailed Changes

### 1. `utils/distributed.py`
Purpose:
Make the training code tolerant to machines where NVML is unavailable.

Changes:
- Replaced unconditional `pynvml.nvmlInit()` with guarded initialization.
- Added `_NVML_AVAILABLE` flag.
- Made `Device.__init__` fail gracefully when NVML is absent.
- Made `set_affinity()` fall back to the current CPU affinity mask when NVML is absent.
- Made `is_local_master()` return `True` when CUDA is unavailable.

Reason:
The original code crashed at import time on this machine because `libnvidia-ml.so.1` was not available.

### 2. `utils/datasets.py`
Purpose:
Allow DynamicVLA to read the local `level_level2` LeRobot v2 dataset directly and stay compatible with the exported metadata format.

Changes:
- Added a `root` argument to `get_dataset(...)`.
- Passed `root` through to `LeRobotDataset(...)` so the dataset can be loaded from an explicit local path.
- Added a local compatibility layer for LeRobot stats aggregation:
  - `_normalize_lerobot_stats(...)`
  - `_patch_lerobot_stats_compatibility()`
- Patched stats aggregation to normalize scalar values into array form before LeRobot aggregates them.
- Patched `LeRobotDatasetMetadata.get_data_file_path()` and `get_video_file_path()` compatibility so both `{episode_chunk}` and `{chunk_index}` templates work.

Reason:
The local dataset at
`/projects/hdd/ssd/ICLR2027/baseline/DynamicVLA/data/level_level2/level_level2`
uses a valid LeRobot v2 structure, but its metadata formatting differs from what the installed `lerobot==0.3.3` code expects.

### 3. `core/train.py`
Purpose:
Wire local dataset roots into training.

Changes:
- Updated the training dataset construction call to pass `root=cfg.DATASET.get("ROOT")`.
- Updated the testing dataset construction call inside the training pipeline to pass `root=cfg.DATASET.get("ROOT")`.

Reason:
Without this change, training only looked for datasets in the default Hugging Face LeRobot cache location.

### 4. `core/test.py`
Purpose:
Keep the standalone test/evaluation path aligned with local dataset loading.

Changes:
- Updated test dataset construction to pass `root=cfg.DATASET.get("ROOT")`.

Reason:
Without this change, standalone `core.test(...)` could still ignore the local dataset root even after `core.train.py` was patched.

### 5. `utils/instruction_generator.py`
Purpose:
Support datasets whose task metadata already contains plain natural-language instructions.

Changes:
- Wrapped `json.loads(...)` in a `try/except`.
- If task metadata is a plain string, return it directly.
- If task metadata is a dict with a non-template task string, return that string directly.
- Kept the original template-based generation path for `pick`, `place`, and `long-horizon`.

Reason:
The local `tasks.jsonl` stores full text instructions such as:
`Pick up the first cardboard package from the conveyor and place it into the box...`
The original code assumed every task string was JSON-formatted metadata for template expansion.

### 6. `configs/dynamicvla_level_level2.yaml` (new file)
Purpose:
Provide a dataset-specific training config for the local `level_level2` dataset.

Key settings:
- `DATASET.NAME: level_level2`
- `DATASET.ROOT: /projects/hdd/ssd/ICLR2027/baseline/DynamicVLA/data/level_level2/level_level2`
- Uses local camera keys:
  - `observation.images.overview`
  - `observation.images.front`
  - `observation.images.wrist`
- Keeps DynamicVLA training defaults otherwise, except for local wandb settings.

Reason:
The upstream default config expected different camera keys and a different dataset source.

### 7. `pretrained/dynamic-vla-DOM/` (new directory)
Purpose:
Store the upstream pretrained DynamicVLA checkpoint locally for finetuning and smoke testing.

Contents:
- `config.json`
- `model.safetensors`
- model card / license files

Reason:
This checkpoint is used as the initialization source for local finetuning on `level_level2`.

### 8. `scripts/submit_train_level_level2.sh` (new file)
Purpose:
Provide a Slurm submission script for finetuning on `level_level2`.

Changes:
- Requests one `6000ada` GPU.
- Activates the local conda environment.
- Runs `run.py` with `configs/dynamicvla_level_level2.yaml`.
- Initializes from `pretrained/dynamic-vla-DOM`.

### 9. `scripts/smoke_test_level_level2.py` (new file)
Purpose:
Provide a reusable smoke test covering config alignment, dataset integrity, batch loading, policy construction, and checkpoint loading.

Checks performed by the script:
- Compare local config against upstream `configs/dynamicvla.yaml`.
- Validate dataset root and required metadata files.
- Validate required dataset features.
- Load one real training batch.
- Build a DynamicVLA policy using local dataset metadata.
- Run one CPU forward pass.
- Optionally load the pretrained checkpoint.
- Report wandb mode.

Important fix in this script:
- The checkpoint-loading path was changed to use `load_dynamicvla(...)` instead of a naive `from_pretrained(...)` path.
- This avoids loading mismatched normalization buffers from the DOM checkpoint when testing against the local `10D state / 10D action` dataset.

Reason:
A direct checkpoint load initially failed with shape mismatches in the normalizer buffers because the DOM checkpoint was built for `6D state / 7D action`, while the local dataset uses `10D state / 10D action`.

### 10. `scripts/submit_smoke_test_level_level2.sh` (new file)
Purpose:
Provide a Slurm submission script for the smoke test.

Changes:
- Requests one `6000ada` GPU.
- Activates the local conda environment.
- Runs `scripts/smoke_test_level_level2.py --load-checkpoint`.

## Validation Performed

The following checks were completed after the above changes:
- Training environment created at:
  `/projects/hdd/ssd/ICLR2027/baseline/.conda/dynamicvla-train310`
- Installed core packages:
  - `torch 2.7.1+cu126`
  - `torchvision 0.22.1+cu126`
  - `transformers 5.2.0`
  - `lerobot 0.3.3`
- Adjusted FFmpeg to a TorchCodec-compatible version:
  - `ffmpeg 7.1.1`
- Verified `python run.py --help` works.
- Verified local dataset loading works with the new config.
- Verified smoke test passes both without and with pretrained checkpoint loading.

Observed dataset smoke-test output:
- train samples: `8967`
- train episodes: `49`
- test episodes: `1`
- sample keys include:
  - `observation.images.overview`
  - `observation.images.front`
  - `observation.images.wrist`
  - `observation.state`
  - `action`
  - `task`
- successful checkpoint-backed smoke test output includes:
  - `loaded_checkpoint=.../pretrained/dynamic-vla-DOM`
  - `SMOKE TEST PASSED`

## Current Commands

Training:
```bash
conda activate /projects/hdd/ssd/ICLR2027/baseline/.conda/dynamicvla-train310
cd /projects/hdd/ssd/ICLR2027/baseline/DynamicVLA
torchrun --nnodes=1 --nproc_per_node=1 --standalone run.py   -c configs/dynamicvla_level_level2.yaml   -p /projects/hdd/ssd/ICLR2027/baseline/DynamicVLA/pretrained/dynamic-vla-DOM
```

Smoke test:
```bash
cd /projects/hdd/ssd/ICLR2027/baseline/DynamicVLA
conda run -p /projects/hdd/ssd/ICLR2027/baseline/.conda/dynamicvla-train310   python scripts/smoke_test_level_level2.py --load-checkpoint
```

## Notes

- This file records local workspace changes only; it is not part of the upstream DynamicVLA repository.
- The local dataset uses `10D state / 10D action`, while the downloaded DOM checkpoint uses `6D state / 7D action` in its saved feature definition.
- Because of that mismatch, checkpoint normalizer buffers must not be loaded blindly across datasets.
- If you make more local changes later, append them here with date, file path, purpose, and reason.
