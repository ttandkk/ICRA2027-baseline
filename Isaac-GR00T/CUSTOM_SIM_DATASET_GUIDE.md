# Custom Simulation and Dataset Integration Guide

This note summarizes the recommended path for connecting Isaac-GR00T to a custom
simulation environment and custom robot dataset in this project.

## 1. Prepare a Custom Dataset

The easiest route is to convert your data into LeRobot v2 format. A minimal
dataset should look like this:

```text
your_dataset/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.front/episode_000000.mp4
  meta/info.json
  meta/episodes.jsonl
  meta/tasks.jsonl
  meta/modality.json
  meta/stats.json
```

The important pieces are:

- `meta/modality.json`: declares the camera, state, action, and language keys.
- A Python modality config: maps dataset keys to GR00T modalities.
- Dataset statistics: required for state/action normalization.

Use H.264 videos if possible. In this cluster, the bundled DROID AV1 videos could
not be decoded by OpenCV on the GPU node, so `demo_data/droid_sample_h264` was
created as a working converted copy.

## 2. Define the Modality Config

For a new robot, register a config under `EmbodimentTag.NEW_EMBODIMENT`. Use
`examples/SO100/so100_config.py` as the closest template.

The config should define four top-level modalities:

- `video`: camera streams, usually `delta_indices=[0]`.
- `state`: proprioceptive state, usually `delta_indices=[0]`.
- `action`: action chunks, for example `delta_indices=list(range(0, 16))`.
- `language`: task instruction key.

Action configuration is especially important. For each action key, choose:

- `ActionRepresentation.RELATIVE` or `ABSOLUTE`.
- `ActionType.EEF` or `NON_EEF`.
- `ActionFormat.DEFAULT`, `XYZ_ROT6D`, or `XYZ_ROTVEC`.

The order of `action_configs` must match the order of `action.modality_keys`.

## 3. Generate or Regenerate Statistics

After creating the dataset and modality config, generate stats:

```bash
cd /projects/hdd/ssd/ICLR2027/baseline/Isaac-GR00T

uv run python gr00t/data/stats.py \
  --dataset-path /path/to/your_dataset \
  --embodiment-tag NEW_EMBODIMENT
```

Regenerate stats whenever action keys, state keys, or action horizon changes.
For example, changing action `delta_indices` from 16 steps to 8 steps requires
new `meta/relative_stats.json`.

## 4. Fine-Tune for a New Embodiment

If your robot/action space does not match one of the pretrained tags, do not
expect base-model zero-shot to work directly. Fine-tune with
`NEW_EMBODIMENT`:

```bash
cd /projects/hdd/ssd/ICLR2027/baseline/Isaac-GR00T

NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640 SAVE_STEPS=1000 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /path/to/your_dataset \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /path/to/your_config.py \
  --output-dir /path/to/output
```

For a smoke test, start with a tiny dataset of 3-5 episodes and a small number
of training steps before launching a full run.

## 5. Run Direct Policy Inference

The simplest way to connect a custom simulator is to write your own rollout loop
and call `Gr00tPolicy` directly.

```python
from gr00t.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

policy = Gr00tPolicy(
    model_path="/path/to/checkpoint",
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    device="cuda:0",
)

obs = {
    "video": {
        "front": front_rgb[None, None],  # (B, T, H, W, 3), uint8
    },
    "state": {
        "arm": arm_state[None, None],    # (B, T, D), float32
    },
    "language": {
        "annotation.human.task_description": [["pick up the cube"]],
    },
}

action, info = policy.get_action(obs)
next_action = action["arm"][:, 0, :]
```

Observation requirements:

- Video arrays: `np.uint8`, RGB, shape `(B, T, H, W, 3)`.
- State arrays: `np.float32`, shape `(B, T, D)`.
- Language: list of lists of strings, shape-like `(B, 1)`.

Actions are returned in physical units, not normalized values. Use the first
action step for receding-horizon control, or execute several steps from the
predicted action chunk.

## 6. Integrate with GR00T's Simulation Evaluation Framework

For a more standard closed-loop evaluation path, expose your simulator as a
Gymnasium environment and connect it to GR00T's server/client rollout stack.

Implementation points:

- Register your Gymnasium environment, e.g. `your_sim/task_name`.
- Add your env prefix to `gr00t/eval/sim/env_utils.py`.
- Add a factory path in `gr00t/eval/rollout_policy.py`.
- Ensure env observation/action keys match the checkpoint modality config.

Server:

```bash
cd /projects/hdd/ssd/ICLR2027/baseline/Isaac-GR00T

uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /path/to/checkpoint \
  --embodiment-tag NEW_EMBODIMENT \
  --use-sim-policy-wrapper
```

Client:

```bash
uv run python gr00t/eval/rollout_policy.py \
  --env-name your_sim/task_name \
  --n-episodes 10 \
  --n-action-steps 8 \
  --max-episode-steps 300
```

Use existing examples as references:

- LIBERO: `examples/LIBERO/README.md`
- DROID: `examples/DROID/README.md`
- SimplerEnv: `examples/SimplerEnv/README.md`
- SO100 custom config: `examples/SO100/so100_config.py`

## 7. Recommended Bring-Up Order

1. Convert 3-5 custom episodes to LeRobot v2.
2. Write `meta/modality.json` and a `NEW_EMBODIMENT` modality config.
3. Generate stats.
4. Run `standalone_inference_script.py` or a tiny direct `Gr00tPolicy` smoke test.
5. Fine-tune briefly and confirm checkpoint loading.
6. Connect the custom simulator using a direct rollout loop.
7. Only then wire it into the full GR00T server/client evaluation framework.

This order keeps failures easy to localize: dataset schema, video decoding,
state/action dimensions, action representation, checkpoint loading, and finally
closed-loop simulation.
