#!/usr/bin/env python
"""Validate the LeRobot dataset contract used by ``submit_train_act.sh``.

Examples
--------
From the same environment as the training job::

    conda run -p /projects/hdd/ssd/conda/envs/lerobot \
      python baseline/ACT/test_dataset.py

The script loads one real training sample with ACT's action horizon, validates
the state/action/image shapes, and checks that the ACT processor's action
normalization and post-processing are a numerical round trip.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


ACT_DIR = Path(__file__).resolve().parent
BASE_DIR = ACT_DIR.parent
LEROBOT_SRC = BASE_DIR / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.utils.feature_utils import dataset_to_policy_features  # noqa: E402


DEFAULT_DATASET_ROOT = BASE_DIR.parent / "dataset" / "factory_conveyor_level2_seeded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(os.getenv("DATASET_ROOT", DEFAULT_DATASET_ROOT)))
    parser.add_argument(
        "--dataset-repo-id",
        default=os.getenv("DATASET_REPO_ID", "local/factory_conveyor_level2_seeded"),
        help="Must match the repo id passed to lerobot-train.",
    )
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "100")))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def shape(value: torch.Tensor) -> tuple[int, ...]:
    require(isinstance(value, torch.Tensor), f"Expected a torch.Tensor, got {type(value).__name__}")
    return tuple(value.shape)


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    require(root.is_dir(), f"Dataset directory does not exist: {root}")
    require(args.chunk_size > 0, f"--chunk-size must be positive, got {args.chunk_size}")

    # This is exactly the future-action window requested by ACTConfig.action_delta_indices.
    # Keeping it explicit makes this script independent of a full training config.
    fps = 30
    action_delta_timestamps = [step / fps for step in range(args.chunk_size)]
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=root,
        delta_timestamps={ACTION: action_delta_timestamps},
        download_videos=False,
    )
    require(0 <= args.sample_index < len(dataset), f"--sample-index must be in [0, {len(dataset) - 1}]")
    sample = dataset[args.sample_index]

    features = dataset.meta.features
    policy_features = dataset_to_policy_features(features)
    expected_action_shape = tuple(features[ACTION]["shape"])
    expected_state_shape = tuple(features["observation.state"]["shape"])
    image_keys = sorted(key for key in features if key.startswith("observation.images."))

    # Dataset sample dimensions: state/image are one frame; action is an ACT future chunk.
    require(shape(sample["observation.state"]) == expected_state_shape,
            f"observation.state: expected {expected_state_shape}, got {shape(sample['observation.state'])}")
    require(shape(sample[ACTION]) == (args.chunk_size, *expected_action_shape),
            f"action: expected {(args.chunk_size, *expected_action_shape)}, got {shape(sample[ACTION])}")
    for key in image_keys:
        expected = tuple(features[key]["shape"])
        require(shape(sample[key]) == expected, f"{key}: expected {expected}, got {shape(sample[key])}")

    # Build the same ACT normalization mapping as training, without allocating the model.
    config = ACTConfig(device="cpu", chunk_size=args.chunk_size, n_action_steps=args.chunk_size)
    config.input_features = {key: value for key, value in policy_features.items() if key != ACTION}
    config.output_features = {ACTION: policy_features[ACTION]}
    preprocessor, postprocessor = make_act_pre_post_processors(config, dataset.meta.stats)

    # ``LeRobotDataset`` returns one action chunk as (T, D). During training,
    # PyTorch's DataLoader stacks that into (B, T, D); mirror that contract.
    training_batch = dict(sample)
    training_batch[ACTION] = sample[ACTION].unsqueeze(0)
    processed = preprocessor(training_batch)
    require(shape(processed["observation.state"]) == (1, *expected_state_shape),
            f"preprocessed state must have a batch dimension, got {shape(processed['observation.state'])}")
    require(shape(processed[ACTION]) == (1, args.chunk_size, *expected_action_shape),
            f"preprocessed action must be (1, {args.chunk_size}, {expected_action_shape}), got {shape(processed[ACTION])}")
    for key in image_keys:
        expected = (1, *tuple(features[key]["shape"]))
        require(shape(processed[key]) == expected, f"preprocessed {key}: expected {expected}, got {shape(processed[key])}")

    for key in ["observation.state", ACTION]:
        require(torch.isfinite(processed[key]).all().item(), f"Non-finite values after preprocessing: {key}")

    # ACT's postprocessor receives model-space actions, so feeding it the preprocessed
    # action directly exercises NormalizerProcessorStep and UnnormalizerProcessorStep together.
    restored_action = postprocessor(processed[ACTION])
    expected_action = training_batch[ACTION]
    max_error = (restored_action - expected_action).abs().max().item()
    require(
        torch.allclose(restored_action, expected_action, rtol=1e-5, atol=args.atol),
        f"action preprocess/postprocess round trip failed: max_abs_error={max_error:.3e}",
    )

    print("PASS: ACT dataset and processor checks succeeded")
    print(f"  dataset: {root} ({len(dataset)} frames, {dataset.meta.total_episodes} episodes, {dataset.meta.fps} fps)")
    print(f"  sample[{args.sample_index}] state: {shape(sample['observation.state'])}")
    print(f"  sample[{args.sample_index}] action: {shape(sample[ACTION])}")
    for key in image_keys:
        print(f"  sample[{args.sample_index}] {key}: {shape(sample[key])}")
    print(f"  preprocessed action: {shape(processed[ACTION])}")
    print(f"  action round-trip max abs error: {max_error:.3e}")


if __name__ == "__main__":
    main()
