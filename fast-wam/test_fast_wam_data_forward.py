#!/usr/bin/env python
"""Inspect one FastWAM training batch and execute a no-gradient forward pass.

This script mirrors the data path used by ``submit_fast_wam_train.sh``:
LeRobot v3 dataset -> PyAV temporal decoding -> collation -> float conversion ->
FastWAM preprocessor -> camera concatenation/text conditioning -> loss.

Run on a GPU with enough free memory (the full FastWAM model is about 6B parameters):

    cd /projects/haitian003ssd/ICRA2027-baseline/lerobot
    /projects/haitian003ssd/ICRA2027-baseline/.conda/lerobot-baselines/bin/python \
      ../fast-wam/test_fast_wam_data_forward.py --device cuda:0
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.utils.collate import lerobot_collate_fn


DEFAULT_ROOT = "/projects/haitian003ssd/Dataset/factory_conveyor_level2_seeded"
DEFAULT_REPO_ID = "local/factory_conveyor_level2_seeded"
DEFAULT_CAMERAS = (
    "observation.images.overview",
    "observation.images.front",
    "observation.images.wrist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 672), metavar=("H", "W"))
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec", "video_reader"))
    parser.add_argument("--pretrained-path", default="lerobot/fastwam_base")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--data-only", action="store_true", help="Stop after data loading and uint8-to-float conversion; do not load the model.")
    return parser.parse_args()


def print_value(name: str, value: Any) -> None:
    """Print a compact, deterministic summary without dumping large tensors."""
    if isinstance(value, torch.Tensor):
        finite = bool(torch.isfinite(value).all()) if value.is_floating_point() else True
        value_range = "n/a"
        if value.numel() and (value.is_floating_point() or value.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
            value_range = f"[{value.min().item():.5g}, {value.max().item():.5g}]"
        print(f"  {name}: tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device} finite={finite} range={value_range}")
    elif isinstance(value, (list, tuple)):
        preview = list(value[:2]) if value else []
        print(f"  {name}: {type(value).__name__} len={len(value)} preview={preview!r}")
    else:
        print(f"  {name}: {type(value).__name__} value={value!r}")


def print_batch(title: str, batch: Mapping[str, Any]) -> None:
    print(f"\n=== {title} ===")
    for key in sorted(batch):
        print_value(key, batch[key])


def make_config(args: argparse.Namespace, dataset: LeRobotDataset) -> FastWAMConfig:
    image_height, image_width = args.image_size
    if image_width % len(DEFAULT_CAMERAS):
        raise ValueError("Image width must be divisible by the number of cameras.")
    features = dataset.meta.features
    missing = [key for key in DEFAULT_CAMERAS if key not in features]
    if missing:
        raise ValueError(f"Dataset is missing expected cameras: {missing}")
    action_dim = int(features["action"]["shape"][0])
    proprio_dim = int(features["observation.state"]["shape"][0])
    per_camera_width = image_width // len(DEFAULT_CAMERAS)
    input_features: dict[str, PolicyFeature] = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(proprio_dim,)),
    }
    input_features.update(
        {
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_height, per_camera_width))
            for key in DEFAULT_CAMERAS
        }
    )
    return FastWAMConfig(
        pretrained_path=args.pretrained_path,
        model_id=args.model_id,
        device=args.device,
        torch_dtype="bfloat16",
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        action_horizon=args.action_horizon,
        n_action_steps=10,
        image_size=tuple(args.image_size),
        input_features=input_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        use_gradient_checkpointing=False,
    )


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    if not root.joinpath("meta", "info.json").is_file():
        raise FileNotFoundError(f"No LeRobot metadata at {root}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    offsets = [index / 30 for index in range(0, 33, 4)]
    delta_timestamps = {
        "observation.state": offsets,
        **{camera: offsets for camera in DEFAULT_CAMERAS},
        "action": [index / 30 for index in range(args.action_horizon)],
    }
    print("=== Dataset construction ===")
    print(f"root={root}\nrepo_id={args.dataset_repo_id}\nvideo_backend={args.video_backend}\ndelta_timestamps={delta_timestamps}")
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    print(f"frames={len(dataset)} episodes={dataset.num_episodes} fps={dataset.meta.fps}")

    raw_sample = dataset[args.sample_index]
    print_batch(f"Raw dataset sample index={args.sample_index}", raw_sample)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=[args.sample_index + offset for offset in range(args.batch_size)],
        num_workers=args.num_workers,
        collate_fn=lerobot_collate_fn,
        pin_memory=args.device.startswith("cuda"),
    )
    batch = next(iter(loader))
    assert batch is not None
    print_batch("Collated batch (before image conversion)", batch)

    for camera in dataset.meta.camera_keys:
        if camera in batch and batch[camera].dtype == torch.uint8:
            batch[camera] = batch[camera].to(dtype=torch.float32).div_(255.0)
    print_batch("Collated batch (after uint8 -> float [0, 1])", batch)

    if args.data_only:
        print("\nData-only check completed successfully (model loading skipped).")
        return

    config = make_config(args, dataset)
    print("\n=== FastWAM configuration ===")
    print(f"action_dim={config.action_dim} proprio_dim={config.proprio_dim} image_size={config.image_size}")
    print(f"observation_delta_indices={config.observation_delta_indices}")
    print(f"action_delta_indices={config.action_delta_indices}")
    print(f"configured_camera_order={list(config.image_features)}")

    print("\n=== Loading policy and training preprocessor ===")
    started = time.perf_counter()
    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=config.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": torch.device(args.device).type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**config.input_features, **config.output_features},
                "norm_map": config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": config.output_features,
                "norm_map": config.normalization_mapping,
            },
        },
    )
    print(f"policy/preprocessor load time: {time.perf_counter() - started:.2f}s")
    processed = preprocessor(batch)
    print_batch("After training preprocessor", processed)
    policy.eval()
    torch.cuda.reset_peak_memory_stats() if args.device.startswith("cuda") else None
    print("\n=== FastWAM native sample ===")
    with torch.inference_mode():
        native_sample = policy._batch_to_training_sample(processed)
        for key in ("video", "image_is_pad", "action", "action_is_pad", "proprio", "context", "context_mask"):
            if key in native_sample:
                print_value(key, native_sample[key])
        started = time.perf_counter()
        loss, metrics = policy.model.training_loss(native_sample)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(args.device))
        elapsed = time.perf_counter() - started
    print("\n=== Forward result ===")
    print(f"loss={loss.item():.6f} elapsed_s={elapsed:.3f}")
    print(f"metrics={metrics}")
    if args.device.startswith("cuda"):
        print(f"peak_cuda_memory_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}")


if __name__ == "__main__":
    main()
