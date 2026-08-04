#!/usr/bin/env python3
"""Validate the DynamicVLA v2.1 data-loading path without loading model weights.

DynamicVLA uses a custom LeRobot wrapper rather than the upstream v3 loader.
Run this against the v2.1 ``*_old`` dataset before submitting a training job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import easydict
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.datasets
import utils.helpers


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dynamicvla_level_level2.yaml"
DEFAULT_ROOT = Path("/projects/hdd/ssd/ICLR2027/dataset/factory_conveyor_level2_seeded_old")


def load_config(path: Path) -> easydict.EasyDict:
    return easydict.EasyDict(yaml.safe_load(path.read_text()))


def describe(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        finite = bool(torch.isfinite(value).all()) if value.is_floating_point() else True
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, finite={finite})"
    if isinstance(value, str):
        return repr(value)
    return f"{type(value).__name__}: {value!r}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_dataset(cfg: easydict.EasyDict, split: str):
    return utils.datasets.get_dataset(
        cfg.DATASET.NAME,
        root=cfg.DATASET.ROOT,
        split=split,
        pin_memory=False,
        delta_action=cfg.DATASET.USE_DELTA_ACTION,
        required_features=cfg.DATASET.REQUIRED_FEATURES,
        feature_aliases=cfg.DATASET.get("FEATURE_ALIASES"),
        image_transforms=utils.datasets.ImageTransforms(cfg.DATASET.IMG_SIZE),
        delta_timestamps=utils.helpers.get_delta_timestamps(cfg.POLICY, cfg.DATASET.DELTA_TIMESTAMPS),
    )


def check_item(item: dict[str, Any], required_features: list[str], label: str) -> None:
    print(f"{label} keys: {sorted(item)}")
    for key in [*required_features, "task"]:
        require(key in item, f"{label} is missing `{key}`")
        print(f"  {key}: {describe(item[key])}")
    require(isinstance(item["task"], str) and item["task"].strip(), f"{label} has no task text")
    require(item["observation.state"].shape[-1] == 10, f"{label} state is not 10D")
    require(item["action"].shape[-1] == 10, f"{label} action is not 10D")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.DATASET.ROOT = str(args.dataset_root)
    info_path, tasks_path = args.dataset_root / "meta" / "info.json", args.dataset_root / "meta" / "tasks.jsonl"
    require(info_path.exists(), f"Missing {info_path}")
    require(tasks_path.exists(), f"Missing {tasks_path}; DynamicVLA requires LeRobot v2.1 task metadata")
    info = json.loads(info_path.read_text())
    require(info["codebase_version"] == "v2.1", f"DynamicVLA requires v2.1; found {info['codebase_version']}")

    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    require(tasks and all(task.get("task") for task in tasks), "tasks.jsonl has missing language text")
    print(f"dataset root: {args.dataset_root}")
    print(f"LeRobot version: {info['codebase_version']}")
    print(f"episodes / frames: {info['total_episodes']} / {info['total_frames']}")
    print(f"tasks ({len(tasks)}):")
    for task in tasks:
        print(f"  [{task['task_index']}] {task['task']}")

    train_dataset = build_dataset(cfg, "train")
    test_dataset = build_dataset(cfg, "test")
    print(f"DynamicVLA split: train_episodes={len(train_dataset.episodes)}, test_episodes={len(test_dataset.episodes)}")
    print(f"DynamicVLA split: train_frames={len(train_dataset)}, test_frames={len(test_dataset)}")

    check_item(train_dataset[0], cfg.DATASET.REQUIRED_FEATURES, "train sample")
    check_item(test_dataset[0], cfg.DATASET.REQUIRED_FEATURES, "test sample")

    batch = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)))
    print(f"batch size: {args.batch_size}")
    for key in [*cfg.DATASET.REQUIRED_FEATURES, "task"]:
        print(f"  {key}: {describe(batch[key])}")
    require(batch["observation.state"].shape[-1] == 10, "Batched state is not 10D")
    require(batch["action"].shape[-1] == 10, "Batched action is not 10D")
    require(all(isinstance(task, str) and task.strip() for task in batch["task"]), "Batch has empty task text")
    print("PASS: DynamicVLA read v2.1 episodes, generated language tasks, and produced valid train/test batches.")


if __name__ == "__main__":
    main()
