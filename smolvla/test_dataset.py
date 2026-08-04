#!/usr/bin/env python
"""Validate the local factory-conveyor LeRobot v3 dataset for SmolVLA training.

This is intentionally model-free: it verifies the exact dataset object consumed
by ``lerobot-train`` and prints a concise description of a real batch.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_ROOT = "/projects/hdd/ssd/ICLR2027/dataset/factory_conveyor_level2_seeded"
DEFAULT_REPO_ID = "local/factory_conveyor_level2_seeded"
REQUIRED_FEATURES = ("observation.images.front", "observation.images.wrist", "observation.state", "action")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--dataset-repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.dataset_repo_id, root=args.dataset_root)
    meta = dataset.meta
    print(f"dataset root: {args.dataset_root}")
    print(f"LeRobot version: {meta.info.codebase_version}")
    print(f"episodes / frames: {meta.total_episodes} / {meta.total_frames}")
    print(f"fps: {meta.fps}")
    print(f"features: {list(meta.features)}")
    print(f"tasks ({len(meta.tasks)}):")
    for task_index, task in meta.tasks.iterrows():
        print(f"  [{int(task.task_index)}] {task_index}")

    missing = set(REQUIRED_FEATURES) - set(meta.features)
    require(not missing, f"Missing required features: {sorted(missing)}")
    require(meta.info.codebase_version == "v3.0", "Expected a LeRobot v3.0 dataset")
    require(len(meta.tasks) > 0, "Dataset has no language tasks")

    sample_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for index in sample_indices:
        sample = dataset[index]
        print(f"sample[{index}]:")
        for key in (*REQUIRED_FEATURES, "task"):
            require(key in sample, f"sample[{index}] is missing `{key}`")
            print(f"  {key}: {describe(sample[key])}")
        require(tuple(sample["action"].shape) == (10,), "Expected a 10D action")
        require(tuple(sample["observation.state"].shape) == (10,), "Expected a 10D robot state")
        require(isinstance(sample["task"], str) and sample["task"].strip(), "Task text is empty")

    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)))
    require(isinstance(batch, Mapping), "Dataloader did not return a mapping")
    print(f"batch size: {args.batch_size}")
    for key in (*REQUIRED_FEATURES, "task"):
        print(f"  {key}: {describe(batch[key])}")
    require(tuple(batch["action"].shape) == (args.batch_size, 10), "Unexpected batched action shape")
    require(tuple(batch["observation.state"].shape) == (args.batch_size, 10), "Unexpected batched state shape")
    print("PASS: dataset is readable and has task text, two camera streams, 10D state, and 10D action.")


if __name__ == "__main__":
    main()
