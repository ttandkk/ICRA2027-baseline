#!/usr/bin/env python3
"""Validate the LeRobot v3 → VLA-Adapter data pipeline without model weights.

The test uses VLA-Adapter's actual LeRobot loader, action tokenizer, batch
transform and collator.  It verifies that task text, front/wrist images,
10-dimensional proprioception and 8x10 action chunks enter one training batch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.constants import IGNORE_INDEX, NUM_ACTIONS_CHUNK, NUM_TOKENS
    from prismatic.vla.datasets import LeRobotBatchTransform, LeRobotV3Dataset
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing VLA-Adapter dependency `{error.name}`. Activate the complete VLA-Adapter training environment "
        "(including timm, transformers, pandas, pyarrow, and opencv-python-headless) before running this test."
    ) from error


DEFAULT_ROOT = Path("/projects/hdd/ssd/ICLR2027/dataset/factory_conveyor_level2_seeded")


class TestTokenizer:
    """Minimal tokenizer implementing the contract used by ActionTokenizer."""

    vocab_size = 151643
    model_max_length = 256
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, text: str, add_special_tokens: bool = True):
        # Keep prompt tokens outside the action-token range, as a real tokenizer
        # would.  ActionTokenizer supplies the final 64 action token IDs.
        token_count = min(64, max(4, len(text.split()) + 2))
        return type("Tokenized", (), {"input_ids": list(range(10, 10 + token_count))})()


def image_transform(image) -> torch.Tensor:
    image = image.resize((224, 224))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-wrist", action="store_true", help="Test only the front camera path.")
    args = parser.parse_args()

    tokenizer = TestTokenizer()
    transform = LeRobotBatchTransform(
        action_tokenizer=ActionTokenizer(tokenizer),
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        use_wrist_image=not args.no_wrist,
        use_proprio=True,
        use_minivlm=True,
    )
    dataset = LeRobotV3Dataset(args.dataset_root, transform)
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        padding_side=tokenizer.padding_side,
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collator)))

    print(f"dataset root: {args.dataset_root}")
    print(f"examples: {len(dataset)}")
    print(f"tasks ({len(dataset.task_instructions)}):")
    for task_index, task in sorted(dataset.task_instructions.items()):
        print(f"  [{task_index}] {task}")
    print(f"dataset statistics: action={len(dataset.stats['action']['q01'])}D proprio={len(dataset.stats['observation.state']['q01'])}D")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            finite = bool(torch.isfinite(value).all()) if value.is_floating_point() else True
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype} finite={finite}")
        else:
            print(f"{key}: {value}")

    require(len(dataset.task_instructions) == 10, "Expected 10 language tasks")
    require(batch["actions"].shape == (args.batch_size, NUM_ACTIONS_CHUNK, 10), "Expected a Bx8x10 action batch")
    require(batch["proprio"].shape == (args.batch_size, 10), "Expected a Bx10 proprio batch")
    expected_channels = 3 if args.no_wrist else 6
    require(batch["pixel_values"].shape == (args.batch_size, expected_channels, 224, 224), "Unexpected image batch shape")
    require(batch["input_ids"].shape == batch["labels"].shape, "Input/label shape mismatch")
    require(int(batch["input_ids"].max()) < tokenizer.vocab_size, "Token ID exceeds tokenizer vocabulary")
    action_labels = (batch["labels"] != IGNORE_INDEX).sum(dim=1)
    require(bool(torch.all(action_labels >= NUM_TOKENS)), "Action labels were truncated")
    require(bool(torch.isfinite(batch["actions"]).all()), "Non-finite actions")
    require(bool(torch.isfinite(batch["proprio"]).all()), "Non-finite proprioception")
    print("PASS: LeRobot v3 data enters the VLA-Adapter loader, tokenizer, image transform, and collator correctly.")


if __name__ == "__main__":
    main()
