#!/usr/bin/env python
"""Smoke test LeRobot v2 data against VLA-Adapter's action batch contract.

This intentionally avoids loading model weights or simulator dependencies. It reads a
few frames from a LeRobot v2 dataset, tokenizes actions with VLA-Adapter's
ActionTokenizer, and runs PaddedCollatorForActionPrediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from prismatic.models.backbones.llm.prompting.qwen_prompter import QwenPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX


class SmokeTokenizer:
    """Tiny tokenizer stub sufficient for ActionTokenizer and collator smoke tests."""

    vocab_size = 32000
    model_max_length = 256
    pad_token_id = 0
    padding_side = "right"

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(0xE000 + int(token_id)) for token_id in token_ids)

    def batch_decode(self, batch_token_ids: list[list[int]]) -> list[str]:
        return [self.decode(token_ids) for token_ids in batch_token_ids]

    def __call__(self, text: str, add_special_tokens: bool = True) -> Any:
        ids = [2] if add_special_tokens else []
        ids.extend((ord(ch) % (self.vocab_size - 256)) + 10 for ch in text)
        return type("Tokenized", (), {"input_ids": ids})()


def load_jsonl_map(path: Path, key: str, value: str) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[int(row[key])] = row[value]
    return out


def find_episode_files(dataset_root: Path) -> list[Path]:
    files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot v2 parquet episodes found under {dataset_root / 'data'}")
    return files


def read_video_frame(dataset_root: Path, video_key: str, episode_index: int, frame_index: int) -> Image.Image:
    chunk_index = episode_index // 1000
    video_path = dataset_root / "videos" / f"chunk-{chunk_index:03d}" / video_key / f"episode_{episode_index:06d}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video file: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        cap.release()


def image_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = image.resize((size, size), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def build_sample(
    dataset_root: Path,
    row: pd.Series,
    task_by_index: dict[int, str],
    tokenizer: SmokeTokenizer,
    action_tokenizer: ActionTokenizer,
    image_key: str,
    image_size: int,
) -> dict[str, Any]:
    episode_index = int(row["episode_index"])
    frame_index = int(row["frame_index"])
    task_index = int(row["task_index"])
    instruction = task_by_index.get(task_index, f"task {task_index}")

    action = np.asarray(row["action"], dtype=np.float32)
    proprio = np.asarray(row["observation.state"], dtype=np.float32)
    image = read_video_frame(dataset_root, image_key, episode_index, frame_index)
    pixel_values = image_to_tensor(image, image_size)

    action_text = action_tokenizer(action, use_minivlm=False)
    prompt_builder = QwenPromptBuilder("openvla")
    conversation = [
        {"from": "human", "value": f"What action should the robot take to {instruction.lower()}?"},
        {"from": "gpt", "value": action_text},
    ]
    for turn in conversation:
        prompt_builder.add_turn(turn["from"], turn["value"])

    input_ids = torch.tensor(tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[: -(len(action) + 1)] = IGNORE_INDEX

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels,
        "actions": action,
        "proprio": proprio,
        "dataset_name": dataset_root.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("fc_003_pick_moving_object_to_static_container"),
        help="Path to a local LeRobot v2 dataset root.",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--image-key", default="observation.images.front")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    task_by_index = load_jsonl_map(dataset_root / "meta" / "tasks.jsonl", "task_index", "task")
    episode_files = find_episode_files(dataset_root)

    tokenizer = SmokeTokenizer()
    action_tokenizer = ActionTokenizer(tokenizer)
    samples: list[dict[str, Any]] = []

    for episode_file in episode_files:
        df = pd.read_parquet(episode_file)
        if df.empty:
            continue
        stride = max(1, len(df) // max(1, args.num_samples))
        for _, row in df.iloc[::stride].iterrows():
            samples.append(
                build_sample(dataset_root, row, task_by_index, tokenizer, action_tokenizer, args.image_key, args.image_size)
            )
            if len(samples) >= args.num_samples:
                break
        if len(samples) >= args.num_samples:
            break

    if not samples:
        raise RuntimeError("No samples could be built from the dataset")

    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        padding_side=tokenizer.padding_side,
    )
    batch = collator(samples)

    print("LeRobot v2 smoke test passed")
    print(f"dataset_root: {dataset_root}")
    print(f"codebase_version: {info.get('codebase_version')}")
    print(f"total_episodes: {info.get('total_episodes')}, total_frames: {info.get('total_frames')}")
    print(f"image_key: {args.image_key}")
    print(f"pixel_values: {tuple(batch['pixel_values'].shape)} {batch['pixel_values'].dtype}")
    print(f"input_ids: {tuple(batch['input_ids'].shape)}")
    print(f"labels: {tuple(batch['labels'].shape)}")
    print(f"actions: {tuple(batch['actions'].shape)}")
    print(f"proprio: {None if batch['proprio'] is None else tuple(batch['proprio'].shape)}")
    print(f"dataset_names: {batch.get('dataset_names')}")


if __name__ == "__main__":
    main()
