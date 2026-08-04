"""LeRobot v3 dataset support for VLA-Adapter fine-tuning."""

from __future__ import annotations

import bisect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from prismatic.models.backbones.llm.prompting import QwenPromptBuilder
from prismatic.vla.constants import IGNORE_INDEX, NUM_ACTIONS_CHUNK, NUM_TOKENS


def _load_task_instructions(path: Optional[Path]) -> Dict[int, str]:
    """Load an optional ``{task_index: instruction}`` JSON mapping."""
    if path is None:
        return {}
    return {int(key): str(value) for key, value in json.loads(path.read_text(encoding="utf-8")).items()}


def _load_dataset_task_instructions(path: Path) -> Dict[int, str]:
    """Read LeRobot v3 ``tasks.parquet``.

    LeRobot stores the language text in the dataframe index (named ``task``)
    and the numeric identifier in the ``task_index`` column.
    """
    if not path.exists():
        return {}
    tasks = pd.read_parquet(path)
    if "task_index" not in tasks.columns:
        raise ValueError(f"LeRobot tasks file is missing `task_index`: {path}")
    if "task" in tasks.columns:
        task_texts = tasks["task"]
    elif tasks.index.name == "task":
        task_texts = tasks.index.to_series(index=tasks.index)
    else:
        raise ValueError(f"LeRobot tasks file has no task text column or `task` index: {path}")
    return {int(task_index): str(task) for task, task_index in zip(task_texts, tasks["task_index"], strict=True)}


class _VideoReader:
    """Seek into LeRobot's consecutively packed MP4 files."""

    def __init__(self, root: Path, video_key: str) -> None:
        self.paths = sorted((root / "videos" / video_key).glob("chunk-*/file-*.mp4"))
        if not self.paths:
            raise FileNotFoundError(f"No videos found for `{video_key}` under {root / 'videos'}")
        self.ends: list[int] = []
        total = 0
        for path in self.paths:
            capture = cv2.VideoCapture(str(path))
            try:
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()
            if frames <= 0:
                raise RuntimeError(f"Could not determine frame count for {path}")
            total += frames
            self.ends.append(total)
        self.capture: Optional[cv2.VideoCapture] = None
        self.open_path: Optional[Path] = None

    def read(self, global_frame_index: int) -> Image.Image:
        file_index = bisect.bisect_right(self.ends, int(global_frame_index))
        if file_index >= len(self.paths):
            raise IndexError(f"Video frame {global_frame_index} is outside the packed video streams")
        path = self.paths[file_index]
        local_index = int(global_frame_index) - (self.ends[file_index - 1] if file_index else 0)
        if self.open_path != path:
            if self.capture is not None:
                self.capture.release()
            self.capture = cv2.VideoCapture(str(path))
            self.open_path = path
        assert self.capture is not None
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, local_index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {local_index} from {path}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


@dataclass
class LeRobotBatchTransform:
    action_tokenizer: Any
    base_tokenizer: Any
    image_transform: Callable[[Image.Image], torch.Tensor]
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_minivlm: bool = True

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        actions = sample["actions"]
        token_groups = [self.action_tokenizer(actions[0], self.use_minivlm)]
        token_groups.extend(self.action_tokenizer(actions[1:], self.use_minivlm))
        action_tokens = [token for group in token_groups for token in group]
        if not action_tokens:
            raise ValueError("LeRobot sample contains no action tokens")
        if len(action_tokens) >= NUM_TOKENS:
            action_tokens = action_tokens[:NUM_TOKENS]
        else:
            action_tokens += random.choices(action_tokens, k=NUM_TOKENS - len(action_tokens))

        prompt_builder = QwenPromptBuilder("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {sample['instruction'].lower()}?")
        prompt_builder.add_turn("gpt", "")
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        if len(input_ids) >= 3:
            del input_ids[-3:]
        input_ids = input_ids + action_tokens
        labels = torch.tensor(input_ids, dtype=torch.long)
        labels[: -(NUM_TOKENS + 1)] = IGNORE_INDEX

        result: Dict[str, Any] = {
            "pixel_values": self.image_transform(sample["image"]),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": labels,
            "dataset_name": sample["dataset_name"],
            "actions": actions,
        }
        if self.use_wrist_image:
            if sample.get("wrist_image") is None:
                raise ValueError("num_images_in_input > 1 requires a valid LeRobot wrist camera key")
            result["pixel_values_wrist"] = self.image_transform(sample["wrist_image"])
        if self.use_proprio:
            result["proprio"] = sample["proprio"]
        return result


class LeRobotV3Dataset(Dataset):
    """Map-style LeRobot v3 dataset yielding VLA-Adapter training samples."""

    def __init__(
        self,
        dataset_root: Path,
        batch_transform: LeRobotBatchTransform,
        image_key: str = "observation.images.front",
        wrist_image_key: str = "observation.images.wrist",
        task_instruction: Optional[str] = None,
        task_instructions_path: Optional[Path] = None,
    ) -> None:
        self.root = Path(dataset_root)
        self.batch_transform = batch_transform
        self.name = self.root.name
        info_path, stats_path = self.root / "meta" / "info.json", self.root / "meta" / "stats.json"
        if not info_path.exists() or not stats_path.exists():
            raise FileNotFoundError("Expected LeRobot v3 meta/info.json and meta/stats.json")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError(f"Expected a LeRobot v3 dataset, got {self.info.get('codebase_version')!r}")
        self.stats = json.loads(stats_path.read_text(encoding="utf-8"))

        parquet_paths = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No LeRobot parquet files found under {self.root / 'data'}")
        self.frames = pd.concat((pd.read_parquet(path) for path in parquet_paths), ignore_index=True)
        required = {"index", "episode_index", "task_index", "action", "observation.state"}
        missing = required - set(self.frames.columns)
        if missing:
            raise ValueError(f"LeRobot parquet files are missing columns: {sorted(missing)}")
        self.frames.sort_values("index", inplace=True, ignore_index=True)
        self.episode_end_rows = {
            int(episode): int(rows[-1])
            for episode, rows in self.frames.groupby("episode_index", sort=False).indices.items()
        }

        self.primary_reader = _VideoReader(self.root, image_key)
        self.wrist_reader = _VideoReader(self.root, wrist_image_key) if batch_transform.use_wrist_image else None
        self.task_instruction = task_instruction
        self.task_instructions = _load_dataset_task_instructions(self.root / "meta" / "tasks.parquet")
        self.task_instructions.update(_load_task_instructions(task_instructions_path))
        self.dataset_statistics = {self.name: {"action": self.stats["action"], "proprio": self.stats["observation.state"]}}
        self.action_low = np.asarray(self.stats["action"]["q01"], dtype=np.float32)
        self.action_high = np.asarray(self.stats["action"]["q99"], dtype=np.float32)
        self.proprio_low = np.asarray(self.stats["observation.state"]["q01"], dtype=np.float32)
        self.proprio_high = np.asarray(self.stats["observation.state"]["q99"], dtype=np.float32)

    @staticmethod
    def _normalize(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return np.clip(2 * (values - low) / (high - low + 1e-8) - 1, -1, 1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, row: int) -> Dict[str, Any]:
        current = self.frames.iloc[row]
        episode_end = self.episode_end_rows[int(current["episode_index"])]
        action_rows = self.frames.iloc[[min(row + offset, episode_end) for offset in range(NUM_ACTIONS_CHUNK)]]
        actions = np.stack(action_rows["action"].map(lambda x: np.asarray(x, dtype=np.float32)).to_list())
        task_index = int(current["task_index"])
        return self.batch_transform(
            {
                "actions": self._normalize(actions, self.action_low, self.action_high),
                "proprio": self._normalize(np.asarray(current["observation.state"], dtype=np.float32), self.proprio_low, self.proprio_high),
                "image": self.primary_reader.read(int(current["index"])),
                "wrist_image": self.wrist_reader.read(int(current["index"])) if self.wrist_reader is not None else None,
                "instruction": self.task_instruction or self.task_instructions.get(task_index, f"complete factory conveyor task {task_index}"),
                "dataset_name": self.name,
            }
        )
