"""Dataset adapter for MotionForge's local LeRobot-v2.1-style conveyor data.

The MotionForge export stores one Parquet file and one MP4 per camera per
episode.  Its metadata is close to LeRobot v2.1, but is not consumable by the
version of LeRobot pinned by OpenPI (notably, episode statistics contain scalar
metadata).  This adapter reads the files directly without modifying the source
dataset.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.common.datasets.video_utils import decode_video_frames


class FactoryConveyorDataset:
    """Random-access action-chunk dataset for a local MotionForge export."""

    _PARQUET_COLUMNS = ("timestamp", "task_index", "observation.state", "action")

    def __init__(
        self,
        root: str | Path,
        *,
        action_horizon: int,
        video_backend: str | None = "pyav",
        cache_episodes: int = 8,
    ):
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.video_backend = video_backend
        self.cache_episodes = cache_episodes

        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing MotionForge dataset metadata: {info_path}")
        self.info = json.loads(info_path.read_text())
        if self.info.get("codebase_version") != "v2.1":
            raise ValueError(
                "FactoryConveyorDataset expects the MotionForge LeRobot-v2.1-style export; "
                f"got {self.info.get('codebase_version')!r}."
            )
        self.fps = float(self.info["fps"])
        self.features = self.info["features"]
        self._validate_features()

        self.tasks = {
            int(item["task_index"]): item["task"]
            for item in self._read_jsonl(self.root / "meta" / "tasks.jsonl")
        }
        episodes = self._read_jsonl(self.root / "meta" / "episodes.jsonl")
        self._episode_lengths = {int(item["episode_index"]): int(item["length"]) for item in episodes}
        self._indices = [
            (episode_index, frame_index)
            for episode_index, length in self._episode_lengths.items()
            for frame_index in range(length)
        ]
        self._episode_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def _validate_features(self) -> None:
        expected = {
            "observation.state": ("float32", (10,)),
            "action": ("float32", (10,)),
            "observation.images.overview": ("video", (3, 240, 320)),
            "observation.images.wrist": ("video", (3, 160, 160)),
        }
        for key, (dtype, shape) in expected.items():
            feature = self.features.get(key)
            if feature is None or feature.get("dtype") != dtype or tuple(feature.get("shape", ())) != shape:
                raise ValueError(f"Unexpected or missing feature {key!r}: {feature}")

    def __len__(self) -> int:
        return len(self._indices)

    def _episode_path(self, episode_index: int) -> Path:
        return self.root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"

    def _video_path(self, episode_index: int, key: str) -> Path:
        return self.root / "videos" / f"chunk-{episode_index // 1000:03d}" / key / f"episode_{episode_index:06d}.mp4"

    def _load_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index in self._episode_cache:
            self._episode_cache.move_to_end(episode_index)
            return self._episode_cache[episode_index]

        table = pq.read_table(self._episode_path(episode_index), columns=list(self._PARQUET_COLUMNS))
        rows = table.to_pydict()
        episode = {
            "timestamp": np.asarray(rows["timestamp"], dtype=np.float32),
            "task_index": np.asarray(rows["task_index"], dtype=np.int64),
            "state": np.asarray(rows["observation.state"], dtype=np.float32),
            "action": np.asarray(rows["action"], dtype=np.float32),
        }
        expected_length = self._episode_lengths[episode_index]
        if len(episode["timestamp"]) != expected_length:
            raise ValueError(f"Episode {episode_index} has {len(episode['timestamp'])} frames; expected {expected_length}.")
        self._episode_cache[episode_index] = episode
        if len(self._episode_cache) > self.cache_episodes:
            self._episode_cache.popitem(last=False)
        return episode

    def _load_video_frame(self, episode_index: int, video_key: str, timestamp: float) -> np.ndarray:
        path = self._video_path(episode_index, video_key)
        if not path.is_file():
            raise FileNotFoundError(f"Missing video for episode {episode_index}: {path}")
        # The pinned LeRobot decoder returns a CPU float32 tensor in CHW format, [0, 1].
        frame = decode_video_frames(path, [float(timestamp)], 1.0 / self.fps, self.video_backend)[0]
        return frame.numpy()

    def __getitem__(self, index: int) -> dict:
        episode_index, frame_index = self._indices[index]
        episode = self._load_episode(episode_index)
        last = len(episode["action"]) - 1
        action_indices = np.minimum(np.arange(frame_index, frame_index + self.action_horizon), last)
        task_index = int(episode["task_index"][frame_index])
        if task_index not in self.tasks:
            raise KeyError(f"No task text for task_index={task_index}")
        timestamp = float(episode["timestamp"][frame_index])
        return {
            "overview": self._load_video_frame(episode_index, "observation.images.overview", timestamp),
            "wrist": self._load_video_frame(episode_index, "observation.images.wrist", timestamp),
            "state": episode["state"][frame_index],
            "actions": episode["action"][action_indices],
            "prompt": self.tasks[task_index],
        }
