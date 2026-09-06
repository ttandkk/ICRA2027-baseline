#!/usr/bin/env python3
"""Bridge MotionForge benchmark observations to a LeRobot SmolVLA policy."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

_motionforge_root = Path(
    os.environ.get(
        "MOTIONFORGE_ROOT",
        Path(__file__).resolve().parents[2] / "MotionForge",
    )
)
_motionforge_source = str(_motionforge_root / "source" / "motionforge")
if _motionforge_source not in sys.path:
    sys.path.insert(0, _motionforge_source)

from motionforge.benchmark.client import BenchmarkClientBridge, ClientBridgeConfig
from motionforge.benchmark.protocol import ObservationRequest, ResetMessage

IMAGE_KEYS = (
    "observation.images.overview",
    "observation.images.front",
    "observation.images.wrist",
)
STATE_KEY = "observation.state"
TASK_KEY = "task"


def seed_policy_rng(seed: int) -> None:
    if not 0 <= int(seed) < 2**63 - 1:
        raise ValueError(f"RESET seed must be in [0, 2**63 - 2], got {seed}.")
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass(slots=True)
class SmolVLAInference:
    """SmolVLA model plus the processors serialized with its checkpoint."""

    checkpoint: Path
    device: str

    config: Any = field(init=False)
    policy: Any = field(init=False)
    preprocessor: Any = field(init=False)
    postprocessor: Any = field(init=False)

    def __post_init__(self) -> None:
        self.checkpoint = self.checkpoint.expanduser().resolve()
        validate_checkpoint_files(self.checkpoint)
        self.config = load_smolvla_config(self.checkpoint, self.device)

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        # The local checkpoint contains the complete VLM and action-expert state.
        # Construct the architecture without downloading redundant base-model weights,
        # then enforce a strict load from the fine-tuned checkpoint.
        self.config.load_vlm_weights = False
        self.config.compile_model = False
        self.policy = SmolVLAPolicy.from_pretrained(
            self.checkpoint,
            config=self.config,
            local_files_only=True,
            strict=True,
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.policy.eval()

        action_shape = tuple(self.config.output_features["action"].shape)
        if action_shape != (10,):
            raise ValueError(f"Expected checkpoint action shape (10,), got {action_shape}.")
        if int(self.config.chunk_size) != 50 or int(self.config.n_action_steps) != 50:
            raise ValueError(
                "Expected the training checkpoint contract chunk_size=50 and "
                f"n_action_steps=50, got {self.config.chunk_size} and "
                f"{self.config.n_action_steps}."
            )

    @property
    def action_horizon(self) -> int:
        return int(self.config.chunk_size)

    def reset(self, reset: ResetMessage) -> None:
        """Clear all model-side state for every server RESET phase."""
        seed_policy_rng(int(reset.seed))
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def predict_chunk(self, message: dict[str, Any]) -> np.ndarray:
        observation = build_smolvla_observation(message, self.config.input_features)
        with torch.inference_mode():
            processed_observation = self.preprocessor(observation)
            normalized_actions = self.policy.predict_action_chunk(processed_observation)
            actions = self.postprocessor(normalized_actions)

        if not isinstance(actions, torch.Tensor):
            raise TypeError(
                f"SmolVLA postprocessor returned {type(actions).__name__}, expected Tensor."
            )
        array = actions.detach().cpu().numpy().astype(np.float32, copy=False)
        expected_shape = (1, self.action_horizon, 10)
        if tuple(array.shape) != expected_shape:
            raise ValueError(f"Expected postprocessed actions {expected_shape}, got {array.shape}.")
        array = np.ascontiguousarray(array[0])
        if not np.isfinite(array).all():
            raise ValueError("SmolVLA produced non-finite actions.")
        return array

    def predict(self, observation: ObservationRequest) -> np.ndarray:
        return self.predict_chunk(observation.to_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--motionforge-host", default="127.0.0.1")
    parser.add_argument("--motionforge-obs-port", type=int, default=3196)
    parser.add_argument("--motionforge-act-port", type=int, default=3198)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--validate-checkpoint",
        action="store_true",
        help="Load the checkpoint, run one synthetic observation, and exit without opening ZMQ.",
    )
    args = parser.parse_args()
    if not 1 <= args.motionforge_obs_port <= 65535:
        parser.error("--motionforge-obs-port must be in [1, 65535]")
    if not 1 <= args.motionforge_act_port <= 65535:
        parser.error("--motionforge-act-port must be in [1, 65535]")
    if args.motionforge_obs_port == args.motionforge_act_port:
        parser.error("observation and action ports must differ")
    if args.num_episodes < 1:
        parser.error("--num-episodes must be >= 1")
    if args.print_every < 0:
        parser.error("--print-every must be >= 0")
    return args


def validate_checkpoint_files(checkpoint: Path) -> None:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required files: {missing}")


def load_smolvla_config(checkpoint: Path, device: str) -> Any:
    import draccus
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    with (checkpoint / "config.json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    policy_type = payload.pop("type", None)
    if policy_type != "smolvla":
        raise ValueError(f"Expected a SmolVLA checkpoint, config type is {policy_type!r}.")
    config = draccus.decode(SmolVLAConfig, payload)
    config.device = device
    return config


def build_smolvla_observation(
    message: dict[str, Any], input_features: dict[str, Any]
) -> dict[str, torch.Tensor | str]:
    expected_keys = set(input_features)
    required_keys = {STATE_KEY, *IMAGE_KEYS}
    if expected_keys != required_keys:
        raise ValueError(
            "Checkpoint input keys differ from the MotionForge SmolVLA contract: "
            f"{sorted(expected_keys)}"
        )

    task = require_task(message)
    observation: dict[str, torch.Tensor | str] = {TASK_KEY: task}
    state = np.asarray(require_field(message, STATE_KEY), dtype=np.float32).reshape(-1)
    expected_state_shape = tuple(input_features[STATE_KEY].shape)
    if tuple(state.shape) != expected_state_shape:
        raise ValueError(f"{STATE_KEY}: expected {expected_state_shape}, got {state.shape}.")
    if not np.isfinite(state).all():
        raise ValueError(f"{STATE_KEY} contains non-finite values.")
    observation[STATE_KEY] = torch.from_numpy(np.ascontiguousarray(state))

    for key in IMAGE_KEYS:
        expected_chw = tuple(input_features[key].shape)
        observation[key] = image_to_chw_float(require_field(message, key), key, expected_chw)
    return observation


def require_task(message: dict[str, Any]) -> str:
    value = message.get(TASK_KEY, message.get("language_instruction"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("MotionForge observation must include a non-empty task instruction.")
    return value.strip()


def image_to_chw_float(value: Any, key: str, expected_chw: tuple[int, ...]) -> torch.Tensor:
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"{key}: expected HWC image, got shape {image.shape}.")
    if image.shape[-1] == 4:
        image = image[..., :3]
    expected_hwc = (expected_chw[1], expected_chw[2], expected_chw[0])
    if tuple(image.shape) != expected_hwc:
        raise ValueError(f"{key}: expected HWC {expected_hwc}, got {image.shape}.")
    if image.dtype != np.uint8:
        raise TypeError(f"{key}: expected uint8 from MotionForge, got {image.dtype}.")
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(chw).to(dtype=torch.float32).div_(255.0)


def require_field(message: dict[str, Any], key: str) -> Any:
    if key not in message:
        raise KeyError(f"MotionForge observation is missing {key!r}.")
    return message[key]


def installed_lerobot_version() -> str:
    try:
        return version("lerobot")
    except PackageNotFoundError:
        return "unknown"


def synthetic_message(inference: SmolVLAInference) -> dict[str, Any]:
    message: dict[str, Any] = {
        TASK_KEY: "Pick up the moving cardboard package from the conveyor and place it into the box.",
        STATE_KEY: np.zeros(
            tuple(inference.config.input_features[STATE_KEY].shape), dtype=np.float32
        ),
    }
    for key in IMAGE_KEYS:
        channels, height, width = inference.config.input_features[key].shape
        if channels != 3:
            raise ValueError(
                f"Synthetic validation only supports RGB, got {key} shape="
                f"{channels, height, width}."
            )
        message[key] = np.zeros((height, width, channels), dtype=np.uint8)
    return message


def run_validation(inference: SmolVLAInference) -> int:
    started_at = time.perf_counter()
    actions = inference.predict_chunk(synthetic_message(inference))
    elapsed_s = time.perf_counter() - started_at
    print(
        "[MOTIONFORGE-SMOLVLA] validation_passed "
        f"checkpoint={inference.checkpoint} shape={actions.shape} "
        f"min={actions.min():.6f} max={actions.max():.6f} inference_s={elapsed_s:.3f}",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    current_version = installed_lerobot_version()
    inference = SmolVLAInference(checkpoint=args.model_path, device=args.device)
    print(
        "[MOTIONFORGE-SMOLVLA] loaded "
        f"checkpoint={inference.checkpoint} device={args.device} "
        f"lerobot={current_version} action_horizon={inference.action_horizon} "
        "wire_horizon=server_required_16",
        flush=True,
    )
    if args.validate_checkpoint:
        return run_validation(inference)

    print(
        "[MOTIONFORGE-SMOLVLA] listening "
        f"host={args.motionforge_host} obs_port={args.motionforge_obs_port} "
        f"act_port={args.motionforge_act_port}",
        flush=True,
    )
    bridge = BenchmarkClientBridge(
        policy=inference,
        config=ClientBridgeConfig(
            host=args.motionforge_host,
            obs_port=int(args.motionforge_obs_port),
            act_port=int(args.motionforge_act_port),
            num_episodes=int(args.num_episodes),
            print_every=int(args.print_every),
        ),
        log=lambda message: print(f"[MOTIONFORGE-SMOLVLA] {message}", flush=True),
    )
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("[MOTIONFORGE-SMOLVLA] interrupted", flush=True)
        return 130

    print(f"[MOTIONFORGE-SMOLVLA] done episodes={args.num_episodes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
