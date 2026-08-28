#!/usr/bin/env python3
"""Bridge MotionForge benchmark observations to a LeRobot X-VLA policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# Keep legacy FC runners compatible without duplicating the common bridge code in
# this baseline directory. CM runners also provide this path through PYTHONPATH.
_motionforge_root = Path(
    os.environ.get(
        "MOTIONFORGE_ROOT",
        Path(__file__).resolve().parents[2] / "MotionForge",
    )
)
_motionforge_source = str(_motionforge_root / "source" / "motionforge")
if _motionforge_source not in sys.path:
    sys.path.insert(0, _motionforge_source)

import numpy as np
import torch

from motionforge.benchmark.client import BenchmarkClientBridge, ClientBridgeConfig
from motionforge.benchmark.protocol import ObservationPacket, PROTOCOL_VERSION

STATE_KEY = "observation.state"
TASK_KEY = "task"
RAW_IMAGE_SHAPES = {
    "observation.images.overview": (240, 320, 3),
    "observation.images.front": (240, 320, 3),
    "observation.images.wrist": (160, 160, 3),
}
RAW_TO_POLICY_IMAGE_KEYS = {
    "observation.images.front": "observation.images.image",
    "observation.images.overview": "observation.images.image2",
    "observation.images.wrist": "observation.images.image3",
}
POLICY_IMAGE_SHAPES = {
    "observation.images.image": (3, 256, 256),
    "observation.images.image2": (3, 256, 256),
    "observation.images.image3": (3, 224, 224),
}
MOTIONFORGE_STATE_SHAPE = (10,)


@dataclass(slots=True)
class XVLAInference:
    """X-VLA model plus the processors serialized with its checkpoint."""

    checkpoint: Path
    device: str
    action_hz: float = 30.0

    config: Any = field(init=False)
    policy: Any = field(init=False)
    preprocessor: Any = field(init=False)
    postprocessor: Any = field(init=False)

    def __post_init__(self) -> None:
        self.checkpoint = self.checkpoint.expanduser().resolve()
        validate_checkpoint_files(self.checkpoint)
        self.config = load_xvla_config(self.checkpoint, self.device)
        validate_xvla_contract(self.checkpoint, self.config)

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        # X-VLA embeds its Florence configuration and full trained weights in this
        # checkpoint, so model loading must remain local and strict. The serialized
        # tokenizer path belongs to the training host; use the portable model name
        # from config while preserving every other serialized processor setting.
        self.policy = XVLAPolicy.from_pretrained(
            self.checkpoint,
            config=self.config,
            local_files_only=True,
            strict=True,
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": self.config.tokenizer_name},
                "device_processor": {"device": self.device},
            },
        )
        self.policy.eval()

    @property
    def action_horizon(self) -> int:
        return int(self.config.chunk_size)

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def predict_chunk(self, message: dict[str, Any]) -> np.ndarray:
        observation = build_xvla_observation(message)
        with torch.inference_mode():
            processed_observation = self.preprocessor(observation)
            normalized_actions = self.policy.predict_action_chunk(processed_observation)
            actions = self.postprocessor(normalized_actions)

        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"X-VLA postprocessor returned {type(actions).__name__}, expected Tensor.")
        array = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        expected_shape = (1, self.action_horizon, 10)
        if tuple(array.shape) != expected_shape:
            raise ValueError(f"Expected postprocessed actions {expected_shape}, got {array.shape}.")
        array = np.ascontiguousarray(array[0])
        if not np.isfinite(array).all():
            raise ValueError("X-VLA produced non-finite actions.")
        return array

    def predict(self, observation: ObservationPacket) -> np.ndarray:
        return self.predict_chunk(observation.to_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--motionforge-host", default="127.0.0.1")
    parser.add_argument("--motionforge-obs-port", type=int, default=3196)
    parser.add_argument("--motionforge-act-port", type=int, default=3198)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--action-hz", type=float, default=30.0)
    parser.add_argument("--max-inference-hz", type=float, default=30.0)
    parser.add_argument(
        "--send-horizon",
        type=int,
        default=16,
        help="Number of leading X-VLA actions sent to MotionForge; GR00T FC sends 16.",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=8,
        help="Execution horizon advertised in packet metadata; GR00T FC uses 8.",
    )
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
    if args.action_hz <= 0 or args.max_inference_hz <= 0:
        parser.error("action and inference frequencies must be positive")
    if args.send_horizon < 1:
        parser.error("--send-horizon must be >= 1")
    if not 1 <= args.execution_horizon <= args.send_horizon:
        parser.error("--execution-horizon must be in [1, --send-horizon]")
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
        "policy_preprocessor_step_7_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required files: {missing}")


def load_xvla_config(checkpoint: Path, device: str) -> Any:
    import draccus
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig

    with (checkpoint / "config.json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    policy_type = payload.pop("type", None)
    if policy_type != "xvla":
        raise ValueError(f"Expected an X-VLA checkpoint, config type is {policy_type!r}.")
    config = draccus.decode(XVLAConfig, payload)
    config.device = device
    return config


def validate_xvla_contract(checkpoint: Path, config: Any) -> None:
    expected_feature_keys = {STATE_KEY, *POLICY_IMAGE_SHAPES}
    if set(config.input_features) != expected_feature_keys:
        raise ValueError(
            "Checkpoint input keys differ from the MotionForge X-VLA contract: "
            f"{sorted(config.input_features)}"
        )
    for key, expected_shape in POLICY_IMAGE_SHAPES.items():
        actual_shape = tuple(config.input_features[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(f"{key}: expected checkpoint shape {expected_shape}, got {actual_shape}.")

    # This released checkpoint declares 8 state features, but its saved training
    # statistics and MotionForge dataset samples are 10-D. State normalization is
    # IDENTITY and XVLA pads the supplied vector to max_state_dim, so the trained
    # runtime contract is to preserve all 10 MotionForge values.
    declared_state_shape = tuple(config.input_features[STATE_KEY].shape)
    if declared_state_shape != (8,):
        raise ValueError(
            f"Expected the known checkpoint-declared state shape (8,), got {declared_state_shape}."
        )
    stats_path = checkpoint / "policy_preprocessor_step_7_normalizer_processor.safetensors"
    from safetensors import safe_open

    with safe_open(stats_path, framework="pt", device="cpu") as stats:
        state_stats_shape = tuple(stats.get_tensor(f"{STATE_KEY}.mean").shape)
        action_stats_shape = tuple(stats.get_tensor("action.mean").shape)
    if state_stats_shape != MOTIONFORGE_STATE_SHAPE:
        raise ValueError(
            "Expected 10-D state statistics from the training data, got "
            f"{state_stats_shape}."
        )
    if action_stats_shape != (10,):
        raise ValueError(f"Expected 10-D action statistics, got {action_stats_shape}.")

    action_shape = tuple(config.output_features["action"].shape)
    if action_shape != (10,):
        raise ValueError(f"Expected checkpoint action shape (10,), got {action_shape}.")
    if int(config.chunk_size) != 30 or int(config.n_action_steps) != 30:
        raise ValueError(
            "Expected the training checkpoint contract chunk_size=30 and "
            f"n_action_steps=30, got {config.chunk_size} and {config.n_action_steps}."
        )
    if int(config.max_state_dim) != 20 or int(config.max_action_dim) != 20:
        raise ValueError(
            "Expected X-VLA model dimensions max_state_dim=max_action_dim=20, got "
            f"{config.max_state_dim} and {config.max_action_dim}."
        )
    if str(config.action_mode).lower() != "auto":
        raise ValueError(f"Expected action_mode='auto', got {config.action_mode!r}.")
    tokenizer_name = str(config.tokenizer_name).strip()
    if not tokenizer_name:
        raise ValueError("Checkpoint config must provide a portable tokenizer_name.")

    with (checkpoint / "policy_preprocessor.json").open(encoding="utf-8") as stream:
        processor_payload = json.load(stream)
    steps = processor_payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("policy_preprocessor.json must contain a steps list.")
    rename_steps = [
        step for step in steps if step.get("registry_name") == "rename_observations_processor"
    ]
    if len(rename_steps) != 1:
        raise ValueError("Expected exactly one rename_observations_processor step.")
    rename_map = rename_steps[0].get("config", {}).get("rename_map")
    if rename_map != RAW_TO_POLICY_IMAGE_KEYS:
        raise ValueError(f"Unexpected X-VLA observation rename map: {rename_map!r}.")


def build_xvla_observation(message: dict[str, Any]) -> dict[str, torch.Tensor | str]:
    require_protocol(message)
    observation: dict[str, torch.Tensor | str] = {TASK_KEY: require_task(message)}

    state = np.asarray(require_field(message, STATE_KEY), dtype=np.float32).reshape(-1)
    if tuple(state.shape) != MOTIONFORGE_STATE_SHAPE:
        raise ValueError(f"{STATE_KEY}: expected {MOTIONFORGE_STATE_SHAPE}, got {state.shape}.")
    if not np.isfinite(state).all():
        raise ValueError(f"{STATE_KEY} contains non-finite values.")
    observation[STATE_KEY] = torch.from_numpy(np.ascontiguousarray(state))

    for key, expected_hwc in RAW_IMAGE_SHAPES.items():
        observation[key] = image_to_chw_uint8(require_field(message, key), key, expected_hwc)
    return observation


def require_task(message: dict[str, Any]) -> str:
    value = message.get(TASK_KEY, message.get("language_instruction"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("MotionForge observation must include a non-empty task instruction.")
    return value.strip()


def image_to_chw_uint8(value: Any, key: str, expected_hwc: tuple[int, ...]) -> torch.Tensor:
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"{key}: expected HWC image, got shape {image.shape}.")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if tuple(image.shape) != expected_hwc:
        raise ValueError(f"{key}: expected HWC {expected_hwc}, got {image.shape}.")
    if image.dtype != np.uint8:
        raise TypeError(f"{key}: expected uint8 from MotionForge, got {image.dtype}.")
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def require_field(message: dict[str, Any], key: str) -> Any:
    if key not in message:
        raise KeyError(f"MotionForge observation is missing {key!r}.")
    return message[key]


def require_protocol(message: dict[str, Any]) -> None:
    protocol = message.get("protocol_version")
    if protocol != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported MotionForge protocol {protocol!r}; expected {PROTOCOL_VERSION!r}."
        )


def installed_lerobot_version() -> str:
    try:
        return version("lerobot")
    except PackageNotFoundError:
        return "unknown"


def synthetic_message() -> dict[str, Any]:
    message: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "index": 0,
        "request_id": 0,
        "request_kind": "validation",
        TASK_KEY: "Pick up the moving cardboard package from the conveyor and place it into the box.",
        STATE_KEY: np.zeros(MOTIONFORGE_STATE_SHAPE, dtype=np.float32),
    }
    for key, shape in RAW_IMAGE_SHAPES.items():
        message[key] = np.zeros(shape, dtype=np.uint8)
    return message


def run_validation(inference: XVLAInference) -> int:
    started_at = time.perf_counter()
    actions = inference.predict_chunk(synthetic_message())
    elapsed_s = time.perf_counter() - started_at
    print(
        "[MOTIONFORGE-XVLA] validation_passed "
        f"checkpoint={inference.checkpoint} shape={actions.shape} "
        f"min={actions.min():.6f} max={actions.max():.6f} inference_s={elapsed_s:.3f}",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    current_version = installed_lerobot_version()
    inference = XVLAInference(
        checkpoint=args.model_path,
        device=args.device,
        action_hz=float(args.action_hz),
    )
    if args.send_horizon > inference.action_horizon:
        raise ValueError(
            f"--send-horizon={args.send_horizon} exceeds checkpoint action horizon "
            f"{inference.action_horizon}."
        )
    print(
        "[MOTIONFORGE-XVLA] loaded "
        f"checkpoint={inference.checkpoint} device={args.device} "
        f"lerobot={current_version} action_horizon={inference.action_horizon} "
        f"send_horizon={args.send_horizon} execution_horizon={args.execution_horizon}",
        flush=True,
    )
    if args.validate_checkpoint:
        return run_validation(inference)

    print(
        "[MOTIONFORGE-XVLA] listening "
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
            legacy_send_horizon=int(args.send_horizon),
            legacy_execution_horizon=int(args.execution_horizon),
            legacy_max_inference_hz=float(args.max_inference_hz),
        ),
        log=lambda message: print(f"[MOTIONFORGE-XVLA] {message}", flush=True),
    )
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("[MOTIONFORGE-XVLA] interrupted", flush=True)
        return 130

    print(f"[MOTIONFORGE-XVLA] done episodes={args.num_episodes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
