#!/usr/bin/env python3
"""Bridge MotionForge benchmark observations to a LeRobot PI0.5 policy."""

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
DEFAULT_TOKENIZER = "google/paligemma-3b-pt-224"


def seed_policy_rng(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass(slots=True)
class PI05Inference:
    """PI0.5 model, checkpoint processors, and reproducible flow-matching noise."""

    checkpoint: Path
    device: str
    tokenizer_path: str

    config: Any = field(init=False)
    policy: Any = field(init=False)
    preprocessor: Any = field(init=False)
    postprocessor: Any = field(init=False)
    noise_generator: torch.Generator = field(init=False)
    _current_policy_seed: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.checkpoint = self.checkpoint.expanduser().resolve()
        validate_checkpoint_files(self.checkpoint)
        self.config = load_pi05_config(self.checkpoint, self.device)
        validate_pi05_contract(self.config)

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        # Compilation and gradient checkpointing are training settings. Disabling them
        # avoids a large first-inference compile and has no effect on checkpoint weights.
        self.config.compile_model = False
        self.config.gradient_checkpointing = False
        self.policy = PI05Policy.from_pretrained(
            self.checkpoint,
            config=self.config,
            local_files_only=True,
            strict=True,
        )
        verify_policy_weights(self.policy, self.checkpoint)

        preprocessor_overrides = {
            "device_processor": {"device": self.device},
            "tokenizer_processor": {"tokenizer_name": self.tokenizer_path},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides=preprocessor_overrides,
        )
        self.policy.eval()

        parameter = next(self.policy.parameters())
        self.noise_generator = torch.Generator(device=parameter.device)
        self._reset_state(seed=0)

    @property
    def action_horizon(self) -> int:
        return int(self.config.chunk_size)

    @property
    def current_policy_seed(self) -> int:
        return int(self._current_policy_seed)

    def _reset_state(self, *, seed: int) -> None:
        if not 0 <= int(seed) < 2**63 - 1:
            raise ValueError(f"RESET seed must be in [0, 2**63 - 2], got {seed}.")
        self._current_policy_seed = int(seed)
        seed_policy_rng(self.current_policy_seed)
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.noise_generator.manual_seed(self.current_policy_seed)

    def reset(self, reset: ResetMessage) -> None:
        """Reset model state and RNG to the seed selected by the server."""
        self._reset_state(seed=int(reset.seed))

    def predict_chunk(self, message: dict[str, Any]) -> np.ndarray:
        observation = build_pi05_observation(message, self.config.input_features)
        parameter = next(self.policy.parameters())
        noise = torch.randn(
            (1, self.action_horizon, int(self.config.max_action_dim)),
            dtype=torch.float32,
            device=parameter.device,
            generator=self.noise_generator,
        )
        with torch.inference_mode():
            processed_observation = self.preprocessor(observation)
            normalized_actions = self.policy.predict_action_chunk(
                processed_observation,
                noise=noise,
            )
            actions = self.postprocessor(normalized_actions)

        if not isinstance(actions, torch.Tensor):
            raise TypeError(
                f"PI0.5 postprocessor returned {type(actions).__name__}, expected Tensor."
            )
        array = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        expected_shape = (1, self.action_horizon, 10)
        if tuple(array.shape) != expected_shape:
            raise ValueError(f"Expected postprocessed actions {expected_shape}, got {array.shape}.")
        array = np.ascontiguousarray(array[0])
        if not np.isfinite(array).all():
            raise ValueError("PI0.5 produced non-finite actions.")
        return array

    def predict(self, observation: ObservationRequest) -> np.ndarray:
        return self.predict_chunk(observation.to_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
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
    if not args.tokenizer_path.strip():
        parser.error("--tokenizer-path must not be empty")
    return args


def validate_checkpoint_files(checkpoint: Path) -> None:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required files: {missing}")


def load_pi05_config(checkpoint: Path, device: str) -> Any:
    import draccus
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    with (checkpoint / "config.json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    policy_type = payload.pop("type", None)
    if policy_type != "pi05":
        raise ValueError(f"Expected a PI0.5 checkpoint, config type is {policy_type!r}.")
    config = draccus.decode(PI05Config, payload)
    config.device = device
    return config


def validate_pi05_contract(config: Any) -> None:
    expected_inputs = {
        STATE_KEY: (10,),
        "observation.images.overview": (3, 240, 320),
        "observation.images.front": (3, 240, 320),
        "observation.images.wrist": (3, 160, 160),
    }
    actual_inputs = {
        key: tuple(feature.shape) for key, feature in config.input_features.items()
    }
    if actual_inputs != expected_inputs:
        raise ValueError(
            "Checkpoint input features differ from the MotionForge PI0.5 contract: "
            f"{actual_inputs}."
        )
    action_shape = tuple(config.output_features["action"].shape)
    if action_shape != (10,):
        raise ValueError(f"Expected checkpoint action shape (10,), got {action_shape}.")
    if int(config.chunk_size) != 50 or int(config.n_action_steps) != 50:
        raise ValueError(
            "Expected chunk_size=50 and n_action_steps=50, got "
            f"{config.chunk_size} and {config.n_action_steps}."
        )
    if int(config.max_state_dim) != 32 or int(config.max_action_dim) != 32:
        raise ValueError(
            "Expected max_state_dim=max_action_dim=32, got "
            f"{config.max_state_dim} and {config.max_action_dim}."
        )
    if int(config.num_inference_steps) != 10:
        raise ValueError(
            f"Expected num_inference_steps=10, got {config.num_inference_steps}."
        )
    if config.use_relative_actions:
        raise ValueError("The MotionForge PI0.5 checkpoint must use absolute actions.")


def verify_policy_weights(policy: Any, checkpoint: Path) -> None:
    """Detect PI05Policy.from_pretrained returning an uninitialized model after an error."""

    from safetensors import safe_open

    parameter_key = "model.action_out_proj.bias"
    state = policy.state_dict()
    if parameter_key not in state:
        raise KeyError(f"Loaded policy is missing verification parameter {parameter_key!r}.")
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as stream:
        if parameter_key not in stream.keys():
            raise KeyError(f"Checkpoint is missing verification parameter {parameter_key!r}.")
        expected = stream.get_tensor(parameter_key)
    actual = state[parameter_key].detach().cpu()
    if not torch.equal(actual, expected.to(dtype=actual.dtype)):
        raise RuntimeError(
            "The loaded PI0.5 parameter does not match model.safetensors; refusing to run "
            "with a potentially uninitialized policy."
        )


def build_pi05_observation(
    message: dict[str, Any], input_features: dict[str, Any]
) -> dict[str, torch.Tensor | str]:
    expected_keys = set(input_features)
    required_keys = {STATE_KEY, *IMAGE_KEYS}
    if expected_keys != required_keys:
        raise ValueError(
            "Checkpoint input keys differ from the MotionForge PI0.5 contract: "
            f"{sorted(expected_keys)}"
        )

    observation: dict[str, torch.Tensor | str] = {TASK_KEY: require_task(message)}
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


def synthetic_message(inference: PI05Inference) -> dict[str, Any]:
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


def run_validation(inference: PI05Inference) -> int:
    started_at = time.perf_counter()
    actions = inference.predict_chunk(synthetic_message(inference))
    elapsed_s = time.perf_counter() - started_at
    print(
        "[MOTIONFORGE-PI05] validation_passed "
        f"checkpoint={inference.checkpoint} shape={actions.shape} "
        f"min={actions.min():.6f} max={actions.max():.6f} inference_s={elapsed_s:.3f}",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    inference = PI05Inference(
        checkpoint=args.model_path,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
    )
    print(
        "[MOTIONFORGE-PI05] loaded "
        f"checkpoint={inference.checkpoint} device={args.device} "
        f"lerobot={installed_lerobot_version()} tokenizer={args.tokenizer_path} "
        f"action_horizon={inference.action_horizon} wire_horizon=server_required_16 "
        "policy_seed=server_reset",
        flush=True,
    )
    if args.validate_checkpoint:
        return run_validation(inference)

    print(
        "[MOTIONFORGE-PI05] listening "
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
        log=lambda message: print(f"[MOTIONFORGE-PI05] {message}", flush=True),
    )
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("[MOTIONFORGE-PI05] interrupted", flush=True)
        return 130

    print(f"[MOTIONFORGE-PI05] done episodes={args.num_episodes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
