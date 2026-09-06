#!/usr/bin/env python3
"""Bridge MotionForge benchmark observations to a LeRobot FastWAM policy."""

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

STATE_KEY = "observation.state"
TASK_KEY = "task"
RAW_IMAGE_SHAPES = {
    "observation.images.overview": (240, 320, 3),
    "observation.images.front": (240, 320, 3),
    "observation.images.wrist": (160, 160, 3),
}
POLICY_IMAGE_SHAPES = {
    "observation.images.front": (3, 224, 224),
    "observation.images.overview": (3, 224, 224),
    "observation.images.wrist": (3, 224, 224),
}
MOTIONFORGE_STATE_SHAPE = (10,)
MOTIONFORGE_ACTION_SHAPE = (10,)
EXPECTED_PREPROCESSOR_STEPS = (
    "rename_observations_processor",
    "to_batch_processor",
    "device_processor",
    "normalizer_processor",
)
EXPECTED_POSTPROCESSOR_STEPS = (
    "unnormalizer_processor",
    "device_processor",
)


def seed_policy_rng(seed: int) -> None:
    if not 0 <= int(seed) < 2**63 - 1:
        raise ValueError(f"RESET seed must be in [0, 2**63 - 2], got {seed}.")
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass(slots=True)
class FastWAMInference:
    """FastWAM model plus the processors serialized with its checkpoint."""

    checkpoint: Path
    device: str

    config: Any = field(init=False)
    policy: Any = field(init=False)
    preprocessor: Any = field(init=False)
    postprocessor: Any = field(init=False)

    def __post_init__(self) -> None:
        self.checkpoint = self.checkpoint.expanduser().resolve()
        validate_checkpoint_files(self.checkpoint)
        self.config = load_fastwam_config(self.checkpoint, self.device)
        validate_fastwam_contract(self.checkpoint, self.config)

        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

        # The trainable FastWAM weights and processor state are local and must match
        # exactly. FastWAM itself remains responsible for loading its frozen Wan VAE,
        # UMT5 text encoder, and tokenizer according to the serialized LeRobot config.
        self.policy = FastWAMPolicy.from_pretrained(
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

    @property
    def action_horizon(self) -> int:
        return int(self.config.action_horizon)

    @property
    def trained_action_steps(self) -> int:
        return int(self.config.n_action_steps)

    def reset(self, reset: ResetMessage) -> None:
        """Clear all model-side state for every server RESET phase."""
        seed_policy_rng(int(reset.seed))
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def predict_chunk(self, message: dict[str, Any]) -> np.ndarray:
        observation = build_fastwam_observation(message)
        with torch.inference_mode():
            processed_observation = self.preprocessor(observation)
            normalized_actions = self.policy.predict_action_chunk(processed_observation)
            actions = self.postprocessor(normalized_actions)

        if not isinstance(actions, torch.Tensor):
            raise TypeError(
                f"FastWAM postprocessor returned {type(actions).__name__}, expected Tensor."
            )
        array = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        expected_shape = (1, self.action_horizon, MOTIONFORGE_ACTION_SHAPE[0])
        if tuple(array.shape) != expected_shape:
            raise ValueError(f"Expected postprocessed actions {expected_shape}, got {array.shape}.")
        array = np.ascontiguousarray(array[0])
        if not np.isfinite(array).all():
            raise ValueError("FastWAM produced non-finite actions.")
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
        "train_config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required files: {missing}")


def load_fastwam_config(checkpoint: Path, device: str) -> Any:
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig

    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if not isinstance(config, FastWAMConfig):
        raise TypeError(
            f"Expected FastWAMConfig from checkpoint, got {type(config).__name__}."
        )
    config.device = device
    return config


def validate_fastwam_contract(checkpoint: Path, config: Any) -> None:
    with (checkpoint / "config.json").open(encoding="utf-8") as stream:
        checkpoint_payload = json.load(stream)
    with (checkpoint / "train_config.json").open(encoding="utf-8") as stream:
        train_payload = json.load(stream)

    train_policy = train_payload.get("policy")
    if not isinstance(train_policy, dict):
        raise ValueError("train_config.json must contain a policy object.")
    expected_saved_policy = dict(train_policy)
    # Exports may clear the resume source or retain the recorded training source.
    # All other settings, and any retained source, must match the training record.
    if checkpoint_payload.get("pretrained_path") is None:
        expected_saved_policy["pretrained_path"] = None
    if checkpoint_payload != expected_saved_policy:
        differing = sorted(
            key
            for key in set(checkpoint_payload) | set(expected_saved_policy)
            if checkpoint_payload.get(key) != expected_saved_policy.get(key)
        )
        raise ValueError(
            "Checkpoint config differs from its recorded training policy for keys: "
            f"{differing}."
        )

    expected_feature_keys = {STATE_KEY, *POLICY_IMAGE_SHAPES}
    if set(config.input_features) != expected_feature_keys:
        raise ValueError(
            "Checkpoint input keys differ from the MotionForge FastWAM contract: "
            f"{sorted(config.input_features)}"
        )
    for key, expected_shape in POLICY_IMAGE_SHAPES.items():
        actual_shape = tuple(config.input_features[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(f"{key}: expected checkpoint shape {expected_shape}, got {actual_shape}.")
    state_shape = tuple(config.input_features[STATE_KEY].shape)
    if state_shape != MOTIONFORGE_STATE_SHAPE:
        raise ValueError(f"Expected state shape {MOTIONFORGE_STATE_SHAPE}, got {state_shape}.")
    if set(config.output_features) != {"action"}:
        raise ValueError(f"Expected only the action output, got {sorted(config.output_features)}.")
    action_shape = tuple(config.output_features["action"].shape)
    if action_shape != MOTIONFORGE_ACTION_SHAPE:
        raise ValueError(f"Expected action shape {MOTIONFORGE_ACTION_SHAPE}, got {action_shape}.")

    expected_scalars = {
        "action_dim": 10,
        "proprio_dim": 10,
        "action_horizon": 32,
        "n_action_steps": 10,
        "num_video_frames": 33,
        "action_video_freq_ratio": 4,
        "num_inference_steps": 10,
        "inference_seed": 42,
        "tokenizer_max_len": 128,
    }
    for name, expected in expected_scalars.items():
        actual = getattr(config, name)
        if actual != expected:
            raise ValueError(f"Expected {name}={expected!r}, got {actual!r}.")
    if tuple(config.image_size) != (224, 672):
        raise ValueError(f"Expected image_size=(224, 672), got {config.image_size}.")
    if int(config.model_video_frames) != 9:
        raise ValueError(f"Expected model_video_frames=9, got {config.model_video_frames}.")

    expected_strings = {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "tokenizer_model_id": "google/umt5-xxl",
        "text_encoder_model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "torch_dtype": "bfloat16",
        "rand_device": "cpu",
    }
    for name, expected in expected_strings.items():
        actual = str(getattr(config, name))
        if actual != expected:
            raise ValueError(f"Expected {name}={expected!r}, got {actual!r}.")
    if not bool(config.load_text_encoder):
        raise ValueError("FastWAM checkpoint must load its trained text-conditioning path.")
    if list(config.toggle_action_dimensions):
        raise ValueError(
            "MotionForge FastWAM checkpoint must not apply LIBERO action toggles, got "
            f"{config.toggle_action_dimensions}."
        )
    if config.video_dit_config.get("video_attention_mask_mode") != "first_frame_causal":
        raise ValueError("FastWAM action inference requires first_frame_causal video attention.")

    _validate_processor_configs(checkpoint)
    _validate_checkpoint_tensors(checkpoint)


def _validate_processor_configs(checkpoint: Path) -> None:
    with (checkpoint / "policy_preprocessor.json").open(encoding="utf-8") as stream:
        preprocessor = json.load(stream)
    with (checkpoint / "policy_postprocessor.json").open(encoding="utf-8") as stream:
        postprocessor = json.load(stream)

    pre_steps = preprocessor.get("steps")
    post_steps = postprocessor.get("steps")
    if not isinstance(pre_steps, list) or not isinstance(post_steps, list):
        raise ValueError("Serialized processor configs must contain steps lists.")
    pre_names = tuple(step.get("registry_name") for step in pre_steps)
    post_names = tuple(step.get("registry_name") for step in post_steps)
    if pre_names != EXPECTED_PREPROCESSOR_STEPS:
        raise ValueError(f"Unexpected FastWAM preprocessor steps: {pre_names}.")
    if post_names != EXPECTED_POSTPROCESSOR_STEPS:
        raise ValueError(f"Unexpected FastWAM postprocessor steps: {post_names}.")
    if pre_steps[0].get("config", {}).get("rename_map") != {}:
        raise ValueError("FastWAM checkpoint must preserve MotionForge camera keys without renaming.")

    expected_norm_map = {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    pre_normalizer = pre_steps[-1].get("config", {})
    post_unnormalizer = post_steps[0].get("config", {})
    if pre_normalizer.get("norm_map") != expected_norm_map:
        raise ValueError(f"Unexpected preprocessor normalization map: {pre_normalizer.get('norm_map')!r}.")
    if post_unnormalizer.get("norm_map") != expected_norm_map:
        raise ValueError(
            f"Unexpected postprocessor normalization map: {post_unnormalizer.get('norm_map')!r}."
        )


def _validate_checkpoint_tensors(checkpoint: Path) -> None:
    from safetensors import safe_open

    pre_stats_path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    post_stats_path = checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    required_pre_shapes = {
        **{
            f"{key}.{stat}": (3, 1, 1)
            for key in POLICY_IMAGE_SHAPES
            for stat in ("mean", "std")
        },
        **{
            f"{STATE_KEY}.{stat}": MOTIONFORGE_STATE_SHAPE
            for stat in ("mean", "std", "min", "max", "q01", "q99")
        },
        **{
            f"action.{stat}": MOTIONFORGE_ACTION_SHAPE
            for stat in ("mean", "std", "min", "max", "q01", "q99")
        },
    }
    action_stats: dict[str, torch.Tensor] = {}
    with safe_open(pre_stats_path, framework="pt", device="cpu") as stats:
        available = set(stats.keys())
        missing = sorted(set(required_pre_shapes) - available)
        if missing:
            raise ValueError(f"FastWAM preprocessor statistics are missing: {missing}.")
        for key, expected_shape in required_pre_shapes.items():
            tensor = stats.get_tensor(key)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{key}: expected statistics shape {expected_shape}, got {tensor.shape}.")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{key}: checkpoint statistics contain non-finite values.")
            if key.startswith("action."):
                action_stats[key] = tensor
    with safe_open(post_stats_path, framework="pt", device="cpu") as stats:
        expected_post_keys = set(action_stats)
        if set(stats.keys()) != expected_post_keys:
            raise ValueError(
                "FastWAM postprocessor action statistics keys differ from the preprocessor."
            )
        for key, expected in action_stats.items():
            actual = stats.get_tensor(key)
            if tuple(actual.shape) != MOTIONFORGE_ACTION_SHAPE or not torch.equal(actual, expected):
                raise ValueError(f"{key}: preprocessor/postprocessor action statistics differ.")

    expected_model_shapes = {
        "model.proprio_encoder.weight": (4096, 10),
        "model.proprio_encoder.bias": (4096,),
        "model.mot.mixtures.action.head.weight": (10, 1024),
        "model.mot.mixtures.action.head.bias": (10,),
    }
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        available = set(weights.keys())
        missing = sorted(set(expected_model_shapes) - available)
        if missing:
            raise ValueError(f"FastWAM model weights are missing contract tensors: {missing}.")
        for key, expected_shape in expected_model_shapes.items():
            actual_shape = tuple(weights.get_slice(key).get_shape())
            if actual_shape != expected_shape:
                raise ValueError(f"{key}: expected model shape {expected_shape}, got {actual_shape}.")


def build_fastwam_observation(message: dict[str, Any]) -> dict[str, torch.Tensor | str]:
    observation: dict[str, torch.Tensor | str] = {TASK_KEY: require_task(message)}

    state = np.asarray(require_field(message, STATE_KEY), dtype=np.float32).reshape(-1)
    if tuple(state.shape) != MOTIONFORGE_STATE_SHAPE:
        raise ValueError(f"{STATE_KEY}: expected {MOTIONFORGE_STATE_SHAPE}, got {state.shape}.")
    if not np.isfinite(state).all():
        raise ValueError(f"{STATE_KEY} contains non-finite values.")
    observation[STATE_KEY] = torch.from_numpy(np.ascontiguousarray(state))

    for key, expected_hwc in RAW_IMAGE_SHAPES.items():
        observation[key] = image_to_chw_float(require_field(message, key), key, expected_hwc)
    return observation


def require_task(message: dict[str, Any]) -> str:
    value = message.get(TASK_KEY, message.get("language_instruction"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("MotionForge observation must include a non-empty task instruction.")
    return value.strip()


def image_to_chw_float(value: Any, key: str, expected_hwc: tuple[int, ...]) -> torch.Tensor:
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
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    # FastWAM's serialized VISUAL normalization is IDENTITY; training images
    # arrived as float32 in [0, 1], and the model performs its own resize.
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


def synthetic_message() -> dict[str, Any]:
    message: dict[str, Any] = {
        TASK_KEY: "Pick up the moving cardboard package from the conveyor and place it into the box.",
        STATE_KEY: np.zeros(MOTIONFORGE_STATE_SHAPE, dtype=np.float32),
    }
    for key, shape in RAW_IMAGE_SHAPES.items():
        message[key] = np.zeros(shape, dtype=np.uint8)
    return message


def run_validation(inference: FastWAMInference) -> int:
    started_at = time.perf_counter()
    actions = inference.predict_chunk(synthetic_message())
    elapsed_s = time.perf_counter() - started_at
    print(
        "[MOTIONFORGE-FASTWAM] validation_passed "
        f"checkpoint={inference.checkpoint} shape={actions.shape} "
        f"min={actions.min():.6f} max={actions.max():.6f} inference_s={elapsed_s:.3f}",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    current_version = installed_lerobot_version()
    inference = FastWAMInference(checkpoint=args.model_path, device=args.device)
    print(
        "[MOTIONFORGE-FASTWAM] loaded "
        f"checkpoint={inference.checkpoint} device={args.device} "
        f"lerobot={current_version} action_horizon={inference.action_horizon} "
        f"trained_action_steps={inference.trained_action_steps} "
        "wire_horizon=server_required_16",
        flush=True,
    )
    if args.validate_checkpoint:
        return run_validation(inference)

    print(
        "[MOTIONFORGE-FASTWAM] listening "
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
        log=lambda message: print(f"[MOTIONFORGE-FASTWAM] {message}", flush=True),
    )
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("[MOTIONFORGE-FASTWAM] interrupted", flush=True)
        return 130

    print(f"[MOTIONFORGE-FASTWAM] done episodes={args.num_episodes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
