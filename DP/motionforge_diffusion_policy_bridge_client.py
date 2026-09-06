#!/usr/bin/env python3
"""Bridge MotionForge benchmark observations to a LeRobot Diffusion Policy."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque
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

IMAGE_NATIVE_HWC_SHAPES = {
    "observation.images.overview": (240, 320, 3),
    "observation.images.front": (240, 320, 3),
    "observation.images.wrist": (160, 160, 3),
}
IMAGE_KEYS = tuple(IMAGE_NATIVE_HWC_SHAPES)
STATE_KEY = "observation.state"
EXPECTED_IMAGE_TRANSFORMS_BY_DATASET = {
    "local/factory_conveyor_level2_seeded": {
        "enable": True,
        "max_num_transforms": 1,
        "random_order": False,
        "tfs": {
            "resize": {
                "weight": 1.0,
                "type": "Resize",
                "kwargs": {"size": [240, 320], "antialias": True},
            }
        },
    },
    "local/pi05_merged_v30": {
        "enable": True,
        "max_num_transforms": 1,
        "random_order": False,
        "tfs": {
            "resize": {
                "weight": 1.0,
                "type": "Resize",
                "kwargs": {"size": [240, 320]},
            }
        },
    },
    "local/lerobot_ht": {
        "enable": True,
        "max_num_transforms": 1,
        "random_order": False,
        "tfs": {
            "resize": {
                "weight": 1.0,
                "type": "Resize",
                "kwargs": {"size": [240, 320]},
            }
        },
    },
    "local/lerobot_hri": {
        "enable": True,
        "max_num_transforms": 1,
        "random_order": False,
        "tfs": {
            "resize": {
                "weight": 1.0,
                "type": "Resize",
                "kwargs": {"size": [240, 320]},
            }
        },
    },
}


def seed_policy_rng(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass(slots=True)
class DiffusionPolicyInference:
    """Diffusion Policy model, processors, observation history, and noise generator."""

    checkpoint: Path
    device: str
    num_inference_steps: int | None = None

    config: Any = field(init=False)
    policy: Any = field(init=False)
    preprocessor: Any = field(init=False)
    postprocessor: Any = field(init=False)
    image_transform: Any = field(init=False)
    observation_history: deque[dict[str, torch.Tensor]] = field(init=False)
    noise_generator: torch.Generator = field(init=False)
    _current_policy_seed: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.checkpoint = self.checkpoint.expanduser().resolve()
        validate_checkpoint_files(self.checkpoint)
        self.config = load_diffusion_config(self.checkpoint, self.device)
        validate_diffusion_contract(self.checkpoint, self.config)
        self.image_transform = load_training_image_transform(self.checkpoint)
        checkpoint_inference_steps = (
            int(self.config.num_train_timesteps)
            if self.config.num_inference_steps is None
            else int(self.config.num_inference_steps)
        )
        if self.num_inference_steps is None:
            self.num_inference_steps = checkpoint_inference_steps
        if not 1 <= int(self.num_inference_steps) <= int(self.config.num_train_timesteps):
            raise ValueError(
                "Diffusion inference steps must be in [1, num_train_timesteps], got "
                f"{self.num_inference_steps} and {self.config.num_train_timesteps}."
            )
        self.config.num_inference_steps = int(self.num_inference_steps)

        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        # The checkpoint contains both complete ResNet18 encoders. Avoid downloading
        # their initialization only to overwrite it during the strict checkpoint load.
        self.config.pretrained_backbone_weights = None
        self.config.compile_model = False
        self.policy = DiffusionPolicy.from_pretrained(
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

        parameter = next(self.policy.parameters())
        self.noise_generator = torch.Generator(device=parameter.device)
        self.observation_history = deque(maxlen=int(self.config.n_obs_steps))
        self._reset_state(seed=0)

    @property
    def action_horizon(self) -> int:
        return int(self.config.n_action_steps)

    @property
    def diffusion_horizon(self) -> int:
        return int(self.config.horizon)

    @property
    def observation_horizon(self) -> int:
        return int(self.config.n_obs_steps)

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
        self.observation_history.clear()
        self.noise_generator.manual_seed(self.current_policy_seed)

    def reset(self, reset: ResetMessage) -> None:
        """Reset policy history and RNG exactly to the server-provided episode seed."""
        self._reset_state(seed=int(reset.seed))

    def predict_chunk(self, message: dict[str, Any]) -> np.ndarray:
        observation = build_diffusion_observation(
            message,
            self.config.input_features,
            self.image_transform,
        )
        with torch.inference_mode():
            processed_observation = self.preprocessor(observation)
            history_batch = self._append_and_stack_observation(processed_observation)
            parameter = next(self.policy.parameters())
            noise = torch.randn(
                (1, self.diffusion_horizon, 10),
                dtype=parameter.dtype,
                device=parameter.device,
                generator=self.noise_generator,
            )
            normalized_actions = self.policy.predict_action_chunk(history_batch, noise=noise)
            actions = self.postprocessor(normalized_actions)

        if not isinstance(actions, torch.Tensor):
            raise TypeError(
                "Diffusion Policy postprocessor returned "
                f"{type(actions).__name__}, expected Tensor."
            )
        array = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        expected_shape = (1, self.action_horizon, 10)
        if tuple(array.shape) != expected_shape:
            raise ValueError(f"Expected postprocessed actions {expected_shape}, got {array.shape}.")
        array = np.ascontiguousarray(array[0])
        if not np.isfinite(array).all():
            raise ValueError("Diffusion Policy produced non-finite actions.")
        return array

    def predict(self, observation: ObservationRequest) -> np.ndarray:
        return self.predict_chunk(observation.to_dict())

    def _append_and_stack_observation(
        self, processed_observation: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        current: dict[str, torch.Tensor] = {}
        for key, feature in self.config.input_features.items():
            value = processed_observation.get(key)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Processed {key} is {type(value).__name__}, expected Tensor.")
            expected_shape = (1, *tuple(feature.shape))
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"Processed {key}: expected {expected_shape}, got {tuple(value.shape)}.")
            current[key] = value

        if not self.observation_history:
            self.observation_history.extend(current for _ in range(self.observation_horizon))
        else:
            self.observation_history.append(current)
        if len(self.observation_history) != self.observation_horizon:
            raise RuntimeError(
                "Diffusion observation history is incomplete: "
                f"{len(self.observation_history)}/{self.observation_horizon}."
            )
        return {
            key: torch.stack([frame[key] for frame in self.observation_history], dim=1)
            for key in self.config.input_features
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help=(
            "DDPM denoising steps used at runtime; defaults to the checkpoint setting "
            "(100 when config.num_inference_steps is null)."
        ),
    )
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
    if args.num_inference_steps is not None and not 1 <= args.num_inference_steps <= 100:
        parser.error("--num-inference-steps must be in [1, 100]")
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


def load_diffusion_config(checkpoint: Path, device: str) -> Any:
    import draccus
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    with (checkpoint / "config.json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    policy_type = payload.pop("type", None)
    if policy_type != "diffusion":
        raise ValueError(f"Expected a Diffusion Policy checkpoint, config type is {policy_type!r}.")
    config = draccus.decode(DiffusionConfig, payload)
    config.device = device
    return config


def load_training_config(checkpoint: Path) -> dict[str, Any]:
    """Load the training metadata needed to reproduce dataset preprocessing."""
    with (checkpoint / "train_config.json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError("train_config.json must contain a JSON object.")
    return payload


def load_training_image_transform(checkpoint: Path) -> Any:
    """Rebuild the serialized LeRobot image transform used during training."""
    import draccus

    from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig

    train_config = load_training_config(checkpoint)
    transform_payload = train_config["dataset"]["image_transforms"]
    transform_config = draccus.decode(ImageTransformsConfig, transform_payload)
    return ImageTransforms(transform_config)


def validate_diffusion_contract(checkpoint: Path, config: Any) -> None:
    with (checkpoint / "config.json").open(encoding="utf-8") as stream:
        checkpoint_config = json.load(stream)
    train_config = load_training_config(checkpoint)
    if train_config.get("policy") != checkpoint_config:
        raise ValueError("train_config.json policy does not exactly match config.json.")
    dataset_config = train_config.get("dataset")
    if not isinstance(dataset_config, dict):
        raise ValueError("train_config.json must contain a dataset object.")
    dataset_repo_id = dataset_config.get("repo_id")
    expected_image_transforms = EXPECTED_IMAGE_TRANSFORMS_BY_DATASET.get(dataset_repo_id)
    if expected_image_transforms is None:
        raise ValueError(
            "Checkpoint dataset is not an approved MotionForge three-view Diffusion "
            f"Policy dataset: {dataset_repo_id!r}."
        )
    image_transforms = dataset_config.get("image_transforms")
    if image_transforms != expected_image_transforms:
        raise ValueError(
            "Checkpoint training image transforms differ from the required "
            f"{dataset_repo_id!r} three-view resize contract: {image_transforms!r}."
        )

    expected_shapes = {
        STATE_KEY: (10,),
        "observation.images.overview": (3, 240, 320),
        "observation.images.front": (3, 240, 320),
        "observation.images.wrist": (3, 240, 320),
    }
    if set(config.input_features) != set(expected_shapes):
        raise ValueError(
            "Checkpoint input keys differ from the MotionForge Diffusion Policy contract: "
            f"{sorted(config.input_features)}"
        )
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(config.input_features[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(f"{key}: expected {expected_shape}, got {actual_shape}.")

    action_shape = tuple(config.output_features["action"].shape)
    if action_shape != (10,):
        raise ValueError(f"Expected checkpoint action shape (10,), got {action_shape}.")
    if (
        int(config.n_obs_steps) != 2
        or int(config.horizon) != 64
        or int(config.n_action_steps) != 32
    ):
        raise ValueError(
            "Expected n_obs_steps=2, diffusion horizon=64, and n_action_steps=32, got "
            f"{config.n_obs_steps}, {config.horizon}, and {config.n_action_steps}."
        )
    if str(config.noise_scheduler_type).upper() != "DDPM":
        raise ValueError(f"Expected DDPM scheduler, got {config.noise_scheduler_type!r}.")
    if int(config.num_train_timesteps) != 100 or config.num_inference_steps is not None:
        raise ValueError(
            "Expected 100 inference steps inherited from num_train_timesteps, got "
            f"train={config.num_train_timesteps}, inference={config.num_inference_steps}."
        )
    if not bool(config.use_separate_rgb_encoder_per_camera):
        raise ValueError("Checkpoint must use a separate RGB encoder for each camera.")
    if config.resize_shape is not None or config.crop_shape is not None:
        raise ValueError(
            "Expected checkpoint-native image resolution without resize or crop, got "
            f"resize={config.resize_shape}, crop={config.crop_shape}."
        )

    from safetensors import safe_open

    stats_path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    with safe_open(stats_path, framework="pt", device="cpu") as stats:
        expected_stats_shapes = {
            f"{STATE_KEY}.min": (10,),
            "action.min": (10,),
            "observation.images.overview.mean": (3, 1, 1),
            "observation.images.front.mean": (3, 1, 1),
            "observation.images.wrist.mean": (3, 1, 1),
        }
        for key, expected_shape in expected_stats_shapes.items():
            actual_shape = tuple(stats.get_tensor(key).shape)
            if actual_shape != expected_shape:
                raise ValueError(f"{key}: expected stats shape {expected_shape}, got {actual_shape}.")

    with (checkpoint / "policy_preprocessor.json").open(encoding="utf-8") as stream:
        processor_payload = json.load(stream)
    steps = processor_payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("policy_preprocessor.json must contain a steps list.")
    step_names = [step.get("registry_name") for step in steps]
    expected_steps = [
        "rename_observations_processor",
        "to_batch_processor",
        "device_processor",
        "normalizer_processor",
    ]
    if step_names != expected_steps:
        raise ValueError(f"Unexpected Diffusion Policy preprocessor steps: {step_names!r}.")
    rename_map = steps[0].get("config", {}).get("rename_map")
    if rename_map != {}:
        raise ValueError(f"Expected an empty observation rename map, got {rename_map!r}.")


def build_diffusion_observation(
    message: dict[str, Any],
    input_features: dict[str, Any],
    image_transform: Any,
) -> dict[str, torch.Tensor]:
    expected_keys = {STATE_KEY, *IMAGE_KEYS}
    if set(input_features) != expected_keys:
        raise ValueError(
            "Checkpoint input keys differ from the MotionForge Diffusion Policy contract: "
            f"{sorted(input_features)}"
        )

    observation: dict[str, torch.Tensor] = {}
    state = np.asarray(require_field(message, STATE_KEY), dtype=np.float32).reshape(-1)
    expected_state_shape = tuple(input_features[STATE_KEY].shape)
    if tuple(state.shape) != expected_state_shape:
        raise ValueError(f"{STATE_KEY}: expected {expected_state_shape}, got {state.shape}.")
    if not np.isfinite(state).all():
        raise ValueError(f"{STATE_KEY} contains non-finite values.")
    observation[STATE_KEY] = torch.from_numpy(np.ascontiguousarray(state))

    for key in IMAGE_KEYS:
        expected_chw = tuple(input_features[key].shape)
        observation[key] = image_to_chw_float(
            require_field(message, key),
            key,
            expected_chw,
            image_transform,
        )
    return observation


def image_to_chw_float(
    value: Any,
    key: str,
    expected_chw: tuple[int, ...],
    image_transform: Any,
) -> torch.Tensor:
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"{key}: expected HWC image, got shape {image.shape}.")
    if image.shape[-1] == 4:
        image = image[..., :3]
    native_hwc = IMAGE_NATIVE_HWC_SHAPES[key]
    if tuple(image.shape) != native_hwc:
        raise ValueError(f"{key}: expected native HWC {native_hwc}, got {image.shape}.")
    if image.dtype != np.uint8:
        raise TypeError(f"{key}: expected uint8 from MotionForge, got {image.dtype}.")
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    transformed = image_transform(torch.from_numpy(chw))
    if not isinstance(transformed, torch.Tensor):
        raise TypeError(
            f"{key}: training image transform returned {type(transformed).__name__}, "
            "expected Tensor."
        )
    if transformed.dtype != torch.uint8:
        raise TypeError(
            f"{key}: training image transform returned {transformed.dtype}, expected uint8."
        )
    if tuple(transformed.shape) != expected_chw:
        raise ValueError(
            f"{key}: expected transformed CHW {expected_chw}, got {tuple(transformed.shape)}."
        )
    normalized = transformed.contiguous().to(dtype=torch.float32).div_(255.0)
    if not torch.isfinite(normalized).all():
        raise ValueError(f"{key}: transformed image contains non-finite values.")
    return normalized


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
    return {
        STATE_KEY: np.zeros((10,), dtype=np.float32),
        **{
            key: np.zeros(shape, dtype=np.uint8)
            for key, shape in IMAGE_NATIVE_HWC_SHAPES.items()
        },
    }


def run_validation(inference: DiffusionPolicyInference) -> int:
    started_at = time.perf_counter()
    actions = inference.predict_chunk(synthetic_message())
    elapsed_s = time.perf_counter() - started_at
    print(
        "[MOTIONFORGE-DIFFUSION] validation_passed "
        f"checkpoint={inference.checkpoint} shape={actions.shape} "
        f"min={actions.min():.6f} max={actions.max():.6f} "
        f"policy_seed={inference.current_policy_seed} inference_s={elapsed_s:.3f}",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    current_version = installed_lerobot_version()
    inference = DiffusionPolicyInference(
        checkpoint=args.model_path,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
    )
    print(
        "[MOTIONFORGE-DIFFUSION] loaded "
        f"checkpoint={inference.checkpoint} device={args.device} "
        f"lerobot={current_version} observation_horizon={inference.observation_horizon} "
        f"diffusion_horizon={inference.diffusion_horizon} "
        f"num_inference_steps={inference.num_inference_steps} "
        f"action_horizon={inference.action_horizon} wire_horizon=server_required_16 "
        "policy_seed=server_reset",
        flush=True,
    )
    if args.validate_checkpoint:
        return run_validation(inference)

    print(
        "[MOTIONFORGE-DIFFUSION] listening "
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
        log=lambda message: print(f"[MOTIONFORGE-DIFFUSION] {message}", flush=True),
    )
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("[MOTIONFORGE-DIFFUSION] interrupted", flush=True)
        return 130

    print(f"[MOTIONFORGE-DIFFUSION] done episodes={args.num_episodes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
