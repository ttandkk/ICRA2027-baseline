#!/usr/bin/env python3
"""Bridge MotionForge benchmark observations to a LeRobot FastWAM policy."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROTOCOL_VERSION = "motionforge.benchmark.v1"
ACTION_KEY = "eef_xyz_rot6d_gripper"
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


@dataclass(slots=True)
class MotionForgeTransport:
    """Minimal ZMQ transport for MotionForge benchmark packets."""

    host: str
    obs_port: int
    act_port: int
    recv_timeout_ms: int
    send_timeout_ms: int

    _zmq: Any = field(init=False)
    _context: Any = field(init=False)
    _obs_socket: Any = field(init=False)
    _act_socket: Any = field(init=False)

    def __post_init__(self) -> None:
        import zmq

        self._zmq = zmq
        self._context = zmq.Context()
        self._obs_socket = self._context.socket(zmq.SUB)
        self._obs_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._obs_socket.setsockopt(zmq.RCVHWM, 16)
        self._obs_socket.connect(f"tcp://{self.host}:{self.obs_port}")

        self._act_socket = self._context.socket(zmq.PUSH)
        self._act_socket.setsockopt(zmq.SNDHWM, 16)
        self._act_socket.setsockopt(zmq.SNDTIMEO, self.send_timeout_ms)
        self._act_socket.connect(f"tcp://{self.host}:{self.act_port}")

    def receive(self) -> dict[str, Any] | None:
        if self._obs_socket.poll(timeout=self.recv_timeout_ms) == 0:
            return None
        message = self._obs_socket.recv_pyobj()
        if not isinstance(message, dict):
            raise TypeError(f"Expected a dict packet, got {type(message).__name__}.")
        return message

    def send(self, packet: dict[str, Any]) -> None:
        try:
            self._act_socket.send_pyobj(packet)
        except self._zmq.Again as exc:
            raise TimeoutError(
                f"Timed out sending an action to tcp://{self.host}:{self.act_port}."
            ) from exc

    def close(self) -> None:
        self._obs_socket.close(linger=0)
        self._act_socket.close(linger=0)
        self._context.term()


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

    def reset(self) -> None:
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
        help="Number of leading FastWAM actions sent to MotionForge.",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=8,
        help="Execution horizon advertised in packet metadata.",
    )
    parser.add_argument("--recv-timeout-ms", type=int, default=100)
    parser.add_argument("--send-timeout-ms", type=int, default=10000)
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
    if args.num_episodes < 0:
        parser.error("--num-episodes must be >= 0")
    if args.action_hz <= 0 or args.max_inference_hz <= 0:
        parser.error("action and inference frequencies must be positive")
    if args.send_horizon < 1:
        parser.error("--send-horizon must be >= 1")
    if not 1 <= args.execution_horizon <= args.send_horizon:
        parser.error("--execution-horizon must be in [1, --send-horizon]")
    if args.recv_timeout_ms < 1 or args.send_timeout_ms < 1:
        parser.error("transport timeouts must be positive")
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
    # save_pretrained intentionally removes the auto base-initialization route;
    # every other setting must remain byte-for-byte equivalent at the JSON level.
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
    require_protocol(message)
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


def require_protocol(message: dict[str, Any]) -> None:
    protocol = message.get("protocol_version")
    if protocol != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported MotionForge protocol {protocol!r}; expected {PROTOCOL_VERSION!r}."
        )


def action_packet(
    *,
    actions: np.ndarray,
    message: dict[str, Any],
    action_horizon: int,
    execution_horizon: int,
    inference_duration_s: float,
    action_hz: float,
    max_inference_hz: float,
) -> dict[str, Any]:
    send_horizon = int(actions.shape[0])
    if tuple(actions.shape[1:]) != MOTIONFORGE_ACTION_SHAPE:
        raise ValueError(
            f"Expected actions [K,{MOTIONFORGE_ACTION_SHAPE[0]}], got {actions.shape}."
        )
    if not np.isfinite(actions).all():
        raise ValueError("Action packet contains non-finite actions.")
    if not 1 <= execution_horizon <= send_horizon <= action_horizon:
        raise ValueError(
            "Expected 1 <= execution_horizon <= send_horizon <= action_horizon, got "
            f"{execution_horizon}, {send_horizon}, and {action_horizon}."
        )
    observation_index = int(require_field(message, "index"))
    packet: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "action": actions.tolist(),
        "action_key": ACTION_KEY,
        "action_frame": "robot",
        "action_representation": "ABSOLUTE",
        "observation_index": observation_index,
        "action_hz": float(action_hz),
        "action_alignment": "observation_aligned",
        "inference_duration_s": float(inference_duration_s),
        "max_inference_hz": float(max_inference_hz),
        "created_time": time.time(),
        "metadata": {
            "client": "motionforge_fastwam_bridge",
            "action_horizon": int(action_horizon),
            "send_horizon": send_horizon,
            "execution_horizon": int(execution_horizon),
            "request_kind": message.get("request_kind"),
        },
    }
    if message.get("request_id") is not None:
        packet["request_id"] = int(message["request_id"])
    return packet


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
    if args.send_horizon > inference.action_horizon:
        raise ValueError(
            f"--send-horizon={args.send_horizon} exceeds checkpoint action horizon "
            f"{inference.action_horizon}."
        )
    print(
        "[MOTIONFORGE-FASTWAM] loaded "
        f"checkpoint={inference.checkpoint} device={args.device} "
        f"lerobot={current_version} action_horizon={inference.action_horizon} "
        f"trained_action_steps={inference.trained_action_steps} "
        f"send_horizon={args.send_horizon} execution_horizon={args.execution_horizon}",
        flush=True,
    )
    if args.validate_checkpoint:
        return run_validation(inference)

    transport = MotionForgeTransport(
        host=args.motionforge_host,
        obs_port=args.motionforge_obs_port,
        act_port=args.motionforge_act_port,
        recv_timeout_ms=args.recv_timeout_ms,
        send_timeout_ms=args.send_timeout_ms,
    )
    episodes = 0
    observations = 0
    actions_sent = 0
    episode_observations = 0
    processed_request_ids: set[int] = set()
    print(
        "[MOTIONFORGE-FASTWAM] listening "
        f"host={args.motionforge_host} obs_port={args.motionforge_obs_port} "
        f"act_port={args.motionforge_act_port}",
        flush=True,
    )
    try:
        while True:
            message = transport.receive()
            if message is None:
                continue
            if "episode_result" in message:
                episodes += 1
                print(
                    "[MOTIONFORGE-FASTWAM] episode_result "
                    f"episode={episodes} observations={episode_observations} "
                    f"result={message['episode_result']}",
                    flush=True,
                )
                inference.reset()
                processed_request_ids.clear()
                episode_observations = 0
                if args.num_episodes and episodes >= args.num_episodes:
                    break
                continue

            require_protocol(message)
            request_id_value = message.get("request_id")
            request_id = None if request_id_value is None else int(request_id_value)
            if request_id is not None and request_id in processed_request_ids:
                print(
                    f"[MOTIONFORGE-FASTWAM] duplicate_request request_id={request_id}",
                    flush=True,
                )
                continue
            if request_id == 0 and message.get("request_kind") == "warmup":
                inference.reset()
                processed_request_ids.clear()

            observations += 1
            episode_observations += 1
            started_at = time.perf_counter()
            predicted_actions = inference.predict_chunk(message)
            inference_duration_s = time.perf_counter() - started_at
            actions = np.ascontiguousarray(predicted_actions[: args.send_horizon])
            packet = action_packet(
                actions=actions,
                message=message,
                action_horizon=inference.action_horizon,
                execution_horizon=args.execution_horizon,
                inference_duration_s=inference_duration_s,
                action_hz=args.action_hz,
                max_inference_hz=args.max_inference_hz,
            )
            transport.send(packet)
            actions_sent += 1
            if request_id is not None:
                processed_request_ids.add(request_id)
            if args.print_every and (actions_sent == 1 or actions_sent % args.print_every == 0):
                print(
                    "[MOTIONFORGE-FASTWAM] action_sent "
                    f"count={actions_sent} request_id={request_id} "
                    f"observation_index={packet['observation_index']} "
                    f"predicted_shape={predicted_actions.shape} sent_shape={actions.shape} "
                    f"inference_s={inference_duration_s:.3f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("[MOTIONFORGE-FASTWAM] interrupted", flush=True)
        return 130
    finally:
        transport.close()

    print(
        "[MOTIONFORGE-FASTWAM] done "
        f"episodes={episodes} observations={observations} actions_sent={actions_sent}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
