#!/usr/bin/env python3
"""Bridge MotionForge benchmark observations to a LeRobot SmolVLA policy."""

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
IMAGE_KEYS = (
    "observation.images.overview",
    "observation.images.front",
    "observation.images.wrist",
)
STATE_KEY = "observation.state"
TASK_KEY = "task"


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

    def reset(self) -> None:
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
        help="Number of leading SmolVLA actions sent to MotionForge; GR00T FC sends 16.",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=8,
        help="Execution horizon advertised in packet metadata; GR00T FC uses 8.",
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
    require_protocol(message)
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
    if not 1 <= execution_horizon <= send_horizon <= action_horizon:
        raise ValueError(
            "Expected 1 <= execution_horizon <= send_horizon <= action_horizon, got "
            f"{execution_horizon}, {send_horizon}, and {action_horizon}."
        )
    observation_index = int(require_field(message, "index"))
    packet: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        # Use a built-in wire type so NumPy 2.x clients remain pickle-compatible
        # with the MotionForge environment, which currently uses NumPy 1.x.
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
            "client": "motionforge_smolvla_bridge",
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


def synthetic_message(inference: SmolVLAInference) -> dict[str, Any]:
    message: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "index": 0,
        "request_id": 0,
        "request_kind": "validation",
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
    if args.send_horizon > inference.action_horizon:
        raise ValueError(
            f"--send-horizon={args.send_horizon} exceeds checkpoint action horizon "
            f"{inference.action_horizon}."
        )
    print(
        "[MOTIONFORGE-SMOLVLA] loaded "
        f"checkpoint={inference.checkpoint} device={args.device} "
        f"lerobot={current_version} action_horizon={inference.action_horizon} "
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
        "[MOTIONFORGE-SMOLVLA] listening "
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
                    "[MOTIONFORGE-SMOLVLA] episode_result "
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
                    f"[MOTIONFORGE-SMOLVLA] duplicate_request request_id={request_id}",
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
                    "[MOTIONFORGE-SMOLVLA] action_sent "
                    f"count={actions_sent} request_id={request_id} "
                    f"observation_index={packet['observation_index']} "
                    f"predicted_shape={predicted_actions.shape} sent_shape={actions.shape} "
                    f"inference_s={inference_duration_s:.3f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("[MOTIONFORGE-SMOLVLA] interrupted", flush=True)
        return 130
    finally:
        transport.close()

    print(
        "[MOTIONFORGE-SMOLVLA] done "
        f"episodes={episodes} observations={observations} actions_sent={actions_sent}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
