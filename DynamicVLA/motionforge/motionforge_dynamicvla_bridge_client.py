#!/usr/bin/env python3
"""Bridge MotionForge FC benchmark observations to a DynamicVLA policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


DYNAMICVLA_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = DYNAMICVLA_ROOT.parents[1]
MOTIONFORGE_ROOT = Path(
    os.environ.get("MOTIONFORGE_ROOT", WORKSPACE_ROOT / "MotionForge")
).expanduser()

for source_root in (
    DYNAMICVLA_ROOT,
    MOTIONFORGE_ROOT / "source" / "motionforge",
):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import numpy as np
import torch

from motionforge.benchmark.client import BenchmarkClientBridge, ClientBridgeConfig
from motionforge.benchmark.protocol import ObservationPacket, PROTOCOL_VERSION


EXPECTED_LEROBOT_VERSION = "0.3.3"
STATE_KEY = "observation.state"
TASK_KEY = "task"
ACTION_DIM = 10
POLICY_TO_MOTIONFORGE_IMAGE = {
    "observation.images.opst_cam": "observation.images.front",
    "observation.images.wrist_cam": "observation.images.wrist",
}


@dataclass(slots=True)
class DynamicVLAInference:
    """DynamicVLA inference adapter preserving its native preprocessing contract."""

    checkpoint: Path
    device: str = "cuda:0"
    action_hz: float = 30.0

    config: Any = field(init=False)
    policy: Any = field(init=False)
    delta_timestamps: tuple[int, ...] = field(init=False)
    _history: deque[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.checkpoint = self.checkpoint.expanduser().resolve()
        payload = validate_checkpoint_files(self.checkpoint)
        validate_checkpoint_contract(payload)
        torch_device = validate_device(self.device)

        # Reuse the native loader so model construction, normalization buffers,
        # checkpoint key handling, and DynamicVLA-specific configuration stay aligned.
        from scripts.inference import get_vla_model

        self.policy, runtime_cfg = get_vla_model(
            str(self.checkpoint),
            use_delta_action=bool(payload["use_delta_action"]),
            streaming=False,
        )
        self.policy = self.policy.to(torch_device)
        self.policy.config.device = str(torch_device)
        self.policy.eval()
        self.config = self.policy.config
        self.delta_timestamps = tuple(int(value) for value in runtime_cfg["delta_timestamps"])
        validate_runtime_contract(self.config, self.delta_timestamps)
        self._history = deque(maxlen=history_capacity(self.delta_timestamps))
        self.reset()

    @property
    def action_horizon(self) -> int:
        return int(self.config.n_action_steps)

    def reset(self) -> None:
        if hasattr(self, "_history"):
            self._history.clear()
        if hasattr(self, "policy"):
            self.policy.reset()

    def predict(self, observation: ObservationPacket) -> np.ndarray:
        return self.predict_chunk(observation.to_dict())

    def predict_chunk(self, message: dict[str, Any]) -> np.ndarray:
        require_protocol(message)
        self._history.append(message)
        selected = select_observation_history(self._history, self.delta_timestamps)
        batch, latest_state = build_dynamicvla_batch(
            selected,
            input_features=self.config.input_features,
            device=torch.device(self.device),
        )

        with torch.inference_mode():
            # predict_action_chunk() intentionally returns unnormalized delta actions.
            # DynamicVLA's native select_action() converts these back to absolute
            # actions; reproduce that conversion before sending a complete chunk.
            actions = self.policy.predict_action_chunk(batch)
            actions = restore_absolute_actions(
                actions,
                latest_state,
                use_delta_action=bool(self.config.use_delta_action),
            )

        if not isinstance(actions, torch.Tensor):
            raise TypeError(
                f"DynamicVLA returned {type(actions).__name__}, expected torch.Tensor."
            )
        array = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        expected_shape = (1, self.action_horizon, ACTION_DIM)
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"Expected DynamicVLA actions {expected_shape}, got {array.shape}."
            )
        array = np.ascontiguousarray(array[0])
        if not np.isfinite(array).all():
            raise ValueError("DynamicVLA produced non-finite actions.")
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
        help="Actions sent in legacy wall_clock_strict mode.",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=8,
        help="Actions executed before requesting a replacement plan.",
    )
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--validate-checkpoint",
        action="store_true",
        help="Load the checkpoint, run one synthetic inference, and exit.",
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


def validate_checkpoint_files(checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    required = ("config.json", "model.safetensors")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required files: {missing}")
    with (checkpoint / "config.json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint config.json must contain an object.")
    return payload


def validate_checkpoint_contract(payload: dict[str, Any]) -> None:
    expected_features = {
        STATE_KEY: (ACTION_DIM,),
        "observation.images.opst_cam": (3, 240, 320),
        "observation.images.wrist_cam": (3, 160, 160),
    }
    if payload.get("type") != "dynamicvla":
        raise ValueError(f"Expected config type 'dynamicvla', got {payload.get('type')!r}.")
    if int(payload.get("n_obs_steps", -1)) != 2:
        raise ValueError(f"Expected n_obs_steps=2, got {payload.get('n_obs_steps')!r}.")
    if int(payload.get("chunk_size", -1)) != 20:
        raise ValueError(f"Expected chunk_size=20, got {payload.get('chunk_size')!r}.")
    if int(payload.get("n_action_steps", -1)) != 20:
        raise ValueError(
            f"Expected n_action_steps=20, got {payload.get('n_action_steps')!r}."
        )
    if payload.get("use_delta_action") is not True:
        raise ValueError("The FC checkpoint must declare use_delta_action=true.")
    if payload.get("enable_streaming") is not False:
        raise ValueError("The FC checkpoint must declare enable_streaming=false.")

    features = payload.get("input_features")
    if not isinstance(features, dict) or set(features) != set(expected_features):
        raise ValueError(
            "Checkpoint input features differ from the DynamicVLA FC contract: "
            f"{sorted(features) if isinstance(features, dict) else features!r}."
        )
    for key, expected_shape in expected_features.items():
        actual_shape = tuple(features[key].get("shape", ()))
        if actual_shape != expected_shape:
            raise ValueError(f"{key}: expected shape {expected_shape}, got {actual_shape}.")
    action_shape = tuple(
        payload.get("output_features", {}).get("action", {}).get("shape", ())
    )
    if action_shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape ({ACTION_DIM},), got {action_shape}.")
    timestamps = payload.get("delta_timestamps", {}).get("observation")
    if timestamps != [-2, 0]:
        raise ValueError(f"Expected observation delta_timestamps [-2, 0], got {timestamps!r}.")


def validate_runtime_contract(config: Any, delta_timestamps: tuple[int, ...]) -> None:
    if int(config.n_obs_steps) != len(delta_timestamps):
        raise ValueError(
            "Runtime n_obs_steps differs from observation timestamps: "
            f"{config.n_obs_steps} vs {delta_timestamps}."
        )
    if delta_timestamps != (-2, 0):
        raise ValueError(f"Expected runtime timestamps (-2, 0), got {delta_timestamps}.")
    if int(config.n_action_steps) != 20 or int(config.chunk_size) != 20:
        raise ValueError(
            "Runtime action horizon differs from the trained 20-step FC contract."
        )


def validate_device(device: str) -> torch.device:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return torch_device


def history_capacity(delta_timestamps: tuple[int, ...]) -> int:
    if not delta_timestamps or max(delta_timestamps) != 0:
        raise ValueError(
            f"Observation timestamps must be non-empty and end at 0, got {delta_timestamps}."
        )
    if any(value > 0 for value in delta_timestamps):
        raise ValueError(f"Future observation timestamps are unsupported: {delta_timestamps}.")
    return 1 - min(delta_timestamps)


def select_observation_history(
    history: deque[dict[str, Any]], delta_timestamps: tuple[int, ...]
) -> list[dict[str, Any]]:
    if not history:
        raise ValueError("Cannot select observations from empty history.")
    observations = list(history)
    last_index = len(observations) - 1
    return [observations[max(0, min(last_index, last_index + dt))] for dt in delta_timestamps]


def build_dynamicvla_batch(
    messages: list[dict[str, Any]],
    *,
    input_features: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    if not messages:
        raise ValueError("At least one observation is required.")
    for message in messages:
        require_protocol(message)

    state_frames = [
        state_to_tensor(require_field(message, STATE_KEY), input_features[STATE_KEY], device)
        for message in messages
    ]
    state = torch.stack(state_frames, dim=0).unsqueeze(0)
    latest_state = state[:, -1:, :].clone()
    batch: dict[str, Any] = {
        STATE_KEY: state,
        TASK_KEY: [require_task(messages[-1])],
        "index": int(messages[-1].get("index", 0)),
    }

    for policy_key, motionforge_key in POLICY_TO_MOTIONFORGE_IMAGE.items():
        feature = input_features[policy_key]
        image_frames = [
            image_to_tensor(require_field(message, motionforge_key), motionforge_key, feature, device)
            for message in messages
        ]
        batch[policy_key] = torch.stack(image_frames, dim=0).unsqueeze(0)
    return batch, latest_state


def state_to_tensor(value: Any, feature: Any, device: torch.device) -> torch.Tensor:
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    expected_shape = tuple(feature.shape)
    if tuple(state.shape) != expected_shape:
        raise ValueError(f"{STATE_KEY}: expected {expected_shape}, got {state.shape}.")
    if not np.isfinite(state).all():
        raise ValueError(f"{STATE_KEY} contains non-finite values.")
    return torch.from_numpy(np.ascontiguousarray(state)).to(device)


def image_to_tensor(
    value: Any,
    key: str,
    feature: Any,
    device: torch.device,
) -> torch.Tensor:
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"{key}: expected HWC image, got shape {image.shape}.")
    if image.shape[-1] == 4:
        image = image[..., :3]
    expected_chw = tuple(feature.shape)
    expected_hwc = (expected_chw[1], expected_chw[2], expected_chw[0])
    if tuple(image.shape) != expected_hwc:
        raise ValueError(f"{key}: expected HWC {expected_hwc}, got {image.shape}.")
    if image.dtype != np.uint8:
        raise TypeError(f"{key}: expected uint8, got {image.dtype}.")
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(chw).to(device=device, dtype=torch.float32).div_(255.0)


def restore_absolute_actions(
    actions: torch.Tensor,
    latest_state: torch.Tensor,
    *,
    use_delta_action: bool,
) -> torch.Tensor:
    if not isinstance(actions, torch.Tensor):
        raise TypeError(f"Expected actions Tensor, got {type(actions).__name__}.")
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected actions [1, time, {ACTION_DIM}], got {tuple(actions.shape)}.")
    if tuple(latest_state.shape) != (1, 1, ACTION_DIM):
        raise ValueError(
            f"Expected latest_state (1, 1, {ACTION_DIM}), got {tuple(latest_state.shape)}."
        )
    if not use_delta_action:
        return actions
    absolute = actions.clone()
    action_dim = absolute.shape[-1] - 1
    absolute[..., :action_dim] += latest_state[..., :action_dim]
    return absolute


def require_field(message: dict[str, Any], key: str) -> Any:
    if key not in message:
        raise KeyError(f"MotionForge observation is missing {key!r}.")
    return message[key]


def require_task(message: dict[str, Any]) -> str:
    task = message.get(TASK_KEY, message.get("language_instruction"))
    if not isinstance(task, str) or not task.strip():
        raise ValueError("MotionForge observation requires a non-empty task instruction.")
    return task


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


def synthetic_message(inference: DynamicVLAInference) -> dict[str, Any]:
    message: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "index": 0,
        "request_id": 0,
        "request_kind": "validation",
        "task": "Pick up the object and place it in the target container.",
        STATE_KEY: np.zeros((ACTION_DIM,), dtype=np.float32),
    }
    for policy_key, motionforge_key in POLICY_TO_MOTIONFORGE_IMAGE.items():
        channels, height, width = inference.config.input_features[policy_key].shape
        message[motionforge_key] = np.zeros((height, width, channels), dtype=np.uint8)
    return message


def run_validation(inference: DynamicVLAInference) -> int:
    started_at = time.perf_counter()
    actions = inference.predict_chunk(synthetic_message(inference))
    elapsed_s = time.perf_counter() - started_at
    print(
        "[MOTIONFORGE-DYNAMICVLA] validation_passed "
        f"checkpoint={inference.checkpoint} shape={actions.shape} "
        f"min={actions.min():.6f} max={actions.max():.6f} inference_s={elapsed_s:.3f}",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    current_version = installed_lerobot_version()
    if current_version != EXPECTED_LEROBOT_VERSION:
        print(
            "[MOTIONFORGE-DYNAMICVLA] WARNING: "
            f"DynamicVLA requirements pin lerobot=={EXPECTED_LEROBOT_VERSION}, "
            f"runtime reports lerobot=={current_version}.",
            flush=True,
        )

    inference = DynamicVLAInference(
        checkpoint=args.model_path,
        device=args.device,
        action_hz=float(args.action_hz),
    )
    if args.send_horizon > inference.action_horizon:
        raise ValueError(
            f"--send-horizon={args.send_horizon} exceeds DynamicVLA horizon "
            f"{inference.action_horizon}."
        )
    print(
        "[MOTIONFORGE-DYNAMICVLA] loaded "
        f"checkpoint={inference.checkpoint} device={args.device} "
        f"lerobot={current_version} action_horizon={inference.action_horizon} "
        f"send_horizon={args.send_horizon} execution_horizon={args.execution_horizon} "
        "streaming=false action_alignment=observation_aligned",
        flush=True,
    )
    if args.validate_checkpoint:
        return run_validation(inference)

    print(
        "[MOTIONFORGE-DYNAMICVLA] listening "
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
        log=lambda message: print(f"[MOTIONFORGE-DYNAMICVLA] {message}", flush=True),
    )
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("[MOTIONFORGE-DYNAMICVLA] interrupted", flush=True)
        return 130

    print(f"[MOTIONFORGE-DYNAMICVLA] done episodes={args.num_episodes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
