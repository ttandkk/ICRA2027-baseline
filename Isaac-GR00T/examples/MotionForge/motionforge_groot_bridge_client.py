#!/usr/bin/env python3
"""Bridge MotionForge benchmark observations to a GR00T policy."""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_motionforge_root = Path(
    os.environ.get(
        "MOTIONFORGE_ROOT",
        Path(__file__).resolve().parents[4] / "MotionForge",
    )
)
_motionforge_source = str(_motionforge_root / "source" / "motionforge")
if _motionforge_source not in sys.path:
    sys.path.insert(0, _motionforge_source)

from motionforge.benchmark.client import BenchmarkClientBridge, ClientBridgeConfig
from motionforge.benchmark.protocol import ObservationRequest, ResetMessage

MOTIONFORGE_RGB_KEYS = {
    "overview": "observation.images.overview",
    "front": "observation.images.front",
    "wrist": "observation.images.wrist",
}
MOTIONFORGE_STATE_KEY = "observation.state"


@dataclass(slots=True)
class ObservationHistory:
    """Recent MotionForge observations used to satisfy GR00T temporal horizons."""

    maxlen: int
    frames: deque[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.frames = deque(maxlen=max(1, int(self.maxlen)))

    def append(self, message: dict[str, Any]) -> None:
        self.frames.append(message)

    def clear(self) -> None:
        self.frames.clear()

    def latest_sequence(self, horizon: int) -> list[dict[str, Any]]:
        if not self.frames:
            raise RuntimeError("Observation history is empty.")
        horizon = max(1, int(horizon))
        values = list(self.frames)[-horizon:]
        while len(values) < horizon:
            values.insert(0, values[0])
        return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motionforge-host", default="127.0.0.1")
    parser.add_argument("--motionforge-obs-port", type=int, default=3196)
    parser.add_argument("--motionforge-act-port", type=int, default=3198)
    parser.add_argument("--groot-model-path", required=False, default=None, help="Model checkpoint path or HF id.")
    parser.add_argument("--groot-embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--groot-device", default="cuda")
    parser.add_argument("--groot-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--groot-accel-mode",
        choices=("pytorch", "torch_compile", "trt_full_pipeline", "trt_action_head", "trt_dit_only"),
        default="pytorch",
        help="Inference acceleration mode applied to the in-process Gr00tPolicy.",
    )
    parser.add_argument(
        "--groot-trt-engine-path",
        default=None,
        help="TensorRT engine directory produced by scripts/deployment/build_trt_pipeline.py.",
    )
    parser.add_argument("--groot-compile-mode", default="max-autotune")
    parser.add_argument(
        "--groot-denoising-steps",
        type=int,
        default=None,


        help="Override action_head.num_inference_timesteps; leave unset to use the checkpoint default.",
    )
    parser.add_argument(
        "--use-sim-policy-wrapper",
        action="store_true",
        help="Wrap direct Gr00tPolicy with Gr00tSimPolicyWrapper for flat observation/action keys.",
    )
    parser.add_argument(
        "--groot-observation-format",
        choices=("flat", "nested"),
        default="flat",
        help="Use flat when --use-sim-policy-wrapper is enabled.",
    )
    parser.add_argument(
        "--video-map",
        default="",
        help=(
            "Comma-separated GR00T_KEY=MF_VIEW mappings. MF_VIEW is overview/front/wrist. "
            "Unspecified keys are assigned in overview,front,wrist order."
        ),
    )
    parser.add_argument("--state-key", default=None, help="GR00T state key receiving MotionForge observation.state.")
    parser.add_argument("--language-key", default=None, help="GR00T language key receiving MotionForge language_instruction.")
    parser.add_argument("--groot-action-key", default=None, help="GR00T action key to forward; defaults to the only/first key.")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Stop after this many acknowledged episode results.",
    )
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= int(args.motionforge_obs_port) <= 65535:
        parser.error("--motionforge-obs-port must be in [1, 65535]")
    if not 1 <= int(args.motionforge_act_port) <= 65535:
        parser.error("--motionforge-act-port must be in [1, 65535]")
    if int(args.motionforge_obs_port) == int(args.motionforge_act_port):
        parser.error("observation and action ports must differ")
    if int(args.num_episodes) < 1:
        parser.error("--num-episodes must be >= 1")
    if int(args.print_every) < 0:
        parser.error("--print-every must be >= 0")
    return args


def load_groot_policy(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if not args.groot_model_path:
        raise ValueError("--groot-model-path is required.")

    from gr00t.policy.gr00t_policy import Gr00tPolicy

    base_policy = Gr00tPolicy(
        embodiment_tag=args.groot_embodiment_tag,
        model_path=args.groot_model_path,
        device=args.groot_device,
        strict=args.groot_strict,
    )
    apply_groot_acceleration(base_policy, args)
    policy = base_policy
    if args.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(base_policy)
    modality_config = policy.get_modality_config()
    print(
        "[MOTIONFORGE-GR00T] "
        f"loaded groot_direct model={args.groot_model_path} device={args.groot_device} "
        f"accel={args.groot_accel_mode} modalities={list(modality_config.keys())}",
        flush=True,
    )
    return policy, modality_config


def apply_groot_acceleration(policy: Any, args: argparse.Namespace) -> None:
    action_head = getattr(getattr(policy, "model", None), "action_head", None)
    if action_head is None:
        raise RuntimeError("Loaded GR00T policy does not expose model.action_head.")

    if args.groot_denoising_steps is not None:
        if args.groot_denoising_steps <= 0:
            raise ValueError("--groot-denoising-steps must be positive.")
        action_head.num_inference_timesteps = int(args.groot_denoising_steps)

    if args.groot_accel_mode == "pytorch":
        return

    if args.groot_accel_mode == "torch_compile":
        import torch

        if not hasattr(action_head, "model") or not hasattr(action_head.model, "forward"):
            raise RuntimeError("torch_compile requires an unpatched PyTorch action_head.model.forward.")
        action_head.model.forward = torch.compile(
            action_head.model.forward,
            mode=args.groot_compile_mode,
        )
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        print(
            "[MOTIONFORGE-GR00T] "
            f"enabled torch.compile mode={args.groot_compile_mode}",
            flush=True,
        )
        return

    if not args.groot_trt_engine_path:
        raise ValueError(f"--groot-trt-engine-path is required for {args.groot_accel_mode}.")
    deployment_dir = Path(__file__).resolve().parents[2] / "scripts" / "deployment"
    if str(deployment_dir) not in sys.path:
        sys.path.insert(0, str(deployment_dir))

    from trt_model_forward import setup_tensorrt_engines

    trt_modes = {
        "trt_full_pipeline": "n17_full_pipeline",
        "trt_action_head": "action_head",
        "trt_dit_only": "dit_only",
    }
    setup_tensorrt_engines(policy, args.groot_trt_engine_path, mode=trt_modes[args.groot_accel_mode])
    print(
        "[MOTIONFORGE-GR00T] "
        f"enabled {args.groot_accel_mode} engines={args.groot_trt_engine_path}",
        flush=True,
    )


def close_groot_acceleration(policy: Any, args: argparse.Namespace) -> None:
    if policy is None or not args.groot_accel_mode.startswith("trt_"):
        return
    deployment_dir = Path(__file__).resolve().parents[2] / "scripts" / "deployment"
    if str(deployment_dir) not in sys.path:
        sys.path.insert(0, str(deployment_dir))
    try:
        from trt_model_forward import close_tensorrt_engines

        close_tensorrt_engines(policy)
    except Exception as exc:
        print(f"[MOTIONFORGE-GR00T] failed to close TensorRT engines: {exc}", flush=True)


@dataclass(slots=True)
class GR00TInference:
    """GR00T model adapter with RESET-scoped history and RNG state."""

    args: argparse.Namespace
    policy: Any = field(init=False)
    modality_config: dict[str, Any] = field(init=False)
    history: ObservationHistory = field(init=False)
    video_map: dict[str, str] = field(init=False)
    predictions: int = field(init=False, default=0)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.policy, self.modality_config = load_groot_policy(self.args)
        self.history = ObservationHistory(
            maxlen=max(
                _horizon(self.modality_config, "video"),
                _horizon(self.modality_config, "state"),
            )
        )
        self.video_map = parse_video_map(self.args.video_map)

    @property
    def action_horizon(self) -> int:
        return _horizon(self.modality_config, "action")

    def reset(self, reset: ResetMessage) -> None:
        """Clear temporal state and seed model RNGs from the server RESET."""
        seed = int(reset.seed)
        if not 0 <= seed < 2**63 - 1:
            raise ValueError(f"RESET seed must be in [0, 2**63 - 2], got {seed}.")
        self.history.clear()
        seed_policy_rng(seed)
        reset_policy = getattr(self.policy, "reset", None)
        if callable(reset_policy):
            reset_policy()

    def predict(self, observation: ObservationRequest) -> np.ndarray:
        message = observation.to_dict()
        self.history.append(message)
        policy_observation = build_groot_observation(
            message=message,
            history=self.history,
            modality_config=self.modality_config,
            observation_format=self.args.groot_observation_format,
            video_map=self.video_map,
            state_key=self.args.state_key,
            language_key=self.args.language_key,
        )
        self.predictions += 1
        if self.args.print_every and (
            self.predictions == 1 or self.predictions % int(self.args.print_every) == 0
        ):
            print_shape_summary(self.predictions, policy_observation)
        action, _info = self.policy.get_action(policy_observation)
        return motionforge_action_chunk(action=action, action_key=self.args.groot_action_key)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_groot_acceleration(self.policy, self.args)
        close_policy = getattr(self.policy, "close", None)
        if callable(close_policy):
            close_policy()


def seed_policy_rng(seed: int) -> None:
    """Seed the global RNGs used internally by the GR00T policy implementation."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    inference = GR00TInference(args)
    print(
        "[MOTIONFORGE-GR00T] "
        f"scheduling action_horizon={inference.action_horizon} "
        "wire_horizon=server_required_16",
        flush=True,
    )
    print(
        "[MOTIONFORGE-GR00T] "
        f"listening motionforge={args.motionforge_host} obs_port={args.motionforge_obs_port} "
        f"act_port={args.motionforge_act_port}",
        flush=True,
    )
    try:
        bridge = BenchmarkClientBridge(
            policy=inference,
            config=ClientBridgeConfig(
                host=args.motionforge_host,
                obs_port=int(args.motionforge_obs_port),
                act_port=int(args.motionforge_act_port),
                num_episodes=int(args.num_episodes),
                print_every=int(args.print_every),
            ),
            log=lambda message: print(f"[MOTIONFORGE-GR00T] {message}", flush=True),
        )
        bridge.run()
    except KeyboardInterrupt:
        print("[MOTIONFORGE-GR00T] interrupted", flush=True)
        return 130
    finally:
        inference.close()
    print(f"[MOTIONFORGE-GR00T] done episodes={args.num_episodes}", flush=True)
    return 0


def build_groot_observation(
    *,
    message: dict[str, Any],
    history: ObservationHistory,
    modality_config: dict[str, Any] | None,
    observation_format: str,
    video_map: dict[str, str],
    state_key: str | None,
    language_key: str | None,
) -> dict[str, Any]:
    video_keys = _modality_keys(modality_config, "video") or tuple(video_map.keys()) or ("overview", "front", "wrist")
    state_keys = _modality_keys(modality_config, "state") or (state_key or "state",)
    language_keys = _modality_keys(modality_config, "language") or (language_key or "task",)
    video_horizon = _horizon(modality_config, "video")
    state_horizon = _horizon(modality_config, "state")

    video_mapping = complete_video_map(video_keys, video_map)
    state_name = state_key or state_keys[0]
    language_name = language_key or language_keys[0]
    instruction = str(message.get("language_instruction", message.get("task", "")))

    video_sequence = history.latest_sequence(video_horizon)
    state_sequence = history.latest_sequence(state_horizon)
    nested = {"video": {}, "state": {}, "language": {}}
    for key in video_keys:
        motionforge_view = video_mapping[key]
        source_key = MOTIONFORGE_RGB_KEYS[motionforge_view]
        nested["video"][key] = np.stack(
            [_as_rgb_uint8(frame[source_key]) for frame in video_sequence],
            axis=0,
        )[None, ...]
    for key in state_keys:
        nested["state"][key] = np.stack(
            [
                _state_for_key(
                    key,
                    np.asarray(frame[MOTIONFORGE_STATE_KEY], dtype=np.float32),
                    state_keys=state_keys,
                )
                for frame in state_sequence
            ],
            axis=0,
        )[None, ...]
    for key in language_keys:
        if key != language_name and len(language_keys) > 1:
            raise ValueError(
                f"Multiple GR00T language keys detected: {language_keys}. "
                "Pass --language-key for the target key or extend the bridge mapping."
            )
        nested["language"][key] = [[instruction]]

    if observation_format == "nested":
        return nested

    flat: dict[str, Any] = {}
    flat.update({f"video.{key}": value for key, value in nested["video"].items()})
    flat.update({f"state.{key}": value for key, value in nested["state"].items()})
    flat[language_name] = [instruction]
    return flat


def motionforge_action_chunk(
    *, action: dict[str, Any], action_key: str | None
) -> np.ndarray:
    """Select and validate the canonical 10D MotionForge source-action chunk."""
    selected_key, array = select_action(action, action_key)
    if array.ndim == 3:
        if int(array.shape[0]) != 1:
            raise ValueError(
                f"GR00T action {selected_key!r} batch must be 1, got {array.shape}."
            )
        array = array[0]
    if array.ndim != 2 or int(array.shape[1]) != 10:
        raise ValueError(
            f"GR00T action {selected_key!r} must have shape [T, 10] for the formal "
            f"MotionForge action schema, got {array.shape}."
        )
    array = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"GR00T action {selected_key!r} contains non-finite values.")
    return array


def select_action(action: dict[str, Any], action_key: str | None) -> tuple[str, np.ndarray]:
    if action_key is None:
        combined = combine_known_action_parts(action)
        if combined is not None:
            return combined
        if len(action) == 1:
            action_key = next(iter(action))
        else:
            candidates = ", ".join(sorted(action))
            raise ValueError(f"Multiple GR00T action keys found ({candidates}); pass --groot-action-key.")
    if action_key not in action:
        raise KeyError(f"GR00T action key {action_key!r} not found. Available keys: {sorted(action)}")
    return action_key, np.asarray(action[action_key], dtype=np.float32)


def combine_known_action_parts(action: dict[str, Any]) -> tuple[str, np.ndarray] | None:
    for eef_suffix, gripper_suffix in (
        ("eef_pose_rot6d", "gripper"),
        ("eef_9d", "gripper_position"),
    ):
        eef_key = _find_action_key(action, eef_suffix)
        gripper_key = _find_action_key(action, gripper_suffix)
        if eef_key is None or gripper_key is None:
            continue
        eef = np.asarray(action[eef_key], dtype=np.float32)
        gripper = np.asarray(action[gripper_key], dtype=np.float32)
        if eef.ndim == gripper.ndim and eef.shape[:-1] == gripper.shape[:-1]:
            return f"{eef_key}+{gripper_key}", np.concatenate([eef, gripper], axis=-1)
        raise ValueError(
            f"Cannot combine GR00T action keys {eef_key!r} shape={eef.shape} and "
            f"{gripper_key!r} shape={gripper.shape}."
        )
    return None


def _find_action_key(action: dict[str, Any], suffix: str) -> str | None:
    for candidate in (suffix, f"action.{suffix}"):
        if candidate in action:
            return candidate
    return None


def parse_video_map(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not value:
        return mapping
    for item in value.split(","):
        if not item.strip():
            continue
        key, sep, view = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid --video-map item {item!r}; expected GR00T_KEY=overview.")
        view = view.strip()
        if view not in MOTIONFORGE_RGB_KEYS:
            raise ValueError(f"Invalid MotionForge view {view!r}; expected one of {sorted(MOTIONFORGE_RGB_KEYS)}.")
        mapping[key.strip()] = view
    return mapping


def complete_video_map(video_keys: tuple[str, ...], mapping: dict[str, str]) -> dict[str, str]:
    default_views = ("overview", "front", "wrist")
    completed = dict(mapping)
    for index, key in enumerate(video_keys):
        if key not in completed:
            completed[key] = default_views[min(index, len(default_views) - 1)]
    return completed


def _modality_keys(modality_config: dict[str, Any] | None, modality: str) -> tuple[str, ...]:
    if not modality_config or modality not in modality_config:
        return ()
    return tuple(str(key) for key in modality_config[modality].modality_keys)


def _horizon(modality_config: dict[str, Any] | None, modality: str) -> int:
    if not modality_config or modality not in modality_config:
        return 1
    return max(1, len(modality_config[modality].delta_indices))


def _state_for_key(key: str, value: np.ndarray, *, state_keys: tuple[str, ...] = ()) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    joint_position = flat[:-1] if flat.size > 1 else flat
    gripper_width = flat[-1:] if flat.size else np.zeros(1, dtype=np.float32)
    if key in {"state", "observation.state"}:
        return flat
    if key == "eef_9d":
        return _fit_state_width(flat[:9], 9)
    if key in {"gripper", "gripper_position", "gripper_width"}:
        return _fit_state_width(gripper_width, 1)
    if key == "joint_position":
        width = 9 if "gripper_width" in state_keys else 7
        return _fit_state_width(joint_position, width)
    if key == "single_arm":
        return _fit_state_width(joint_position[:5], 5)
    return flat


def _fit_state_width(value: np.ndarray, width: int) -> np.ndarray:
    result = np.zeros(int(width), dtype=np.float32)
    count = min(result.shape[0], int(value.size))
    if count:
        result[:count] = np.asarray(value[:count], dtype=np.float32)
    return result


def _as_rgb_uint8(value: Any, *, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected image with shape HxWxC, got {array.shape}.")
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if target_shape is not None and tuple(array.shape[:2]) != tuple(target_shape):
        array = _resize_with_letterbox(array, target_shape)
    return np.ascontiguousarray(array)


def _resize_with_letterbox(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = (int(value) for value in target_shape)
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target image shape: {target_shape!r}.")
    height, width = array.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid source image shape: {array.shape!r}.")

    scale = min(target_h / height, target_w / width)
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))

    try:
        from PIL import Image

        resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
        resized = np.asarray(
            Image.fromarray(array).resize((resized_w, resized_h), resampling),
            dtype=np.uint8,
        )
    except Exception:
        resized = _resize_nearest(array, resized_h, resized_w)

    output = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = max(0, (target_h - resized_h) // 2)
    left = max(0, (target_w - resized_w) // 2)
    output[top : top + resized_h, left : left + resized_w] = resized[:target_h, :target_w]
    return output


def _resize_nearest(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    y_indices = np.linspace(0, array.shape[0] - 1, int(target_h)).astype(np.int64)
    x_indices = np.linspace(0, array.shape[1] - 1, int(target_w)).astype(np.int64)
    return array[y_indices][:, x_indices].astype(np.uint8, copy=False)


def print_shape_summary(step: int, observation: dict[str, Any]) -> None:
    shapes = {
        key: tuple(value.shape) if isinstance(value, np.ndarray) else type(value).__name__
        for key, value in observation.items()
        if key.startswith(("video.", "state."))
    }
    print(f"[MOTIONFORGE-GR00T] obs={step} shapes={shapes}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
