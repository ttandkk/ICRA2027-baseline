#!/usr/bin/env python3
"""Bridge MotionForge benchmark observations to a GR00T policy."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_VERSION = "motionforge.benchmark.v1"
MOTIONFORGE_RGB_KEYS = {
    "overview": "observation.images.overview",
    "front": "observation.images.front",
    "wrist": "observation.images.wrist",
}
MOTIONFORGE_STATE_KEY = "observation.state"


@dataclass(slots=True)
class MotionForgeTransport:
    """Minimal MotionForge benchmark client transport without importing MotionForge."""

    host: str
    obs_port: int
    act_port: int

    _zmq: Any = field(init=False)
    _context: Any = field(init=False)
    _obs_socket: Any = field(init=False)
    _act_socket: Any = field(init=False)

    def __post_init__(self) -> None:
        import zmq

        self._zmq = zmq
        self._context = zmq.Context()
        self._obs_socket = self._context.socket(zmq.SUB)
        self._obs_socket.connect(f"tcp://{self.host}:{self.obs_port}")
        self._obs_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._obs_socket.RCVHWM = 1
        self._act_socket = self._context.socket(zmq.PUSH)
        self._act_socket.connect(f"tcp://{self.host}:{self.act_port}")

    def recv_latest(self) -> dict[str, Any] | None:
        message = None
        while True:
            try:
                message = self._obs_socket.recv_pyobj(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
        return message

    def send_action(self, packet: dict[str, Any]) -> None:
        self._act_socket.send_pyobj(packet, flags=self._zmq.NOBLOCK)

    def close(self) -> None:
        self._obs_socket.close(linger=0)
        self._act_socket.close(linger=0)
        self._context.term()


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


@dataclass(slots=True)
class BridgeTiming:
    """Accumulated wall-clock timing for bridge-side processing."""

    received_observations: int = 0
    observations: int = 0
    skipped_observations: int = 0
    actions: int = 0
    build_observation_s: float = 0.0
    policy_get_action_s: float = 0.0
    send_action_s: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        observation_count = max(1, int(self.observations))
        action_count = max(1, int(self.actions))
        return {
            "received_observations": int(self.received_observations),
            "observations": int(self.observations),
            "skipped_observations": int(self.skipped_observations),
            "actions": int(self.actions),
            "build_observation_s": self.build_observation_s,
            "policy_get_action_s": self.policy_get_action_s,
            "send_action_s": self.send_action_s,
            "build_observation_ms_per_obs": self.build_observation_s * 1000.0 / observation_count,
            "policy_get_action_ms_per_action": self.policy_get_action_s * 1000.0 / action_count,
            "send_action_ms_per_action": self.send_action_s * 1000.0 / action_count,
        }

    def add_received_observation(self) -> None:
        self.received_observations += 1

    def add_build_observation(self, elapsed_s: float) -> None:
        self.observations += 1
        self.build_observation_s += float(elapsed_s)

    def add_skipped_observation(self) -> None:
        self.skipped_observations += 1

    def add_policy_get_action(self, elapsed_s: float) -> None:
        self.policy_get_action_s += float(elapsed_s)

    def add_send_action(self, elapsed_s: float) -> None:
        self.actions += 1
        self.send_action_s += float(elapsed_s)


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
        "--dry-run",
        action="store_true",
        help="Convert observations and print shapes without calling GR00T or sending actions.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Stop after this many episode_result messages; use 0 to keep listening until interrupted.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many MotionForge observations.")
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=8,
        help=(
            "Number of env observation steps to execute before requesting a fresh GR00T "
            "action chunk. Must be <= len(action.delta_indices). Use 1 for the old "
            "per-observation replan behavior."
        ),
    )
    parser.add_argument(
        "--action-hz",
        type=float,
        default=120.0,
        help="Temporal frequency represented by consecutive policy actions.",
    )
    parser.add_argument(
        "--action-alignment",
        choices=("observation_aligned", "execution_aligned"),
        default="observation_aligned",
        help="Whether action[0] is aligned to observation capture or execution time.",
    )
    parser.add_argument(
        "--max-inference-hz",
        type=float,
        default=30.0,
        help="Maximum request rate advertised to MotionForge.",
    )
    parser.add_argument("--poll-sleep", type=float, default=0.001)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()
    if int(args.execution_horizon) < 1:
        raise ValueError("--execution-horizon must be >= 1.")
    if float(args.action_hz) <= 0.0:
        raise ValueError("--action-hz must be > 0.")
    if float(args.max_inference_hz) <= 0.0:
        raise ValueError("--max-inference-hz must be > 0.")
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

    from gr00t.deployment.modes import InferenceMode
    from trt_model_forward import setup_tensorrt_engines

    trt_modes = {
        "trt_full_pipeline": InferenceMode.n17_full_pipeline,
        "trt_action_head": InferenceMode.action_head,
        "trt_dit_only": InferenceMode.dit_only,
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


def main() -> int:
    args = parse_args()
    if args.poll_sleep < 0.0:
        raise ValueError("--poll-sleep must be >= 0.")
    if args.print_every < 0:
        raise ValueError("--print-every must be >= 0.")
    if args.num_episodes < 0:
        raise ValueError("--num-episodes must be >= 0.")

    motionforge = MotionForgeTransport(
        host=args.motionforge_host,
        obs_port=args.motionforge_obs_port,
        act_port=args.motionforge_act_port,
    )
    policy = None
    modality_config = None
    history = ObservationHistory(maxlen=32)
    steps_seen = 0
    actions_sent = 0
    episodes_seen = 0
    episode_steps_seen = 0
    episode_actions_sent = 0
    episode_timing = BridgeTiming()
    total_timing = BridgeTiming()
    action_horizon = None
    last_policy_observation_index: int | None = None
    processed_request_ids: set[int] = set()
    try:
        if not args.dry_run:
            policy, modality_config = load_groot_policy(args)
            action_horizon = _horizon(modality_config, "action")
            if int(args.execution_horizon) > action_horizon:
                raise ValueError(
                    f"--execution-horizon={args.execution_horizon} exceeds policy action horizon "
                    f"{action_horizon}. Use a value in [1, {action_horizon}]."
                )
            print(
                "[MOTIONFORGE-GR00T] "
                f"scheduling action_horizon={action_horizon} "
                f"execution_horizon={args.execution_horizon}",
                flush=True,
            )
        else:
            print("[MOTIONFORGE-GR00T] dry-run enabled; GR00T policy will not be called", flush=True)

        print(
            "[MOTIONFORGE-GR00T] "
            f"listening motionforge={args.motionforge_host} obs_port={args.motionforge_obs_port} "
            f"act_port={args.motionforge_act_port}",
            flush=True,
        )
        while True:
            message = motionforge.recv_latest()
            if message is None:
                time.sleep(args.poll_sleep)
                continue
            if "episode_result" in message:
                episodes_seen += 1
                print(
                    "[MOTIONFORGE-GR00T] "
                    f"episode_result episode={episodes_seen} observations={episode_steps_seen} "
                    f"actions_sent={episode_actions_sent} result={message['episode_result']}",
                    flush=True,
                )
                print_bridge_timing("episode_timing", episode_timing, episode=episodes_seen)
                history.clear()
                episode_steps_seen = 0
                episode_actions_sent = 0
                episode_timing = BridgeTiming()
                last_policy_observation_index = None
                processed_request_ids.clear()
                if args.num_episodes and episodes_seen >= args.num_episodes:
                    break
                reset_policy = getattr(policy, "reset", None)
                if callable(reset_policy):
                    reset_policy()
                continue

            request_id = optional_request_id(message)
            if request_id == 0 and message.get("request_kind") == "warmup":
                history.clear()
                processed_request_ids.clear()
                last_policy_observation_index = None
                reset_policy = getattr(policy, "reset", None)
                if callable(reset_policy):
                    reset_policy()
            history.append(message)
            steps_seen += 1
            episode_steps_seen += 1
            episode_timing.add_received_observation()
            total_timing.add_received_observation()
            observation_index = observation_step_index(message, fallback=steps_seen - 1)
            request_driven = request_id is not None
            if request_driven and request_id in processed_request_ids:
                episode_timing.add_skipped_observation()
                total_timing.add_skipped_observation()
                continue
            if request_driven:
                should_request_action = True
            else:
                should_request_action = args.dry_run or should_replan(
                    observation_index=observation_index,
                    last_policy_observation_index=last_policy_observation_index,
                    execution_horizon=int(args.execution_horizon),
                )
            if not should_request_action:
                episode_timing.add_skipped_observation()
                total_timing.add_skipped_observation()
                if args.print_every and (steps_seen == 1 or steps_seen % args.print_every == 0):
                    next_index = int(last_policy_observation_index or 0) + int(args.execution_horizon)
                    print(
                        "[MOTIONFORGE-GR00T] "
                        f"skip_replan obs={steps_seen} index={observation_index} "
                        f"next_replan_index={next_index}",
                        flush=True,
                    )
                if args.max_steps is not None and steps_seen >= args.max_steps:
                    break
                continue

            started_at = time.perf_counter()
            observation = build_groot_observation(
                message=message,
                history=history,
                modality_config=modality_config,
                observation_format=args.groot_observation_format,
                video_map=parse_video_map(args.video_map),
                state_key=args.state_key,
                language_key=args.language_key,
            )
            elapsed_s = time.perf_counter() - started_at
            episode_timing.add_build_observation(elapsed_s)
            total_timing.add_build_observation(elapsed_s)
            if args.print_every and (steps_seen == 1 or steps_seen % args.print_every == 0):
                print_shape_summary(steps_seen, observation)
            if args.dry_run:
                if args.max_steps is not None and steps_seen >= args.max_steps:
                    break
                continue

            assert policy is not None
            started_at = time.perf_counter()
            action, info = policy.get_action(observation)
            elapsed_s = time.perf_counter() - started_at
            episode_timing.add_policy_get_action(elapsed_s)
            total_timing.add_policy_get_action(elapsed_s)
            packet = motionforge_action_packet(
                action=action,
                observation_index=observation_index,
                action_key=args.groot_action_key,
                request_id=request_id,
                action_hz=float(args.action_hz),
                action_alignment=str(args.action_alignment),
                inference_duration_s=elapsed_s,
                max_inference_hz=float(args.max_inference_hz),
                metadata={
                    "client": "motionforge_groot_bridge",
                    "groot_info": safe_metadata(info),
                    "action_horizon": action_horizon,
                    "execution_horizon": int(args.execution_horizon),
                    "request_kind": message.get("request_kind"),
                },
            )
            started_at = time.perf_counter()
            motionforge.send_action(packet)
            elapsed_s = time.perf_counter() - started_at
            episode_timing.add_send_action(elapsed_s)
            total_timing.add_send_action(elapsed_s)
            actions_sent += 1
            episode_actions_sent += 1
            last_policy_observation_index = observation_index
            if request_id is not None:
                processed_request_ids.add(request_id)
            if args.max_steps is not None and steps_seen >= args.max_steps:
                break
    finally:
        close_groot_acceleration(policy, args)
        motionforge.close()
        close_policy = getattr(policy, "close", None)
        if callable(close_policy):
            close_policy()
    print(
        "[MOTIONFORGE-GR00T] "
        f"done episodes={episodes_seen} observations={steps_seen} actions_sent={actions_sent}",
        flush=True,
    )
    print_bridge_timing("total_timing", total_timing)
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


def motionforge_action_packet(
    *,
    action: dict[str, Any],
    observation_index: int,
    action_key: str | None,
    metadata: dict[str, Any],
    request_id: int | None = None,
    action_hz: float | None = None,
    action_alignment: str | None = None,
    inference_duration_s: float | None = None,
    max_inference_hz: float | None = None,
) -> dict[str, Any]:
    selected_key, array = select_action(action, action_key)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        width = int(array.shape[0])
    elif array.ndim == 2:
        width = int(array.shape[1])
    else:
        raise ValueError(f"Unsupported GR00T action shape for {selected_key!r}: {array.shape}")
    if width == 10:
        motionforge_action_key = "eef_xyz_rot6d_gripper"
    elif width == 8:
        motionforge_action_key = "eef_pose_gripper"
    else:
        raise ValueError(
            f"GR00T action {selected_key!r} has width {width}; MotionForge accepts width 10 "
            "`eef_xyz_rot6d_gripper` or width 8 `eef_pose_gripper`. Add an action adapter for this checkpoint."
        )
    packet = {
        "protocol_version": PROTOCOL_VERSION,
        "action": np.asarray(array, dtype=np.float32),
        "action_key": motionforge_action_key,
        "action_frame": "robot",
        "action_representation": "ABSOLUTE",
        "observation_index": int(observation_index),
        "created_time": time.time(),
        "metadata": {**metadata, "groot_action_key": selected_key},
    }
    optional = {
        "request_id": request_id,
        "action_hz": action_hz,
        "action_alignment": action_alignment,
        "inference_duration_s": inference_duration_s,
        "max_inference_hz": max_inference_hz,
    }
    packet.update({key: value for key, value in optional.items() if value is not None})
    return packet


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


def observation_step_index(message: dict[str, Any], *, fallback: int) -> int:
    try:
        return int(message.get("index", fallback))
    except (TypeError, ValueError):
        return int(fallback)


def optional_request_id(message: dict[str, Any]) -> int | None:
    value = message.get("request_id")
    return None if value is None else int(value)


def should_replan(
    *,
    observation_index: int,
    last_policy_observation_index: int | None,
    execution_horizon: int,
) -> bool:
    if last_policy_observation_index is None:
        return True
    return int(observation_index) - int(last_policy_observation_index) >= int(execution_horizon)


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


def print_bridge_timing(label: str, timing: BridgeTiming, *, episode: int | None = None) -> None:
    values = timing.to_dict()
    episode_part = "" if episode is None else f" episode={episode}"
    print(
        "[MOTIONFORGE-GR00T] "
        f"{label}{episode_part} "
        f"received_observations={values['received_observations']} "
        f"policy_observations={values['observations']} "
        f"skipped_observations={values['skipped_observations']} "
        f"actions={values['actions']} "
        f"build_observation_s={values['build_observation_s']:.3f} "
        f"policy_get_action_s={values['policy_get_action_s']:.3f} "
        f"send_action_s={values['send_action_s']:.3f} "
        f"build_observation_ms_per_obs={values['build_observation_ms_per_obs']:.3f} "
        f"policy_get_action_ms_per_action={values['policy_get_action_ms_per_action']:.3f} "
        f"send_action_ms_per_action={values['send_action_ms_per_action']:.3f}",
        flush=True,
    )


def safe_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe_metadata(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return {"shape": tuple(int(dim) for dim in value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [safe_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


if __name__ == "__main__":
    raise SystemExit(main())
