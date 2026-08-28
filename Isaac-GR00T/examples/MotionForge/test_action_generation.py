#!/usr/bin/env python3
"""Smoke-test MotionForge GR00T action generation on LeRobot episodes.

The test uses observations from a real dataset episode, calls ``Gr00tPolicy.get_action``,
validates the generated action chunk, and reports an informational comparison with the
recorded action chunk.

Default usage (paths already point at this workspace)::

    python examples/MotionForge/test_action_generation.py

Test every CM sub-dataset instead of only the first one::

    python examples/MotionForge/test_action_generation.py --max-datasets 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_PATH = Path("/project/mohan_ws/ckpts/MotionforgeGroup/gr00t1.7/CM-80000")
DEFAULT_DATASET_PATH = Path("/project/mohan_ws/dataset/lerobot_cm")


def bootstrap_local_runtime() -> None:
    """Expose this workspace's local FFmpeg and backbone assets when available."""
    env = os.environ.copy()
    changed = False

    ffmpeg_lib = WORKSPACE_ROOT / ".cache" / "gr00t-ffmpeg" / "lib"
    library_paths = env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if ffmpeg_lib.is_dir() and str(ffmpeg_lib) not in library_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(ffmpeg_lib), *[path for path in library_paths if path]]
        )
        changed = True

    backbone = WORKSPACE_ROOT / "ckpts" / "nvidia" / "Cosmos-Reason2-2B"
    if "GR00T_BACKBONE_MODEL_PATH" not in env and backbone.is_dir():
        env["GR00T_BACKBONE_MODEL_PATH"] = str(backbone)
        changed = True

    # LD_LIBRARY_PATH is consumed by the dynamic loader at process startup, so re-exec once.
    if changed and env.get("MOTIONFORGE_ACTION_TEST_BOOTSTRAPPED") != "1":
        env["MOTIONFORGE_ACTION_TEST_BOOTSTRAPPED"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="A LeRobot dataset, or a directory containing multiple LeRobot datasets.",
    )
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--device", default="cuda", help="For example: cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--dataset-filter",
        default=None,
        help="Only test dataset directory names containing this string (for example cm_010).",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=1,
        help="Maximum number of discovered datasets to test; 0 means all datasets.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--step-index",
        type=int,
        default=None,
        help="Observation step to test. By default, samples are spread through the episode.",
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=None,
        help="Override the checkpoint's number of diffusion inference steps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optionally write the machine-readable test summary to this path.",
    )
    args = parser.parse_args()

    if args.max_datasets < 0:
        parser.error("--max-datasets must be >= 0")
    if args.episode_index < 0:
        parser.error("--episode-index must be >= 0")
    if args.step_index is not None and args.step_index < 0:
        parser.error("--step-index must be >= 0")
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.denoising_steps is not None and args.denoising_steps < 1:
        parser.error("--denoising-steps must be >= 1")
    return args


def discover_datasets(root: Path, name_filter: str | None, limit: int) -> list[Path]:
    root = root.expanduser().resolve()
    if (root / "meta" / "info.json").is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = sorted(
            path for path in root.iterdir() if (path / "meta" / "info.json").is_file()
        )
    else:
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    if name_filter:
        candidates = [path for path in candidates if name_filter.lower() in path.name.lower()]
    if not candidates:
        suffix = f" matching {name_filter!r}" if name_filter else ""
        raise FileNotFoundError(f"No LeRobot datasets found under {root}{suffix}")
    return candidates if limit == 0 else candidates[:limit]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_policy(args: argparse.Namespace) -> Any:
    import torch
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {model_path}")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {args.device!r} was requested, but torch.cuda.is_available() is False. "
            "Run this checkpoint test on a machine where PyTorch can access an NVIDIA GPU."
        )

    started = time.perf_counter()
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=str(model_path),
        device=args.device,
        strict=True,
    )
    if args.denoising_steps is not None:
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
    elapsed = time.perf_counter() - started
    print(f"[model] loaded {model_path} on {args.device} in {elapsed:.2f}s")
    return policy


def prepare_observation(
    trajectory: Any,
    step_index: int,
    modality_config: dict[str, Any],
    embodiment_tag: Any,
) -> tuple[dict[str, Any], str]:
    from examples.MotionForge.motionforge_groot_bridge_client import _as_rgb_uint8
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t

    observation_config = deepcopy(modality_config)
    observation_config.pop("action")
    step_data = extract_step_data(
        trajectory,
        step_index,
        observation_config,
        embodiment_tag,
        allow_padding=True,
    )

    image_keys = list(observation_config["video"].modality_keys)
    if not image_keys:
        raise ValueError("The checkpoint does not define any video modalities")
    flat_observation: dict[str, Any] = {}
    for key in image_keys:
        sequence = np.asarray(step_data.images[key])
        if sequence.ndim == 3:
            sequence = sequence[None, ...]
        if sequence.ndim != 4:
            raise ValueError(f"Expected THWC image sequence for {key!r}, got {sequence.shape}")
        flat_observation[f"video.{key}"] = np.stack(
            [_as_rgb_uint8(frame) for frame in sequence],
            axis=0,
        )
    for key, value in step_data.states.items():
        flat_observation[f"state.{key}"] = np.asarray(value, dtype=np.float32)
    for key in observation_config["language"].modality_keys:
        flat_observation[key] = step_data.text

    return parse_observation_gr00t(flat_observation, observation_config), step_data.text


def choose_steps(
    trajectory_length: int,
    action_delta_indices: list[int],
    requested_step: int | None,
    num_samples: int,
) -> list[int]:
    min_delta = min(action_delta_indices)
    max_delta = max(action_delta_indices)
    first_valid = max(0, -min_delta)
    last_valid = trajectory_length - 1 - max_delta
    if last_valid < first_valid:
        raise ValueError(
            f"Episode has {trajectory_length} steps, which is shorter than the action horizon "
            f"[{min_delta}, {max_delta}]."
        )
    if requested_step is not None:
        if not first_valid <= requested_step <= last_valid:
            raise IndexError(
                f"--step-index {requested_step} is invalid; valid range is "
                f"[{first_valid}, {last_valid}] for this episode."
            )
        return [requested_step]

    # Interior, evenly spaced samples avoid testing only episode boundaries.
    points = np.linspace(first_valid, last_valid, num_samples + 2)[1:-1]
    return sorted(set(int(round(point)) for point in points))


def recorded_action_chunk(
    trajectory: Any,
    step_index: int,
    action_config: Any,
) -> dict[str, np.ndarray]:
    indices = [step_index + delta for delta in action_config.delta_indices]
    chunk: dict[str, np.ndarray] = {}
    for key in action_config.modality_keys:
        column = trajectory[f"action.{key}"].iloc[indices]
        chunk[key] = np.vstack([np.asarray(value, dtype=np.float32) for value in column])
    return chunk


def validate_and_summarize_action(
    action: dict[str, Any],
    target: dict[str, np.ndarray],
    action_config: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    expected_keys = list(action_config.modality_keys)
    missing = [key for key in expected_keys if key not in action]
    if missing:
        raise AssertionError(f"Generated action is missing keys: {missing}")

    generated_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    key_summary: dict[str, Any] = {}
    expected_horizon = len(action_config.delta_indices)
    for key in expected_keys:
        value = np.asarray(action[key])
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != expected_horizon:
            raise AssertionError(
                f"Action {key!r} has shape {value.shape}; expected "
                f"(1, {expected_horizon}, action_dim)."
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise AssertionError(f"Action {key!r} has non-floating dtype {value.dtype}")
        if not np.isfinite(value).all():
            bad_count = int(value.size - np.isfinite(value).sum())
            raise AssertionError(f"Action {key!r} contains {bad_count} NaN/Inf values")
        if value.shape[2:] != target[key].shape[1:]:
            raise AssertionError(
                f"Action {key!r} dimension {value.shape[2:]} does not match dataset "
                f"dimension {target[key].shape[1:]}"
            )

        generated = value[0].astype(np.float32, copy=False)
        generated_parts.append(generated)
        target_parts.append(target[key])
        key_summary[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(value.min()),
            "max": float(value.max()),
            "mean": float(value.mean()),
        }

    generated_all = np.concatenate(generated_parts, axis=-1)
    target_all = np.concatenate(target_parts, axis=-1)
    return key_summary, generated_all, target_all


def rotation_6d_diagnostics(action_chunk: dict[str, Any]) -> dict[str, float] | None:
    """Check the two predicted rotation columns without turning quality into pass/fail."""
    if "eef_pose_rot6d" not in action_chunk:
        return None
    eef = np.asarray(action_chunk["eef_pose_rot6d"])[0]
    if eef.shape[-1] < 9:
        return None
    col0, col1 = eef[:, 3:6], eef[:, 6:9]
    return {
        "mean_col0_norm": float(np.linalg.norm(col0, axis=-1).mean()),
        "mean_col1_norm": float(np.linalg.norm(col1, axis=-1).mean()),
        "mean_abs_dot": float(np.abs(np.sum(col0 * col1, axis=-1)).mean()),
    }


def test_dataset(
    policy: Any,
    dataset_path: Path,
    episode_index: int,
    requested_step: int | None,
    num_samples: int,
) -> list[dict[str, Any]]:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    modality_config = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=dataset_path, modality_configs=modality_config)
    if episode_index >= len(loader):
        raise IndexError(
            f"Episode {episode_index} does not exist in {dataset_path.name}; "
            f"the dataset contains {len(loader)} episodes."
        )

    load_started = time.perf_counter()
    trajectory = loader[episode_index]
    load_seconds = time.perf_counter() - load_started
    action_config = modality_config["action"]
    steps = choose_steps(
        len(trajectory), action_config.delta_indices, requested_step, num_samples
    )
    print(
        f"\n[dataset] {dataset_path.name}: episodes={len(loader)}, "
        f"episode={episode_index}, frames={len(trajectory)}, load={load_seconds:.2f}s"
    )

    results: list[dict[str, Any]] = []
    for step_index in steps:
        observation, instruction = prepare_observation(
            trajectory, step_index, modality_config, policy.embodiment_tag
        )
        target = recorded_action_chunk(trajectory, step_index, action_config)

        infer_started = time.perf_counter()
        generated_action, _ = policy.get_action(observation)
        inference_seconds = time.perf_counter() - infer_started
        keys, generated, recorded = validate_and_summarize_action(
            generated_action, target, action_config
        )

        error = generated.astype(np.float64) - recorded.astype(np.float64)
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))
        diagnostics = rotation_6d_diagnostics(generated_action)
        result = {
            "status": "PASS",
            "dataset": str(dataset_path),
            "episode_index": episode_index,
            "step_index": step_index,
            "instruction": instruction,
            "inference_seconds": inference_seconds,
            "action_horizon": int(generated.shape[0]),
            "action_dimension": int(generated.shape[1]),
            "action_keys": keys,
            "first_action": generated[0].tolist(),
            "mae_vs_recorded_chunk": mae,
            "rmse_vs_recorded_chunk": rmse,
            "rotation_6d_diagnostics": diagnostics,
        }
        results.append(result)

        shape_text = ", ".join(f"{key}: {value['shape']}" for key, value in keys.items())
        print(f"  [PASS] step={step_index}, inference={inference_seconds:.3f}s")
        print(f"         instruction={instruction!r}")
        print(f"         action shapes={{{shape_text}}}")
        print(f"         first action={np.array2string(generated[0], precision=5)}")
        print(f"         informational MAE={mae:.6f}, RMSE={rmse:.6f}")
        if diagnostics is not None:
            print(
                "         rot6d "
                f"norms=({diagnostics['mean_col0_norm']:.4f}, "
                f"{diagnostics['mean_col1_norm']:.4f}), "
                f"mean|dot|={diagnostics['mean_abs_dot']:.4f}"
            )
    return results


def main() -> int:
    bootstrap_local_runtime()
    args = parse_args()
    set_seed(args.seed)
    datasets = discover_datasets(args.dataset_path, args.dataset_filter, args.max_datasets)
    print(f"[data] discovered {len(datasets)} dataset(s): {[path.name for path in datasets]}")

    policy = load_policy(args)
    modality_config = policy.get_modality_config()
    print(
        "[model] modalities: "
        + ", ".join(
            f"{name}={list(config.modality_keys)}@{list(config.delta_indices)}"
            for name, config in modality_config.items()
        )
    )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for dataset_path in datasets:
        try:
            results.extend(
                test_dataset(
                    policy,
                    dataset_path,
                    args.episode_index,
                    args.step_index,
                    args.num_samples,
                )
            )
        except Exception as exc:  # Continue so an all-dataset run reports every failing task.
            failures.append(
                {"dataset": str(dataset_path), "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"\n[FAIL] {dataset_path.name}: {type(exc).__name__}: {exc}")

    summary = {
        "model_path": str(args.model_path.expanduser().resolve()),
        "embodiment_tag": args.embodiment_tag,
        "device": args.device,
        "passed_samples": len(results),
        "failed_datasets": len(failures),
        "results": results,
        "failures": failures,
    }
    if args.save_json is not None:
        output_path = args.save_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"[output] wrote {output_path}")

    print(
        f"\n[summary] passed_samples={len(results)}, "
        f"failed_datasets={len(failures)}, status={'PASS' if not failures and results else 'FAIL'}"
    )
    return 0 if results and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
