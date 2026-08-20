#!/usr/bin/env python
"""Thin wrapper for training FastWAM with LeRobot.

FastWAM concatenates camera images horizontally. This wrapper reads the local
LeRobot dataset metadata to infer state/action dimensions and builds the
required visual ``policy.input_features`` for selected camera views.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def env_default(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def default_output_dir() -> str:
    run_id = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/train/fastwam_factory_conveyor_level2_{run_id}"


def normalize_path(path: str | None) -> str | None:
    if path is None:
        return None
    expanded = Path(path).expanduser()
    return str(expanded if expanded.is_absolute() else Path.cwd() / expanded)


def parse_image_size(value: str) -> tuple[int, int]:
    try:
        height, width = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError("--image-size must be JSON, e.g. '[224,448]'") from exc
    if not all(isinstance(item, int) and item > 0 for item in (height, width)):
        raise argparse.ArgumentTypeError("--image-size dimensions must be positive integers")
    return height, width


def dataset_features(dataset_root: str) -> dict[str, object]:
    info_path = Path(dataset_root) / "meta" / "info.json"
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))["features"]
    except FileNotFoundError as exc:
        raise ValueError(f"Dataset metadata not found: {info_path}") from exc
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid LeRobot dataset metadata: {info_path}") from exc


def feature_dimension(features: dict[str, object], key: str) -> int:
    feature = features.get(key)
    if not isinstance(feature, dict) or not isinstance(feature.get("shape"), list):
        raise ValueError(f"Dataset feature {key!r} with a one-dimensional shape is required")
    shape = feature["shape"]
    if len(shape) != 1 or not isinstance(shape[0], int) or shape[0] <= 0:
        raise ValueError(f"Dataset feature {key!r} must have a one-dimensional positive shape, got {shape!r}")
    return shape[0]


def fastwam_input_features(
    features: dict[str, object], image_keys: list[str], image_size: tuple[int, int], proprio_dim: int
) -> str:
    height, total_width = image_size
    if not image_keys:
        raise ValueError("At least one --image-key is required")
    if total_width % len(image_keys):
        raise ValueError(
            f"FastWAM image width {total_width} is not divisible by {len(image_keys)} selected cameras; "
            "choose a compatible --image-size or camera count"
        )

    camera_width = total_width // len(image_keys)
    input_features: dict[str, object] = {"observation.state": {"type": "STATE", "shape": [proprio_dim]}}
    for key in image_keys:
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("dtype") not in {"image", "video"}:
            raise ValueError(f"Dataset does not contain an image/video feature {key!r}")
        input_features[key] = {"type": "VISUAL", "shape": [3, height, camera_width]}
    return json.dumps(input_features, separators=(",", ":"))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train FastWAM with LeRobot.")
    parser.add_argument("--dataset-root", default=env_default("DATASET_ROOT"))
    parser.add_argument("--dataset-repo-id", default=env_default("DATASET_REPO_ID", "local/factory_conveyor_level2_seeded"))
    parser.add_argument("--output-dir", default=env_default("OUTPUT_DIR", default_output_dir()))
    parser.add_argument("--job-name", default=env_default("JOB_NAME", "fastwam_factory_conveyor_level2"))
    parser.add_argument("--lerobot-train-bin", default=env_default("LEROBOT_TRAIN_BIN", "lerobot-train"))
    parser.add_argument("--accelerate-bin", default=env_default("ACCELERATE_BIN", "accelerate"))
    parser.add_argument(
        "--ddp-num-processes",
        type=int,
        default=int(env_default("DDP_NUM_PROCESSES", "1")),
        help="Launch distributed data parallel training through Accelerate when greater than one.",
    )
    parser.add_argument(
        "--ddp-mixed-precision",
        default=env_default("DDP_MIXED_PRECISION", "bf16"),
        help="Accelerate mixed precision mode used for distributed launches.",
    )
    parser.add_argument(
        "--distributed-type",
        choices=("ddp", "fsdp"),
        default=env_default("DISTRIBUTED_TYPE", "ddp"),
        help="Use DDP replication or FSDP full parameter/gradient/optimizer sharding.",
    )

    parser.add_argument("--device", default=env_default("DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=int(env_default("BATCH_SIZE", "8")))
    parser.add_argument("--steps", type=int, default=int(env_default("STEPS", "300000")))
    parser.add_argument("--num-workers", type=int, default=int(env_default("NUM_WORKERS", "4")))
    parser.add_argument("--log-freq", type=int, default=int(env_default("LOG_FREQ", "50")))
    parser.add_argument("--save-freq", type=int, default=int(env_default("SAVE_FREQ", "50000")))
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=int(env_default("EVAL_STEPS", env_default("EVAL_FREQ", "0"))),
        help="Dataset evaluation interval; forwarded as LeRobot 0.6 --eval_steps.",
    )
    parser.add_argument("--seed", type=int, default=int(env_default("SEED", "1000")))
    parser.add_argument("--wandb-enable", type=str_to_bool, default=str_to_bool(env_default("WANDB_ENABLE", "true")))
    parser.add_argument("--wandb-project", default=env_default("WANDB_PROJECT", "fastwam"))

    parser.add_argument("--model-id", default=env_default("MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument(
        "--image-keys",
        default=env_default(
            "IMAGE_KEYS", "observation.images.overview,observation.images.front,observation.images.wrist"
        ),
        help="Comma-separated camera features; each is resized to an equal share of --image-size width.",
    )
    parser.add_argument("--image-size", type=parse_image_size, default=parse_image_size(env_default("IMAGE_SIZE", "[224,672]")))
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--proprio-dim", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=int(env_default("ACTION_HORIZON", "32")))
    parser.add_argument("--n-action-steps", type=int, default=int(env_default("N_ACTION_STEPS", "10")))
    parser.add_argument("--torch-dtype", default=env_default("TORCH_DTYPE"))
    parser.add_argument(
        "--use-gradient-checkpointing",
        type=str_to_bool,
        default=str_to_bool(env_default("USE_GRADIENT_CHECKPOINTING", "true")),
        help="Checkpoint both FastWAM DiT experts to reduce training activation memory.",
    )
    parser.add_argument("--optimizer-lr", type=float, default=env_default("OPTIMIZER_LR"))
    parser.add_argument("--toggle-action-dimensions", default=env_default("TOGGLE_ACTION_DIMENSIONS"))
    parser.add_argument("--push-to-hub", type=str_to_bool, default=str_to_bool(env_default("PUSH_TO_HUB", "false")))
    parser.add_argument("--policy-repo-id", default=env_default("POLICY_REPO_ID"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def build_train_command(args: argparse.Namespace) -> list[str]:
    if args.dataset_root is None:
        raise ValueError("--dataset-root or DATASET_ROOT must be set")
    if args.push_to_hub and not args.policy_repo_id:
        raise ValueError("--push-to-hub=true requires --policy-repo-id=HF_USER/MODEL_NAME")
    if args.n_action_steps > args.action_horizon:
        raise ValueError("--n-action-steps must be less than or equal to --action-horizon")

    dataset_root = normalize_path(args.dataset_root)
    output_dir = normalize_path(args.output_dir)
    assert dataset_root is not None and output_dir is not None
    features = dataset_features(dataset_root)
    image_keys = [key.strip() for key in args.image_keys.split(",") if key.strip()]
    dataset_image_keys = sorted(
        key
        for key, feature in features.items()
        if key.startswith("observation.images.")
        and isinstance(feature, dict)
        and feature.get("dtype") in {"image", "video"}
    )
    if sorted(image_keys) != dataset_image_keys:
        raise ValueError(
            "FastWAM uses all image/video features reported by the current LeRobot dataset metadata; "
            f"--image-keys must be exactly {','.join(dataset_image_keys)}"
        )
    action_dim = args.action_dim if args.action_dim is not None else feature_dimension(features, "action")
    proprio_dim = args.proprio_dim if args.proprio_dim is not None else feature_dimension(features, "observation.state")

    cmd = [
        args.lerobot_train_bin,
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        "--policy.type=fastwam",
        f"--policy.model_id={args.model_id}",
        f"--policy.device={args.device}",
        f"--policy.input_features={fastwam_input_features(features, image_keys, args.image_size, proprio_dim)}",
        f"--policy.action_dim={action_dim}",
        f"--policy.proprio_dim={proprio_dim}",
        f"--policy.image_size={json.dumps(list(args.image_size), separators=(',', ':'))}",
        f"--policy.action_horizon={args.action_horizon}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.use_gradient_checkpointing={str(args.use_gradient_checkpointing).lower()}",
        f"--policy.push_to_hub={str(args.push_to_hub).lower()}",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--num_workers={args.num_workers}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        f"--eval_steps={args.eval_freq}",
        f"--seed={args.seed}",
        f"--wandb.enable={str(args.wandb_enable).lower()}",
        f"--wandb.project={args.wandb_project}",
    ]
    if args.torch_dtype:
        cmd.append(f"--policy.torch_dtype={args.torch_dtype}")
    if args.optimizer_lr is not None:
        cmd.append(f"--policy.optimizer_lr={args.optimizer_lr}")
    if args.toggle_action_dimensions:
        cmd.append(f"--policy.toggle_action_dimensions={args.toggle_action_dimensions}")
    if args.policy_repo_id:
        cmd.append(f"--policy.repo_id={args.policy_repo_id}")
    return cmd


def build_launch_command(args: argparse.Namespace) -> list[str]:
    train_cmd = build_train_command(args)
    if args.ddp_num_processes == 1:
        return train_cmd
    if args.ddp_num_processes < 1:
        raise ValueError("--ddp-num-processes must be positive")
    launch_cmd = [
        args.accelerate_bin,
        "launch",
        f"--num_processes={args.ddp_num_processes}",
        f"--mixed_precision={args.ddp_mixed_precision}",
    ]
    if args.distributed_type == "fsdp":
        # Shard FastWAM at its 30 MoT layers rather than replicating all 6B
        # parameters and gradient buckets on every GPU.
        launch_cmd.extend(
            [
                "--use_fsdp",
                "--fsdp_version=1",
                "--fsdp_sharding_strategy=FULL_SHARD",
                "--fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP",
                "--fsdp_transformer_layer_cls_to_wrap=MoTLayer",
                "--fsdp_backward_prefetch=BACKWARD_PRE",
                "--fsdp_state_dict_type=SHARDED_STATE_DICT",
                "--fsdp_use_orig_params=true",
                "--fsdp_sync_module_states=false",
                "--fsdp_cpu_ram_efficient_loading=false",
            ]
        )
    else:
        launch_cmd.append("--multi_gpu")
    return [*launch_cmd, *train_cmd]


def main() -> None:
    args, extra_args = parse_args()
    cmd = build_launch_command(args)
    cmd.extend(extra_args)
    print("Running:", " ".join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
