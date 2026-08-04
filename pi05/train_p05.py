#!/usr/bin/env python
"""Thin wrapper for fine-tuning PI05 with LeRobot.

Cluster paths and environment setup live in submit_pi05_train.sh. This script
only translates environment variables or CLI arguments into lerobot-train flags,
then forwards any extra args unchanged.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_POLICY_PRETRAINED_PATH = "lerobot/pi05_base"


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
    return f"outputs/train/pi05_level_level2_{run_id}"


def normalize_path(path: str | None) -> str | None:
    if path is None:
        return None
    expanded = Path(path).expanduser()
    return str(expanded if expanded.is_absolute() else Path.cwd() / expanded)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Fine-tune PI05 with LeRobot.")
    parser.add_argument("--dataset-root", default=env_default("DATASET_ROOT"))
    parser.add_argument("--dataset-repo-id", default=env_default("DATASET_REPO_ID", "local/pi05_level_level2"))
    parser.add_argument(
        "--policy-pretrained-path",
        default=env_default("POLICY_PRETRAINED_PATH", DEFAULT_POLICY_PRETRAINED_PATH),
        help="LeRobot PI05 pretrained checkpoint, default: lerobot/pi05_base.",
    )
    parser.add_argument("--output-dir", default=env_default("OUTPUT_DIR", default_output_dir()))
    parser.add_argument("--job-name", default=env_default("JOB_NAME", "pi05_level_level2"))
    parser.add_argument("--lerobot-train-bin", default=env_default("LEROBOT_TRAIN_BIN", "lerobot-train"))

    parser.add_argument("--device", default=env_default("DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=int(env_default("BATCH_SIZE", "1")))
    parser.add_argument("--steps", type=int, default=int(env_default("STEPS", "3000")))
    parser.add_argument("--num-workers", type=int, default=int(env_default("NUM_WORKERS", "4")))
    parser.add_argument("--log-freq", type=int, default=int(env_default("LOG_FREQ", "20")))
    parser.add_argument("--save-freq", type=int, default=int(env_default("SAVE_FREQ", "1000")))
    parser.add_argument("--eval-freq", type=int, default=int(env_default("EVAL_FREQ", "0")))
    parser.add_argument("--seed", type=int, default=int(env_default("SEED", "1000")))
    parser.add_argument("--wandb-enable", type=str_to_bool, default=str_to_bool(env_default("WANDB_ENABLE", "false")))
    parser.add_argument("--wandb-project", default=env_default("WANDB_PROJECT", "pi05"))

    parser.add_argument("--compile-model", type=str_to_bool, default=str_to_bool(env_default("COMPILE_MODEL", "true")))
    parser.add_argument(
        "--gradient-checkpointing",
        type=str_to_bool,
        default=str_to_bool(env_default("GRADIENT_CHECKPOINTING", "true")),
    )
    parser.add_argument("--dtype", default=env_default("DTYPE", "bfloat16"))
    parser.add_argument(
        "--freeze-vision-encoder",
        type=str_to_bool,
        default=str_to_bool(env_default("FREEZE_VISION_ENCODER", "false")),
    )
    parser.add_argument(
        "--train-expert-only",
        type=str_to_bool,
        default=str_to_bool(env_default("TRAIN_EXPERT_ONLY", "false")),
    )
    parser.add_argument(
        "--use-relative-actions",
        type=str_to_bool,
        default=str_to_bool(env_default("USE_RELATIVE_ACTIONS", "false")),
    )
    parser.add_argument("--relative-exclude-joints", default=env_default("RELATIVE_EXCLUDE_JOINTS", "[\"gripper\"]"))
    parser.add_argument("--normalization-mapping", default=env_default("NORMALIZATION_MAPPING"))

    parser.add_argument("--push-to-hub", type=str_to_bool, default=str_to_bool(env_default("PUSH_TO_HUB", "false")))
    parser.add_argument("--policy-repo-id", default=env_default("POLICY_REPO_ID"))
    parser.add_argument("--optimizer-lr", type=float, default=env_default("OPTIMIZER_LR"))
    parser.add_argument("--scheduler-decay-lr", type=float, default=env_default("SCHEDULER_DECAY_LR"))
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_known_args()


def build_train_command(args: argparse.Namespace) -> list[str]:
    if args.dataset_root is None:
        raise ValueError("--dataset-root or DATASET_ROOT must be set")
    if args.push_to_hub and not args.policy_repo_id:
        raise ValueError("--push-to-hub=true requires --policy-repo-id=HF_USER/MODEL_NAME")

    dataset_root = normalize_path(args.dataset_root)
    output_dir = normalize_path(args.output_dir)
    assert dataset_root is not None
    assert output_dir is not None

    cmd = [
        args.lerobot_train_bin,
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        "--policy.type=pi05",
        f"--policy.pretrained_path={args.policy_pretrained_path}",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}",
        f"--policy.device={args.device}",
        f"--policy.push_to_hub={str(args.push_to_hub).lower()}",
        f"--policy.compile_model={str(args.compile_model).lower()}",
        f"--policy.gradient_checkpointing={str(args.gradient_checkpointing).lower()}",
        f"--policy.dtype={args.dtype}",
        f"--policy.freeze_vision_encoder={str(args.freeze_vision_encoder).lower()}",
        f"--policy.train_expert_only={str(args.train_expert_only).lower()}",
        f"--policy.use_relative_actions={str(args.use_relative_actions).lower()}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--num_workers={args.num_workers}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        f"--eval_freq={args.eval_freq}",
        f"--seed={args.seed}",
        f"--wandb.enable={str(args.wandb_enable).lower()}",
        f"--wandb.project={args.wandb_project}",
    ]

    if args.use_relative_actions:
        cmd.append(f"--policy.relative_exclude_joints={args.relative_exclude_joints}")
    if args.normalization_mapping:
        cmd.append(f"--policy.normalization_mapping={args.normalization_mapping}")
    if args.policy_repo_id:
        cmd.append(f"--policy.repo_id={args.policy_repo_id}")
    if args.optimizer_lr is not None:
        cmd.append(f"--policy.optimizer_lr={args.optimizer_lr}")
    if args.scheduler_decay_lr is not None:
        cmd.append(f"--policy.scheduler_decay_lr={args.scheduler_decay_lr}")

    return cmd


def main() -> None:
    args, extra_args = parse_args()
    cmd = build_train_command(args)
    cmd.extend(extra_args)

    print("Running:", " ".join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
