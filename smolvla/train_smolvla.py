#!/usr/bin/env python
"""Thin wrapper for fine-tuning SmolVLA with LeRobot.

Cluster paths and environment setup live in submit_smolvla_train.sh. This
script only translates local defaults/environment variables into lerobot-train
CLI arguments, then forwards any extra args unchanged.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_POLICY_PATH = "lerobot/smolvla_base"


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
    return f"outputs/train/smolvla_level_level2_{run_id}"


def normalize_path(path: str | None) -> str | None:
    if path is None:
        return None
    expanded = Path(path).expanduser()
    return str(expanded if expanded.is_absolute() else Path.cwd() / expanded)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA with LeRobot.")
    parser.add_argument("--dataset-root", default=env_default("DATASET_ROOT"))
    parser.add_argument("--dataset-repo-id", default=env_default("DATASET_REPO_ID", "local/level_level2"))
    parser.add_argument("--policy-path", default=env_default("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument("--output-dir", default=env_default("OUTPUT_DIR", default_output_dir()))
    parser.add_argument("--job-name", default=env_default("JOB_NAME", "smolvla_level_level2"))
    parser.add_argument("--lerobot-train-bin", default=env_default("LEROBOT_TRAIN_BIN", "lerobot-train"))

    parser.add_argument("--device", default=env_default("DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=int(env_default("BATCH_SIZE", "8")))
    parser.add_argument("--steps", type=int, default=int(env_default("STEPS", "20000")))
    parser.add_argument("--num-workers", type=int, default=int(env_default("NUM_WORKERS", "4")))
    parser.add_argument("--log-freq", type=int, default=int(env_default("LOG_FREQ", "50")))
    parser.add_argument("--save-freq", type=int, default=int(env_default("SAVE_FREQ", "5000")))
    parser.add_argument("--eval-freq", type=int, default=int(env_default("EVAL_FREQ", "0")))
    parser.add_argument("--seed", type=int, default=int(env_default("SEED", "1000")))
    parser.add_argument("--wandb-enable", type=str_to_bool, default=str_to_bool(env_default("WANDB_ENABLE", "false")))
    parser.add_argument("--wandb-project", default=env_default("WANDB_PROJECT", "smolvla"))

    parser.add_argument(
        "--infer-policy-features",
        type=str_to_bool,
        default=str_to_bool(env_default("INFER_POLICY_FEATURES", "true")),
        help="Set policy input/output features to null so LeRobot infers them from the dataset.",
    )
    parser.add_argument("--push-to-hub", type=str_to_bool, default=str_to_bool(env_default("PUSH_TO_HUB", "false")))
    parser.add_argument("--policy-repo-id", default=env_default("POLICY_REPO_ID"))

    parser.add_argument(
        "--full-finetune",
        action="store_true",
        default=str_to_bool(env_default("FULL_FINETUNE", "false")),
        help="Unfreeze vision encoder and train beyond expert-only mode.",
    )
    parser.add_argument(
        "--peft-lora",
        action="store_true",
        default=str_to_bool(env_default("PEFT_LORA", "false")),
        help="Use LeRobot SmolVLA LoRA fine-tuning.",
    )
    parser.add_argument("--peft-r", type=int, default=int(env_default("PEFT_R", "64")))
    parser.add_argument("--peft-alpha", type=int, default=int(env_default("PEFT_ALPHA", "64")))
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
        f"--policy.path={args.policy_path}",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}",
        f"--policy.device={args.device}",
        f"--policy.push_to_hub={str(args.push_to_hub).lower()}",
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

    if args.infer_policy_features:
        cmd.extend(["--policy.input_features=null", "--policy.output_features=null"])
    if args.policy_repo_id:
        cmd.append(f"--policy.repo_id={args.policy_repo_id}")
    if args.full_finetune:
        cmd.extend(["--policy.freeze_vision_encoder=false", "--policy.train_expert_only=false"])
    if args.peft_lora:
        cmd.extend(
            [
                "--peft.method_type=LORA",
                f"--peft.r={args.peft_r}",
                f"--peft.lora_alpha={args.peft_alpha}",
            ]
        )
        if args.optimizer_lr is None:
            args.optimizer_lr = 1e-4
        if args.scheduler_decay_lr is None:
            args.scheduler_decay_lr = 1e-4
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
