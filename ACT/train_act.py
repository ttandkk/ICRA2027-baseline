#!/usr/bin/env python
"""Train a LeRobot ACT policy on a local LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


def as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "y", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def abs_path(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else Path.cwd() / path)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    run_id = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Train ACT with LeRobot.")
    parser.add_argument("--dataset-root", default=env("DATASET_ROOT"))
    parser.add_argument("--dataset-repo-id", default=env("DATASET_REPO_ID", "local/factory_conveyor_level2_seeded"))
    parser.add_argument("--output-dir", default=env("OUTPUT_DIR", f"outputs/train/act_factory_conveyor_{run_id}"))
    parser.add_argument("--job-name", default=env("JOB_NAME", "act_factory_conveyor_level2_seeded"))
    parser.add_argument("--lerobot-train-bin", default=env("LEROBOT_TRAIN_BIN", "lerobot-train"))
    parser.add_argument("--device", default=env("DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=int(env("BATCH_SIZE", "32")))
    parser.add_argument("--steps", type=int, default=int(env("STEPS", "80000")))
    parser.add_argument("--num-workers", type=int, default=int(env("NUM_WORKERS", "4")))
    parser.add_argument("--log-freq", type=int, default=int(env("LOG_FREQ", "50")))
    parser.add_argument("--save-freq", type=int, default=int(env("SAVE_FREQ", "40000")))
    parser.add_argument("--eval-freq", type=int, default=int(env("EVAL_FREQ", "0")))
    parser.add_argument("--seed", type=int, default=int(env("SEED", "1000")))
    parser.add_argument("--wandb-enable", type=as_bool, default=as_bool(env("WANDB_ENABLE", "true")))
    parser.add_argument("--wandb-project", default=env("WANDB_PROJECT", "act"))
    parser.add_argument("--chunk-size", type=int, default=int(env("CHUNK_SIZE", "100")))
    parser.add_argument("--n-action-steps", type=int, default=int(env("N_ACTION_STEPS", "100")))
    parser.add_argument("--grad-clip-norm", type=float, default=float(env("GRAD_CLIP_NORM", "1.0")))
    parser.add_argument("--optimizer-lr", type=float, default=env("OPTIMIZER_LR"))
    parser.add_argument("--push-to-hub", type=as_bool, default=as_bool(env("PUSH_TO_HUB", "false")))
    parser.add_argument("--policy-repo-id", default=env("POLICY_REPO_ID"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    if args.dataset_root is None:
        raise ValueError("--dataset-root or DATASET_ROOT must be set")
    if args.push_to_hub and not args.policy_repo_id:
        raise ValueError("--push-to-hub=true requires --policy-repo-id")
    cmd = [
        args.lerobot_train_bin,
        f"--dataset.repo_id={args.dataset_repo_id}", f"--dataset.root={abs_path(args.dataset_root)}",
        "--policy.type=act", f"--policy.device={args.device}",
        f"--policy.chunk_size={args.chunk_size}", f"--policy.n_action_steps={args.n_action_steps}",
        f"--optimizer.grad_clip_norm={args.grad_clip_norm}",
        f"--policy.push_to_hub={str(args.push_to_hub).lower()}",
        f"--output_dir={abs_path(args.output_dir)}", f"--job_name={args.job_name}",
        f"--batch_size={args.batch_size}", f"--steps={args.steps}", f"--num_workers={args.num_workers}",
        f"--log_freq={args.log_freq}", f"--save_freq={args.save_freq}", f"--eval_freq={args.eval_freq}",
        f"--seed={args.seed}", f"--wandb.enable={str(args.wandb_enable).lower()}",
        f"--wandb.project={args.wandb_project}",
    ]
    if args.optimizer_lr is not None:
        cmd.append(f"--policy.optimizer_lr={args.optimizer_lr}")
    if args.policy_repo_id:
        cmd.append(f"--policy.repo_id={args.policy_repo_id}")
    cmd.extend(extra)
    print("Running:", " ".join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
