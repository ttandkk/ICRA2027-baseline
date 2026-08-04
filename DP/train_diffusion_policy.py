#!/usr/bin/env python
"""Train a LeRobot Diffusion Policy on a local LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
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


def selected_input_features(dataset_root: str, image_keys: list[str]) -> str:
    # Diffusion Policy requires all selected camera views to share one resolution.
    features = json.loads((Path(dataset_root) / "meta" / "info.json").read_text(encoding="utf-8"))["features"]
    selected = {"observation.state": {"type": "STATE", "shape": features["observation.state"]["shape"]}}
    shapes = set()
    for key in image_keys:
        if key not in features or features[key]["dtype"] not in {"image", "video"}:
            raise ValueError(f"Dataset does not contain an image feature {key!r}")
        shapes.add(tuple(features[key]["shape"]))
        selected[key] = {"type": "VISUAL", "shape": features[key]["shape"]}
    if len(shapes) != 1:
        raise ValueError(f"Selected camera shapes differ: {sorted(shapes)}")
    return json.dumps(selected)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    run_id = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Train Diffusion Policy with LeRobot.")
    parser.add_argument("--dataset-root", default=env("DATASET_ROOT"))
    parser.add_argument("--dataset-repo-id", default=env("DATASET_REPO_ID", "local/factory_conveyor_level2_seeded"))
    parser.add_argument("--output-dir", default=env("OUTPUT_DIR", f"outputs/train/diffusion_factory_conveyor_{run_id}"))
    parser.add_argument("--job-name", default=env("JOB_NAME", "diffusion_factory_conveyor_level2_seeded"))
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
    parser.add_argument("--wandb-project", default=env("WANDB_PROJECT", "diffusion-policy"))
    parser.add_argument("--image-keys", default=env("IMAGE_KEYS", "observation.images.overview,observation.images.front"), help="Comma-separated same-resolution cameras; wrist is intentionally excluded.")
    parser.add_argument("--horizon", type=int, default=int(env("HORIZON", "64")))
    parser.add_argument("--n-obs-steps", type=int, default=int(env("N_OBS_STEPS", "2")))
    parser.add_argument("--n-action-steps", type=int, default=int(env("N_ACTION_STEPS", "32")))
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
    dataset_root = abs_path(args.dataset_root)
    cmd = [
        args.lerobot_train_bin,
        f"--dataset.repo_id={args.dataset_repo_id}", f"--dataset.root={dataset_root}",
        "--policy.type=diffusion", f"--policy.device={args.device}",
        f"--policy.input_features={selected_input_features(dataset_root, [key.strip() for key in args.image_keys.split(",") if key.strip()])}",
        f"--policy.horizon={args.horizon}", f"--policy.n_obs_steps={args.n_obs_steps}",
        f"--policy.n_action_steps={args.n_action_steps}",
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
