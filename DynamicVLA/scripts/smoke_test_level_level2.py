#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import easydict
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.datasets
import utils.helpers
from policies.dynamicvla.modeling_dynamicvla import load_dynamicvla

BASE_CFG = PROJECT_ROOT / "configs" / "dynamicvla.yaml"
DEFAULT_CFG = PROJECT_ROOT / "configs" / "dynamicvla_level_level2.yaml"
DEFAULT_CKPT = PROJECT_ROOT / "pretrained" / "dynamic-vla-DOM"


def load_yaml(path: Path):
    with open(path, "r") as f:
        return easydict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = value


def compare_configs(base_cfg, target_cfg):
    base_flat, target_flat = {}, {}
    flatten("", base_cfg, base_flat)
    flatten("", target_cfg, target_flat)
    diffs = []
    for key in sorted(set(base_flat) | set(target_flat)):
        if base_flat.get(key) != target_flat.get(key):
            diffs.append((key, base_flat.get(key), target_flat.get(key)))
    return diffs


def check_dataset_metadata(cfg):
    root = Path(cfg.DATASET.ROOT)
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    assert root.exists(), f"Dataset root does not exist: {root}"
    assert info_path.exists(), f"Missing dataset info: {info_path}"
    assert tasks_path.exists(), f"Missing dataset tasks: {tasks_path}"

    info = json.loads(info_path.read_text())
    feature_keys = set(info["features"].keys())
    required = cfg.DATASET.REQUIRED_FEATURES
    aliases = cfg.DATASET.get("FEATURE_ALIASES", {})
    missing = [k for k in required if k not in feature_keys and aliases.get(k) not in feature_keys]
    assert not missing, f"Config requires missing dataset features: {missing}"
    assert info["codebase_version"] == "v2.1", info["codebase_version"]
    assert info["fps"] == 30, info["fps"]
    return info


def get_one_batch(cfg):
    dataset = utils.datasets.get_dataset(
        cfg.DATASET.NAME,
        root=cfg.DATASET.get("ROOT"),
        split="train",
        pin_memory=cfg.DATASET.PIN_MEMORY,
        delta_action=cfg.DATASET.USE_DELTA_ACTION,
        required_features=cfg.DATASET.REQUIRED_FEATURES,
        feature_aliases=cfg.DATASET.get("FEATURE_ALIASES"),
        image_transforms=utils.datasets.ImageTransforms(
            cfg.DATASET.IMG_SIZE, cfg.TRAIN.IMAGE_TRANSFORMS
        ),
        delta_timestamps=utils.helpers.get_delta_timestamps(
            cfg.POLICY, cfg.DATASET.DELTA_TIMESTAMPS
        ),
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    batch = next(iter(loader))
    if isinstance(batch["task"], list) and batch["task"] and isinstance(batch["task"][0], (tuple, list)):
        batch["task"] = batch["task"][0]
    return dataset, batch


def build_policy(cfg, dataset_meta):
    policy = utils.helpers.get_policy(
        cfg.POLICY,
        dataset_meta,
        cfg.DATASET.IMG_SIZE,
        cfg.DATASET.REQUIRED_FEATURES,
        cfg.DATASET.get("FEATURE_ALIASES"),
    )
    policy.device = "cpu"
    return policy


def run_forward(policy, batch):
    batch = {
        k: (v.to("cpu") if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    with torch.no_grad():
        loss, loss_dict = policy.forward(batch)
    return float(loss.detach().cpu().item()), loss_dict




def check_pretrained_alignment(cfg, info, batch, checkpoint: Path):
    ckpt_cfg_path = checkpoint / "config.json"
    assert ckpt_cfg_path.exists(), f"Missing checkpoint config: {ckpt_cfg_path}"
    ckpt_cfg = json.loads(ckpt_cfg_path.read_text())

    pretrained_images = [
        key
        for key, ft in ckpt_cfg["input_features"].items()
        if ft["type"] == "VISUAL"
    ]
    config_images = [
        key for key in cfg.DATASET.REQUIRED_FEATURES if key.startswith("observation.images.")
    ]
    assert config_images == pretrained_images, (
        f"Image feature order mismatch: config={config_images}, "
        f"pretrained={pretrained_images}"
    )

    assert cfg.POLICY.N_OBS_STEPS == ckpt_cfg["n_obs_steps"]
    assert cfg.POLICY.CHUNK_SIZE == ckpt_cfg["chunk_size"]
    assert cfg.POLICY.N_ACTION_STEPS == ckpt_cfg["n_action_steps"]
    assert list(cfg.POLICY.RESIZE_IMGS_WITH_PADDING) == ckpt_cfg["resize_imgs_with_padding"]

    state_dim = batch["observation.state"].shape[-1]
    action_dim = batch["action"].shape[-1]
    assert state_dim <= ckpt_cfg["max_state_dim"], (state_dim, ckpt_cfg["max_state_dim"])
    assert action_dim <= ckpt_cfg["max_action_dim"], (action_dim, ckpt_cfg["max_action_dim"])

    aliases = cfg.DATASET.get("FEATURE_ALIASES", {})
    image_shapes = {
        key: info["features"][aliases.get(key, key)]["shape"] for key in config_images
    }
    return ckpt_cfg, {
        "pretrained_images": pretrained_images,
        "image_shapes": image_shapes,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_state_dim": ckpt_cfg["max_state_dim"],
        "max_action_dim": ckpt_cfg["max_action_dim"],
    }

def maybe_check_wandb(cfg):
    if not cfg.WANDB.ENABLED:
        return "disabled"
    api_key = os.environ.get("WANDB_API_KEY")
    if cfg.WANDB.MODE == "online" and not api_key:
        return "online-without-api-key"
    return cfg.WANDB.MODE


def main():
    parser = argparse.ArgumentParser(description="DynamicVLA level_level2 smoke test")
    parser.add_argument("--config", default=str(DEFAULT_CFG))
    parser.add_argument("--base-config", default=str(BASE_CFG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--load-checkpoint", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    base_cfg = load_yaml(Path(args.base_config))

    print("[1/6] Comparing config against upstream default...")
    diffs = compare_configs(base_cfg, cfg)
    allowed_prefixes = {
        "CONST.N_WORKERS",
        "DATASET.NAME",
        "DATASET.ROOT",
        "DATASET.PIN_MEMORY",
        "DATASET.REQUIRED_FEATURES",
        "DATASET.FEATURE_ALIASES.observation.images.opst_cam",
        "DATASET.FEATURE_ALIASES.observation.images.wrist_cam",
        "TRAIN.CKPT_SAVE_FREQ.EPOCH",
        "WANDB.ENABLED",
        "WANDB.ENTITY",
    }
    unexpected = [d for d in diffs if d[0] not in allowed_prefixes]
    print(f"  total_differences={len(diffs)}")
    for key, base_val, target_val in diffs:
        print(f"  diff {key}: {base_val!r} -> {target_val!r}")
    if unexpected:
        raise AssertionError(f"Unexpected config differences: {[d[0] for d in unexpected]}")

    print("[2/6] Validating dataset metadata...")
    info = check_dataset_metadata(cfg)
    print(f"  total_episodes={info['total_episodes']} total_frames={info['total_frames']} fps={info['fps']}")

    print("[3/6] Loading one training batch...")
    dataset, batch = get_one_batch(cfg)
    print(f"  dataset_len={len(dataset)} train_episodes={len(dataset.episodes)}")
    print(f"  batch_keys={sorted(batch.keys())}")
    print(f"  state_shape={tuple(batch['observation.state'].shape)} action_shape={tuple(batch['action'].shape)}")
    print(f"  task={batch['task'][0] if isinstance(batch['task'], list) else batch['task']}")

    print("[4/6] Checking pretrained alignment...")
    ckpt = Path(args.checkpoint)
    _, alignment = check_pretrained_alignment(cfg, info, batch, ckpt)
    print(f"  image_features={alignment['pretrained_images']}")
    print(f"  source_image_shapes={alignment['image_shapes']}")
    print(
        f"  state_dim={alignment['state_dim']} <= max_state_dim={alignment['max_state_dim']} "
        f"action_dim={alignment['action_dim']} <= max_action_dim={alignment['max_action_dim']}"
    )

    print("[5/6] Building policy and running CPU forward...")
    policy = build_policy(cfg, dataset.meta)
    if args.load_checkpoint:
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")
        model_file = ckpt / "model.safetensors"
        if not model_file.exists():
            raise FileNotFoundError(f"Checkpoint weights not found: {model_file}")
        load_dynamicvla(
            policy,
            str(model_file),
            device="cpu",
            checkpoint_keys_mapping="model._orig_mod.//model.",
        )
        policy.device = "cpu"
        print(f"  loaded_checkpoint={ckpt}")
    loss, loss_dict = run_forward(policy, batch)
    print(f"  forward_loss={loss:.6f}")
    print(f"  loss_dict_keys={sorted(loss_dict.keys())}")

    print("[6/6] Checking wandb mode...")
    wandb_status = maybe_check_wandb(cfg)
    print(f"  wandb_status={wandb_status}")
    if wandb_status == "online-without-api-key":
        print("  warning=wandb online is enabled but WANDB_API_KEY is not set")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
