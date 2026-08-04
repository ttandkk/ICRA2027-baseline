#!/usr/bin/env python3
"""Run a dummy SmolVLA inference benchmark without a simulator."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="lerobot/smolvla_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--image-width", type=int, default=360)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--bench-steps", type=int, default=50)
    parser.add_argument("--num-denoise-steps", type=int, default=None)
    parser.add_argument("--max-cameras", type=int, default=None)
    parser.add_argument("--task", default="pick up the object and place it in the target area")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("chunk", "select_action", "split"),
        default="chunk",
        help=(
            "chunk benchmarks one full action generation call; select_action includes action queue pops; "
            "split reports VLM prefix/cache time and action-head denoising time separately."
        ),
    )
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_batch(policy: SmolVLAPolicy, args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    config = policy.config
    tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name)
    prompts = [args.task if args.task.endswith("\n") else f"{args.task}\n"] * args.batch_size
    tokens = tokenizer(
        prompts,
        padding=config.pad_language_to,
        max_length=config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )

    batch: dict[str, torch.Tensor] = {
        OBS_LANGUAGE_TOKENS: tokens["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].to(device=device, dtype=torch.bool),
    }

    state_feature = config.robot_state_feature
    if state_feature is None:
        raise ValueError("Policy config does not define observation.state.")
    batch[OBS_STATE] = torch.zeros(args.batch_size, *state_feature.shape, device=device)

    image_features = config.image_features
    if not image_features:
        raise ValueError("Policy config does not define any visual input feature.")
    image_items = list(image_features.items())
    if args.max_cameras is not None:
        image_items = image_items[: args.max_cameras]
    for key, feature in image_items:
        channels = feature.shape[0]
        batch[key] = torch.rand(
            args.batch_size,
            channels,
            args.image_height,
            args.image_width,
            device=device,
        )

    return batch


@torch.inference_mode()
def benchmark_chunk(
    policy: SmolVLAPolicy, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device
):
    latencies = []
    actions = None
    for _ in range(args.warmup_steps):
        policy.predict_action_chunk(batch)
    sync(device)

    for _ in range(args.bench_steps):
        start = time.perf_counter()
        actions = policy.predict_action_chunk(batch)
        sync(device)
        latencies.append(time.perf_counter() - start)
    return latencies, actions


@torch.inference_mode()
def benchmark_select_action(
    policy: SmolVLAPolicy, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device
):
    latencies = []
    action = None
    policy.reset()
    for _ in range(args.warmup_steps):
        policy.select_action(batch)
    sync(device)

    for _ in range(args.bench_steps):
        start = time.perf_counter()
        action = policy.select_action(batch)
        sync(device)
        latencies.append(time.perf_counter() - start)
    return latencies, action


@torch.inference_mode()
def benchmark_split(
    policy: SmolVLAPolicy, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device
):
    def run_once():
        model = policy.model
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        bsize = state.shape[0]

        actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
        noise = model.sample_noise(actions_shape, device)

        sync(device)
        total_start = time.perf_counter()

        prefix_start = time.perf_counter()
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=model.config.use_cache,
            fill_kv_cache=True,
        )
        sync(device)
        prefix_s = time.perf_counter() - prefix_start

        action_start = time.perf_counter()
        x_t = noise
        dt = -1.0 / model.config.num_steps
        for step in range(model.config.num_steps):
            current_time = 1.0 + step * dt
            timestep = torch.tensor(current_time, dtype=torch.float32, device=device).expand(bsize)
            v_t = model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=timestep,
            )
            x_t = x_t + dt * v_t
        sync(device)
        action_s = time.perf_counter() - action_start
        total_s = time.perf_counter() - total_start
        return prefix_s, action_s, total_s, x_t

    for _ in range(args.warmup_steps):
        run_once()

    prefix_latencies = []
    action_latencies = []
    total_latencies = []
    action = None
    for _ in range(args.bench_steps):
        prefix_s, action_s, total_s, action = run_once()
        prefix_latencies.append(prefix_s)
        action_latencies.append(action_s)
        total_latencies.append(total_s)

    return {
        "vlm_prefix": prefix_latencies,
        "action_head_denoise": action_latencies,
        "total": total_latencies,
    }, action


def latency_summary(latencies: list[float]) -> dict[str, float]:
    sorted_latencies = sorted(latencies)
    mean_s = statistics.fmean(latencies)
    median_s = statistics.median(latencies)
    p90_s = sorted_latencies[int(0.9 * (len(sorted_latencies) - 1))]
    min_s = min(latencies)
    max_s = max(latencies)
    return {
        "mean_ms": mean_s * 1000.0,
        "median_ms": median_s * 1000.0,
        "p90_ms": p90_s * 1000.0,
        "min_ms": min_s * 1000.0,
        "max_ms": max_s * 1000.0,
        "hz": 1.0 / mean_s,
    }


def summarize(latencies: list[float], args: argparse.Namespace, policy: SmolVLAPolicy, action: torch.Tensor) -> dict:
    summary = latency_summary(latencies)
    mean_s = summary["mean_ms"] / 1000.0
    return {
        "policy_path": args.policy_path,
        "mode": args.mode,
        "device": str(policy.config.device),
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "bench_steps": args.bench_steps,
        "n_action_steps": policy.config.n_action_steps,
        "chunk_size": policy.config.chunk_size,
        "action_shape": list(action.shape),
        **summary,
        "amortized_action_ms": summary["mean_ms"] / policy.config.n_action_steps,
        "amortized_action_hz": policy.config.n_action_steps / mean_s,
    }


def summarize_split(
    latencies: dict[str, list[float]], args: argparse.Namespace, policy: SmolVLAPolicy, action: torch.Tensor
) -> dict:
    result = {
        "policy_path": args.policy_path,
        "mode": args.mode,
        "device": str(policy.config.device),
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "bench_steps": args.bench_steps,
        "n_action_steps": policy.config.n_action_steps,
        "chunk_size": policy.config.chunk_size,
        "num_denoise_steps": policy.model.config.num_steps,
        "num_vlm_layers": policy.model.vlm_with_expert.num_vlm_layers,
        "num_expert_layers": policy.model.vlm_with_expert.num_expert_layers,
        "action_shape": list(action.shape),
        "vlm_prefix": latency_summary(latencies["vlm_prefix"]),
        "action_head_denoise": latency_summary(latencies["action_head_denoise"]),
        "total": latency_summary(latencies["total"]),
    }
    result["total"]["amortized_action_ms"] = result["total"]["mean_ms"] / policy.config.n_action_steps
    result["total"]["amortized_action_hz"] = policy.config.n_action_steps / (result["total"]["mean_ms"] / 1000.0)
    result["action_head_denoise"]["per_denoise_step_ms"] = (
        result["action_head_denoise"]["mean_ms"] / policy.model.config.num_steps
    )
    return result


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    print(f"Loading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.eval()
    policy.config.device = device
    if args.num_denoise_steps is not None:
        policy.config.num_steps = args.num_denoise_steps
        policy.model.config.num_steps = args.num_denoise_steps

    batch = build_batch(policy, args, device)
    print(f"Image features: {list(policy.config.image_features)}")
    print(f"State shape: {tuple(batch[OBS_STATE].shape)}")

    if args.mode == "chunk":
        latencies, action = benchmark_chunk(policy, batch, args, device)
        result = summarize(latencies, args, policy, action)
    elif args.mode == "select_action":
        latencies, action = benchmark_select_action(policy, batch, args, device)
        result = summarize(latencies, args, policy, action)
    else:
        split_latencies, action = benchmark_split(policy, batch, args, device)
        result = summarize_split(split_latencies, args, policy, action)

    print(json.dumps(result, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
