#!/usr/bin/env python3
"""Small multi-GPU NCCL all-reduce validation used before distributed training."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    value = torch.full((8 * 1024 * 1024,), float(local_rank + 1), device=local_rank, dtype=torch.float32)
    expected = world_size * (world_size + 1) / 2
    for _ in range(10):
        value.fill_(float(local_rank + 1))
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(local_rank)
        if not torch.all(value == expected):
            raise RuntimeError(f"rank {local_rank}: all-reduce result is invalid")

    if local_rank == 0:
        print(f"NCCL all-reduce passed on {world_size} GPUs", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
