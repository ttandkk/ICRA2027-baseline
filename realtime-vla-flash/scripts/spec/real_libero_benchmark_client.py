import argparse
import json
import statistics
import time

import numpy as np
from openpi_client import websocket_client_policy


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a running real-weight LIBERO FLASH server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    records = []

    for index in range(args.runs):
        observation = {
            "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/state": rng.standard_normal(8).astype(np.float32),
            "prompt": args.prompt,
            "__executed_steps__": 12,
        }
        if index == 0:
            observation["__reset_policy_state__"] = True

        start = time.perf_counter()
        output = client.infer(observation)
        roundtrip_ms = (time.perf_counter() - start) * 1000.0
        timing = dict(output.get("policy_timing", {}))
        records.append(
            {
                "index": index,
                "roundtrip_ms": roundtrip_ms,
                "accepted_prefix_len": int(output.get("accepted_prefix_len", 0)),
                "timing": timing,
            }
        )
        print(
            f"run={index:02d} route={timing.get('route_type', 'unknown')} "
            f"sample_actions_ms={float(timing.get('sample_actions_ms', 0.0)):.3f} "
            f"roundtrip_ms={roundtrip_ms:.3f} accepted={output.get('accepted_prefix_len', 0)}",
            flush=True,
        )

    draft_records = [record for record in records if record["timing"].get("route_type") != "full"]
    summary = {
        "runs": len(records),
        "full_rounds": len(records) - len(draft_records),
        "draft_rounds": len(draft_records),
        "all_sample_actions_ms": _summary(
            [float(record["timing"]["sample_actions_ms"]) for record in records]
        ),
        "all_roundtrip_ms": _summary([record["roundtrip_ms"] for record in records]),
        "records": records,
    }
    if draft_records:
        summary["draft_sample_actions_ms"] = _summary(
            [float(record["timing"]["sample_actions_ms"]) for record in draft_records]
        )
        summary["draft_roundtrip_ms"] = _summary([record["roundtrip_ms"] for record in draft_records])

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
