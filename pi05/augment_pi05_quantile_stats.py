#!/usr/bin/env python
"""Add q01/q10/q50/q90/q99 stats needed by PI05 quantile normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FEATURES = ("observation.state", "action")
DEFAULT_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def quantile_key(q: float) -> str:
    return f"q{int(q * 100):02d}"


def load_feature_array(parquet_files: list[Path], feature: str) -> np.ndarray:
    chunks = []
    for parquet_file in parquet_files:
        series = pd.read_parquet(parquet_file, columns=[feature])[feature]
        chunks.append(np.stack(series.to_numpy()).astype(np.float64, copy=False))
    if not chunks:
        raise ValueError(f"No parquet files found for feature {feature!r}")
    return np.concatenate(chunks, axis=0)


def add_quantiles(dataset_root: Path, features: tuple[str, ...], overwrite: bool) -> bool:
    stats_path = dataset_root / "meta" / "stats.json"
    data_dir = dataset_root / "data"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing stats file: {stats_path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    changed = False
    for feature in features:
        if feature not in stats:
            raise KeyError(f"Feature {feature!r} not found in {stats_path}")
        feature_stats = stats[feature]
        missing = [quantile_key(q) for q in DEFAULT_QUANTILES if quantile_key(q) not in feature_stats]
        if not overwrite and not missing:
            continue

        values = load_feature_array(parquet_files, feature)
        quantiles = np.quantile(values, DEFAULT_QUANTILES, axis=0)
        for idx, q in enumerate(DEFAULT_QUANTILES):
            feature_stats[quantile_key(q)] = quantiles[idx].tolist()
        changed = True
        print(f"Added quantile stats for {feature}: shape={values.shape}", flush=True)

    if changed:
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
            f.write("\n")
        print(f"Updated {stats_path}", flush=True)
    else:
        print(f"Quantile stats already present in {stats_path}", flush=True)
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_quantiles(args.dataset_root, tuple(args.features), args.overwrite)


if __name__ == "__main__":
    main()
