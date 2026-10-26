"""Make predictions.csv.  Usage:
    python -m cwfc.predict --data challenge_data --output predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .models import CANDIDATES, persistence
from .targets import TARGETS, build_targets


def rolling_origin_mae(y, model, horizon, min_train, step):
    """Expanding-window validation INSIDE the training data only."""
    errors, origin = [], min_train
    while origin + horizon <= len(y):
        try:
            pred = model(y[:origin], horizon)
        except ValueError:
            return float("inf")
        errors.append(float(np.mean(np.abs(y[origin : origin + horizon] - pred))))
        origin += step
    return float(np.mean(errors)) if errors else float("inf")


def run(data_dir, output, horizon=24, verbose=True):
    series = build_targets(Path(data_dir) / "vm_cpu_readings.csv.gz")
    if len(series) < horizon + 2:
        raise ValueError("not enough timestamps for the requested horizon")
    train, holdout = series.iloc[:-horizon], series.iloc[-horizon:]

    # The holdout is used ONLY for its timestamps, never for its target values.
    result = pd.DataFrame({"timestamp": holdout["timestamp"].to_numpy()})
    for target in TARGETS:
        y = train[target].to_numpy(dtype=float)
        min_train = max(12, len(y) // 2)
        step = max(1, horizon // 2)
        scores = {
            name: rolling_origin_mae(y, m, horizon, min_train, step)
            for name, m in CANDIDATES.items()
        }
        best = min(scores, key=scores.get)
        model = CANDIDATES[best] if np.isfinite(scores[best]) else persistence
        if verbose:
            print(f"[{target}] validation MAE per model:")
            for k, v in sorted(scores.items(), key=lambda kv: kv[1]):
                print(f"   {k:16s} {v:.4f}")
            print(f"   -> chosen: {best}")
        result[target] = model(y, horizon)

    result["active_vm_count"] = result["active_vm_count"].clip(lower=0)
    result["aggregate_cpu_utilization"] = result["aggregate_cpu_utilization"].clip(0, 100)
    result.to_csv(output, index=False)
    if verbose:
        print(f"saved {len(result)} rows to {output}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="challenge_data")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--horizon", type=int, default=24)
    a = ap.parse_args()
    run(a.data, a.output, a.horizon)


if __name__ == "__main__":
    main()
