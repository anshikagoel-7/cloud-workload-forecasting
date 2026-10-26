"""Validation metrics for every candidate model. Uses TRAINING data only.

    python -m cwfc.evaluate --data challenge_data --out reports/validation_metrics.csv

Optional, run ONCE at the very end (after predictions.csv is final):
    python -m cwfc.evaluate --data challenge_data --final-holdout predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import mae, normalized_mae, rmse, smape
from .models import CANDIDATES, persistence
from .targets import TARGETS, build_targets


def collect_folds(y, model, horizon, min_train, step):
    """Return actual, predicted and 'is peak' flags, stacked over all validation folds."""
    actual, pred, peak = [], [], []
    origin = min_train
    while origin + horizon <= len(y):
        hist = y[:origin]
        a = y[origin : origin + horizon]
        actual.append(a)
        pred.append(model(hist, horizon))
        # "peak" = in the top 20% of what was already known at the forecast origin
        peak.append(a >= np.quantile(hist, 0.8))
        origin += step
    return np.concatenate(actual), np.concatenate(pred), np.concatenate(peak)


def validation_table(train, horizon=24):
    rows = []
    for target in TARGETS:
        y = train[target].to_numpy(dtype=float)
        min_train, step = max(12, len(y) // 2), max(1, horizon // 2)
        base_a, base_p, _ = collect_folds(y, persistence, horizon, min_train, step)
        for name, model in CANDIDATES.items():
            try:
                a, p, is_peak = collect_folds(y, model, horizon, min_train, step)
            except ValueError:
                continue
            rows.append({
                "target": target,
                "model": name,
                "MAE": mae(a, p),
                "RMSE": rmse(a, p),
                "sMAPE_%": smape(a, p),
                "MAE_vs_persistence": normalized_mae(a, p, base_p),
                "peak_MAE": mae(a[is_peak], p[is_peak]) if is_peak.any() else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="challenge_data")
    ap.add_argument("--out", default="reports/validation_metrics.csv")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--final-holdout", default=None,
                    help="path to predictions.csv; scores it on the real holdout (use ONCE)")
    a = ap.parse_args()

    series = build_targets(Path(a.data) / "vm_cpu_readings.csv.gz")
    train, holdout = series.iloc[: -a.horizon], series.iloc[-a.horizon :]

    if a.final_holdout:
        pred = pd.read_csv(a.final_holdout)
        for t in TARGETS:
            base = persistence(train[t].to_numpy(float), a.horizon)
            print(f"{t}: MAE={mae(holdout[t], pred[t]):.4f} "
                  f"RMSE={rmse(holdout[t], pred[t]):.4f} "
                  f"persistence_MAE={mae(holdout[t], base):.4f}")
        return

    table = validation_table(train, a.horizon)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
