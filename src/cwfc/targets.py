"""Build the two aggregate target series from VM-level readings."""
from __future__ import annotations

import pandas as pd

TARGETS = ["active_vm_count", "aggregate_cpu_utilization"]


def build_targets(readings_path) -> pd.DataFrame:
    """Return one row per timestamp: active_vm_count and aggregate_cpu_utilization."""
    df = pd.read_csv(readings_path, usecols=["timestamp", "vm_id", "avg_cpu"])
    # "valid" observation = avg_cpu present and inside [0, 100]
    df = df[df["avg_cpu"].between(0, 100)]
    # a VM may be counted at most once per timestamp
    df = df.drop_duplicates(["timestamp", "vm_id"])
    out = (
        df.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return out
