import numpy as np
import pandas as pd

from cwfc.predict import run


def make_tiny_dataset(folder, n_vms=20, n_steps=60):
    rng = np.random.default_rng(0)
    rows = []
    for v in range(n_vms):
        for t in range(n_steps):
            rows.append((t * 300, f"vm{v}", 1, 9, float(np.clip(10 + rng.normal(), 0, 100))))
    df = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
    df.to_csv(folder / "vm_cpu_readings.csv.gz", index=False)


def test_predictions_follow_the_contract(tmp_path):
    make_tiny_dataset(tmp_path)
    out = tmp_path / "predictions.csv"
    run(tmp_path, out, horizon=24, verbose=False)
    saved = pd.read_csv(out)
    assert list(saved.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(saved) == 24
    assert saved["timestamp"].is_monotonic_increasing
    assert saved["timestamp"].is_unique
    assert np.isfinite(saved.to_numpy(dtype=float)).all()
    assert (saved["active_vm_count"] >= 0).all()
    assert saved["aggregate_cpu_utilization"].between(0, 100).all()
