import numpy as np

from cwfc.models import ar_ridge, exp_smoothing, moving_average, persistence


def test_models_return_horizon_length():
    y = np.linspace(1, 2, 40)
    for f in (persistence, lambda y, h: moving_average(y, h, 5),
              lambda y, h: exp_smoothing(y, h, 0.3), lambda y, h: ar_ridge(y, h, 3)):
        assert f(y, 7).shape == (7,)


def test_persistence_repeats_last_value():
    assert (persistence(np.array([1.0, 2.0, 5.0]), 3) == 5.0).all()


def test_validation_table_has_all_columns():
    import pandas as pd
    from cwfc.evaluate import validation_table

    rng = np.random.default_rng(1)
    train = pd.DataFrame({
        "active_vm_count": np.full(80, 100.0),
        "aggregate_cpu_utilization": 10 + rng.normal(size=80),
    })
    table = validation_table(train, horizon=12)
    assert {"MAE", "RMSE", "MAE_vs_persistence", "peak_MAE"} <= set(table.columns)
    assert len(table) > 0
