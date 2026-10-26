# Cloud Workload Forecasting

Forecast `active_vm_count` and `aggregate_cpu_utilization` for the final 2 hours (24 x 5-minute steps) of the Azure Public Dataset V2 subset (data-v1).

## Setup

Python 3.11+.

    python3 -m venv .venv            # Windows: py -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
    python -m pip install -e ".[dev,notebook]"

## Data

Download `data-v1.tar.gz` from
https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1
and extract it into `challenge_data/`:

    tar -xzf data-v1.tar.gz -C challenge_data

The data files are not committed; they are git-ignored.

## Generate predictions (one command)

    python -m cwfc.predict --data challenge_data --output predictions.csv

## Other commands

    python -m cwfc.evaluate --data challenge_data --out reports/validation_metrics.csv
    python -m pytest -q

## Main modelling decisions

- Targets are built at the native 5-minute cadence. No resampling and no filled values (the data has no gaps).
- `active_vm_count` counts each VM once per timestamp. `aggregate_cpu_utilization` is the mean `avg_cpu` across VMs, using only readings inside 0 to 100.
- Training uses the first 110 steps. The last 24 steps are used only for their timestamps, never for their target values.
- Rolling-origin validation runs inside the training data only (horizon 24, first origin 55, step 12, giving 3 folds).
- Candidate models: persistence, moving averages (3, 12, 36), exponential smoothing (0.2, 0.5) and ridge autoregression (3, 6 lags).
- The model is chosen per target by validation MAE. Persistence won for both targets.
- `timestamp` in `predictions.csv` holds the source timestamps in seconds. Each prediction maps one-to-one to a holdout timestamp.
- Clipping: `active_vm_count` >= 0 and CPU within 0 to 100.

## Repo layout

- `src/cwfc/`: code (targets, models, validation, prediction, evaluation)
- `tests/`: pytest tests
- `notebooks/01_eda.ipynb`: exploratory analysis
- `reports/`: figures, validation metrics and `final_report.md`
