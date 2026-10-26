"""Simple forecasting models. Each takes (y, h): history array, horizon -> h forecasts."""
from __future__ import annotations

import numpy as np


def persistence(y, h):
    return np.repeat(y[-1], h)


def moving_average(y, h, window):
    if len(y) < window:
        raise ValueError("not enough history")
    return np.repeat(np.mean(y[-window:]), h)


def exp_smoothing(y, h, alpha):
    level = y[0]
    for v in y[1:]:
        level = alpha * v + (1 - alpha) * level
    return np.repeat(level, h)


def ar_ridge(y, h, p, lam=1.0):
    """Autoregression on the last p values (ridge-regularised), forecast step by step."""
    if len(y) < p + 5:
        raise ValueError("not enough history")
    mu = float(np.mean(y))
    z = np.asarray(y, dtype=float) - mu
    X = np.column_stack([z[i : len(z) - p + i] for i in range(p)])
    t = z[p:]
    w = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ t)
    hist = list(z[-p:])
    out = []
    for _ in range(h):
        nxt = float(np.dot(w, hist[-p:]))
        out.append(nxt)
        hist.append(nxt)
    return np.array(out) + mu


CANDIDATES = {
    "persistence": persistence,
    "moving_avg_3": lambda y, h: moving_average(y, h, 3),
    "moving_avg_12": lambda y, h: moving_average(y, h, 12),
    "moving_avg_36": lambda y, h: moving_average(y, h, 36),
    "exp_smooth_0.2": lambda y, h: exp_smoothing(y, h, 0.2),
    "exp_smooth_0.5": lambda y, h: exp_smoothing(y, h, 0.5),
    "ar_ridge_3": lambda y, h: ar_ridge(y, h, 3),
    "ar_ridge_6": lambda y, h: ar_ridge(y, h, 6),
}
