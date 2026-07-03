"""Log returns and portfolio aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns; drops the first (NaN) row."""
    return np.log(prices / prices.shift(1)).dropna(how="any")


def normalise_weights(weights: list[float] | np.ndarray, n_assets: int) -> np.ndarray:
    """Validate and rescale weights to sum to 1 (long-only or long/short)."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (n_assets,):
        raise ValueError(f"Expected {n_assets} weights, got {w.shape}")
    total = w.sum()
    if abs(total) < 1e-12:
        raise ValueError("Weights sum to zero; cannot normalise")
    return w / total


def portfolio_returns(returns: pd.DataFrame, weights) -> pd.Series:
    """Weighted portfolio return series (weights normalised to sum to 1)."""
    w = normalise_weights(weights, returns.shape[1])
    return pd.Series(returns.values @ w, index=returns.index, name="portfolio")
