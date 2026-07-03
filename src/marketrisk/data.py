"""
Price data: reproducible synthetic multi-asset paths, CSV, or live Yahoo Finance.

Offline-first (same house style as the credit-risk project): the bundled
synthetic generator needs no network or API key, so every example, test, and
the dashboard run anywhere; `--source yfinance` upgrades to real prices when
the optional dependency is installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Annualised drift / vol and a plausible cross-asset correlation structure for
# the synthetic book: broad equity, rates (bond), gold, and an FX carry proxy.
_SYNTH_ASSETS = ("EQUITY", "BOND", "GOLD", "FX")
_ANNUAL_MU = np.array([0.08, 0.03, 0.05, 0.01])
_ANNUAL_VOL = np.array([0.18, 0.07, 0.15, 0.10])
_CORR = np.array([
    [1.00, -0.25, 0.10, 0.20],
    [-0.25, 1.00, 0.15, -0.05],
    [0.10, 0.15, 1.00, 0.05],
    [0.20, -0.05, 0.05, 1.00],
])
TRADING_DAYS = 252


def synthetic_prices(n_days: int = 1250, seed: int = 42,
                     tickers: tuple[str, ...] = _SYNTH_ASSETS) -> pd.DataFrame:
    """Correlated geometric-Brownian daily prices (deterministic per seed).

    Returns a DataFrame indexed by business day, one column per ticker,
    starting at 100.0. 1250 days ~ 5 trading years, enough for a 250-day
    rolling backtest with plenty of out-of-sample points.
    """
    k = len(tickers)
    mu, vol = _ANNUAL_MU[:k], _ANNUAL_VOL[:k]
    corr = _CORR[:k, :k]
    cov = np.outer(vol, vol) * corr / TRADING_DAYS
    drift = mu / TRADING_DAYS - 0.5 * np.diag(cov)

    rng = np.random.default_rng(seed)
    shocks = rng.multivariate_normal(np.zeros(k), cov, size=n_days)
    log_prices = np.log(100.0) + np.cumsum(drift + shocks, axis=0)

    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(np.exp(log_prices), index=idx, columns=list(tickers))


def load_csv(path: str) -> pd.DataFrame:
    """Load a wide CSV of prices (first column = date, one column per asset)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if df.empty:
        raise ValueError(f"No numeric price data found in {path}")
    return df.sort_index()


def fetch_yfinance(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance (optional dependency)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "yfinance is not installed. `pip install yfinance`, or use the "
            "synthetic/CSV sources which need no network."
        ) from exc
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return prices.dropna(how="all").ffill().dropna()


def resolve_prices(source: str = "synthetic", *, csv_path: str | None = None,
                   tickers: list[str] | None = None, start: str = "2018-01-01",
                   end: str | None = None, seed: int = 42) -> pd.DataFrame:
    """Single entry point the CLI/dashboard use to obtain a price panel."""
    if source == "synthetic":
        return synthetic_prices(seed=seed)
    if source == "csv":
        if not csv_path:
            raise ValueError("source='csv' requires csv_path")
        return load_csv(csv_path)
    if source == "yfinance":
        if not tickers:
            raise ValueError("source='yfinance' requires tickers")
        return fetch_yfinance(tickers, start, end)
    raise ValueError(f"Unknown source: {source!r} (use synthetic | csv | yfinance)")
