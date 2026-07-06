"""
Stress testing & scenario analysis: hypothetical shocks and historical replay.

Two complementary approaches, both standard on a risk desk:

* **Hypothetical scenarios** — instantaneous return shocks per asset
  (e.g. equity −30%, rates rally +5%). P&L is the weight-linear combination,
  the same first-order approximation as delta-normal VaR.
* **Historical stress (empirical)** — replay the worst k-day windows the
  portfolio itself has lived through in the sample: no distributional
  assumption, and the dates make the story auditable.

VaR says "how bad is a normal bad day"; stress answers "what does a crisis
do to *this* book" — the two are reported together for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Preset shock library (fractions of value, per asset column when present).
# Calibrated to the rough magnitude of the named episodes for a daily-to-weekly
# horizon; assets missing from a portfolio are simply skipped and reported.
PRESET_SCENARIOS: dict[str, dict[str, float]] = {
    "Equity crash (GFC-style)":   {"EQUITY": -0.30, "BOND": +0.03, "GOLD": +0.05, "FX": -0.05},
    "Pandemic shock (2020-style)": {"EQUITY": -0.25, "BOND": +0.02, "GOLD": -0.03, "FX": -0.08},
    "Rates spike (+200bp)":       {"EQUITY": -0.08, "BOND": -0.10, "GOLD": -0.04, "FX": +0.02},
    "Flight to quality":          {"EQUITY": -0.15, "BOND": +0.05, "GOLD": +0.10, "FX": -0.03},
    "USD squeeze":                {"EQUITY": -0.10, "BOND": -0.02, "GOLD": -0.06, "FX": +0.08},
}


@dataclass
class ScenarioResult:
    name: str
    asset_pnl: pd.Series          # per-asset contribution (weight * shock)
    total_pnl: float              # portfolio return under the scenario
    applied: list[str]            # assets the scenario actually shocked
    skipped: list[str]            # scenario assets not in the portfolio


def apply_scenario(weights, shocks: dict[str, float], names: list[str],
                   name: str = "custom") -> ScenarioResult:
    """First-order P&L of an instantaneous shock: pnl_i = w_i * shock_i."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(names),):
        raise ValueError(f"weights {w.shape} do not match {len(names)} assets")
    shock_vec = np.array([shocks.get(n, 0.0) for n in names], dtype=float)
    pnl = w * shock_vec
    applied = [n for n in names if n in shocks]
    skipped = [k for k in shocks if k not in names]
    return ScenarioResult(name=name,
                          asset_pnl=pd.Series(pnl, index=names),
                          total_pnl=float(pnl.sum()),
                          applied=applied, skipped=skipped)


def run_preset_scenarios(weights, names: list[str]) -> list[ScenarioResult]:
    """Apply the whole preset library to the portfolio."""
    return [apply_scenario(weights, shocks, names, name)
            for name, shocks in PRESET_SCENARIOS.items()]


def historical_worst_windows(returns: pd.DataFrame, weights,
                             horizon: int = 5, top: int = 3) -> pd.DataFrame:
    """The `top` non-overlapping worst `horizon`-day windows for THIS portfolio.

    Empirical stress: cumulative portfolio return over every rolling window,
    then greedily pick the worst ones, skipping overlaps so each row is a
    distinct episode. Columns: start, end, cum_return.
    """
    w = np.asarray(weights, dtype=float)
    port = pd.Series(returns.values @ w, index=returns.index)
    if len(port) <= horizon + 1:
        raise ValueError("Not enough observations for the requested horizon")
    cum = port.rolling(horizon).sum().dropna()          # log-return sum ~ window return

    order = np.argsort(cum.values)                       # worst first
    chosen: list[int] = []
    for idx in order:
        if all(abs(idx - c) >= horizon for c in chosen):
            chosen.append(int(idx))
        if len(chosen) == top:
            break

    rows = []
    for idx in chosen:
        end = cum.index[idx]
        start_pos = returns.index.get_loc(end) - horizon + 1
        rows.append({"start": returns.index[start_pos], "end": end,
                     "cum_return": float(cum.iloc[idx])})
    return pd.DataFrame(rows)


def stress_report(returns: pd.DataFrame, weights,
                  horizon: int = 5, top: int = 3) -> dict:
    """Everything the CLI/dashboard need: presets + empirical worst windows."""
    names = list(returns.columns)
    presets = run_preset_scenarios(weights, names)
    worst = historical_worst_windows(returns, weights, horizon, top)
    return {
        "presets": presets,
        "worst_windows": worst,
        "worst_loss": float(worst["cum_return"].min()) if len(worst) else 0.0,
    }
