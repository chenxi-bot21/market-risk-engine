"""
VaR backtesting: rolling out-of-sample VaR, Kupiec POF, Christoffersen
independence / conditional coverage, and the Basel traffic-light zones.

A VaR model is only as good as its violations: at 99% you *want* ~1% of days
to breach — too many means the model understates risk (Kupiec rejects), and
clustered breaches mean it reacts too slowly (Christoffersen rejects), even
if the count looks right.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .var import historical_var, parametric_var


# --------------------------------------------------------------------------- #
# Rolling out-of-sample VaR                                                   #
# --------------------------------------------------------------------------- #
def rolling_var_backtest(returns: pd.Series, window: int = 250,
                         alpha: float = 0.99,
                         method: str = "historical") -> pd.DataFrame:
    """Walk-forward: at each t, estimate VaR from the previous `window` returns
    only (no look-ahead), then record whether day t actually breached it."""
    r = returns.to_numpy(dtype=float)
    if r.size <= window + 30:
        raise ValueError(f"Need > {window + 30} observations for a {window}-day backtest")

    n_out = r.size - window
    var = np.empty(n_out)
    for i in range(n_out):
        est = r[i:i + window]
        if method == "historical":
            var[i] = historical_var(est, alpha)
        elif method == "parametric":
            var[i] = parametric_var(est.mean(), est.std(ddof=1), alpha)
        else:
            raise ValueError(f"Unknown method: {method!r}")

    realized = r[window:]
    return pd.DataFrame({
        "return": realized,
        "var": var,
        "violation": realized < -var,
    }, index=returns.index[window:])


# --------------------------------------------------------------------------- #
# Statistical tests on the violation series                                   #
# --------------------------------------------------------------------------- #
def _safe_loglik(n: int, p: float) -> float:
    """n * ln(p) with the 0 * ln(0) = 0 convention."""
    return 0.0 if n == 0 else n * np.log(p)


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    reject_95: bool           # reject the model at the 5% level?


def kupiec_pof(violations, alpha: float = 0.99) -> TestResult:
    """Kupiec proportion-of-failures: is the breach *count* consistent with
    the promised coverage? LR ~ chi2(1) under H0."""
    v = np.asarray(violations, dtype=bool)
    T, x = v.size, int(v.sum())
    p = 1.0 - alpha
    pi = x / T
    ll_h0 = _safe_loglik(T - x, 1.0 - p) + _safe_loglik(x, p)
    ll_h1 = _safe_loglik(T - x, 1.0 - pi) + _safe_loglik(x, pi) if 0 < pi < 1 else ll_h0
    lr = max(0.0, -2.0 * (ll_h0 - ll_h1))
    pval = float(stats.chi2.sf(lr, df=1))
    return TestResult("Kupiec POF", float(lr), pval, pval < 0.05)


def christoffersen(violations, alpha: float = 0.99) -> list[TestResult]:
    """Christoffersen tests: independence (are breaches clustered?) and
    conditional coverage (count + independence jointly). chi2(1) / chi2(2)."""
    v = np.asarray(violations, dtype=bool).astype(int)
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    ll_h0 = _safe_loglik(n00 + n10, 1.0 - pi) + _safe_loglik(n01 + n11, pi)
    ll_h1 = (_safe_loglik(n00, 1.0 - pi01) + _safe_loglik(n01, pi01)
             + _safe_loglik(n10, 1.0 - pi11) + _safe_loglik(n11, pi11))
    lr_ind = max(0.0, -2.0 * (ll_h0 - ll_h1))
    p_ind = float(stats.chi2.sf(lr_ind, df=1))

    pof = kupiec_pof(v.astype(bool), alpha)
    lr_cc = pof.statistic + lr_ind
    p_cc = float(stats.chi2.sf(lr_cc, df=2))

    return [
        TestResult("Christoffersen independence", float(lr_ind), p_ind, p_ind < 0.05),
        TestResult("Conditional coverage (POF+ind)", float(lr_cc), p_cc, p_cc < 0.05),
    ]


def basel_traffic_light(violations, lookback: int = 250) -> dict:
    """Basel committee zones on the last `lookback` days of a 99% VaR model:
    green 0-4 breaches, yellow 5-9 (capital add-on), red 10+ (model rejected)."""
    v = np.asarray(violations, dtype=bool)
    recent = v[-lookback:]
    n = int(recent.sum())
    zone = "green" if n <= 4 else ("yellow" if n <= 9 else "red")
    return {"violations": n, "observations": int(recent.size), "zone": zone}


def backtest_report(returns: pd.Series, window: int = 250, alpha: float = 0.99,
                    method: str = "historical") -> dict:
    """Run the full battery and return everything the report/dashboard needs."""
    bt = rolling_var_backtest(returns, window, alpha, method)
    v = bt["violation"].to_numpy()
    tests = [kupiec_pof(v, alpha)] + christoffersen(v, alpha)
    return {
        "backtest": bt,
        "tests": tests,
        "traffic_light": basel_traffic_light(v),
        "expected_violations": round((1.0 - alpha) * v.size, 1),
        "observed_violations": int(v.sum()),
    }
