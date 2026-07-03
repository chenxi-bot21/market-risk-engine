"""
Volatility models: RiskMetrics EWMA and GARCH(1,1) by maximum likelihood.

GARCH is implemented directly on scipy (no `arch` dependency) so the package
stays light and runs on any Python with the scientific stack. Returns are
scaled by 100 inside the optimiser for numerical stability and scaled back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_SCALE = 100.0


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA daily volatility: s2_t = lam*s2_{t-1} + (1-lam)*r2_{t-1}.

    Seeded with the sample variance of the first 30 observations. The first
    value corresponds to the second return date (needs one lag).
    """
    r = returns.to_numpy(dtype=float)
    if r.size < 40:
        raise ValueError("Need at least 40 observations for EWMA")
    s2 = np.empty(r.size)
    s2[0] = r[:30].var(ddof=1)
    for t in range(1, r.size):
        s2[t] = lam * s2[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return pd.Series(np.sqrt(s2), index=returns.index, name="ewma_vol")


@dataclass
class GarchResult:
    omega: float          # in return^2 units (unscaled)
    alpha: float          # ARCH coefficient
    beta: float           # GARCH coefficient
    persistence: float    # alpha + beta (< 1 for stationarity)
    long_run_vol: float   # sqrt(omega / (1 - persistence)), daily
    loglik: float
    cond_vol: pd.Series   # in-sample conditional daily volatility

    def forecast(self, horizon: int = 10) -> np.ndarray:
        """h-step-ahead daily vol forecast, mean-reverting to long-run vol:
        s2_{t+h} = lr2 + persistence^h * (s2_t - lr2)."""
        lr2 = self.long_run_vol ** 2
        s2_t = float(self.cond_vol.iloc[-1]) ** 2
        h = np.arange(1, horizon + 1)
        return np.sqrt(lr2 + self.persistence ** h * (s2_t - lr2))


def _garch_neg_loglik(params: np.ndarray, eps2: np.ndarray, s2_0: float) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10                     # infeasible -> huge penalty
    s2 = np.empty(eps2.size)
    s2[0] = s2_0
    for t in range(1, eps2.size):
        s2[t] = omega + alpha * eps2[t - 1] + beta * s2[t - 1]
    if np.any(s2 <= 0):                 # numerical safety
        return 1e10
    return 0.5 * float(np.sum(np.log(s2) + eps2 / s2))


def fit_garch(returns: pd.Series) -> GarchResult:
    """Fit GARCH(1,1) with normal errors by MLE (L-BFGS-B, feasibility penalty)."""
    r = returns.to_numpy(dtype=float)
    if r.size < 250:
        raise ValueError("Need at least ~250 observations to fit GARCH")
    eps = (r - r.mean()) * _SCALE
    eps2 = eps ** 2
    var0 = float(eps2.mean())

    x0 = np.array([0.05 * var0, 0.08, 0.90])
    bounds = [(1e-8, 10.0 * var0), (1e-6, 0.5), (0.4, 0.998)]
    res = minimize(_garch_neg_loglik, x0, args=(eps2, var0),
                   method="L-BFGS-B", bounds=bounds)
    omega_s, alpha, beta = res.x

    # Rebuild the conditional-variance path with the fitted parameters.
    s2 = np.empty(eps2.size)
    s2[0] = var0
    for t in range(1, eps2.size):
        s2[t] = omega_s + alpha * eps2[t - 1] + beta * s2[t - 1]

    persistence = float(alpha + beta)
    omega = float(omega_s) / _SCALE**2
    long_run = float(np.sqrt(omega / max(1.0 - persistence, 1e-8)))
    cond_vol = pd.Series(np.sqrt(s2) / _SCALE, index=returns.index, name="garch_vol")
    return GarchResult(omega=omega, alpha=float(alpha), beta=float(beta),
                       persistence=persistence, long_run_vol=long_run,
                       loglik=-float(res.fun), cond_vol=cond_vol)
