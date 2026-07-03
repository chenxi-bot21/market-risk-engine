"""
Value-at-Risk and Expected Shortfall — historical, parametric, Monte Carlo.

Sign convention: VaR and ES are reported as **positive loss numbers** (a
fraction of portfolio value). "99% one-day VaR = 0.021" means: with 99%
confidence the one-day loss will not exceed 2.1%. ES (a.k.a. CVaR) is the
expected loss *given* the loss exceeds VaR, so ES >= VaR always.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Historical (non-parametric)                                                 #
# --------------------------------------------------------------------------- #
def historical_var(returns, alpha: float = 0.99) -> float:
    """Empirical quantile: the loss exceeded on (1-alpha) of past days."""
    r = np.asarray(returns, dtype=float)
    _check(r, alpha)
    return float(-np.quantile(r, 1.0 - alpha))


def historical_es(returns, alpha: float = 0.99) -> float:
    """Mean loss on the days worse than the VaR quantile."""
    r = np.asarray(returns, dtype=float)
    _check(r, alpha)
    cutoff = np.quantile(r, 1.0 - alpha)
    tail = r[r <= cutoff]
    return float(-tail.mean())


# --------------------------------------------------------------------------- #
# Parametric (variance-covariance)                                            #
# --------------------------------------------------------------------------- #
def parametric_var(mu: float, sigma: float, alpha: float = 0.99) -> float:
    """Normal (delta-normal) VaR: -mu + sigma * z_alpha."""
    z = stats.norm.ppf(alpha)
    return float(-mu + sigma * z)


def parametric_es(mu: float, sigma: float, alpha: float = 0.99) -> float:
    """Normal ES: -mu + sigma * phi(z_alpha) / (1 - alpha)."""
    z = stats.norm.ppf(alpha)
    return float(-mu + sigma * stats.norm.pdf(z) / (1.0 - alpha))


def cornish_fisher_var(mu: float, sigma: float, skew: float, excess_kurt: float,
                       alpha: float = 0.99) -> float:
    """Modified VaR: adjust the normal quantile for skewness and fat tails.

    VaR lives in the LOWER tail of returns, so the Cornish-Fisher expansion is
    applied to the (negative) lower quantile z = ppf(1-alpha) — this is what
    makes negative skew correctly *increase* VaR (skew ``s``, EXCESS kurtosis
    ``k``):
        z_cf = z + (z^2-1)s/6 + (z^3-3z)k/24 - (2z^3-5z)s^2/36
    With s=0, k=0 this reduces exactly to the normal VaR.
    """
    z = stats.norm.ppf(1.0 - alpha)              # lower-tail quantile (< 0)
    z_cf = (z
            + (z**2 - 1.0) * skew / 6.0
            + (z**3 - 3.0 * z) * excess_kurt / 24.0
            - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0)
    return float(-(mu + sigma * z_cf))


# --------------------------------------------------------------------------- #
# Monte Carlo                                                                 #
# --------------------------------------------------------------------------- #
def monte_carlo_var_es(mean: np.ndarray, cov: np.ndarray, weights: np.ndarray,
                       alpha: float = 0.99, n_sims: int = 20_000,
                       seed: int = 7) -> tuple[float, float]:
    """Simulate multivariate-normal asset returns, take empirical VaR/ES of
    the simulated portfolio P&L. Deterministic per seed."""
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(np.asarray(mean, float),
                                   np.asarray(cov, float), size=n_sims)
    pnl = sims @ np.asarray(weights, float)
    return historical_var(pnl, alpha), historical_es(pnl, alpha)


# --------------------------------------------------------------------------- #
# One-call summary                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class VarSummary:
    alpha: float
    historical_var: float
    historical_es: float
    parametric_var: float
    parametric_es: float
    cornish_fisher_var: float
    monte_carlo_var: float
    monte_carlo_es: float

    def as_dict(self) -> dict[str, float]:
        return {
            "Historical VaR": self.historical_var,
            "Historical ES": self.historical_es,
            "Parametric (Normal) VaR": self.parametric_var,
            "Parametric (Normal) ES": self.parametric_es,
            "Cornish-Fisher VaR": self.cornish_fisher_var,
            "Monte Carlo VaR": self.monte_carlo_var,
            "Monte Carlo ES": self.monte_carlo_es,
        }


def var_summary(asset_returns, weights, alpha: float = 0.99,
                n_sims: int = 20_000, seed: int = 7) -> VarSummary:
    """Compute all methods on one portfolio: the cross-method comparison is
    the point — divergence between historical and normal VaR is the fat-tail
    signal Cornish-Fisher then quantifies."""
    R = np.asarray(asset_returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    port = R @ w
    mu, sigma = float(port.mean()), float(port.std(ddof=1))
    skew = float(stats.skew(port))
    ekurt = float(stats.kurtosis(port))          # scipy returns EXCESS kurtosis
    mc_var, mc_es = monte_carlo_var_es(R.mean(axis=0), np.cov(R, rowvar=False),
                                       w, alpha, n_sims, seed)
    return VarSummary(
        alpha=alpha,
        historical_var=historical_var(port, alpha),
        historical_es=historical_es(port, alpha),
        parametric_var=parametric_var(mu, sigma, alpha),
        parametric_es=parametric_es(mu, sigma, alpha),
        cornish_fisher_var=cornish_fisher_var(mu, sigma, skew, ekurt, alpha),
        monte_carlo_var=mc_var,
        monte_carlo_es=mc_es,
    )


def _check(r: np.ndarray, alpha: float) -> None:
    if r.ndim != 1 or r.size < 30:
        raise ValueError("Need a 1-D return series with at least 30 observations")
    if not 0.5 < alpha < 1.0:
        raise ValueError("alpha must be in (0.5, 1), e.g. 0.99")
