"""
Extreme Value Theory tail estimation: Peaks-over-Threshold with a GPD fit.

Historical VaR can't see beyond the worst day in the sample, and the normal
distribution actively understates tails. EVT fixes the far tail (99.5%+, where
regulators live): by the Pickands–Balkema–de Haan theorem, losses beyond a high
threshold u converge to a Generalized Pareto Distribution, so we fit GPD to the
exceedances and extrapolate quantiles the sample barely contains.

    VaR_q = u + (β/ξ) [ ((1−q) / (N_u/n))^(−ξ) − 1 ]
    ES_q  = VaR_q / (1−ξ) + (β − ξu) / (1−ξ)          (ξ < 1)

ξ > 0 ⇒ heavy (Pareto-type) tail — for daily equity returns typically ~0.1–0.3.
Reference: McNeil, Frey & Embrechts, *Quantitative Risk Management*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto


@dataclass
class GPDTail:
    threshold: float        # u, in LOSS units (positive)
    xi: float               # GPD shape (ξ): >0 heavy tail, ≈0 exponential
    beta: float             # GPD scale (β)
    n_obs: int              # total observations
    n_exceed: int           # observations beyond the threshold
    threshold_q: float      # quantile used to set u (e.g. 0.95 of losses)

    def var(self, alpha: float = 0.995) -> float:
        """EVT VaR at confidence `alpha` (must lie beyond the threshold quantile)."""
        self._check_alpha(alpha)
        ratio = (1.0 - alpha) / (self.n_exceed / self.n_obs)
        if abs(self.xi) < 1e-9:                     # ξ→0: exponential tail limit
            return float(self.threshold - self.beta * np.log(ratio))
        return float(self.threshold + self.beta / self.xi * (ratio ** (-self.xi) - 1.0))

    def es(self, alpha: float = 0.995) -> float:
        """EVT Expected Shortfall; requires ξ < 1 (finite tail mean)."""
        if self.xi >= 1.0:
            raise ValueError(f"ES undefined: fitted xi={self.xi:.3f} >= 1 (infinite mean)")
        v = self.var(alpha)
        return float(v / (1.0 - self.xi) + (self.beta - self.xi * self.threshold) / (1.0 - self.xi))

    def _check_alpha(self, alpha: float) -> None:
        if alpha <= self.threshold_q:
            raise ValueError(
                f"alpha={alpha} must exceed the threshold quantile ({self.threshold_q}) — "
                "inside the threshold, use historical/parametric VaR instead")


def fit_gpd_tail(returns, threshold_q: float = 0.95, min_exceedances: int = 30) -> GPDTail:
    """Fit a GPD to loss exceedances over the `threshold_q` empirical loss quantile.

    threshold_q trades bias (too low: GPD asymptotics don't hold) against
    variance (too high: too few exceedances); 0.95 is the common default.
    """
    losses = -np.asarray(returns, dtype=float)
    if losses.ndim != 1 or losses.size < 250:
        raise ValueError("Need a 1-D return series with at least 250 observations")
    if not 0.80 <= threshold_q < 0.99:
        raise ValueError("threshold_q should be in [0.80, 0.99)")

    u = float(np.quantile(losses, threshold_q))
    exceed = losses[losses > u] - u
    if exceed.size < min_exceedances:
        raise ValueError(f"Only {exceed.size} exceedances over u={u:.5f}; "
                         f"need >= {min_exceedances} (lower threshold_q or more data)")

    xi, _loc, beta = genpareto.fit(exceed, floc=0.0)
    return GPDTail(threshold=u, xi=float(xi), beta=float(beta),
                   n_obs=int(losses.size), n_exceed=int(exceed.size),
                   threshold_q=threshold_q)


def evt_var_es(returns, alpha: float = 0.995,
               threshold_q: float = 0.95) -> tuple[float, float]:
    """One-call convenience: fit the tail and return (VaR, ES) at `alpha`."""
    tail = fit_gpd_tail(returns, threshold_q)
    return tail.var(alpha), tail.es(alpha)
