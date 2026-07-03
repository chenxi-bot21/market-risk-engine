"""
Parametric risk decomposition: marginal and component VaR (Euler allocation).

Because delta-normal VaR is homogeneous of degree 1 in the weights, Euler's
theorem gives an exact additive split: sum(component VaR_i) == portfolio VaR.
That additivity is what makes component VaR the standard desk-level risk
attribution — each asset's share of the total, netting diversification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def component_var(weights, cov, alpha: float = 0.99,
                  names: list[str] | None = None) -> pd.DataFrame:
    """Marginal, component, and %-of-total VaR per asset (mean ignored, as is
    conventional for short horizons).

    marginal_i  = z_alpha * (Sigma w)_i / sigma_p     (dVaR/dw_i)
    component_i = w_i * marginal_i                     (sums to total VaR)
    """
    w = np.asarray(weights, dtype=float)
    S = np.asarray(cov, dtype=float)
    z = stats.norm.ppf(alpha)

    sigma_p = float(np.sqrt(w @ S @ w))
    if sigma_p <= 0:
        raise ValueError("Portfolio volatility is zero; check weights/covariance")
    marginal = z * (S @ w) / sigma_p
    component = w * marginal
    total = z * sigma_p

    idx = names if names is not None else [f"asset_{i}" for i in range(w.size)]
    df = pd.DataFrame({
        "weight": w,
        "marginal_var": marginal,
        "component_var": component,
        "pct_of_total": component / total,
    }, index=idx)
    df.attrs["total_var"] = total
    return df
