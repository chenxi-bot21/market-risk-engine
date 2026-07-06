# Methodology — how the engine decides

Design goal: the standard market-risk toolkit a desk / model-validation team
actually uses, implemented transparently enough to defend in an interview.
Sign convention throughout: **VaR and ES are positive loss fractions**.

## 1. VaR & Expected Shortfall (four methods, on purpose)

| Method | Estimate | Strength | Weakness |
|---|---|---|---|
| **Historical** | empirical (1−α) quantile of the last N returns | no distributional assumption; captures fat tails that actually happened | jumpy; limited by the window; "ghost effects" as big days roll out |
| **Parametric (delta-normal)** | −μ + σ·z_α | smooth, fast, analytic ES | understates tail risk when returns are fat-tailed/skewed |
| **Cornish-Fisher** | normal quantile expanded with sample skew s and excess kurtosis k | keeps the parametric form but corrects the tail | expansion degrades for extreme s, k |
| **Monte Carlo** | simulate asset returns ~ N(μ, Σ), empirical VaR of simulated P&L | extends to non-linear positions/scenarios; seeded → reproducible | only as good as the distributional assumption |

Cornish-Fisher is applied at the **lower** quantile `z = Φ⁻¹(1−α) < 0`:

```
z_cf = z + (z²−1)s/6 + (z³−3z)k/24 − (2z³−5z)s²/36,   VaR = −(μ + σ·z_cf)
```

so negative skew correctly **increases** VaR (a subtle sign bug the test suite
guards against). ES for the normal case is the closed form
`ES = −μ + σ·φ(z_α)/(1−α)`; historical/MC ES average the tail beyond the
quantile. ES ≥ VaR is asserted in tests for every method.

**Why run all four:** the *spread* between historical and normal VaR is a
diagnostic — a wide gap means the Gaussian assumption is understating the tail,
and Cornish-Fisher quantifies how much of the gap is skew/kurtosis.

## 2. Volatility

- **EWMA (RiskMetrics)**: `σ²_t = λσ²_{t−1} + (1−λ)r²_{t−1}`, λ=0.94 — the
  industry default for fast-reacting daily vol.
- **GARCH(1,1)** by normal MLE: `σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}`,
  optimised with scipy L-BFGS-B (returns pre-scaled ×100 for conditioning,
  infeasible regions penalised so α+β < 1 is enforced). Reported:
  persistence (α+β), long-run vol `√(ω/(1−α−β))`, and the mean-reverting
  h-step forecast `σ²_{t+h} = σ²_LR + (α+β)^h (σ²_t − σ²_LR)`.
  Implemented from scratch (no `arch` package) — fewer deps, and the
  likelihood is simple enough that owning it is worth more than importing it.
  The fitter is validated by **parameter recovery on simulated GARCH data**.

## 3. Risk decomposition (component VaR)

Delta-normal VaR is homogeneous of degree 1 in weights, so Euler's theorem
gives an exact additive attribution:

```
σ_p = √(wᵀΣw);  MVaR_i = z_α (Σw)_i / σ_p;  CVaR_i = w_i · MVaR_i;  Σ CVaR_i = VaR_p
```

Component VaR is *the* desk-level answer to "which position drives my risk?" —
it nets diversification (a negatively-correlated asset can show a **negative**
component). Additivity is tested to 1e-12.

## 4. Backtesting (the regulatory part)

A VaR model is judged by its **violations** (days the loss exceeded VaR):

1. **Walk-forward** — at each day t, VaR is estimated from the previous
   `window` days only (no look-ahead), then compared with the day-t return.
2. **Kupiec POF (1995)** — likelihood-ratio test that the violation *count*
   matches 1−α. LR ~ χ²(1).
3. **Christoffersen (1998)** — independence test on the violation sequence's
   transition matrix (clustered breaches = model reacts too slowly), plus the
   joint conditional-coverage test, LR ~ χ²(2).
4. **Basel traffic light** — the supervisory rule on the last 250 days of a
   99% model: ≤4 breaches green, 5–9 yellow (capital multiplier add-on),
   ≥10 red (model rejected).

The test suite constructs violation sequences with known properties
(correct coverage → Kupiec passes; 5% breaches on a 99% model → rejected;
10 consecutive breaches → Christoffersen independence rejected; the same 10
spread out → passes) so the statistics are verified behaviourally, not just
numerically.

## 5. Stress testing & scenario analysis

VaR answers "how bad is a normal bad day"; stress answers "what does a crisis
do to *this* book". Two standard flavours, both in `scenarios.py`:

- **Hypothetical scenarios** — instantaneous per-asset return shocks
  (`P&L_i = w_i · shock_i`, the same first-order approximation as delta-normal
  VaR). A preset library covers the canonical episodes (GFC-style equity crash,
  2020-style pandemic shock, +200bp rates spike, flight-to-quality, USD
  squeeze); assets absent from the portfolio are skipped and reported, so the
  same scenarios run on any book.
- **Historical stress (empirical replay)** — the worst non-overlapping k-day
  windows the portfolio itself experienced in-sample: no distributional
  assumption, and each row carries auditable start/end dates. Tested against an
  engineered crash (the finder must locate the exact planted window).

## 6. Data

Offline-first: a **seeded, correlated-GBM synthetic book** (equity / bond /
gold / FX with a plausible correlation matrix, e.g. negative equity–bond) so
every example and test is reproducible with zero network. CSV and `yfinance`
sources plug in behind the same `resolve_prices` entry point — same pattern as
the credit-risk project's `datasets.resolve_dataset`.

## Limitations (honest list)

- Linear positions only — no options Greeks / full revaluation (a natural
  extension: plug simulated risk-factor paths into a pricing layer).
- Monte Carlo assumes multivariate normality; a Student-t or copula engine
  would capture joint fat tails.
- GARCH errors are normal; GJR/EGARCH asymmetry and t-errors are the obvious
  next steps and slot into the same `GarchResult` interface.
- No EVT (peaks-over-threshold) tail estimator yet — listed as future work
  rather than half-implemented.
