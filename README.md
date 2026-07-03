# marketrisk — Multi-Asset VaR / ES Engine

<p>
  <a href="https://github.com/chenxi-bot21/market-risk-engine/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/chenxi-bot21/market-risk-engine/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-41%20passing-2E4057">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

A market-risk engine covering the full desk workflow: **Value-at-Risk and
Expected Shortfall** by four methods (historical, parametric-normal,
Cornish-Fisher, Monte Carlo), **EWMA and GARCH(1,1)** volatility, **Euler
risk decomposition** (component VaR), and a **regulatory-grade backtest**
(rolling out-of-sample VaR + Kupiec POF, Christoffersen independence /
conditional coverage, Basel traffic light).

Pairs with the [credit-risk PD/IFRS 9 project](https://github.com/chenxi-bot21/credit-risk-model)
to cover both halves of a bank risk stack: **credit + market risk**.

Runs **fully offline** (reproducible synthetic multi-asset data, no API key);
upgrades to real prices via `yfinance` when installed. GARCH is implemented
directly on `scipy` MLE — no `arch` dependency.

➡️ **How each method works and why: [METHODOLOGY.md](METHODOLOGY.md)**

## Quickstart

```bash
pip install -e .            # numpy + pandas + scipy only
python -m marketrisk run    # synthetic 4-asset book, writes output/market_risk_report.md
```

```
Assets: EQUITY, BOND, GOLD, FX  |  1249 return days  |  weights: [0.25 0.25 0.25 0.25]

99% one-day VaR / ES
  Historical VaR                1.117%
  Historical ES                 1.234%
  Parametric (Normal) VaR       1.022%
  Parametric (Normal) ES        1.172%
  Cornish-Fisher VaR            1.057%
  Monte Carlo VaR               1.019%
  Monte Carlo ES                1.160%

GARCH(1,1): persistence=0.620, long-run vol=0.442%/day
Backtest: 16 violations vs 10.0 expected → Basel zone: YELLOW
  Kupiec POF                       LR= 3.089  p=0.079  pass
  Christoffersen independence      LR= 0.521  p=0.470  pass
  Conditional coverage (POF+ind)   LR= 3.610  p=0.164  pass
```

Note how the methods *disagree*: historical VaR (1.117%) sits above normal VaR
(1.022%) because the data has fatter tails than a Gaussian — exactly the gap
Cornish-Fisher (1.057%) partially recovers. That cross-method comparison is the
point of running all four.

### Your own portfolio

```bash
# CSV of prices (date index, one column per asset), custom weights & level
python -m marketrisk run --source csv --csv prices.csv --weights 0.5,0.3,0.2 --alpha 0.975

# Real prices from Yahoo Finance (optional dependency)
pip install -e ".[live]"
python -m marketrisk run --source yfinance --tickers SPY,TLT,GLD --start 2019-01-01
```

### Dashboard

```bash
pip install -e ".[app]"     # streamlit + plotly
streamlit run app.py
```

Four tabs: VaR/ES method comparison · EWMA vs GARCH volatility (+ forecast) ·
backtest chart with violations and test table · component-VaR decomposition.

## What's inside

| Module | Contents |
|---|---|
| `var.py` | Historical, parametric-normal, **Cornish-Fisher** (skew/kurtosis-adjusted), and **Monte Carlo** VaR & ES; one-call `var_summary` |
| `volatility.py` | RiskMetrics **EWMA** (λ=0.94) and **GARCH(1,1) by MLE** (scipy L-BFGS-B, feasibility-penalised), with mean-reverting vol forecasts |
| `decompose.py` | Marginal & **component VaR** via Euler allocation — components sum *exactly* to total VaR (tested to 1e-12) |
| `backtest.py` | Walk-forward VaR (no look-ahead), **Kupiec POF**, **Christoffersen** independence & conditional coverage, **Basel traffic light** |
| `data.py` | Reproducible correlated-GBM synthetic book / CSV / `yfinance` (pluggable) |
| `report.py`, `cli.py` | Markdown risk report + `python -m marketrisk` CLI |

## Engineering

- `src/` package layout, typed dataclasses, **41 unit tests** including
  closed-form checks (normal VaR/ES vs analytic values), Euler-additivity to
  12 decimals, seeded Monte-Carlo determinism, GARCH parameter recovery on
  simulated GARCH data, and statistical-test behaviour (clustered violations
  must fail Christoffersen; correct coverage must pass Kupiec).
- CI on Python 3.11 / 3.12; core deps are just numpy + pandas + scipy.

```bash
python -m unittest discover -s tests -t .
```

## License

MIT © Chenxi Zhao
