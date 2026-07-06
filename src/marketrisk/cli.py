"""
CLI: run the full market-risk analysis end-to-end and write a report.

    python -m marketrisk run                          # synthetic data, defaults
    python -m marketrisk run --alpha 0.975 --window 500
    python -m marketrisk run --source csv --csv prices.csv --weights 0.5,0.3,0.2
    python -m marketrisk run --source yfinance --tickers SPY,TLT,GLD --start 2019-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import backtest as bt
from . import data, returns as ret
from .decompose import component_var
from .report import build_report
from .scenarios import stress_report
from .var import var_summary
from .volatility import fit_garch


def _parse_weights(text: str | None, n: int) -> np.ndarray:
    if not text:
        return np.full(n, 1.0 / n)
    w = np.array([float(x) for x in text.split(",")], dtype=float)
    return ret.normalise_weights(w, n)


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="marketrisk", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="compute VaR/ES, decomposition, GARCH, backtest")
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "csv", "yfinance"])
    p.add_argument("--csv", default=None, help="price CSV (source=csv)")
    p.add_argument("--tickers", default=None, help="comma list (source=yfinance)")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--weights", default=None, help="comma list; default equal-weight")
    p.add_argument("--alpha", type=float, default=0.99)
    p.add_argument("--window", type=int, default=250, help="backtest window (days)")
    p.add_argument("--method", default="historical", choices=["historical", "parametric"],
                   help="VaR method for the rolling backtest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stress-horizon", type=int, default=5,
                   help="window (days) for historical worst-case stress replay")
    p.add_argument("--output", default="output", help="report directory")
    args = ap.parse_args(argv)

    tickers = args.tickers.split(",") if args.tickers else None
    prices = data.resolve_prices(args.source, csv_path=args.csv, tickers=tickers,
                                 start=args.start, seed=args.seed)
    R = ret.log_returns(prices)
    w = _parse_weights(args.weights, R.shape[1])
    port = ret.portfolio_returns(R, w)

    print(f"Assets: {', '.join(prices.columns)}  |  {len(R)} return days  "
          f"|  weights: {np.round(w, 3)}")

    summary = var_summary(R.values, w, alpha=args.alpha, seed=args.seed)
    decomp = component_var(w, np.cov(R.values, rowvar=False), args.alpha,
                           names=list(prices.columns))
    garch = fit_garch(port)
    bt_out = bt.backtest_report(port, window=args.window, alpha=args.alpha,
                                method=args.method)
    stress = stress_report(R, w, horizon=args.stress_horizon)

    print(f"\n{int(args.alpha * 100)}% one-day VaR / ES")
    for k, v in summary.as_dict().items():
        print(f"  {k:<28s} {v * 100:6.3f}%")
    print(f"\nGARCH(1,1): persistence={garch.persistence:.3f}, "
          f"long-run vol={garch.long_run_vol * 100:.3f}%/day")
    tl = bt_out["traffic_light"]
    print(f"Backtest: {bt_out['observed_violations']} violations vs "
          f"{bt_out['expected_violations']} expected → Basel zone: {tl['zone'].upper()}")
    for t in bt_out["tests"]:
        print(f"  {t.name:<32s} LR={t.statistic:6.3f}  p={t.p_value:.3f}"
              f"  {'REJECT' if t.reject_95 else 'pass'}")
    print(f"\nStress scenarios (hypothetical):")
    for r in stress["presets"]:
        print(f"  {r.name:<28s} P&L {r.total_pnl * 100:7.2f}%")
    ww = stress["worst_windows"]
    print(f"Worst {args.stress_horizon}-day historical window: "
          f"{ww['cum_return'].iloc[0] * 100:.2f}% "
          f"({ww['start'].iloc[0].date()} → {ww['end'].iloc[0].date()})")

    report = build_report(
        summary, decomp,
        garch_info={"omega": garch.omega, "alpha": garch.alpha, "beta": garch.beta,
                    "persistence": garch.persistence,
                    "long_run_vol": garch.long_run_vol,
                    "current_vol": float(garch.cond_vol.iloc[-1])},
        backtest_info={**bt_out, "window": args.window, "method": args.method,
                       "n_out": len(bt_out["backtest"])},
        portfolio_desc=f"{', '.join(prices.columns)} @ {np.round(w, 3).tolist()}",
        stress_info={**stress, "horizon": args.stress_horizon},
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "market_risk_report.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {path}")
    return 0


def main() -> None:  # console-script entry point
    sys.exit(run())
