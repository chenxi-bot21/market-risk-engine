"""marketrisk — multi-asset VaR/ES engine with GARCH volatility and backtesting."""
from .backtest import (basel_traffic_light, backtest_report, christoffersen,
                       kupiec_pof, rolling_var_backtest)
from .data import resolve_prices, synthetic_prices
from .decompose import component_var
from .returns import log_returns, portfolio_returns
from .scenarios import (apply_scenario, historical_worst_windows,
                        run_preset_scenarios, stress_report)
from .var import (cornish_fisher_var, historical_es, historical_var,
                  monte_carlo_var_es, parametric_es, parametric_var, var_summary)
from .volatility import ewma_volatility, fit_garch

__version__ = "1.0.0"
__all__ = [
    "basel_traffic_light", "backtest_report", "christoffersen", "kupiec_pof",
    "rolling_var_backtest", "resolve_prices", "synthetic_prices", "component_var",
    "log_returns", "portfolio_returns", "cornish_fisher_var", "historical_es",
    "historical_var", "monte_carlo_var_es", "parametric_es", "parametric_var",
    "var_summary", "ewma_volatility", "fit_garch", "apply_scenario",
    "historical_worst_windows", "run_preset_scenarios", "stress_report",
]
