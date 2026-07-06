"""
Market Risk Engine — interactive dashboard.

    streamlit run app.py

Needs the `app` extras: pip install -e ".[app]"   (streamlit + plotly)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from marketrisk import (backtest_report, component_var, ewma_volatility,
                        fit_garch, log_returns, portfolio_returns,
                        stress_report, synthetic_prices, var_summary)
from marketrisk.data import load_csv

st.set_page_config(page_title="Market Risk Engine", layout="wide")
st.title("Market Risk Engine — VaR / ES, GARCH, Backtesting")

# ---------------- Sidebar: data + parameters ----------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Price source", ["Synthetic (bundled)", "Upload CSV"])
    if source == "Upload CSV":
        up = st.file_uploader("Wide CSV: date index + one column per asset", type="csv")
        if up is None:
            st.stop()
        prices = load_csv(up)
    else:
        seed = st.number_input("Random seed", 1, 9999, 42)
        prices = synthetic_prices(seed=int(seed))

    st.header("Parameters")
    alpha = st.select_slider("Confidence level", [0.90, 0.95, 0.975, 0.99], value=0.99)
    window = st.slider("Backtest window (days)", 100, 500, 250, step=50)
    method = st.selectbox("Backtest VaR method", ["historical", "parametric"])

    st.header("Weights")
    raw_w = [st.slider(c, 0.0, 1.0, round(1.0 / prices.shape[1], 2), 0.05)
             for c in prices.columns]
    w = np.array(raw_w)
    if w.sum() == 0:
        st.error("All weights are zero.")
        st.stop()
    w = w / w.sum()
    st.caption(f"Normalised: {np.round(w, 3).tolist()}")


@st.cache_data(show_spinner=False)
def _analyse(prices_key: pd.DataFrame, w_key: tuple, alpha_key: float,
             window_key: int, method_key: str):
    R = log_returns(prices_key)
    wv = np.array(w_key)
    port = portfolio_returns(R, wv)
    summary = var_summary(R.values, wv, alpha=alpha_key)
    decomp = component_var(wv, np.cov(R.values, rowvar=False), alpha_key,
                           names=list(prices_key.columns))
    garch = fit_garch(port)
    ewma = ewma_volatility(port)
    bt = backtest_report(port, window=window_key, alpha=alpha_key, method=method_key)
    stress = stress_report(R, wv)
    return R, port, summary, decomp, garch, ewma, bt, stress


R, port, summary, decomp, garch, ewma, bt, stress = _analyse(
    prices, tuple(w), alpha, window, method)
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["VaR / ES", "Volatility", "Backtest", "Decomposition", "Stress"])

# ---------------- Tab 1: VaR / ES ----------------
with tab1:
    d = summary.as_dict()
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Historical VaR ({int(alpha*100)}%)", f"{d['Historical VaR']*100:.3f}%")
    c2.metric("Historical ES", f"{d['Historical ES']*100:.3f}%")
    c3.metric("Cornish-Fisher VaR", f"{d['Cornish-Fisher VaR']*100:.3f}%")
    fig = go.Figure(go.Bar(x=list(d.keys()), y=[v * 100 for v in d.values()],
                           marker_color=["#1a3d6d" if "VaR" in k else "#7a9cc6"
                                         for k in d]))
    fig.update_layout(yaxis_title="% of portfolio value", height=380,
                      margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Divergence between Historical and Normal VaR signals fat tails; "
               "Cornish-Fisher quantifies the skew/kurtosis correction.")

# ---------------- Tab 2: Volatility ----------------
with tab2:
    fig = go.Figure()
    fig.add_scatter(x=ewma.index, y=ewma * 100, name="EWMA (λ=0.94)",
                    line=dict(width=1.2))
    fig.add_scatter(x=garch.cond_vol.index, y=garch.cond_vol * 100,
                    name="GARCH(1,1)", line=dict(width=1.2))
    fig.update_layout(yaxis_title="daily vol (%)", height=380,
                      margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("α (ARCH)", f"{garch.alpha:.3f}")
    c2.metric("β (GARCH)", f"{garch.beta:.3f}")
    c3.metric("Persistence", f"{garch.persistence:.3f}")
    c4.metric("Long-run vol", f"{garch.long_run_vol*100:.3f}%")
    horizon = st.slider("Forecast horizon (days)", 5, 60, 20)
    fc = garch.forecast(horizon)
    fig2 = go.Figure(go.Scatter(y=fc * 100, mode="lines+markers",
                                name="σ forecast"))
    fig2.add_hline(y=garch.long_run_vol * 100, line_dash="dot",
                   annotation_text="long-run")
    fig2.update_layout(yaxis_title="daily vol (%)", xaxis_title="days ahead",
                       height=300, margin=dict(t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Tab 3: Backtest ----------------
with tab3:
    df = bt["backtest"]
    tl = bt["traffic_light"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed violations", bt["observed_violations"])
    c2.metric("Expected", bt["expected_violations"])
    c3.metric("Basel zone (250d)", tl["zone"].upper())
    fig = go.Figure()
    fig.add_scatter(x=df.index, y=df["return"] * 100, name="daily return",
                    mode="lines", line=dict(width=0.8, color="#888"))
    fig.add_scatter(x=df.index, y=-df["var"] * 100, name=f"-VaR ({method})",
                    mode="lines", line=dict(width=1.2, color="#1a3d6d"))
    viol = df[df["violation"]]
    fig.add_scatter(x=viol.index, y=viol["return"] * 100, name="violation",
                    mode="markers", marker=dict(color="crimson", size=7))
    fig.update_layout(yaxis_title="%", height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.table(pd.DataFrame([{
        "Test": t.name, "LR": round(t.statistic, 3),
        "p-value": round(t.p_value, 3),
        "Reject @5%": "YES" if t.reject_95 else "no",
    } for t in bt["tests"]]))

# ---------------- Tab 4: Decomposition ----------------
with tab4:
    fig = go.Figure(go.Bar(x=decomp.index,
                           y=decomp["component_var"] * 100,
                           marker_color="#1a3d6d"))
    fig.update_layout(yaxis_title=f"component VaR (% of value, {int(alpha*100)}%)",
                      height=380, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    show = decomp.copy()
    for col in ("marginal_var", "component_var"):
        show[col] = (show[col] * 100).round(4).astype(str) + "%"
    show["pct_of_total"] = (decomp["pct_of_total"] * 100).round(1).astype(str) + "%"
    st.table(show)
    st.caption(f"Euler allocation: components sum exactly to total VaR "
               f"({decomp.attrs['total_var']*100:.3f}%), netting diversification.")

# ---------------- Tab 5: Stress ----------------
with tab5:
    st.subheader("Hypothetical scenarios")
    pres = stress["presets"]
    fig = go.Figure(go.Bar(x=[r.name for r in pres],
                           y=[r.total_pnl * 100 for r in pres],
                           marker_color=["crimson" if r.total_pnl < 0 else "#1a3d6d"
                                         for r in pres]))
    fig.update_layout(yaxis_title="portfolio P&L (%)", height=360,
                      margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Worst historical 5-day windows (empirical replay)")
    ww = stress["worst_windows"].copy()
    ww["cum_return"] = (ww["cum_return"] * 100).round(2).astype(str) + "%"
    ww["start"] = ww["start"].dt.date
    ww["end"] = ww["end"].dt.date
    st.table(ww)
    st.caption("VaR asks 'how bad is a normal bad day'; stress asks 'what does "
               "a crisis do to this book'. Hypothetical shocks are first-order "
               "(weight x shock); historical windows are replayed from the data "
               "with auditable dates.")
