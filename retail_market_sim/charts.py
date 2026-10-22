"""Plotly figures for the scenario dashboard.

Every function is pure (DataFrames in, a figure out), so the charts can be tested
without Streamlit. Colours are colour-blind friendly: teal for increases, orange
for decreases, grey for baseline and dark blue for the scenario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INCREASE = "#2a9d8f"
DECREASE = "#e76f51"
BASELINE = "#8d99ae"
SCENARIO = "#264653"
CAUTION = "#e9c46a"
_MAX_LOG_RESPONSE = 50.0


def _style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white", title=dict(text=title, x=0.01), height=height,
        margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", y=-0.15),
    )
    return fig


def _empty_figure(message: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color="#6c757d"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, title, 300)


def cell_labels(frame: pd.DataFrame) -> pd.Series:
    """Readable, mostly unique label for each SKU-retailer cell."""
    return frame["SKU"].astype(str) + " @ " + frame["Retailer"].astype(str)


def price_response_curve(
    price0: float, units0: float, elasticity: float, proposed_price: float,
    observed_min: float | None = None, observed_max: float | None = None,
    title: str = "Price response", span: tuple = (0.6, 1.4), points: int = 81,
) -> go.Figure:
    """Revenue and units against price for one cell, anchored to its observed baseline.

    The shaded band is the historical price range for that cell; the line outside the
    band is an extrapolation of the constant-elasticity model.
    """
    values = np.array([price0, units0, elasticity, proposed_price], dtype=float)
    if not np.isfinite(values).all() or price0 <= 0 or proposed_price <= 0 or units0 < 0:
        raise ValueError("Price response needs finite positive prices, non-negative units and a finite elasticity.")

    def response(ratio):
        return units0 * np.exp(np.clip(elasticity * np.log(ratio), -_MAX_LOG_RESPONSE, _MAX_LOG_RESPONSE))

    low = max(min(span[0], 0.95 * proposed_price / price0), 0.05)
    high = min(max(span[1], 1.05 * proposed_price / price0), 20.0)
    ratios = np.linspace(low, high, points)
    prices = ratios * price0
    units = response(ratios)
    revenue = prices * units
    q0, q1 = float(units0), float(response(proposed_price / price0))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08, subplot_titles=("Revenue", "Units"))
    fig.add_trace(go.Scatter(x=prices, y=revenue, mode="lines", name="Modelled response",
                             line=dict(color=SCENARIO, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=prices, y=units, mode="lines", name="Modelled response",
                             line=dict(color=SCENARIO, width=2), showlegend=False), row=2, col=1)
    for name, price, qty, colour in (("Baseline", price0, q0, BASELINE), ("Scenario", proposed_price, q1, DECREASE)):
        fig.add_trace(go.Scatter(x=[price], y=[price * qty], mode="markers", name=name,
                                 marker=dict(size=12, color=colour, line=dict(width=1, color="white"))), row=1, col=1)
        fig.add_trace(go.Scatter(x=[price], y=[qty], mode="markers", name=name, showlegend=False,
                                 marker=dict(size=12, color=colour, line=dict(width=1, color="white"))), row=2, col=1)

    if observed_min is not None and observed_max is not None and np.isfinite([observed_min, observed_max]).all():
        for row in (1, 2):
            fig.add_vrect(x0=min(observed_min, observed_max), x1=max(observed_min, observed_max),
                          fillcolor=INCREASE, opacity=0.12, line_width=0, row=row, col=1,
                          annotation_text="Observed price range" if row == 1 else None,
                          annotation_position="top left")
    else:
        fig.add_annotation(text="No historical price range available", xref="paper", yref="paper",
                           x=0.99, y=1.0, showarrow=False, font=dict(size=11, color="#6c757d"))
    fig.update_xaxes(title_text="Price", row=2, col=1)
    fig.update_yaxes(title_text="Revenue", row=1, col=1)
    fig.update_yaxes(title_text="Units", row=2, col=1)
    return _style(fig, title, 560)


def revenue_by_group_chart(scenario: pd.DataFrame, group_column: str, title: str | None = None) -> go.Figure:
    """Baseline against scenario revenue for each value of one grouping column."""
    grouped = (scenario.groupby(group_column, observed=True, as_index=False)
               .agg(Baseline=("Baseline_Revenue", "sum"), Scenario=("Simulated_Revenue", "sum"))
               .sort_values("Baseline", ascending=False))
    grouped["Delta"] = grouped["Scenario"] - grouped["Baseline"]
    grouped["Pct"] = np.where(grouped["Baseline"] > 0, grouped["Delta"] / grouped["Baseline"].where(grouped["Baseline"] > 0) * 100, np.nan)
    custom = np.column_stack([grouped["Delta"], grouped["Pct"]])
    hover = "%{x}<br>%{fullData.name}: %{y:,.0f}<br>Change vs baseline: %{customdata[0]:+,.0f} (%{customdata[1]:+.2f}%)<extra></extra>"
    fig = go.Figure([
        go.Bar(x=grouped[group_column].astype(str), y=grouped["Baseline"], name="Baseline",
               marker_color=BASELINE, customdata=custom, hovertemplate=hover),
        go.Bar(x=grouped[group_column].astype(str), y=grouped["Scenario"], name="Scenario",
               marker_color=SCENARIO, customdata=custom, hovertemplate=hover),
    ])
    fig.update_layout(barmode="group", yaxis_title="Revenue")
    return _style(fig, title or f"Revenue by {group_column.lower()}")


def revenue_movers_chart(scenario: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Cells with the largest absolute revenue change, coloured by direction."""
    title = "Biggest revenue movers"
    movers = scenario.loc[scenario["Revenue_Delta"].abs() > 1e-9].copy()
    if movers.empty:
        return _empty_figure("No revenue change yet. Edit a price above to see the biggest movers.", title)
    movers["label"] = cell_labels(movers)
    order = movers["Revenue_Delta"].abs().sort_values(ascending=False).index[:top_n]
    movers = movers.loc[order].iloc[::-1]
    fig = go.Figure(go.Bar(
        x=movers["Revenue_Delta"], y=movers["label"], orientation="h",
        marker_color=[INCREASE if v >= 0 else DECREASE for v in movers["Revenue_Delta"]],
        text=[f"{v:+,.0f}" for v in movers["Revenue_Delta"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Revenue change: %{x:+,.0f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Revenue change vs baseline", showlegend=False)
    return _style(fig, title, max(300, 60 + 40 * len(movers)))


def elasticity_chart(estimates: pd.DataFrame) -> go.Figure:
    """Fitted own-price elasticity per SKU; weak price support is highlighted."""
    est = estimates.sort_values("Elasticity_MAP", ascending=False).copy()
    weak = est["Weak_Price_Support"].astype(bool) if "Weak_Price_Support" in est.columns else pd.Series(False, index=est.index)
    fig = go.Figure()
    for flag, name, colour in ((False, "Adequate price variation", SCENARIO), (True, "Weak price support", CAUTION)):
        part = est.loc[weak == flag]
        if part.empty:
            continue
        fig.add_trace(go.Bar(
            x=part["Elasticity_MAP"], y=part["SKU"].astype(str), orientation="h", name=name,
            marker_color=colour, customdata=part["Category"].astype(str) if "Category" in part.columns else None,
            hovertemplate="%{y}<br>Elasticity: %{x:.2f}<extra></extra>",
        ))
    fig.add_vline(x=-1, line_dash="dash", line_color="#6c757d", annotation_text="Unit elasticity",
                  annotation_position="top")
    fig.update_layout(xaxis_title="Own-price elasticity (more negative = more price-sensitive)")
    return _style(fig, "Fitted price sensitivity by SKU", max(320, 80 + 28 * len(est)))


def share_change_chart(shares: pd.DataFrame, label_columns: list) -> go.Figure:
    """Revenue-share change in percentage points, coloured by direction."""
    title = "Value share change (percentage points)"
    frame = shares.dropna(subset=["Share_Change_pp"]).copy()
    if frame.empty or (frame["Share_Change_pp"].abs() < 1e-9).all():
        return _empty_figure("No share change yet. Edit a price above to see share shifts.", title)
    frame["label"] = frame[label_columns].astype(str).agg(" / ".join, axis=1)
    frame = frame.sort_values("Share_Change_pp")
    fig = go.Figure(go.Bar(
        x=frame["Share_Change_pp"], y=frame["label"], orientation="h",
        marker_color=[INCREASE if v >= 0 else DECREASE for v in frame["Share_Change_pp"]],
        text=[f"{v:+.2f}" for v in frame["Share_Change_pp"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Share change: %{x:+.2f} pp<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Percentage points", showlegend=False)
    return _style(fig, title, max(300, 60 + 34 * len(frame)))


def revenue_bridge_chart(baseline_revenue: float, effects) -> go.Figure:
    """Sequential Volume -> Mix -> Price bridge from baseline to scenario revenue."""
    volume, mix, price = (float(effects[k]) for k in ("Volume_Effect", "Mix_Effect", "Price_Effect"))
    total = baseline_revenue + volume + mix + price
    levels = [baseline_revenue, baseline_revenue + volume, baseline_revenue + volume + mix, total]
    spread = max(levels) - min(levels)
    pad = max(spread, 0.02 * abs(baseline_revenue), 1.0)
    fig = go.Figure(go.Waterfall(
        x=["Baseline", "Volume", "Mix", "Price", "Scenario"],
        measure=["absolute", "relative", "relative", "relative", "total"],
        y=[baseline_revenue, volume, mix, price, total],
        text=[f"{baseline_revenue:,.0f}", f"{volume:+,.0f}", f"{mix:+,.0f}", f"{price:+,.0f}", f"{total:,.0f}"],
        textposition="outside",
        increasing=dict(marker=dict(color=INCREASE)), decreasing=dict(marker=dict(color=DECREASE)),
        totals=dict(marker=dict(color=SCENARIO)), connector=dict(line=dict(color="#adb5bd")),
    ))
    fig.update_yaxes(range=[min(levels) - pad, max(levels) + pad], title_text="Revenue (axis truncated)")
    fig.update_layout(showlegend=False)
    return _style(fig, "Revenue bridge: baseline to scenario")
