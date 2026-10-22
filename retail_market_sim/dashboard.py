"""Streamlit adapter; the analytical package remains UI-independent."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data import CELL_KEYS, PRODUCT_KEYS, get_data_hash, load_data, validate_baseline
from .dashboard_adapter import matrix_to_price_changes, price_support
from .decomposition import three_factor_decomposition, value_shares
from .model import ModelConfig, fit_model_map
from .simulation import simulate_scenario

DATA_PATH = Path(__file__).resolve().parents[1] / "sample_data.csv"
MODEL_VERSION = "retail-map-v2"


@st.cache_data(show_spinner=False, max_entries=4)
def cached_load(path: str, modified_ns: int) -> tuple:
    return load_data(path, modified_ns)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_fit(data: pd.DataFrame, config: ModelConfig, model_version: str) -> dict:
    """Cache copies of fit results; never silently publish a failed optimization."""
    fitted = fit_model_map(data, config)
    if not fitted["converged"]:
        raise RuntimeError(f"MAP optimization failed: {fitted['message']}")
    return fitted


def main() -> None:
    st.set_page_config(page_title="Retail Price Scenarios", page_icon="🛒", layout="wide")
    st.title("Retail price scenario explorer")
    st.caption("Same-period, observed-baseline-anchored responses. Not a forecast or causal guarantee.")
    with st.sidebar.expander("Advanced model settings"):
        prior = st.number_input("Prior mean elasticity", -5.0, -0.1, -1.5, 0.1)
        maxeval = st.selectbox("MAP evaluation limit", [3000, 6000, 12000], index=1)
    config = ModelConfig(prior_elasticity=float(prior), max_evaluations=int(maxeval))
    try:
        data, notes = cached_load(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
        baseline = data.loc[data["Period"].eq(data["Period"].max())].copy()
        validate_baseline(baseline)
        with st.spinner("Loading or fitting the demand model…"):
            fitted = cached_fit(data, config, MODEL_VERSION)
    except Exception as exc:
        st.error(f"Unable to prepare a validated fit: {exc}")
        st.stop()
    st.sidebar.info("Fit results are cached. Price edits do not refit the model. MAP supplies no uncertainty intervals.")
    signature = get_data_hash(data)
    retailers = sorted(baseline["Retailer"].unique())
    columns = {f"Price | {r}": r for r in retailers}
    source = baseline.assign(_column=baseline["Retailer"].map({r: c for c, r in columns.items()}))
    matrix = source.pivot(index=PRODUCT_KEYS, columns="_column", values="Price").reset_index()
    matrix.columns.name = None
    reset_key = f"reset_{signature}"
    st.session_state.setdefault(reset_key, 0)
    st.subheader("Scenario prices")
    st.caption("Prices are per sold pack. Absent listings cannot be created by editing a blank cell.")
    if st.button("Reset prices"):
        st.session_state[reset_key] += 1
    edited = st.data_editor(
        matrix, key=f"editor_{signature}_{st.session_state[reset_key]}",
        disabled=PRODUCT_KEYS, num_rows="fixed", hide_index=True,
        column_config={c: st.column_config.NumberColumn(label=r, min_value=0.000001, format="%.4f") for c, r in columns.items()},
    )
    if (matrix[list(columns)].isna() & edited[list(columns)].notna()).to_numpy().any():
        st.warning("Edits to absent listings are ignored; new listings require a demand assumption.")
    try:
        changes = matrix_to_price_changes(baseline, edited, columns)
        scenario = simulate_scenario(baseline, fitted["estimates"], changes)
        support = price_support(data, scenario)
        decomposition = three_factor_decomposition(scenario, [])
    except Exception as exc:
        st.error(f"Invalid scenario: {exc}")
        st.stop()
    r0 = float(scenario["Baseline_Revenue"].sum())
    r1 = float(scenario["Simulated_Revenue"].sum())
    q0 = float(scenario["Baseline_Unit_Sales"].sum())
    q1 = float(scenario["Simulated_Unit_Sales"].sum())
    changed = scenario["Price_Change_Pct"].abs().gt(1e-7)
    outside = support["Outside_Observed_Range"] & support["Price_Change_Pct"].abs().gt(1e-7)
    a, b, c, d = st.columns(4)
    a.metric("Observed-market revenue change", f"{r1-r0:+,.2f}", f"{100*(r1/r0-1):+.2f}%")
    b.metric("Pack/item quantity change", f"{q1-q0:+,.1f}")
    c.metric("Edited cells", len(changes))
    d.metric("Edited cells outside observed prices", int(outside.sum()))
    st.caption("Revenue uses the dataset's currency. Quantities count packs, not litres. Observed-market totals are not an external market estimate.")
    if changes:
        selected = scenario.loc[changed]
        st.subheader("Edited-cell impact")
        st.dataframe(selected[CELL_KEYS + ["Baseline_Price", "Simulated_Price", "Price_Change_Pct", "Revenue_Delta"]], hide_index=True)
        st.info("These are fitted own-price associations. Substitution, promotional confounding, inventory constraints and competitor reactions are excluded; consider testing, not automatic rollout.")
    if outside.any():
        st.warning("Some proposed prices lie outside their SKU-retailer historical positive-sales price range.")
    if fitted["weak_support_count"]:
        st.warning(f"{fitted['weak_support_count']} SKU estimates fail heuristic support screens; this is not a causal-validity or significance test.")
    if fitted["positive_elasticities"]:
        st.warning("Positive elasticity estimates require investigation before business use.")
    row = decomposition.iloc[0]
    fig = go.Figure(go.Waterfall(
        x=["Baseline", "Volume", "Mix", "Price", "Scenario"],
        measure=["absolute", "relative", "relative", "relative", "total"],
        y=[r0, row["Volume_Effect"], row["Mix_Effect"], row["Price_Effect"], 0],
    ))
    fig.update_layout(title="Sequential Volume → Mix → Price attribution", yaxis_title="Revenue")
    st.plotly_chart(fig)
    st.caption("Accounting allocation, not separate causal mechanisms. Volume is based on numeric litres-per-pack in this beverage dataset.")
    for tab, grouping in zip(st.tabs(["Category", "Retailer", "Brand"]), [["Category"], ["Retailer"], ["Category", "Brand"]]):
        with tab:
            st.dataframe(three_factor_decomposition(scenario, grouping), hide_index=True)
    st.subheader("Value shares")
    category = st.selectbox("Category for conditional shares", sorted(scenario["Category"].unique()))
    total_tab, conditional_tab = st.tabs(["Observed-market shares", "Within-category shares"])
    with total_tab:
        for grouping in [["Retailer"], ["Category"], ["Category", "Brand"]]:
            st.dataframe(value_shares(scenario, grouping), hide_index=True)
    with conditional_tab:
        for grouping in [["Category", "Retailer"], ["Category", "Brand"]]:
            shares = value_shares(scenario, grouping, parent_columns=["Category"])
            st.dataframe(shares.loc[shares["Category"].eq(category)], hide_index=True)
        retailer = st.selectbox("Retailer for brand-within-retailer shares", retailers)
        shares = value_shares(scenario, ["Category", "Retailer", "Brand"], parent_columns=["Category", "Retailer"])
        st.dataframe(shares.loc[shares["Category"].eq(category) & shares["Retailer"].eq(retailer)], hide_index=True)
    st.caption("Unedited products retain demand; their shares can change through the denominator. Share changes are percentage points.")
    with st.expander("Evidence and fit diagnostics"):
        st.json({"method": "MAP", "converged": fitted["converged"], "iterations": fitted["iterations"]})
        st.dataframe(fitted["estimates"], hide_index=True)
        st.dataframe(support, hide_index=True)
        st.write("Prior sensitivity is not established by a support flag; it requires refitting under alternative priors.")
    with st.expander("Data checks"):
        for note in notes:
            st.write(note)
    st.subheader("Scenario details and exports")
    st.dataframe(scenario, hide_index=True)
    settings = {
        "model_version": MODEL_VERSION, "data_hash": signature,
        "config": vars(config), "inference_method": "MAP",
        "baseline_period": int(baseline["Period"].max()),
        "uncertainty_intervals": False, "price_changes": [{"cell": list(k), "price": v} for k, v in changes.items()],
    }
    st.download_button("Download scenario CSV", scenario.to_csv(index=False), "scenario.csv", "text/csv")
    st.download_button("Download model settings", json.dumps(settings, indent=2), "model_settings.json", "application/json")
