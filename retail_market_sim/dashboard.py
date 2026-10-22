"""Streamlit adapter; the analytical package remains UI-independent."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from . import charts
from .dashboard_adapter import matrix_to_price_changes, price_support
from .data import CELL_KEYS, PRODUCT_KEYS, get_data_hash, load_data, validate_baseline
from .decomposition import three_factor_decomposition, value_shares
from .model import ModelConfig, fit_model_map
from .simulation import simulate_scenario

DATA_PATH = Path(__file__).resolve().parents[1] / "sample_data.csv"
MODEL_VERSION = "retail-map-v2"

SHARE_VIEWS = {
    "Retailer (share of total market)": (["Retailer"], None),
    "Category (share of total market)": (["Category"], None),
    "Brand (share within category)": (["Category", "Brand"], ["Category"]),
    "Retailer (share within category)": (["Category", "Retailer"], ["Category"]),
}


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


def _optional_float(value):
    return None if pd.isna(value) else float(value)


def _price_response_tab(scenario: pd.DataFrame, support: pd.DataFrame, changed: pd.Series) -> None:
    labels = charts.cell_labels(scenario).tolist()
    default = int(changed.to_numpy().argmax()) if changed.any() else int(scenario["Baseline_Revenue"].to_numpy().argmax())
    position = st.selectbox("SKU-retailer cell", range(len(labels)), index=default, format_func=lambda i: labels[i])
    row, history = scenario.iloc[position], support.iloc[position]
    st.plotly_chart(charts.price_response_curve(
        price0=float(row["Baseline_Price"]), units0=float(row["Baseline_Unit_Sales"]),
        elasticity=float(row["Elasticity_MAP"]), proposed_price=float(row["Simulated_Price"]),
        observed_min=_optional_float(history["Historical_Min_Price"]),
        observed_max=_optional_float(history["Historical_Max_Price"]), title=labels[position]))
    st.caption("Constant-elasticity response anchored to the observed baseline month. "
               "The shaded band is the cell's historical price range; the line outside it is extrapolation.")


def _value_share_tab(scenario: pd.DataFrame) -> None:
    view = st.radio("Show", list(SHARE_VIEWS), horizontal=True)
    group, parent = SHARE_VIEWS[view]
    st.plotly_chart(charts.share_change_chart(value_shares(scenario, group, parent), group))
    st.caption("Unedited products keep their baseline demand, so share changes mostly reflect the "
               "denominator moving. Substitution is not modelled.")


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
        st.info("These are fitted own-price associations. Substitution, promotional confounding, inventory constraints and competitor reactions are excluded; consider testing, not automatic rollout.")
    if outside.any():
        st.warning("Some proposed prices lie outside their SKU-retailer historical positive-sales price range.")
    if fitted["weak_support_count"]:
        st.warning(f"{fitted['weak_support_count']} SKU estimates fail heuristic support screens; this is not a causal-validity or significance test.")
    if fitted["positive_elasticities"]:
        st.warning("Positive elasticity estimates require investigation before business use.")

    st.subheader("Impact")
    price_tab, group_tab, mover_tab, bridge_tab, share_tab, elasticity_tab = st.tabs(
        ["Price response", "Revenue by group", "Biggest movers", "Revenue bridge", "Value share", "Elasticities"])
    with price_tab:
        _price_response_tab(scenario, support, changed)
    with group_tab:
        group = st.radio("Group by", ["Category", "Retailer", "Brand"], horizontal=True)
        st.plotly_chart(charts.revenue_by_group_chart(scenario, group))
    with mover_tab:
        st.plotly_chart(charts.revenue_movers_chart(scenario))
    with bridge_tab:
        st.plotly_chart(charts.revenue_bridge_chart(r0, decomposition.iloc[0]))
        st.caption("Accounting allocation, not separate causal mechanisms. Volume is based on numeric litres per pack.")
    with share_tab:
        _value_share_tab(scenario)
    with elasticity_tab:
        st.plotly_chart(charts.elasticity_chart(fitted["estimates"]))
        st.caption("Weak price support means little within-cell price variation; those estimates lean on the prior.")

    with st.expander("Tables and diagnostics"):
        if changes:
            st.dataframe(scenario.loc[changed, CELL_KEYS + ["Baseline_Price", "Simulated_Price", "Price_Change_Pct", "Revenue_Delta"]], hide_index=True)
        for tab, grouping in zip(st.tabs(["Category", "Retailer", "Brand"]), [["Category"], ["Retailer"], ["Category", "Brand"]]):
            with tab:
                st.dataframe(three_factor_decomposition(scenario, grouping), hide_index=True)
        st.json({"method": "MAP", "converged": fitted["converged"], "iterations": fitted["iterations"]})
        st.dataframe(fitted["estimates"], hide_index=True)
        st.dataframe(support, hide_index=True)
        for note in notes:
            st.write(note)
    settings = {
        "model_version": MODEL_VERSION, "data_hash": signature,
        "config": vars(config), "inference_method": "MAP",
        "baseline_period": int(baseline["Period"].max()),
        "uncertainty_intervals": False, "price_changes": [{"cell": list(k), "price": v} for k, v in changes.items()],
    }
    st.download_button("Download scenario CSV", scenario.to_csv(index=False), "scenario.csv", "text/csv")
    st.download_button("Download model settings", json.dumps(settings, indent=2), "model_settings.json", "application/json")
