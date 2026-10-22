"""
app.py — Bayesian Retail Price Simulation Dashboard

Run:
    streamlit run app.py

Place sample_data.csv in the same directory as this file.
"""

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from retail_market_sim import (
    load_data,
    fit_model_map,
    ModelConfig,
    simulate_scenario,
    three_factor_decomposition,
    value_shares,
    validate_baseline,
    get_data_hash,
)

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="Bayesian Retail Price Simulator",
    page_icon="🛒",
    layout="wide",
)

DATA_PATH = Path(__file__).resolve().parent / "sample_data.csv"
MODEL_VERSION = "retail-map-v1"
RANDOM_SEED = 42


# =============================================================================
# Application
# =============================================================================

def main() -> None:
    st.title("Bayesian Retail Price Simulation")
    st.caption(
        "Edit latest-month SKU–retailer prices to simulate own-price demand "
        "responses across the observed market."
    )

    if not DATA_PATH.exists():
        st.error("Place sample_data.csv in the same directory as app.py.")
        st.stop()

    try:
        data, data_notes = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"Unable to load POS data: {exc}")
        st.stop()

    latest_period = int(data["Period"].max())
    latest_label = data.loc[data["Period"].eq(latest_period), "Month"].iloc[0]

    # Latest GLOBAL month defines the market snapshot
    baseline = data.loc[data["Period"].eq(latest_period)].copy().reset_index(drop=True)

    try:
        validate_baseline(baseline)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.header("Bayesian model")
    st.sidebar.caption("MAP estimation is cached. Editing prices does not refit the model.")

    prior_elasticity = st.sidebar.number_input(
        "Prior mean own-price elasticity",
        min_value=-5.0,
        max_value=-0.1,
        value=-1.5,
        step=0.1,
        help=(
            "A regularizing prior, not a fixed scenario elasticity. "
            "Sparse SKU estimates may remain close to this prior."
        ),
    )
    max_evaluations = st.sidebar.selectbox(
        "Maximum MAP evaluations",
        options=[3000, 6000, 12000],
        index=1,
    )
    st.sidebar.info(
        "First fit may take time for PyMC compilation. "
        "The dashboard uses MAP point estimates; it does not display "
        "posterior credible intervals."
    )

    config = ModelConfig(prior_elasticity=float(prior_elasticity), max_evaluations=int(max_evaluations))

    with st.spinner("Compiling and fitting the cached PyMC demand model…"):
        try:
            fitted = fit_model_map(data, config)
        except Exception as exc:
            st.error(f"Bayesian fitting failed: {exc}")
            st.info("Check the data and PyMC installation.")
            st.stop()

    estimates = fitted["estimates"]

    if not fitted["converged"]:
        st.warning(
            "MAP optimization did not report convergence. Results are provisional. "
            f"Optimizer: {fitted['message']}. Try increasing the evaluation limit."
        )

    if fitted["positive_elasticities"]:
        st.warning(
            f"{fitted['positive_elasticities']} SKU MAP elasticities are positive. "
            "These are retained transparently, not forced negative. "
            "Inspect confounding and price variation before using their scenarios."
        )

    if fitted["weak_support_count"]:
        st.caption(
            f"{fitted['weak_support_count']} SKU estimates have limited "
            "within-retailer price variation or few observations and may "
            "be strongly influenced by the prior."
        )

    # -------------------------------------------------------------------------
    # Editable wide price matrix
    # -------------------------------------------------------------------------
    st.subheader(f"Price matrix — {latest_label}")
    st.caption(
        "Rows identify products; columns identify retailers. "
        "Blank cells indicate no observation in the latest month and cannot "
        "create new listings. All reported market totals use this full snapshot."
    )

    retailers = sorted(baseline["Retailer"].unique().tolist())
    retailer_to_column = {r: f"Price | {r}" for r in retailers}
    column_to_retailer = {c: r for r, c in retailer_to_column.items()}

    matrix_source = baseline.assign(
        _PriceColumn=baseline["Retailer"].map(retailer_to_column)
    )
    price_matrix = (
        matrix_source.pivot(
            index=["Category", "Brand", "SKU"],
            columns="_PriceColumn",
            values="Price",
        )
        .reindex(columns=list(column_to_retailer))
        .reset_index()
    )
    price_matrix.columns.name = None

    data_signature = get_data_hash(data)

    reset_key = f"matrix_reset_{data_signature}"
    if reset_key not in st.session_state:
        st.session_state[reset_key] = 0

    if st.button("Reset all prices to baseline"):
        st.session_state[reset_key] += 1

    editor_key = f"price_editor_{data_signature}_{st.session_state[reset_key]}"

    editor_config = {
        column: st.column_config.NumberColumn(
            label=retailer,
            min_value=0.000001,
            format="%.6f",
            help="Enter a positive price per sold pack/unit.",
        )
        for column, retailer in column_to_retailer.items()
    }

    edited_matrix = st.data_editor(
        price_matrix,
        key=editor_key,
        disabled=["Category", "Brand", "SKU"],
        column_config=editor_config,
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
    )

    # Edits to absent cells are ignored by the baseline left join
    absent_cells = price_matrix[list(column_to_retailer)].isna()
    filled_absent_cells = edited_matrix[list(column_to_retailer)].notna() & absent_cells
    if filled_absent_cells.to_numpy().any():
        st.warning(
            "Prices entered in originally blank cells are ignored. "
            "New listings require a separate baseline demand assumption."
        )

    try:
        scenario = simulate_scenario(baseline, edited_matrix, column_to_retailer, estimates)
    except Exception as exc:
        st.error(f"Invalid scenario: {exc}")
        st.stop()

    changed = ~np.isclose(
        scenario["Simulated_Price"],
        scenario["Baseline_Price"],
        rtol=1e-9,
        atol=1e-10,
    )
    st.caption(
        f"{int(changed.sum()):,} of {len(scenario):,} "
        "latest-month SKU–retailer prices changed."
    )

    if scenario["Price_Change_Pct"].abs().gt(50).any():
        st.warning(
            "Some price changes exceed ±50%. Constant-elasticity extrapolation "
            "this far from baseline may be unreliable."
        )

    # -------------------------------------------------------------------------
    # Market-wide KPIs
    # -------------------------------------------------------------------------
    r0 = float(scenario["Baseline_Revenue"].sum())
    r1 = float(scenario["Simulated_Revenue"].sum())
    delta = r1 - r0
    delta_pct = delta / r0 * 100
    q0 = float(scenario["Baseline_Unit_Sales"].sum())
    q1 = float(scenario["Simulated_Unit_Sales"].sum())

    st.subheader("Market-wide impact")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline revenue", f"{r0:,.2f}")
    k2.metric(
        "Simulated revenue",
        f"{r1:,.2f}",
        delta=f"{delta:+,.2f} ({delta_pct:+.2f}%)",
    )
    k3.metric("Absolute revenue change", f"{delta:+,.2f}")
    k4.metric("Revenue change (%)", f"{delta_pct:+.2f}%")

    u1, u2 = st.columns(2)
    u1.metric("Baseline unit sales", f"{q0:,.0f}")
    u2.metric(
        "Simulated unit sales",
        f"{q1:,.1f}",
        delta=f"{q1 - q0:+,.1f}",
    )
    st.caption(
        "Revenue is shown in the dataset's currency units. Unit sales count "
        "sold packs/items, not a combined kg/L measure. "
        "Simulated quantities are expectations and need not be integers."
    )

    # -------------------------------------------------------------------------
    # Three-factor decomposition (Volume -> Mix -> Price)
    # -------------------------------------------------------------------------
    market_decomp = three_factor_decomposition(scenario, [])
    ve = float(market_decomp["Volume_Effect"].iloc[0])
    me = float(market_decomp["Mix_Effect"].iloc[0])
    pe = float(market_decomp["Price_Effect"].iloc[0])
    recon_error = float(market_decomp["Revenue_Delta"].iloc[0] - market_decomp["Decomposition_Sum"].iloc[0])

    waterfall = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=[
                "Baseline revenue",
                "Volume effect",
                "Mix effect",
                "Price effect",
                "Simulated revenue",
            ],
            y=[r0, ve, me, pe, 0],
            text=[
                f"{r0:,.2f}",
                f"{ve:+,.2f}",
                f"{me:+,.2f}",
                f"{pe:+,.2f}",
                f"{r1:,.2f}",
            ],
            textposition="outside",
            connector={"line": {"color": "#94A3B8"}},
            increasing={"marker": {"color": "#16A34A"}},
            decreasing={"marker": {"color": "#DC2626"}},
            totals={"marker": {"color": "#2563EB"}},
        )
    )
    waterfall.update_layout(
        title="Revenue bridge (Volume → Mix → Price) using equivalized volumes",
        yaxis_title="Revenue",
        showlegend=False,
        margin=dict(t=70, b=40),
    )
    st.plotly_chart(waterfall, use_container_width=True)
    st.caption(
        f"Volume: {ve:+,.2f} | Mix: {me:+,.2f} | Price: {pe:+,.2f} | "
        f"Reconciliation residual: {recon_error:.8f}"
    )

    # Two-factor cell-level bridge for reference
    with st.expander("Two-factor cell-level bridge (exact for matched SKU–retailer cells)"):
        cell_pe = float(scenario["Price_Effect"].sum())
        cell_ve = float(scenario["Volume_Effect"].sum())
        st.write(
            f"Price effect (Σ(P₁−P₀)Q₁): {cell_pe:+,.2f} | "
            f"Volume effect (Σ(Q₁−Q₀)P₀): {cell_ve:+,.2f} | "
            f"Sum: {cell_pe + cell_ve:+,.2f} | ΔRevenue: {delta:+,.2f}"
        )

    # -------------------------------------------------------------------------
    # Decomposition by segment
    # -------------------------------------------------------------------------
    st.subheader("Decomposition by segment")
    dec_tab_cat, dec_tab_ret, dec_tab_brand = st.tabs(["By Category", "By Retailer", "By Brand"])

    cat_decomp = three_factor_decomposition(scenario, ["Category"])
    ret_decomp = three_factor_decomposition(scenario, ["Retailer"])
    brand_decomp = three_factor_decomposition(scenario, ["Category", "Brand"])
    brand_decomp["Portfolio"] = brand_decomp["Category"] + " / " + brand_decomp["Brand"]

    segments = [
        (dec_tab_cat, "Category", cat_decomp, "Category"),
        (dec_tab_ret, "Retailer", ret_decomp, "Retailer"),
        (dec_tab_brand, "Brand", brand_decomp, "Portfolio"),
    ]

    for tab, label, df, key_col in segments:
        with tab:
            st.dataframe(
                df[[key_col, "R0", "R1", "Revenue_Delta", "Volume_Effect", "Mix_Effect", "Price_Effect", "Decomposition_Sum"]],
                hide_index=True,
                use_container_width=True,
            )
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Volume", x=df[key_col], y=df["Volume_Effect"], marker_color="#64748B"))
            fig.add_trace(go.Bar(name="Mix", x=df[key_col], y=df["Mix_Effect"], marker_color="#F59E0B"))
            fig.add_trace(go.Bar(name="Price", x=df[key_col], y=df["Price_Effect"], marker_color="#2563EB"))
            fig.update_layout(barmode="relative", title=f"{label} decomposition", yaxis_title="Revenue effect", margin=dict(t=50, b=50))
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # Market shares
    # -------------------------------------------------------------------------
    st.subheader("Value-share shifts")
    st.caption(
        "Overall retailer and category shares use total observed market revenue. "
        "Overall brand shares identify each Category/Brand portfolio separately. "
        "Conditional views recalculate shares within the selected category."
    )

    retailer_shares = value_shares(scenario, ["Retailer"])
    category_shares = value_shares(scenario, ["Category"])
    brand_shares = value_shares(scenario, ["Category", "Brand"])
    brand_shares["Portfolio"] = brand_shares["Category"] + " / " + brand_shares["Brand"]

    tab_retailer, tab_brand, tab_category, tab_conditional = st.tabs(
        ["Retailers", "Brands", "Categories", "Within-category shares"]
    )

    with tab_retailer:
        st.plotly_chart(
            share_chart(retailer_shares, "Retailer", "Retailer shares of total market revenue"),
            use_container_width=True,
        )
        st.dataframe(retailer_shares, hide_index=True, use_container_width=True)

    with tab_brand:
        st.plotly_chart(
            share_chart(brand_shares, "Portfolio", "Category-brand portfolios: share of total market revenue"),
            use_container_width=True,
        )
        st.dataframe(brand_shares, hide_index=True, use_container_width=True)

    with tab_category:
        st.plotly_chart(
            share_chart(category_shares, "Category", "Category shares of total market revenue"),
            use_container_width=True,
        )
        st.dataframe(category_shares, hide_index=True, use_container_width=True)

    with tab_conditional:
        selected_category = st.selectbox(
            "Category for conditional share analysis",
            sorted(scenario["Category"].unique()),
        )

        retailer_category = value_shares(
            scenario, ["Category", "Retailer"], parent_columns=["Category"]
        )
        brand_category = value_shares(
            scenario, ["Category", "Brand"], parent_columns=["Category"]
        )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                share_chart(
                    retailer_category.loc[retailer_category["Category"].eq(selected_category)],
                    "Retailer",
                    "Retailer shares within category",
                ),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                share_chart(
                    brand_category.loc[brand_category["Category"].eq(selected_category)],
                    "Brand",
                    "Brand shares within category, all retailers",
                ),
                use_container_width=True,
            )

        selected_retailer = st.selectbox(
            "Retailer for brand-within-retailer shares",
            sorted(scenario["Retailer"].unique()),
        )
        brand_retailer_category = value_shares(
            scenario,
            ["Category", "Retailer", "Brand"],
            parent_columns=["Category", "Retailer"],
        )
        conditional_table = brand_retailer_category.loc[
            brand_retailer_category["Category"].eq(selected_category)
            & brand_retailer_category["Retailer"].eq(selected_retailer)
        ]
        st.dataframe(conditional_table, hide_index=True, use_container_width=True)
        st.caption("Share_Change_pp is measured in percentage points.")

    # -------------------------------------------------------------------------
    # Scenario detail and downloads
    # -------------------------------------------------------------------------
    st.subheader("SKU–retailer simulation details")

    display_columns = [
        "Month", "Category", "Retailer", "Brand", "SKU", "Pack_Size",
        "Elasticity_MAP", "Baseline_Price", "Simulated_Price", "Price_Change_Pct",
        "Baseline_Unit_Sales", "Simulated_Unit_Sales",
        "Baseline_Revenue", "Simulated_Revenue", "Revenue_Delta",
        "Price_Effect", "Volume_Effect",
    ]

    st.dataframe(scenario[display_columns], hide_index=True, use_container_width=True)
    st.download_button(
        "Download scenario CSV",
        data=scenario[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="retail_price_scenario.csv",
        mime="text/csv",
    )

    # -------------------------------------------------------------------------
    # Diagnostics and transparent model specification
    # -------------------------------------------------------------------------
    with st.expander("Estimated SKU elasticities and fit diagnostics"):
        st.write({
            "Fit method": "PyMC maximum a posteriori (MAP)",
            "Converged": fitted["converged"],
            "Optimizer message": fitted["message"],
            "Optimizer iterations": fitted["iterations"],
            "Positive-unit training observations": fitted["training_rows"],
            "Observed periods": int(data["Period"].nunique()),
            "Latest-month cells": len(baseline),
        })
        st.dataframe(estimates, hide_index=True, use_container_width=True)
        st.caption(
            "Weak_Price_Support flags fewer than 12 positive observations "
            "or within-pair log-price standard deviation below 0.02. "
            "These are screening thresholds, not statistical significance tests."
        )

        settings = {
            "model_version": MODEL_VERSION,
            "prior_elasticity": float(prior_elasticity),
            "maximum_MAP_evaluations": int(max_evaluations),
            "baseline_period": latest_period,
            "baseline_label": str(latest_label),
            "data_signature": data_signature,
            "uncertainty_intervals": False,
        }
        st.download_button(
            "Download model settings",
            data=json.dumps(settings, indent=2).encode("utf-8"),
            file_name="retail_model_settings.json",
            mime="application/json",
        )

    with st.expander("Data checks"):
        if data_notes:
            for note in data_notes:
                st.write(f"- {note}")

        historical_pairs = data[["Category", "Retailer", "Brand", "SKU"]].drop_duplicates()
        absent_latest = len(historical_pairs) - len(baseline)
        st.write(
            f"{absent_latest:,} historically observed SKU–retailer identities "
            "are absent from the latest snapshot. They are not backfilled."
        )
        st.write(
            "Pack_Size is preserved as metadata. Without an explicit, "
            "validated measurement dimension, kg, litres, and pieces are "
            "not summed into a single physical-volume KPI."
        )

    with st.expander("Economic model, accounting, and limitations"):
        st.latex(
            r"\ln Q_i = \alpha_{s(i)} + \mu_{r(i)} + a_{s(i),r(i)}"
            r" + \beta_{s(i)}\left(\ln P_i"
            r" - \overline{\ln P}_{s(i),r(i)}\right)"
            r" + \lambda_{c(i),m(i)} + \tau_{c(i)}T_i + u_i"
        )
        st.latex(
            r"\beta_s \mid \bar{\beta}_{c(s)}"
            r"\sim \mathcal{N}\!\left(\bar{\beta}_{c(s)},\,0.5^2\right)"
        )
        st.latex(
            r"Q_1 = Q_0\exp\!\left("
            r"\widehat{\beta}_{s,\mathrm{MAP}}\ln(P_1/P_0)\right),"
            r"\qquad R_1=P_1Q_1"
        )
        st.latex(
            r"\Delta R = \sum_i (P_{1,i}-P_{0,i})Q_{1,i}"
            r" + \sum_i (Q_{1,i}-Q_{0,i})P_{0,i}"
        )

        st.markdown(
            """
            - Retailer, SKU–retailer, seasonal, and trend controls enter model
              estimation. During a same-month scenario, they are held fixed.
            - Seasonal effects are pooled within category for stability with
              only two annual cycles.
            - The calculation is anchored to observed quantities. An unchanged
              price gives unchanged units and revenue, avoiding a baseline
              mismatch caused by model fitting error.
            - MAP is a point estimate under the specified priors. It is not
              posterior sampling and supplies no credible or predictive interval.
            - Price coefficients may be confounded by unobserved promotions,
              stockouts, distribution changes, and demand shocks. They are not
              automatically causal elasticities.
            - Unedited products retain their baseline quantities and revenue.
              Their shares may still change because the denominator changes.
            - No substitution, cannibalization, halo, retailer switching,
              inventory constraints, or competitor reactions are simulated.
              Market-wide totals are sums of independent own-price responses.
            - A zero-unit baseline cell remains at zero units under this
              multiplicative scenario. Launches and latent demand require a
              separate model.
            - Constant-elasticity responses can become unrealistic for large
              price changes. This dashboard is a scenario tool, not a guarantee.
            - The exact SKU-level price and volume effects already sum to the
              revenue change. Adding a residual "mix effect" to these same
              formulas would double-count: that residual is algebraically zero.
            - Aggregate price, when needed, is Revenue / Units: a
              quantity-weighted arithmetic mean, not a revenue-weighted
              arithmetic mean of prices.
            """
        )


def share_chart(table: pd.DataFrame, label_column: str, title: str) -> go.Figure:
    """Side-by-side baseline and simulated value-share bars."""
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Baseline",
            x=table[label_column],
            y=table["Baseline_Share_Pct"],
            marker_color="#64748B",
            hovertemplate="%{x}<br>Value share: %{y:.2f}%<extra>Baseline</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            name="Simulated",
            x=table[label_column],
            y=table["Simulated_Share_Pct"],
            marker_color="#2563EB",
            hovertemplate="%{x}<br>Value share: %{y:.2f}%<extra>Simulated</extra>",
        )
    )
    figure.update_layout(
        title=title,
        barmode="group",
        yaxis_title="Value share (%)",
        xaxis_title="",
        legend_title_text="Scenario",
        margin=dict(t=60, b=50),
    )
    return figure


if __name__ == "__main__":
    main()