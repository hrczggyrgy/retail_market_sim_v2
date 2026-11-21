"""Streamlit adapter; the analytical package remains UI-independent."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data import CELL_KEYS, PRODUCT_KEYS, get_data_hash, load_data, validate_baseline
from .dashboard_adapter import matrix_to_price_changes, price_support
from .decomposition import three_factor_decomposition, three_factor_decomposition_report, value_shares
from .model import ModelConfig, fit_model_map
from .simulation import build_named_scenario, simulate_scenario
from .dashboard_cache import get_cached_scenario_bundle, _hash_config

DATA_PATH = Path(__file__).resolve().parents[1] / "sample_data.csv"
MODEL_VERSION = "retail-map-v2"


def _generate_plain_language_explanation(
    target_row: pd.Series,
    r0: float,
    r1: float,
    q0: float,
    q1: float,
    baseline_price: float,
    simulated_price: float,
    elasticity: float,
    support_status: str,
    hist_min: float,
    hist_max: float,
    zero_anchor: bool,
) -> str:
    """Generate a plain-language explanation of the scenario results."""
    # Target cell changes
    price_pct = 100 * (simulated_price / baseline_price - 1)
    pack_change_pct = 100 * (target_row["Simulated_Unit_Sales"] / target_row["Baseline_Unit_Sales"] - 1)
    target_rev_change_pct = 100 * (target_row["Simulated_Revenue"] / target_row["Baseline_Revenue"] - 1)
    market_rev_change_pct = 100 * (r1 / r0 - 1)
    
    sku_name = target_row["SKU"]
    brand_name = target_row["Brand"]
    retailer_name = target_row["Retailer"]
    
    # Build explanation
    explanation_parts = []
    
    # What was changed
    explanation_parts.append(
        f"A {price_pct:+.1f}% price change for **{sku_name} ({brand_name})** at **{retailer_name}** "
        f"(baseline: {baseline_price:.2f} → new: {simulated_price:.2f})."
    )
    
    # Target effect
    if not zero_anchor:
        explanation_parts.append(
            f"The model estimates an own-price elasticity of **{elasticity:.2f}**. "
            f"This implies a **{pack_change_pct:+.1f}%** change in packs "
            f"({target_row['Baseline_Unit_Sales']:,.0f} → {target_row['Simulated_Unit_Sales']:,.0f}). "
            f"Target SKU revenue changes by **{target_rev_change_pct:+.1f}%**."
        )
    else:
        explanation_parts.append(
            f"This SKU-retailer cell has **no historical sales (zero anchor)**, so no elasticity can be estimated "
            f"and the quantity response is assumed to be zero."
        )
    
    # Market effect
    explanation_parts.append(
        f"**Observed-market revenue** (all SKUs × retailers in this dataset) changes by "
        f"**{market_rev_change_pct:+.2f}%**. This is smaller than the target SKU change because "
        f"the target is one cell within a larger market; other products retain their baseline packs, "
        f"but their revenue shares shift as the market denominator changes."
    )
    
    # Support status
    if zero_anchor:
        explanation_parts.append(
            "⚠️ **Zero anchor warning:** This cell has no positive-sales history. The model cannot estimate a price response."
        )
    elif support_status == "out_of_range":
        explanation_parts.append(
            f"⚠️ **Extrapolation warning:** The proposed price ({simulated_price:.2f}) lies outside the "
            f"observed historical range ({hist_min:.2f}–{hist_max:.2f}). Results are model extrapolations "
            f"without data support."
        )
    elif support_status == "unknown":
        explanation_parts.append(
            "ℹ️ **No historical data:** This cell has no observed positive-sales history. Price response cannot be validated."
        )
    else:
        explanation_parts.append(
            f"✅ **Supported:** The proposed price lies within the observed historical range "
            f"({hist_min:.2f}–{hist_max:.2f})."
        )
    
    return "\n\n".join(explanation_parts)


def render_decision_summary(
    scenario: pd.DataFrame,
    support: pd.DataFrame,
    changes: dict,
    baseline_price: float,
    simulated_price: float,
    r0: float,
    r1: float,
    q0: float,
    q1: float,
) -> None:
    """Render the decision summary card immediately after a scenario is computed."""
    if not changes:
        st.info("No price changes defined. Select a preset or edit prices to see a decision summary.")
        return
    
    # Get the primary edited cell (first one if multiple)
    target_key = list(changes.keys())[0]
    target_mask = np.ones(len(scenario), dtype=bool)
    for col, val in zip(CELL_KEYS, target_key):
        target_mask &= scenario[col].eq(val).to_numpy()
    target_row = scenario.loc[target_mask].iloc[0]
    
    # Get support info for target cell
    target_support = support[
        (support["Category"] == target_key[0]) &
        (support["Retailer"] == target_key[1]) &
        (support["Brand"] == target_key[2]) &
        (support["SKU"] == target_key[3])
    ].iloc[0]
    
    elasticity = float(target_row["Elasticity_MAP"])
    support_status = target_support["Support_Status"]
    hist_min = target_support["Historical_Min_Price"]
    hist_max = target_support["Historical_Max_Price"]
    zero_anchor = bool(target_support["Zero_Anchor"])
    
    # Calculate changes
    price_pct = 100 * (simulated_price / baseline_price - 1)
    pack_change_pct = 100 * (target_row["Simulated_Unit_Sales"] / target_row["Baseline_Unit_Sales"] - 1)
    target_rev_change = float(target_row["Revenue_Delta"])
    target_rev_change_pct = 100 * (target_row["Simulated_Revenue"] / target_row["Baseline_Revenue"] - 1)
    market_rev_change = r1 - r0
    market_rev_change_pct = 100 * (r1 / r0 - 1)
    
    # Header
    st.markdown("---")
    st.subheader("📋 Decision Summary")
    
    # Scenario metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Target:** {target_key[3]} ({target_key[2]}) at {target_key[1]}")
        st.markdown(f"**Category:** {target_key[0]}")
    with col2:
        st.markdown(f"**Price change:** {price_pct:+.1f}% ({baseline_price:.2f} → {simulated_price:.2f})")
        st.markdown(f"**Elasticity:** {elasticity:.2f}")
    with col3:
        support_label = {
            "in_range": "🟢 Within historical range",
            "out_of_range": "🔴 Outside historical range",
            "unknown": "❓ No historical data",
        }.get(support_status, support_status)
        if zero_anchor:
            support_label = "⚪ Zero anchor (not estimable)"
        st.markdown(f"**Data support:** {support_label}")
        st.markdown(f"**Baseline period:** {int(scenario['Period'].max())}")
    
    # Main answer - three large cards
    st.markdown("### What happens?")
    a, b, c = st.columns(3)
    
    with a:
        st.metric(
            "Target SKU Revenue",
            f"{target_row['Simulated_Revenue']:,.2f}",
            f"{target_rev_change:+,.2f} ({target_rev_change_pct:+.1f}%)",
            delta_color="normal" if target_rev_change >= 0 else "inverse"
        )
        st.caption(f"Packs: {target_row['Baseline_Unit_Sales']:,.0f} → {target_row['Simulated_Unit_Sales']:,.0f} ({pack_change_pct:+.1f}%)")
    
    with b:
        st.metric(
            "Observed-Market Revenue",
            f"{r1:,.2f}",
            f"{market_rev_change:+,.2f} ({market_rev_change_pct:+.2f}%)",
            delta_color="normal" if market_rev_change >= 0 else "inverse"
        )
        st.caption(f"Market packs: {q0:,.1f} → {q1:,.1f}")
    
    with c:
        # Evidence quality card
        if zero_anchor:
            st.metric("Evidence Quality", "⚪ Zero Anchor", "Not estimable")
            st.caption("No historical sales for this SKU-retailer")
        elif support_status == "out_of_range":
            st.metric("Evidence Quality", "🔴 Extrapolation", "Outside range")
            st.caption(f"Historical: {hist_min:.2f}–{hist_max:.2f}")
        elif support_status == "unknown":
            st.metric("Evidence Quality", "❓ No Data", "Cannot validate")
            st.caption("No positive-sales history")
        else:
            st.metric("Evidence Quality", "🟢 Supported", "In range")
            st.caption(f"Historical: {hist_min:.2f}–{hist_max:.2f}")
    
    # Plain-language explanation
    st.markdown("### Why?")
    explanation = _generate_plain_language_explanation(
        target_row, r0, r1, q0, q1,
        baseline_price, simulated_price, elasticity,
        support_status, hist_min, hist_max, zero_anchor
    )
    st.markdown(explanation)
    
    st.markdown("---")


def render_market_explanation(
    scenario: pd.DataFrame,
    market_decomp: pd.DataFrame,
    changes: dict,
    r0: float,
    r1: float,
) -> None:
    """Render an interactive market waterfall explaining the market revenue change."""
    if not changes:
        return
    
    st.subheader("🔍 Why did market revenue change?")
    st.caption("This bridge breaks down the accounting components. Hover for definitions.")
    
    # Get the market decomposition row
    row = market_decomp.iloc[0]
    
    # Compute target cell contributions
    target_key = list(changes.keys())[0]
    target_mask = np.ones(len(scenario), dtype=bool)
    for col, val in zip(CELL_KEYS, target_key):
        target_mask &= scenario[col].eq(val).to_numpy()
    target_row = scenario.loc[target_mask].iloc[0]
    
    # Target cell effects
    target_price_effect = float(target_row["Price_Effect"])
    target_volume_effect = float(target_row["Volume_Effect"])
    target_rev_delta = float(target_row["Revenue_Delta"])
    
    # Other cells' unchanged revenue
    other_cells_revenue_baseline = r0 - float(target_row["Baseline_Revenue"])
    other_cells_revenue_scenario = r1 - float(target_row["Simulated_Revenue"])
    other_cells_change = other_cells_revenue_scenario - other_cells_revenue_baseline
    
    # Share denominator effect = market change - target change - other cells change
    # Actually, other cells' packs are unchanged, so their revenue change comes from price effects
    # but since their prices didn't change, their revenue change is zero
    # The "share denominator effect" is the market revenue change not explained by target cell
    
    # Build waterfall data
    waterfall_data = {
        "Baseline Market Revenue": r0,
        "Target Price Effect": target_price_effect,
        "Target Volume Effect": target_volume_effect,
        "Other Cells' Revenue Change": other_cells_change,
        "Scenario Market Revenue": r1,
    }
    
    # Interactive waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Market Revenue Bridge",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=list(waterfall_data.keys()),
        y=list(waterfall_data.values()),
        text=[f"{v:+,.2f}" for v in waterfall_data.values()],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ca02c"}},
        decreasing={"marker": {"color": "#d62728"}},
        totals={"marker": {"color": "#1f77b4"}},
        hovertemplate="<b>%{x}</b><br>Value: %{y:,.2f}<extra></extra>",
    ))
    
    fig.update_layout(
        title="Market Revenue: From Baseline to Scenario",
        yaxis_title="Revenue",
        height=400,
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanatory text
    st.markdown("**How to read this:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        - **Baseline Market Revenue**: {r0:,.2f} (sum of all SKU×retailer baseline revenue)
        - **Target Price Effect**: {target_price_effect:+,.2f} (price change × scenario quantity)
        - **Target Volume Effect**: {target_volume_effect:+,.2f} (quantity change × baseline price)
        """)
    with col2:
        st.markdown(f"""
        - **Other Cells' Revenue Change**: {other_cells_change:+,.2f} (revenue change from unedited cells)
        - **Scenario Market Revenue**: {r1:,.2f} (new total)
        - **Net Market Change**: {r1 - r0:+,.2f} ({(r1/r0-1)*100:+.2f}%)
        """)
    
    # Key insight
    st.info(
        f"💡 **Key insight:** The target SKU's revenue changed by {target_rev_delta:+,.2f} "
        f"({target_rev_delta/float(target_row['Baseline_Revenue'])*100:+.1f}%), "
        f"but the observed-market revenue only changed by {r1-r0:+,.2f} ({(r1/r0-1)*100:+.2f}%). "
        f"This is because the target is one cell among many; other products' packs are unchanged "
        f"in this model, so their revenue shares shift purely through the denominator effect."
    )


def render_target_cell_story(
    target_row: pd.Series,
    target_support: pd.Series,
    baseline_price: float,
    simulated_price: float,
    elasticity: float,
) -> None:
    """Render an enhanced target-cell story with interpretation and warnings."""
    st.subheader(f"🎯 Target Cell Story: {target_row['SKU']} ({target_row['Brand']}) at {target_row['Retailer']}")
    
    support_status = target_support["Support_Status"]
    hist_min = target_support["Historical_Min_Price"]
    hist_max = target_support["Historical_Max_Price"]
    zero_anchor = bool(target_support["Zero_Anchor"])
    
    price_pct = 100 * (simulated_price / baseline_price - 1)
    pack_change_pct = 100 * (target_row["Simulated_Unit_Sales"] / target_row["Baseline_Unit_Sales"] - 1)
    target_rev_change_pct = 100 * (target_row["Simulated_Revenue"] / target_row["Baseline_Revenue"] - 1)
    
    # One-sentence interpretation
    if zero_anchor:
        interpretation = (
            "⚠️ This SKU-retailer has **no historical sales**, so the model cannot estimate a price response. "
            "Quantity is assumed unchanged."
        )
    elif support_status == "out_of_range":
        interpretation = (
            f"🔴 **Extrapolation:** The proposed price ({simulated_price:.2f}) is outside the observed "
            f"historical range ({hist_min:.2f}–{hist_max:.2f}). The model estimates "
            f"{'high' if abs(elasticity) > 2 else 'moderate'} price sensitivity (elasticity = {elasticity:.2f}), "
            f"but this estimate has **no data support** at this price level."
        )
    elif support_status == "unknown":
        interpretation = (
            "❓ This cell has **no historical positive-sales data**. The model cannot validate the price response."
        )
    else:
        sensitivity = "highly" if abs(elasticity) > 2 else "moderately" if abs(elasticity) > 1 else "low"
        direction = "increase" if pack_change_pct > 0 else "decrease"
        interpretation = (
            f"✅ This SKU is **{sensitivity} price-sensitive** (elasticity = {elasticity:.2f}) within the observed range. "
            f"A {price_pct:+.1f}% price change is estimated to {direction} packs by {abs(pack_change_pct):.1f}%, "
            f"{'more than offsetting' if target_rev_change_pct > 0 else 'not offsetting'} the lower price per pack."
        )
    
    st.markdown(f"### {interpretation}")
    
    # Detailed charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Price per pack**")
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(
            x=["Baseline", "Scenario"],
            y=[baseline_price, simulated_price],
            mode="lines+markers+text",
            text=[f"{baseline_price:.2f}", f"{simulated_price:.2f}"],
            textposition="top center",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=14, color="#1f77b4", symbol="circle"),
            name="Price",
            showlegend=False,
        ))
        # Add historical range band if available
        if not zero_anchor and not np.isnan(hist_min) and not np.isnan(hist_max):
            price_fig.add_hrect(
                y0=hist_min, y1=hist_max,
                fillcolor="rgba(44, 160, 44, 0.15)",
                line_width=0,
                annotation_text="Historical range",
                annotation_position="top left",
            )
        price_fig.update_layout(
            yaxis_title="Price",
            height=280,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(price_fig, use_container_width=True)
    
    with col2:
        st.markdown("**Packs (quantity)**")
        qty_fig = go.Figure()
        qty_fig.add_trace(go.Scatter(
            x=["Baseline", "Scenario"],
            y=[target_row["Baseline_Unit_Sales"], target_row["Simulated_Unit_Sales"]],
            mode="lines+markers+text",
            text=[f"{target_row['Baseline_Unit_Sales']:,.0f}", f"{target_row['Simulated_Unit_Sales']:,.0f}"],
            textposition="top center",
            line=dict(color="#2ca02c", width=3),
            marker=dict(size=14, color="#2ca02c", symbol="square"),
            name="Quantity",
            showlegend=False,
        ))
        qty_fig.update_layout(
            yaxis_title="Packs",
            height=280,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(qty_fig, use_container_width=True)
    
    # Revenue and elasticity details
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Revenue**")
        rev_fig = go.Figure()
        baseline_rev = baseline_price * target_row["Baseline_Unit_Sales"]
        scenario_rev = simulated_price * target_row["Simulated_Unit_Sales"]
        rev_fig.add_trace(go.Scatter(
            x=["Baseline", "Scenario"],
            y=[baseline_rev, scenario_rev],
            mode="lines+markers+text",
            text=[f"{baseline_rev:,.2f}", f"{scenario_rev:,.2f}"],
            textposition="top center",
            line=dict(color="#ff7f0e", width=3),
            marker=dict(size=14, color="#ff7f0e", symbol="diamond"),
            name="Revenue",
            showlegend=False,
        ))
        rev_fig.update_layout(
            yaxis_title="Revenue",
            height=280,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(rev_fig, use_container_width=True)
    
    with col4:
        st.markdown("**Elasticity & Support**")
        st.metric("Elasticity (β)", f"{elasticity:.3f}")
        
        if zero_anchor:
            st.warning("⚪ **Zero anchor** — no price activation possible (no historical sales)")
        else:
            if support_status == "in_range":
                st.success("🟢 **Within historical range**")
            elif support_status == "out_of_range":
                st.error("🔴 **Outside historical range** — extrapolation")
            else:
                st.info("❓ **No historical data**")
            
            if not np.isnan(hist_min) and not np.isnan(hist_max):
                st.caption(f"Historical price range: {hist_min:.2f} – {hist_max:.2f}")
                st.caption(f"Proposed price: {simulated_price:.2f}")
                if simulated_price < hist_min:
                    st.caption(f"📍 {hist_min - simulated_price:.2f} below historical minimum")
                elif simulated_price > hist_max:
                    st.caption(f"📍 {simulated_price - hist_max:.2f} above historical maximum")
                else:
                    st.caption("📍 Within historical range")
        
        st.caption(f"Price change: {price_pct:+.1f}%")
        st.caption(f"Pack change: {pack_change_pct:+.1f}%")
        st.caption(f"Revenue delta: {target_row['Revenue_Delta']:+,.2f} ({target_rev_change_pct:+.1f}%)")
        
        # Model limitations note
        with st.expander("⚠️ What this model does not know"):
            st.markdown("""
            - **No cross-price effects:** Competitor prices are held fixed; substitution is excluded.
            - **No inventory constraints:** Unlimited supply assumed.
            - **No competitor reactions:** Rivals do not respond to your price change.
            - **No promotional confounding:** The elasticity is a fitted association, not a causal guarantee.
            - **No profit optimization:** Revenue ≠ profit; costs are not modeled.
            - **Same-period anchoring:** Uses observed baseline demand, not a future forecast.
            """)


def render_market_share_explanation(
    scenario: pd.DataFrame,
    changes: dict,
) -> None:
    """Render a compact 'Who gained or lost share?' panel."""
    if not changes:
        return
    
    st.subheader("📊 Who gained or lost revenue share?")
    st.caption("Share changes reflect denominator effects. Unedited products keep their packs; their share falls if the target grows.")
    
    # Get target cell
    target_key = list(changes.keys())[0]
    target_mask = np.ones(len(scenario), dtype=bool)
    for col, val in zip(CELL_KEYS, target_key):
        target_mask &= scenario[col].eq(val).to_numpy()
    target_row = scenario.loc[target_mask].iloc[0]
    
    # Target SKU share change
    target_baseline_share = float(target_row["Baseline_Revenue"]) / scenario["Baseline_Revenue"].sum() * 100
    target_scenario_share = float(target_row["Simulated_Revenue"]) / scenario["Simulated_Revenue"].sum() * 100
    target_share_change = target_scenario_share - target_baseline_share
    
    # Brand-level shares
    brand_shares = scenario.groupby(["Category", "Brand"], as_index=False, observed=True).agg(
        Baseline_Revenue=("Baseline_Revenue", "sum"),
        Simulated_Revenue=("Simulated_Revenue", "sum"),
    )
    brand_shares["Baseline_Share_Pct"] = brand_shares["Baseline_Revenue"] / brand_shares["Baseline_Revenue"].sum() * 100
    brand_shares["Simulated_Share_Pct"] = brand_shares["Simulated_Revenue"] / brand_shares["Simulated_Revenue"].sum() * 100
    brand_shares["Share_Change_pp"] = brand_shares["Simulated_Share_Pct"] - brand_shares["Baseline_Share_Pct"]
    brand_shares = brand_shares.sort_values("Share_Change_pp", ascending=False)
    
    # Retailer-level shares
    retailer_shares = scenario.groupby("Retailer", as_index=False, observed=True).agg(
        Baseline_Revenue=("Baseline_Revenue", "sum"),
        Simulated_Revenue=("Simulated_Revenue", "sum"),
    )
    retailer_shares["Baseline_Share_Pct"] = retailer_shares["Baseline_Revenue"] / retailer_shares["Baseline_Revenue"].sum() * 100
    retailer_shares["Simulated_Share_Pct"] = retailer_shares["Simulated_Revenue"] / retailer_shares["Simulated_Revenue"].sum() * 100
    retailer_shares["Share_Change_pp"] = retailer_shares["Simulated_Share_Pct"] - retailer_shares["Baseline_Share_Pct"]
    retailer_shares = retailer_shares.sort_values("Share_Change_pp", ascending=False)
    
    # Display target SKU share change prominently
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            f"Target SKU Share ({target_key[3]})",
            f"{target_scenario_share:.2f}%",
            f"{target_share_change:+.2f} pp",
            delta_color="normal" if target_share_change >= 0 else "inverse"
        )
    with col2:
        st.metric("Baseline Share", f"{target_baseline_share:.2f}%")
    with col3:
        st.metric("Scenario Share", f"{target_scenario_share:.2f}%")
    
    # Top gaining/losing brands
    tab1, tab2 = st.tabs(["By Brand", "By Retailer"])
    
    with tab1:
        st.markdown("**Brand revenue share changes (percentage points)**")
        display_brands = brand_shares[["Category", "Brand", "Baseline_Share_Pct", "Simulated_Share_Pct", "Share_Change_pp"]].copy()
        display_brands.columns = ["Category", "Brand", "Baseline Share %", "Scenario Share %", "Change (pp)"]
        st.dataframe(
            display_brands.style.format({
                "Baseline Share %": "{:.2f}%",
                "Scenario Share %": "{:.2f}%",
                "Change (pp)": "{:+.2f} pp",
            }).applymap(
                lambda x: "color: green" if isinstance(x, str) and x.startswith("+") else ("color: red" if isinstance(x, str) and x.startswith("-") else ""),
                subset=["Change (pp)"],
            ),
            hide_index=True,
            use_container_width=True,
        )
    
    with tab2:
        st.markdown("**Retailer revenue share changes (percentage points)**")
        display_retailers = retailer_shares[["Retailer", "Baseline_Share_Pct", "Simulated_Share_Pct", "Share_Change_pp"]].copy()
        display_retailers.columns = ["Retailer", "Baseline Share %", "Scenario Share %", "Change (pp)"]
        st.dataframe(
            display_retailers.style.format({
                "Baseline Share %": "{:.2f}%",
                "Scenario Share %": "{:.2f}%",
                "Change (pp)": "{:+.2f} pp",
            }).applymap(
                lambda x: "color: green" if isinstance(x, str) and x.startswith("+") else ("color: red" if isinstance(x, str) and x.startswith("-") else ""),
                subset=["Change (pp)"],
            ),
            hide_index=True,
            use_container_width=True,
        )
    
    # Key note
    st.info(
        "💡 **Important:** Share loss ≠ pack loss. In this model, unedited products keep their baseline packs. "
        "Their revenue share falls because the target SKU's revenue grows, expanding the market denominator. "
        "This is a **share shift**, not a **volume shift**."
    )


def render_price_sweep_decision_chart(
    sweep_df: pd.DataFrame,
    baseline_price: float,
    r0: float,
    q0: float,
    target_cell: tuple,
    hist_min: float,
    hist_max: float,
    zero_anchor: bool,
    elasticity: float,
    current_price: float | None = None,
) -> None:
    """Render an enhanced price sweep as a decision chart."""
    st.subheader("📈 Price Sweep: Decision Chart")
    
    # Revenue sweep with historical band
    fig_rev = go.Figure()
    
    # Historical range band
    if not zero_anchor and not np.isnan(hist_min) and not np.isnan(hist_max):
        fig_rev.add_vrect(
            x0=hist_min, x1=hist_max,
            fillcolor="rgba(44, 160, 44, 0.15)",
            line_width=0,
            annotation_text="Historical support range",
            annotation_position="top left",
        )
    
    # Main revenue curve
    fig_rev.add_trace(go.Scatter(
        x=sweep_df["price"],
        y=sweep_df["revenue"],
        mode="lines",
        line=dict(color="#1f77b4", width=3),
        name="Market Revenue",
        hovertemplate="Price: %{x:.2f}<br>Market Revenue: %{y:,.2f}<extra></extra>",
    ))
    
    # Extrapolated points
    extrap = sweep_df[sweep_df["extrapolated"]]
    if len(extrap) > 0:
        fig_rev.add_trace(go.Scatter(
            x=extrap["price"],
            y=extrap["revenue"],
            mode="markers",
            marker=dict(size=8, color="#ff7f0e", symbol="x"),
            name="Extrapolated (no historical support)",
            hovertemplate="Price: %{x:.2f}<br>Market Revenue: %{y:,.2f}<br>⚠️ Extrapolated<extra></extra>",
        ))
    
    # Baseline marker
    fig_rev.add_trace(go.Scatter(
        x=[baseline_price],
        y=[r0],
        mode="markers",
        marker=dict(size=16, color="#2ca02c", symbol="star", line=dict(width=2, color="white")),
        name="Baseline",
        hovertemplate="Baseline Price: %{x:.2f}<br>Baseline Revenue: %{y:,.2f}<extra></extra>",
    ))
    
    # Current proposal marker
    if current_price is not None:
        # Find the revenue at current price (interpolate from sweep)
        current_revenue = np.interp(current_price, sweep_df["price"], sweep_df["revenue"])
        fig_rev.add_trace(go.Scatter(
            x=[current_price],
            y=[current_revenue],
            mode="markers",
            marker=dict(size=16, color="#d62728", symbol="diamond", line=dict(width=2, color="white")),
            name="Current Proposal",
            hovertemplate="Current Proposal<br>Price: %{x:.2f}<br>Market Revenue: %{y:,.2f}<extra></extra>",
        ))
    
    fig_rev.update_layout(
        title="Market Revenue vs Target Cell Price",
        xaxis_title="Target Cell Price",
        yaxis_title="Market Revenue",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    st.plotly_chart(fig_rev, use_container_width=True)
    
    # Packs sweep
    fig_packs = go.Figure()
    
    if not zero_anchor and not np.isnan(hist_min) and not np.isnan(hist_max):
        fig_packs.add_vrect(
            x0=hist_min, x1=hist_max,
            fillcolor="rgba(44, 160, 44, 0.15)",
            line_width=0,
            annotation_text="Historical support range",
            annotation_position="top left",
        )
    
    fig_packs.add_trace(go.Scatter(
        x=sweep_df["price"],
        y=sweep_df["packs"],
        mode="lines",
        line=dict(color="#2ca02c", width=3),
        name="Market Packs",
        hovertemplate="Price: %{x:.2f}<br>Market Packs: %{y:,.1f}<extra></extra>",
    ))
    
    if len(extrap) > 0:
        fig_packs.add_trace(go.Scatter(
            x=extrap["price"],
            y=extrap["packs"],
            mode="markers",
            marker=dict(size=8, color="#ff7f0e", symbol="x"),
            name="Extrapolated",
            hovertemplate="Price: %{x:.2f}<br>Market Packs: %{y:,.1f}<br>⚠️ Extrapolated<extra></extra>",
        ))
    
    fig_packs.add_trace(go.Scatter(
        x=[baseline_price],
        y=[q0],
        mode="markers",
        marker=dict(size=16, color="#2ca02c", symbol="star", line=dict(width=2, color="white")),
        name="Baseline",
        hovertemplate="Baseline Price: %{x:.2f}<br>Baseline Packs: %{y:,.1f}<extra></extra>",
    ))
    
    # Current proposal marker for packs
    if current_price is not None:
        current_packs = np.interp(current_price, sweep_df["price"], sweep_df["packs"])
        fig_packs.add_trace(go.Scatter(
            x=[current_price],
            y=[current_packs],
            mode="markers",
            marker=dict(size=16, color="#d62728", symbol="diamond", line=dict(width=2, color="white")),
            name="Current Proposal",
            hovertemplate="Current Proposal<br>Price: %{x:.2f}<br>Market Packs: %{y:,.1f}<extra></extra>",
        ))
    
    fig_packs.update_layout(
        title="Market Packs vs Target Cell Price",
        xaxis_title="Target Cell Price",
        yaxis_title="Market Packs",
        height=300,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    st.plotly_chart(fig_packs, use_container_width=True)
    
    # Decision guidance
    if not zero_anchor and not np.isnan(hist_min) and not np.isnan(hist_max):
        # Find optimal price in supported range
        in_range = sweep_df[
            (sweep_df["price"] >= hist_min) & 
            (sweep_df["price"] <= hist_max)
        ]
        if len(in_range) > 0:
            max_rev_idx = in_range["revenue"].idxmax()
            optimal_price = in_range.loc[max_rev_idx, "price"]
            optimal_revenue = in_range.loc[max_rev_idx, "revenue"]
            st.success(
                f"💡 **Within historical range:** Maximum market revenue of {optimal_revenue:,.2f} "
                f"occurs at price {optimal_price:.2f} (elasticity β = {elasticity:.2f})."
            )
    
    if current_price is not None:
        current_revenue = np.interp(current_price, sweep_df["price"], sweep_df["revenue"])
        current_packs = np.interp(current_price, sweep_df["price"], sweep_df["packs"])
        revenue_change_pct = 100 * (current_revenue / r0 - 1) if r0 > 0 else 0
        packs_change_pct = 100 * (current_packs / q0 - 1) if q0 > 0 else 0
        price_change_pct = 100 * (current_price / baseline_price - 1)
        st.info(
            f"📍 **Current proposal:** Price = {current_price:.2f} ({price_change_pct:+.1f}%) → "
            f"Market Revenue = {current_revenue:,.2f} ({revenue_change_pct:+.2f}%) → "
            f"Market Packs = {current_packs:,.1f} ({packs_change_pct:+.1f}%)"
        )
    
    st.caption(
        f"⚠️ Prices outside the shaded range are **extrapolations**. "
        f"The model can calculate them, but historical data do not support them. "
        f"β = {elasticity:.2f} produces flat revenue at β = -1.0. This is not a recommendation."
    )
    
    # Sweep table
    with st.expander("📋 Sweep data table"):
        st.dataframe(
            sweep_df[["price", "revenue", "revenue_change_pct", "packs", "extrapolated", "extrapolation_reason"]].style.format({
                "price": "{:.2f}",
                "revenue": "{:,.2f}",
                "revenue_change_pct": "{:+.2f}%",
                "packs": "{:,.1f}",
            }),
            use_container_width=True,
        )


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
    
    baseline_period = int(baseline["Period"].max())
    
    # Preset selector for quick scenarios
    st.subheader("Quick scenario")
    preset_options = {
        "hold": "Hold (no price changes)",
        "nexus_cola_20pct": "Cola 2L at Nexus Retail Group -20%",
        "custom": "Custom edits…",
    }
    preset = st.selectbox(
        "Choose a preset or customize",
        options=list(preset_options.keys()),
        format_func=lambda k: preset_options[k],
        index=0,
    )
    st.caption(f"📅 Baseline period: {baseline_period}")
    
    # Show preset details
    if preset == "nexus_cola_20pct":
        try:
            preset_changes = build_named_scenario("nexus_cola_20pct", baseline)
            target_cell = list(preset_changes.keys())[0]
            target_price = list(preset_changes.values())[0]
            baseline_price = baseline.loc[
                (baseline["Category"] == target_cell[0]) &
                (baseline["Retailer"] == target_cell[1]) &
                (baseline["Brand"] == target_cell[2]) &
                (baseline["SKU"] == target_cell[3]), "Price"
            ].iloc[0]
            pct_change = 100 * (target_price / baseline_price - 1)
            st.info(
                f"🎯 **Target:** {target_cell[3]} ({target_cell[2]}) at {target_cell[1]}\n"
                f"📊 **Baseline price:** {baseline_price:.2f} → **New price:** {target_price:.2f} ({pct_change:+.1f}%)\n"
                f"📅 **Baseline period:** {int(baseline['Period'].max())}"
            )
        except Exception as exc:
            st.warning(f"Could not load preset details: {exc}")
    elif preset == "hold":
        st.info("📌 **Hold scenario:** All prices remain at baseline. Revenue change = 0.")
    
    st.subheader("Scenario prices")
    st.caption("Prices are per sold pack. Absent listings cannot be created by editing a blank cell.")
    
    # Determine initial matrix based on preset
    if preset == "nexus_cola_20pct":
        # Apply preset to matrix
        try:
            preset_changes = build_named_scenario("nexus_cola_20pct", baseline)
            # Create a copy of matrix with preset applied
            edited_matrix = matrix.copy()
            for (cat, ret, brand, sku), new_price in preset_changes.items():
                mask = (
                    (edited_matrix["Category"] == cat) &
                    (edited_matrix["Brand"] == brand) &
                    (edited_matrix["SKU"] == sku)
                )
                col_name = f"Price | {ret}"
                if col_name in edited_matrix.columns:
                    edited_matrix.loc[mask, col_name] = new_price
        except Exception:
            edited_matrix = matrix
    elif preset == "hold":
        edited_matrix = matrix
    else:
        # Custom: use session state to persist edits across reruns
        if f"custom_matrix_{signature}" not in st.session_state:
            st.session_state[f"custom_matrix_{signature}"] = matrix
        edited_matrix = st.session_state[f"custom_matrix_{signature}"]
    
    if st.button("Reset prices"):
        st.session_state[reset_key] += 1
        if f"custom_matrix_{signature}" in st.session_state:
            st.session_state[f"custom_matrix_{signature}"] = matrix
    
    # Only show editor for custom mode
    if preset == "custom":
        edited = st.data_editor(
            edited_matrix, key=f"editor_{signature}_{st.session_state[reset_key]}",
            disabled=PRODUCT_KEYS, num_rows="fixed", hide_index=True,
            column_config={c: st.column_config.NumberColumn(label=r, min_value=0.000001, format="%.4f") for c, r in columns.items()},
        )
        st.session_state[f"custom_matrix_{signature}"] = edited
        if (matrix[list(columns)].isna() & edited[list(columns)].notna()).to_numpy().any():
            st.warning("Edits to absent listings are ignored; new listings require a demand assumption.")
        try:
            changes = matrix_to_price_changes(baseline, edited, columns)
        except Exception as exc:
            st.error(f"Invalid scenario: {exc}")
            st.stop()
    else:
        # For presets, use the pre-computed changes
        if preset == "nexus_cola_20pct":
            try:
                changes = build_named_scenario("nexus_cola_20pct", baseline)
            except Exception as exc:
                st.error(f"Invalid preset: {exc}")
                st.stop()
        else:  # hold
            changes = {}
        # Show read-only matrix for preset modes
        st.dataframe(
            edited_matrix, hide_index=True,
            column_config={c: st.column_config.NumberColumn(label=r, format="%.4f") for c, r in columns.items()},
        )
    
    # Compute config hash for cache key
    config_hash = _hash_config(config)
    data_hash = signature
    
    # Get cached scenario bundle (simulation + support + decompositions)
    try:
        with st.spinner("Computing scenario…"):
            bundle = get_cached_scenario_bundle(
                baseline=baseline,
                fitted_estimates=fitted["estimates"].data if hasattr(fitted["estimates"], "data") else fitted["estimates"],
                price_changes=changes,
                data=data,
                data_hash=data_hash,
                config_hash=config_hash,
            )
        
        scenario = bundle.scenario
        support = bundle.support
        market_decomp = bundle.market_decomp
        category_decomp = bundle.category_decomp
        retailer_decomp = bundle.retailer_decomp
        brand_decomp = bundle.brand_decomp
    except Exception as e:
        st.error(f"Scenario computation failed: {e}")
        st.stop()
    
    r0 = float(scenario["Baseline_Revenue"].sum())
    r1 = float(scenario["Simulated_Revenue"].sum())
    q0 = float(scenario["Baseline_Unit_Sales"].sum())
    q1 = float(scenario["Simulated_Unit_Sales"].sum())
    
    # Canonical edited keys from the validated price_changes dict (same threshold everywhere)
    edited_keys = list(changes.keys())
    changed_mask = np.zeros(len(scenario), dtype=bool)
    for key in edited_keys:
        mask = np.ones(len(scenario), dtype=bool)
        for col, val in zip(CELL_KEYS, key):
            mask &= scenario[col].eq(val).to_numpy()
        changed_mask |= mask
    
    # Support status for edited cells
    edited_support = support.loc[changed_mask].copy()
    outside = edited_support["Outside_Observed_Range"]
    unknown = edited_support["Support_Status"] == "unknown"
    in_range = edited_support["Support_Status"] == "in_range"
    zero_anchor = edited_support["Zero_Anchor"]
    
    # ============================================================
    # PHASE 1: DECISION SUMMARY (shown first after scenario is computed)
    # ============================================================
    if changes:
        # Get primary target cell info for decision summary
        target_key = edited_keys[0]
        target_mask = np.ones(len(scenario), dtype=bool)
        for col, val in zip(CELL_KEYS, target_key):
            target_mask &= scenario[col].eq(val).to_numpy()
        target_row = scenario.loc[target_mask].iloc[0]
        target_support = support[
            (support["Category"] == target_key[0]) &
            (support["Retailer"] == target_key[1]) &
            (support["Brand"] == target_key[2]) &
            (support["SKU"] == target_key[3])
        ].iloc[0]
        
        baseline_price = float(target_row["Baseline_Price"])
        simulated_price = float(target_row["Simulated_Price"])
        
        render_decision_summary(
            scenario=scenario,
            support=support,
            changes=changes,
            baseline_price=baseline_price,
            simulated_price=simulated_price,
            r0=r0,
            r1=r1,
            q0=q0,
            q1=q1,
        )
        
        # ============================================================
        # PHASE 2: MARKET EXPLANATION (Why did market revenue change?)
        # ============================================================
        render_market_explanation(
            scenario=scenario,
            market_decomp=market_decomp,
            changes=changes,
            r0=r0,
            r1=r1,
        )
        
        # ============================================================
        # PHASE 3: TARGET CELL STORY (Enhanced cell detail)
        # ============================================================
        render_target_cell_story(
            target_row=target_row,
            target_support=target_support,
            baseline_price=baseline_price,
            simulated_price=simulated_price,
            elasticity=float(target_row["Elasticity_MAP"]),
        )
        
        # ============================================================
        # PHASE 4: MARKET SHARE EXPLANATION (Who gained/lost share?)
        # ============================================================
        render_market_share_explanation(
            scenario=scenario,
            changes=changes,
        )
    else:
        # No changes - show simple KPIs
        a, b, c, d = st.columns(4)
        a.metric("Observed-market revenue change", f"{r1-r0:+,.2f}", f"{100*(r1/r0-1):+.2f}%")
        b.metric("Pack/item quantity change", f"{q1-q0:+,.1f}")
        c.metric("Edited cells", len(changes))
        d.metric("Edited cells outside observed prices", int(outside.sum()))
        st.caption("Revenue uses the dataset's currency. Quantities count packs, not litres. Observed-market totals are not an external market estimate.")
        st.info("No price changes defined. Select a preset or edit prices to see the decision narrative.")
    
    # ============================================================
    # COMPARISON MODE: Alternatives & Price Sweep (enhanced)
    # ============================================================
    st.markdown("---")
    st.subheader("📊 Compare alternatives & price sweep")
    st.caption("Define multiple scenarios and sweep target cell price. Not a recommendation.")
    
    # Alternative definitions
    alt_col1, alt_col2, alt_col3 = st.columns(3)
    
    with alt_col1:
        st.markdown("**Alternative A: Hold**")
        alt_a_changes = {}
    
    with alt_col2:
        st.markdown("**Alternative B: Current edit**")
        alt_b_changes = changes
    
    with alt_col3:
        st.markdown("**Alternative C: Custom**")
        # Let user define a third alternative
        enable_alt_c = st.checkbox("Enable Alternative C", key="enable_alt_c")
        if enable_alt_c:
            # Simple multiplier input
            mult = st.number_input("Price multiplier for all edited cells", 0.5, 2.0, 1.0, 0.05, key="alt_c_mult")
            alt_c_changes = {k: v * mult for k, v in changes.items()}
        else:
            alt_c_changes = {}
    
    # Compute scenarios for each alternative
    alternatives = {
        "Hold": alt_a_changes,
        "Current edit": alt_b_changes,
    }
    if enable_alt_c:
        alternatives["Custom"] = alt_c_changes
    
    alt_scenarios = {}
    alt_results = {}
    with st.spinner("Computing alternatives…"):
        for alt_name, alt_changes in alternatives.items():
            if not alt_changes:
                # Hold scenario - no changes
                alt_scenario = scenario.copy()
                alt_scenario["Simulated_Price"] = alt_scenario["Price"]
                alt_scenario["Simulated_Unit_Sales"] = alt_scenario["Baseline_Unit_Sales"]
                alt_scenario["Simulated_Revenue"] = alt_scenario["Baseline_Revenue"]
                alt_scenario["Price_Change_Pct"] = 0.0
                alt_scenario["Revenue_Delta"] = 0.0
            else:
                alt_scenario = simulate_scenario(baseline, fitted["estimates"], alt_changes)
            alt_scenarios[alt_name] = alt_scenario
            
            # Aggregate metrics
            r0_alt = float(alt_scenario["Baseline_Revenue"].sum())
            r1_alt = float(alt_scenario["Simulated_Revenue"].sum())
            q0_alt = float(alt_scenario["Baseline_Unit_Sales"].sum())
            q1_alt = float(alt_scenario["Simulated_Unit_Sales"].sum())
            alt_results[alt_name] = {
                "baseline_revenue": r0_alt,
                "scenario_revenue": r1_alt,
                "revenue_change": r1_alt - r0_alt,
                "revenue_change_pct": 100 * (r1_alt / r0_alt - 1) if r0_alt > 0 else 0,
                "baseline_packs": q0_alt,
                "scenario_packs": q1_alt,
                "packs_change": q1_alt - q0_alt,
            }
    
    # Display alternatives comparison
    alt_df = pd.DataFrame(alt_results).T
    alt_df.index.name = "Alternative"
    st.dataframe(
        alt_df[["baseline_revenue", "scenario_revenue", "revenue_change", "revenue_change_pct", "baseline_packs", "scenario_packs", "packs_change"]].style.format({
            "baseline_revenue": "{:,.2f}",
            "scenario_revenue": "{:,.2f}",
            "revenue_change": "{:+,.2f}",
            "revenue_change_pct": "{:+.2f}%",
            "baseline_packs": "{:,.1f}",
            "scenario_packs": "{:,.1f}",
            "packs_change": "{:+,.1f}",
        }),
        use_container_width=True,
    )
    
    # Price sweep for target cell (enhanced as decision chart)
    if changes:
        # Build selectable list from canonical edited keys (same as changes dict)
        sweep_cell_options = {}
        for key in edited_keys:
            label = f"{key[0]} | {key[1]} | {key[2]} | {key[3]}"
            sweep_cell_options[key] = label
        
        # Default to first edited cell
        sweep_default_idx = 0
        # Try to match the previously selected cell from detailed view
        if 'selected_key' in locals() and selected_key in sweep_cell_options:
            sweep_default_idx = list(sweep_cell_options.keys()).index(selected_key)
        
        sweep_target_key = st.selectbox(
            "Select target cell for price sweep",
            options=list(sweep_cell_options.keys()),
            format_func=lambda k: sweep_cell_options[k],
            index=sweep_default_idx,
            key="sweep_target_cell",
        )
        
        target_cell = sweep_target_key
        # Find target row in edited_detail (which is built from edited_keys)
        # Rebuild edited_detail for sweep
        selected = scenario.loc[changed_mask].copy()
        support_cols = edited_support[CELL_KEYS + ["Support_Status", "Historical_Min_Price", "Historical_Max_Price", "Elasticity", "Zero_Anchor"]]
        edited_detail = selected[CELL_KEYS + ["Baseline_Price", "Simulated_Price", "Price_Change_Pct", "Revenue_Delta", "Baseline_Unit_Sales", "Simulated_Unit_Sales"]].merge(
            support_cols, on=CELL_KEYS, how="left", validate="one_to_one"
        )
        
        target_row = edited_detail[
            (edited_detail["Category"] == target_cell[0]) &
            (edited_detail["Retailer"] == target_cell[1]) &
            (edited_detail["Brand"] == target_cell[2]) &
            (edited_detail["SKU"] == target_cell[3])
        ].iloc[0]
        
        target_support = support[
            (support["Category"] == target_cell[0]) &
            (support["Retailer"] == target_cell[1]) &
            (support["Brand"] == target_cell[2]) &
            (support["SKU"] == target_cell[3])
        ].iloc[0]
        
        baseline_price = float(target_row["Baseline_Price"])
        simulated_price = float(target_row["Simulated_Price"])
        elasticity = float(target_row["Elasticity"])
        hist_min = float(target_support["Historical_Min_Price"])
        hist_max = float(target_support["Historical_Max_Price"])
        zero_anchor = bool(target_support["Zero_Anchor"])
        
        # Default sweep range: historical range when available, else ±50% of baseline
        # Lower bound: small positive floor that includes baseline; upper bound: at least 2x baseline
        floor = max(1e-6, baseline_price * 1e-4)  # allow baseline for sub-cent prices
        default_min = hist_min if not np.isnan(hist_min) else baseline_price * 0.5
        default_max = hist_max if not np.isnan(hist_max) else baseline_price * 1.5
        default_min = max(floor, default_min)
        default_max = max(default_min * 1.01, default_max, baseline_price * 2)
        
        # Sweep range - use dynamic keys based on target cell to reset when target changes
        cell_key_str = f"{target_cell[0]}_{target_cell[1]}_{target_cell[2]}_{target_cell[3]}"
        sweep_min = st.number_input("Sweep min price", floor, default_max * 2, default_min, 0.01, key=f"sweep_min_{cell_key_str}")
        # Clamp max default to be at least the current min (handles user raising min above old default max)
        max_default = max(default_max, sweep_min * 1.01)
        sweep_max = st.number_input("Sweep max price", sweep_min, max_default * 2, max_default, 0.01, key=f"sweep_max_{cell_key_str}")
        n_points = st.slider("Number of points", 5, 50, 20, key=f"sweep_points_{cell_key_str}")
        
        # Show support status for sweep bounds
        if not np.isnan(hist_min) and not np.isnan(hist_max):
            st.caption(f"Historical price range: {hist_min:.2f} – {hist_max:.2f}")
        elif zero_anchor:
            st.caption("⚠️ Zero anchor — no historical price data; sweep bounds default to ±50% of baseline")
        else:
            st.caption("ℹ️ No historical positive-sales data; sweep bounds default to ±50% of baseline")
        
        # Generate sweep
        sweep_prices = np.linspace(sweep_min, sweep_max, n_points)
        sweep_results = []
        
        # Check if posterior is available
        has_posterior = "mcmc_diagnostics" in st.session_state or hasattr(fitted, "posterior")
        
        for p in sweep_prices:
            sweep_changes = {target_cell: float(p)}
            try:
                sweep_scenario = simulate_scenario(baseline, fitted["estimates"], sweep_changes)
                
                r1_sweep = float(sweep_scenario["Simulated_Revenue"].sum())
                q1_sweep = float(sweep_scenario["Simulated_Unit_Sales"].sum())
                rev_change = r1_sweep - r0
                pct_change = 100 * (r1_sweep / r0 - 1) if r0 > 0 else 0
                
                # Mark extrapolation
                extrapolated = False
                extrap_reason = ""
                if not np.isnan(hist_min) and (p < hist_min or p > hist_max):
                    extrapolated = True
                    if p < hist_min:
                        extrap_reason = f"Below historical min ({hist_min:.2f})"
                    else:
                        extrap_reason = f"Above historical max ({hist_max:.2f})"
                
                sweep_results.append({
                    "price": p,
                    "revenue": r1_sweep,
                    "revenue_change": rev_change,
                    "revenue_change_pct": pct_change,
                    "packs": q1_sweep,
                    "extrapolated": extrapolated,
                    "extrapolation_reason": extrap_reason,
                })
            except Exception as e:
                # Skip invalid sweep points but log the error
                sweep_results.append({
                    "price": p,
                    "revenue": np.nan,
                    "revenue_change": np.nan,
                    "revenue_change_pct": np.nan,
                    "packs": np.nan,
                    "extrapolated": True,
                    "extrapolation_reason": f"Error: {str(e)}",
                })
        
        sweep_df = pd.DataFrame(sweep_results)
        
        # Current scenario price for marker
        current_scenario_price = simulated_price
        
        # Render enhanced price sweep as decision chart
        render_price_sweep_decision_chart(
            sweep_df=sweep_df,
            baseline_price=baseline_price,
            r0=r0,
            q0=q0,
            target_cell=target_cell,
            hist_min=hist_min,
            hist_max=hist_max,
            zero_anchor=zero_anchor,
            elasticity=elasticity,
            current_price=current_scenario_price,
        )
    
    st.info("This comparison is based on fitted own-price associations (MAP point estimate). Substitution, promotional confounding, inventory constraints, competitor reactions, and supply constraints are excluded. Posterior uncertainty bands require --sample run.")
    
    if fitted["weak_support_count"]:
        st.warning(f"{fitted['weak_support_count']} SKU estimates fail heuristic support screens (<12 obs or within-pair log-price SD <0.02); this is not a causal-validity or significance test.")
    if fitted["positive_elasticities"]:
        st.warning("Positive elasticity estimates require investigation before business use.")
    
    # Traditional market waterfall (kept for reference)
    with st.expander("📊 Traditional Volume-Mix-Price Waterfall (Accounting View)"):
        row = market_decomp.iloc[0]
        fig = go.Figure(go.Waterfall(
            x=["Baseline", "Volume", "Mix", "Price", "Scenario"],
            measure=["absolute", "relative", "relative", "relative", "total"],
            y=[r0, row["Volume_Effect"], row["Mix_Effect"], row["Price_Effect"], 0],
        ))
        fig.update_layout(title="Sequential Volume → Mix → Price attribution", yaxis_title="Revenue")
        st.plotly_chart(fig)
        baseline_period = int(baseline["Period"].max())
        st.caption(f"Accounting allocation, not separate causal mechanisms. Volume is based on numeric litres-per-pack in this beverage dataset. Baseline: Period {baseline_period}.")
    
    # Cached grouped decompositions
    with st.expander("📊 Grouped Decompositions"):
        decomp_tabs = st.tabs(["Category", "Retailer", "Brand"])
        cached_decomps = {
            "Category": category_decomp,
            "Retailer": retailer_decomp,
            "Brand": brand_decomp,
        }
        for tab, grouping_name in zip(decomp_tabs, ["Category", "Retailer", "Brand"]):
            with tab:
                decomp = cached_decomps[grouping_name]
                display_decomp = decomp.copy()
                for col in ["Volume_Effect", "Mix_Effect", "Price_Effect", "Decomposition_Sum", "Revenue_Delta", "Q0", "Q1", "R0", "R1"]:
                    if col in display_decomp.columns:
                        display_decomp[col] = display_decomp[col].apply(lambda x: "N/A (bridge unavailable)" if pd.isna(x) else f"{x:,.2f}")
                st.dataframe(display_decomp, hide_index=True)
    
    # Value shares (enhanced with market share explanation already shown above)
    with st.expander("📊 Detailed Value Shares"):
        category = st.selectbox("Category for conditional shares", sorted(scenario["Category"].unique()))
        total_tab, conditional_tab = st.tabs(["Observed-market shares", "Within-category shares"])
        with total_tab:
            for grouping in [["Retailer"], ["Category"], ["Category", "Brand"]]:
                shares = value_shares(scenario, grouping)
                display_shares = shares.copy()
                for col in ["Baseline_Share_Pct", "Simulated_Share_Pct", "Share_Change_pp"]:
                    if col in display_shares.columns:
                        display_shares[col] = display_shares[col].apply(
                            lambda x: "N/A (no revenue)" if pd.isna(x) else f"{x:.2f}%")
                st.dataframe(display_shares, hide_index=True)
        with conditional_tab:
            for grouping in [["Category", "Retailer"], ["Category", "Brand"]]:
                shares = value_shares(scenario, grouping, parent_columns=["Category"])
                display_shares = shares.loc[shares["Category"].eq(category)].copy()
                for col in ["Baseline_Share_Pct", "Simulated_Share_Pct", "Share_Change_pp"]:
                    if col in display_shares.columns:
                        display_shares[col] = display_shares[col].apply(
                            lambda x: "N/A (no revenue)" if pd.isna(x) else f"{x:.2f}%")
                st.dataframe(display_shares, hide_index=True)
            retailer = st.selectbox("Retailer for brand-within-retailer shares", retailers)
            shares = value_shares(scenario, ["Category", "Retailer", "Brand"], parent_columns=["Category", "Retailer"])
            display_shares = shares.loc[shares["Category"].eq(category) & shares["Retailer"].eq(retailer)].copy()
            for col in ["Baseline_Share_Pct", "Simulated_Share_Pct", "Share_Change_pp"]:
                if col in display_shares.columns:
                    display_shares[col] = display_shares[col].apply(
                        lambda x: "N/A (no revenue)" if pd.isna(x) else f"{x:.2f}%")
            st.dataframe(display_shares, hide_index=True)
        st.caption("Unedited products retain demand; their shares can change through the denominator. Share changes are percentage points. N/A (no revenue) indicates zero denominator.")
    
    with st.expander("🔬 Evidence and fit diagnostics"):
        st.json({"method": "MAP", "converged": fitted["converged"], "iterations": fitted["iterations"]})
        st.dataframe(fitted["estimates"], hide_index=True)
        st.dataframe(support, hide_index=True)
        st.write("Prior sensitivity is not established by a support flag; it requires refitting under alternative priors.")
    with st.expander("📋 Data checks"):
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
