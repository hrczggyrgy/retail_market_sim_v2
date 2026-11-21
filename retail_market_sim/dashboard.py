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
    changed = scenario["Price_Change_Pct"].abs().gt(1e-7)
    
    # Support status for edited cells
    edited_support = support.loc[changed].copy()
    outside = edited_support["Outside_Observed_Range"]
    unknown = edited_support["Support_Status"] == "unknown"
    in_range = edited_support["Support_Status"] == "in_range"
    zero_anchor = edited_support["Zero_Anchor"]
    
    a, b, c, d = st.columns(4)
    a.metric("Observed-market revenue change", f"{r1-r0:+,.2f}", f"{100*(r1/r0-1):+.2f}%")
    b.metric("Pack/item quantity change", f"{q1-q0:+,.1f}")
    c.metric("Edited cells", len(changes))
    d.metric("Edited cells outside observed prices", int(outside.sum()))
    st.caption("Revenue uses the dataset's currency. Quantities count packs, not litres. Observed-market totals are not an external market estimate.")
    
    if changes:
        # Show detailed edited-cell impact with support status
        selected = scenario.loc[changed].copy()
        # Don't include Price_Change_Pct in support_cols to avoid merge conflict (already in selected)
        support_cols = edited_support[CELL_KEYS + ["Support_Status", "Historical_Min_Price", "Historical_Max_Price", "Elasticity", "Zero_Anchor"]]
        edited_detail = selected[CELL_KEYS + ["Baseline_Price", "Simulated_Price", "Price_Change_Pct", "Revenue_Delta", "Baseline_Unit_Sales", "Simulated_Unit_Sales"]].merge(
            support_cols, on=CELL_KEYS, how="left", validate="one_to_one"
        )
        
        # Format for display
        display_edited = edited_detail.copy()
        # Color code support status
        def format_support(row):
            if row["Zero_Anchor"]:
                return "⚪ Zero anchor (not estimable)"
            elif row["Support_Status"] == "unknown":
                return "❓ No historical data"
            elif row["Support_Status"] == "out_of_range":
                return "🔴 Outside historical range"
            elif row["Support_Status"] == "in_range":
                return "🟢 Within historical range"
            return row["Support_Status"]
        
        display_edited["Support"] = display_edited.apply(format_support, axis=1)
        
        st.subheader("Edited-cell impact")
        st.dataframe(
            display_edited[CELL_KEYS + ["Baseline_Price", "Simulated_Price", "Price_Change_Pct", "Revenue_Delta", "Support", "Elasticity"]],
            hide_index=True,
            column_config={
                "Elasticity": st.column_config.NumberColumn("Elasticity", format="%.3f"),
                "Support": st.column_config.TextColumn("Price Support"),
            }
        )
        
        # Show warning for unsupported edits
        if outside.any():
            st.warning("⚠️ Some proposed prices lie outside their SKU-retailer historical positive-sales price range. These edits are not supported by observed data.")
            with st.expander("Details on unsupported edits"):
                unsupported = edited_detail[edited_detail["Support_Status"] == "out_of_range"]
                st.dataframe(unsupported[CELL_KEYS + ["Baseline_Price", "Simulated_Price", "Historical_Min_Price", "Historical_Max_Price", "Price_Change_Pct"]], hide_index=True)
        
        if unknown.any():
            st.info("ℹ️ Some edited cells have no historical positive-sales data (zero anchor or no history). Price response cannot be validated for these cells.")
        
        st.info("These are fitted own-price associations. Substitution, promotional confounding, inventory constraints and competitor reactions are excluded; consider testing, not automatic rollout.")
    
    # Cell selection for detailed paired analysis
    if changes:
        st.subheader("Cell detail: paired baseline vs scenario")
        # Build selectable list of edited cells
        cell_options = {}
        for _, row in edited_detail.iterrows():
            key = (row["Category"], row["Retailer"], row["Brand"], row["SKU"])
            label = f"{row['Category']} | {row['Retailer']} | {row['Brand']} | {row['SKU']}"
            cell_options[key] = label
        
        selected_key = st.selectbox(
            "Select cell for detailed view",
            options=list(cell_options.keys()),
            format_func=lambda k: cell_options[k],
            index=0,
        )
        
        if selected_key:
            cell_data = edited_detail[
                (edited_detail["Category"] == selected_key[0]) &
                (edited_detail["Retailer"] == selected_key[1]) &
                (edited_detail["Brand"] == selected_key[2]) &
                (edited_detail["SKU"] == selected_key[3])
            ].iloc[0]
            
            col1, col2 = st.columns(2)
            
            # Price comparison
            with col1:
                st.markdown("**Price per pack**")
                price_fig = go.Figure()
                price_fig.add_trace(go.Scatter(
                    x=["Baseline", "Scenario"],
                    y=[cell_data["Baseline_Price"], cell_data["Simulated_Price"]],
                    mode="lines+markers+text",
                    text=[f"{cell_data['Baseline_Price']:.2f}", f"{cell_data['Simulated_Price']:.2f}"],
                    textposition="top center",
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=12, color="#1f77b4", symbol="circle"),
                    name="Price",
                    showlegend=False,
                ))
                price_fig.update_layout(
                    yaxis_title="Price",
                    height=250,
                    margin=dict(l=40, r=20, t=30, b=40),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(price_fig, use_container_width=True)
            
            # Quantity comparison
            with col2:
                st.markdown("**Packs (quantity)**")
                qty_fig = go.Figure()
                qty_fig.add_trace(go.Scatter(
                    x=["Baseline", "Scenario"],
                    y=[cell_data["Baseline_Unit_Sales"], cell_data["Simulated_Unit_Sales"]],
                    mode="lines+markers+text",
                    text=[f"{cell_data['Baseline_Unit_Sales']:,.0f}", f"{cell_data['Simulated_Unit_Sales']:,.0f}"],
                    textposition="top center",
                    line=dict(color="#2ca02c", width=2),
                    marker=dict(size=12, color="#2ca02c", symbol="square"),
                    name="Quantity",
                    showlegend=False,
                ))
                qty_fig.update_layout(
                    yaxis_title="Packs",
                    height=250,
                    margin=dict(l=40, r=20, t=30, b=40),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(qty_fig, use_container_width=True)
            
            # Revenue comparison
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Revenue**")
                rev_fig = go.Figure()
                rev_fig.add_trace(go.Scatter(
                    x=["Baseline", "Scenario"],
                    y=[cell_data["Baseline_Price"] * cell_data["Baseline_Unit_Sales"], cell_data["Simulated_Price"] * cell_data["Simulated_Unit_Sales"]],
                    mode="lines+markers+text",
                    text=[f"{cell_data['Baseline_Price'] * cell_data['Baseline_Unit_Sales']:,.2f}", f"{cell_data['Simulated_Price'] * cell_data['Simulated_Unit_Sales']:,.2f}"],
                    textposition="top center",
                    line=dict(color="#ff7f0e", width=2),
                    marker=dict(size=12, color="#ff7f0e", symbol="diamond"),
                    name="Revenue",
                    showlegend=False,
                ))
                rev_fig.update_layout(
                    yaxis_title="Revenue",
                    height=250,
                    margin=dict(l=40, r=20, t=30, b=40),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(rev_fig, use_container_width=True)
            
            # Elasticity and support details
            with col4:
                st.markdown("**Elasticity & support**")
                st.metric("Elasticity", f"{cell_data['Elasticity']:.3f}")
                if cell_data["Zero_Anchor"]:
                    st.warning("⚪ Zero anchor — no price activation possible")
                else:
                    support_status = cell_data["Support_Status"]
                    if support_status == "in_range":
                        st.success("🟢 Within historical range")
                    elif support_status == "out_of_range":
                        st.error("🔴 Outside historical range")
                    else:
                        st.info("❓ No historical data")
                    if not pd.isna(cell_data.get("Historical_Min_Price", np.nan)):
                        st.caption(f"Historical range: {cell_data['Historical_Min_Price']:.2f} – {cell_data['Historical_Max_Price']:.2f}")
                st.caption(f"Price change: {cell_data['Price_Change_Pct']:+.1f}%")
                st.caption(f"Revenue delta: {cell_data['Revenue_Delta']:+,.2f}")
    
    # ============================================================
    # Comparison mode: Multiple alternatives & price sweep
    # ============================================================
    st.subheader("Compare alternatives & price sweep")
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
    
    # Price sweep for target cell
    if changes:
        st.subheader("Target cell price sweep")
        # Use the same cell selector as the detailed view for consistency
        # Build selectable list of edited cells
        sweep_cell_options = {}
        for _, row in edited_detail.iterrows():
            key = (row["Category"], row["Retailer"], row["Brand"], row["SKU"])
            label = f"{row['Category']} | {row['Retailer']} | {row['Brand']} | {row['SKU']}"
            sweep_cell_options[key] = label
        
        # Default to first edited cell, but allow user to select any
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
        target_row = edited_detail[
            (edited_detail["Category"] == target_cell[0]) &
            (edited_detail["Retailer"] == target_cell[1]) &
            (edited_detail["Brand"] == target_cell[2]) &
            (edited_detail["SKU"] == target_cell[3])
        ].iloc[0]
        
        baseline_price = target_row["Baseline_Price"]
        elasticity = target_row["Elasticity"]
        support_status = target_row["Support_Status"]
        hist_min = target_row.get("Historical_Min_Price", np.nan)
        hist_max = target_row.get("Historical_Max_Price", np.nan)
        zero_anchor = target_row.get("Zero_Anchor", False)
        
        # Default sweep range to historical range when available, otherwise baseline ±50%
        # Ensure min < max by clamping defaults
        default_min = max(0.01, hist_min if not np.isnan(hist_min) else baseline_price * 0.5)
        default_max = min(baseline_price * 2, hist_max if not np.isnan(hist_max) else baseline_price * 1.5)
        if default_min >= default_max:
            default_min = 0.01
            default_max = baseline_price * 2
        
        # Sweep range - use dynamic keys based on target cell to reset when target changes
        cell_key_str = f"{target_cell[0]}_{target_cell[1]}_{target_cell[2]}_{target_cell[3]}"
        sweep_min = st.number_input("Sweep min price", 0.01, baseline_price * 2, default_min, 0.01, key=f"sweep_min_{cell_key_str}")
        sweep_max = st.number_input("Sweep max price", sweep_min, baseline_price * 2, default_max, 0.01, key=f"sweep_max_{cell_key_str}")
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
                
                r1 = float(sweep_scenario["Simulated_Revenue"].sum())
                q1 = float(sweep_scenario["Simulated_Unit_Sales"].sum())
                rev_change = r1 - r0
                pct_change = 100 * (r1 / r0 - 1) if r0 > 0 else 0
                
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
                    "revenue": r1,
                    "revenue_change": rev_change,
                    "revenue_change_pct": pct_change,
                    "packs": q1,
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
        
        # Plot sweep
        fig_sweep = go.Figure()
        
        # Revenue line
        fig_sweep.add_trace(go.Scatter(
            x=sweep_df["price"],
            y=sweep_df["revenue"],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6, color="#1f77b4"),
            name="Market Revenue",
            showlegend=False,
        ))
        
        # Highlight extrapolated region
        extrap = sweep_df[sweep_df["extrapolated"]]
        if len(extrap) > 0:
            fig_sweep.add_trace(go.Scatter(
                x=extrap["price"],
                y=extrap["revenue"],
                mode="markers",
                marker=dict(size=8, color="#ff7f0e", symbol="x"),
                name="Extrapolated (no historical support)",
                showlegend=True,
            ))
        
        # Baseline marker
        fig_sweep.add_trace(go.Scatter(
            x=[baseline_price],
            y=[r0],
            mode="markers",
            marker=dict(size=12, color="#2ca02c", symbol="star"),
            name="Baseline",
            showlegend=True,
        ))
        
        fig_sweep.update_layout(
            title="Price sweep: Market revenue vs target cell price",
            xaxis_title="Target cell price",
            yaxis_title="Market Revenue",
            height=400,
            hovermode="x unified",
        )
        
        st.plotly_chart(fig_sweep, use_container_width=True)
        
        # Packs sweep
        fig_sweep_q = go.Figure()
        fig_sweep_q.add_trace(go.Scatter(
            x=sweep_df["price"],
            y=sweep_df["packs"],
            mode="lines+markers",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=6, color="#2ca02c"),
            name="Market Packs",
            showlegend=False,
        ))
        if len(extrap) > 0:
            fig_sweep_q.add_trace(go.Scatter(
                x=extrap["price"],
                y=extrap["packs"],
                mode="markers",
                marker=dict(size=8, color="#ff7f0e", symbol="x"),
                name="Extrapolated",
                showlegend=True,
            ))
        fig_sweep_q.add_trace(go.Scatter(
            x=[baseline_price],
            y=[q0],
            mode="markers",
            marker=dict(size=12, color="#2ca02c", symbol="star"),
            name="Baseline",
            showlegend=True,
        ))
        fig_sweep_q.update_layout(
            title="Price sweep: Market packs vs target cell price",
            xaxis_title="Target cell price",
            yaxis_title="Market Packs",
            height=300,
        )
        st.plotly_chart(fig_sweep_q, use_container_width=True)
        
        # Sweep table
        st.dataframe(
            sweep_df[["price", "revenue", "revenue_change_pct", "packs", "extrapolated", "extrapolation_reason"]].style.format({
                "price": "{:.2f}",
                "revenue": "{:,.2f}",
                "revenue_change_pct": "{:+.2f}%",
                "packs": "{:,.1f}",
            }),
            use_container_width=True,
        )
        
        st.caption("⚠️ Extrapolated points have no historical price support. β = {:.2f} produces flat revenue at β = -1.0. This is not a recommendation.".format(elasticity))
    
    st.info("This comparison is based on fitted own-price associations (MAP point estimate). Substitution, promotional confounding, inventory constraints, competitor reactions, and supply constraints are excluded. Posterior uncertainty bands require --sample run.")
    
    if fitted["weak_support_count"]:
        st.warning(f"{fitted['weak_support_count']} SKU estimates fail heuristic support screens (<12 obs or within-pair log-price SD <0.02); this is not a causal-validity or significance test.")
    if fitted["positive_elasticities"]:
        st.warning("Positive elasticity estimates require investigation before business use.")
    
    # Market waterfall (from cached market decomp)
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
    
    st.subheader("Value shares")
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
