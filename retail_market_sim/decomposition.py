"""Price-volume-mix decomposition and market-share calculations."""
import numpy as np
import pandas as pd


def _pack_size_to_factor(pack_size) -> float:
    """Return validated litres per pack; invalid units are rejected, never defaulted."""
    value = pd.to_numeric(pd.Series([pack_size]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"Pack_Size must be a finite positive litres-per-pack value; got {pack_size!r}.")
    return float(value)


def _ensure_equiv_volumes(scenario: pd.DataFrame) -> pd.DataFrame:
    if "Equiv_Volume_Baseline" not in scenario.columns:
        scenario = scenario.copy()
        litres = scenario["Pack_Size"].map(_pack_size_to_factor)
        scenario["Equiv_Volume_Baseline"] = litres * scenario["Baseline_Unit_Sales"]
        scenario["Equiv_Volume_Simulated"] = litres * scenario["Simulated_Unit_Sales"]
    return scenario


def three_factor_decomposition(scenario: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    """Exact sequential Volume -> Mix -> Price attribution using litres.

    Strict API: requires positive total equivalized volume in every group.
    Use `three_factor_decomposition_report` for a reporting wrapper that handles
    zero-volume groups gracefully.
    """
    scenario = _ensure_equiv_volumes(scenario)
    if not group_columns:
        scenario = scenario.copy()
        scenario["_total_market"] = 1
        group_columns = ["_total_market"]
        drop_total = True
    else:
        drop_total = False
    g = scenario.groupby(group_columns, as_index=False, observed=True).agg(
        Q0=("Equiv_Volume_Baseline", "sum"), Q1=("Equiv_Volume_Simulated", "sum"),
        R0=("Baseline_Revenue", "sum"), R1=("Simulated_Revenue", "sum"))
    if (g["Q0"] <= 0).any() or (g["Q1"] <= 0).any():
        raise ValueError("Each decomposition group requires positive total equivalized volume in both baseline and scenario.")
    detail = scenario.copy()
    v0 = detail["Equiv_Volume_Baseline"]
    v1 = detail["Equiv_Volume_Simulated"]
    detail["s0"] = v0 / detail.groupby(group_columns, observed=True)["Equiv_Volume_Baseline"].transform("sum")
    detail["s1"] = v1 / detail.groupby(group_columns, observed=True)["Equiv_Volume_Simulated"].transform("sum")
    detail["P0_equiv"] = detail["Baseline_Revenue"] / v0.replace(0, np.nan)
    detail["P1_equiv"] = detail["Simulated_Revenue"] / v1.replace(0, np.nan)
    detail["_mix_term"] = (detail["s1"] - detail["s0"]) * detail["P0_equiv"]
    detail["_price_term"] = v1 * (detail["P1_equiv"] - detail["P0_equiv"])
    terms = detail.groupby(group_columns, as_index=False, observed=True).agg(
        Mix_Sum=("_mix_term", "sum"), Price_Effect=("_price_term", "sum"))
    g = g.merge(terms, on=group_columns, how="left", validate="one_to_one")
    g["Pbar0"] = g["R0"] / g["Q0"]
    g["Volume_Effect"] = (g["Q1"] - g["Q0"]) * g["Pbar0"]
    price_effect = g.pop("Price_Effect")
    g["Mix_Effect"] = g["Q1"] * g.pop("Mix_Sum")
    g["Price_Effect"] = price_effect
    g["Decomposition_Sum"] = g["Volume_Effect"] + g["Mix_Effect"] + g["Price_Effect"]
    g["Revenue_Delta"] = g["R1"] - g["R0"]
    error = g["Decomposition_Sum"] - g["Revenue_Delta"]
    if not np.allclose(error, 0.0, rtol=1e-9, atol=1e-6):
        raise RuntimeError(f"Decomposition reconciliation failed: max error = {error.abs().max():.6f}")
    if drop_total:
        g = g.drop(columns=["_total_market"])
    return g


def _decomposition_group_status(q0: float, q1: float, r0: float, r1: float) -> tuple[str, str]:
    """Determine status and bridge availability for a decomposition group.

    Returns (status, bridge_status) where:
    - status: 'positive', 'zero_to_zero', 'one_sided_zero'
    - bridge_status: 'available', 'no sales', 'bridge unavailable'
    """
    if q0 > 0 and q1 > 0:
        return 'positive', 'available'
    elif q0 == 0 and q1 == 0 and r0 == 0 and r1 == 0:
        return 'zero_to_zero', 'no sales'
    else:
        return 'one_sided_zero', 'bridge unavailable'


def three_factor_decomposition_report(scenario: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    """Reporting wrapper for three-factor decomposition that handles zero-volume groups.

    For zero-to-zero volume and revenue: returns zero absolute change with 'no sales' status
    and undefined mix shares. For one-sided zero: shows absolute revenue change and
    'bridge unavailable' rather than inventing shares.
    """
    scenario = _ensure_equiv_volumes(scenario)

    # Handle empty group_columns (market total) by using a temporary column
    if not group_columns:
        scenario = scenario.copy()
        scenario["_total_market"] = 1
        effective_group_columns = ["_total_market"]
    else:
        effective_group_columns = list(group_columns)

    # Compute group totals first
    g = scenario.groupby(effective_group_columns, as_index=False, observed=True).agg(
        Q0=("Equiv_Volume_Baseline", "sum"), Q1=("Equiv_Volume_Simulated", "sum"),
        R0=("Baseline_Revenue", "sum"), R1=("Simulated_Revenue", "sum"))

    # Compute revenue delta for all groups
    g["Revenue_Delta"] = g["R1"] - g["R0"]

    # Determine status for each group
    statuses = []
    bridge_statuses = []
    for _, row in g.iterrows():
        status, bridge_status = _decomposition_group_status(row["Q0"], row["Q1"], row["R0"], row["R1"])
        statuses.append(status)
        bridge_statuses.append(bridge_status)

    g["_group_status"] = statuses
    g["_bridge_status"] = bridge_statuses

    # For positive groups, compute full decomposition
    positive_mask = g["_group_status"] == 'positive'
    if positive_mask.any():
        pos_groups = g.loc[positive_mask, effective_group_columns].copy()
        # Filter scenario to only positive groups
        pos_scenario = scenario.merge(pos_groups, on=effective_group_columns, how="inner", validate="many_to_one")
        pos_decomp = three_factor_decomposition(pos_scenario, effective_group_columns)
        # Drop duplicate columns from pos_decomp before merge (Q0, Q1, R0, R1, Revenue_Delta are already in g)
        decomp_cols = [c for c in pos_decomp.columns if c not in ["Q0", "Q1", "R0", "R1", "Revenue_Delta"]]
        pos_decomp = pos_decomp[decomp_cols]
        # Merge results back
        g = g.merge(pos_decomp, on=effective_group_columns, how="left", validate="one_to_one")

    # For zero-to-zero groups: zero change, no bridge
    zero_mask = g["_group_status"] == 'zero_to_zero'
    g.loc[zero_mask, "Volume_Effect"] = 0.0
    g.loc[zero_mask, "Mix_Effect"] = 0.0
    g.loc[zero_mask, "Price_Effect"] = 0.0
    g.loc[zero_mask, "Decomposition_Sum"] = 0.0
    g.loc[zero_mask, "Revenue_Delta"] = 0.0

    # For one-sided zero groups: show revenue delta, no bridge
    onesided_mask = g["_group_status"] == 'one_sided_zero'
    g.loc[onesided_mask, "Volume_Effect"] = np.nan
    g.loc[onesided_mask, "Mix_Effect"] = np.nan
    g.loc[onesided_mask, "Price_Effect"] = np.nan
    g.loc[onesided_mask, "Decomposition_Sum"] = np.nan
    # Revenue_Delta is already correct from groupby

    # Reorder columns for output - use original group_columns for display
    display_columns = list(group_columns) if group_columns else []
    cols = display_columns + ["Q0", "Q1", "R0", "R1", "Volume_Effect", "Mix_Effect", "Price_Effect",
                               "Decomposition_Sum", "Revenue_Delta", "_group_status", "_bridge_status"]
    g = g[cols]

    return g


def value_shares(scenario: pd.DataFrame, group_columns: list, parent_columns: list | None = None) -> pd.DataFrame:
    grouped = scenario.groupby(group_columns, as_index=False, observed=True).agg(
        Baseline_Revenue=("Baseline_Revenue", "sum"), Simulated_Revenue=("Simulated_Revenue", "sum"))
    if parent_columns:
        baseline_denominator = grouped.groupby(parent_columns, observed=True)["Baseline_Revenue"].transform("sum")
        simulated_denominator = grouped.groupby(parent_columns, observed=True)["Simulated_Revenue"].transform("sum")
    else:
        baseline_denominator = grouped["Baseline_Revenue"].sum()
        simulated_denominator = grouped["Simulated_Revenue"].sum()

    # Compute shares, handling zero denominators explicitly
    baseline_share = grouped["Baseline_Revenue"] / baseline_denominator * 100
    simulated_share = grouped["Simulated_Revenue"] / simulated_denominator * 100

    # Replace inf/NaN from zero denominators with explicit "N/A (no revenue)"
    # Use a string marker that can be displayed in the UI
    grouped["Baseline_Share_Pct"] = baseline_share.replace([np.inf, -np.inf], np.nan)
    grouped["Simulated_Share_Pct"] = simulated_share.replace([np.inf, -np.inf], np.nan)

    # For zero denominator, mark as "N/A (no revenue)" - we use NaN internally
    # but the display layer should render this appropriately
    zero_baseline = baseline_denominator == 0
    zero_simulated = simulated_denominator == 0
    if zero_baseline.any():
        grouped.loc[zero_baseline, "Baseline_Share_Pct"] = np.nan
    if zero_simulated.any():
        grouped.loc[zero_simulated, "Simulated_Share_Pct"] = np.nan

    grouped["Share_Change_pp"] = grouped["Simulated_Share_Pct"] - grouped["Baseline_Share_Pct"]
    return grouped
