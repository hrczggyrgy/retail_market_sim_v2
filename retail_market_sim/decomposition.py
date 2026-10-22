"""
Price-Volume-Mix decomposition and market share calculations.
"""
import numpy as np
import pandas as pd


# =============================================================================
# Pack Size Conversion
# =============================================================================

# Note: In the current sample data, Pack_Size is already numeric litres.
# This parser is retained for potential future formats.
def _pack_size_to_factor(pack_size: str) -> float:
    """
    Extract numeric pack-size factor from string.

    Current data uses numeric litres directly (e.g., "0.33", "2.00", "1.98").
    This function returns the numeric value if parseable, else 1.0.
    """
    if pd.isna(pack_size):
        return 1.0
    s = str(pack_size).strip()
    try:
        return float(s)
    except ValueError:
        return 1.0


def _ensure_equiv_volumes(scenario: pd.DataFrame) -> pd.DataFrame:
    """Add equivalized volume columns if not present."""
    if "Equiv_Volume_Baseline" not in scenario.columns:
        scenario = scenario.copy()
        scenario["Equiv_Volume_Baseline"] = scenario.apply(
            lambda row: _pack_size_to_factor(row["Pack_Size"]) * row["Baseline_Unit_Sales"], axis=1
        )
        scenario["Equiv_Volume_Simulated"] = scenario.apply(
            lambda row: _pack_size_to_factor(row["Pack_Size"]) * row["Simulated_Unit_Sales"], axis=1
        )
    return scenario


# =============================================================================
# Three-Factor Decomposition (Volume -> Mix -> Price)
# =============================================================================

def three_factor_decomposition(
    scenario: pd.DataFrame,
    group_columns: list,
) -> pd.DataFrame:
    """
    Compute Volume / Mix / Price decomposition at an aggregated level.

    Uses the sequential convention (Volume -> Mix -> Price) with equivalized volumes:
      VE = (Q1 - Q0) * Pbar0
      ME = Q1 * sum_s (s1_s - s0_s) * P0_s
      PE = sum_s V1_s * (P1_s - P0_s)

    where Q = sum V (equivalized volume), s = V/Q (share), P = revenue per equiv volume.

    Parameters
    ----------
    scenario : pd.DataFrame
        Simulation output with Baseline/Simulated columns.
    group_columns : list
        Columns to group by (e.g., ["Category"], ["Retailer"], ["Category", "Brand"]).
        Empty list = total market.

    Returns
    -------
    pd.DataFrame with decomposition components and reconciliation check.
    """
    scenario = _ensure_equiv_volumes(scenario)

    # Handle empty group_columns (total market)
    if not group_columns:
        scenario = scenario.copy()
        scenario["_total_market"] = 1
        group_columns = ["_total_market"]
        drop_total = True
    else:
        drop_total = False

    g = scenario.groupby(group_columns, as_index=False, observed=True).agg(
        Q0=("Equiv_Volume_Baseline", "sum"),
        Q1=("Equiv_Volume_Simulated", "sum"),
        R0=("Baseline_Revenue", "sum"),
        R1=("Simulated_Revenue", "sum"),
    )

    # Need SKU-level details within each group for Mix and Price
    detail = scenario.copy()

    # Merge Q1 into detail for mix calculation
    detail = detail.merge(g[group_columns + ["Q1"]], on=group_columns, how="left")

    # Quantity shares
    detail["s0"] = detail["Equiv_Volume_Baseline"] / detail.groupby(group_columns)["Equiv_Volume_Baseline"].transform("sum")
    detail["s1"] = detail["Equiv_Volume_Simulated"] / detail.groupby(group_columns)["Equiv_Volume_Simulated"].transform("sum")

    # Price per equivalized volume
    detail["P0_equiv"] = detail["Baseline_Revenue"] / detail["Equiv_Volume_Baseline"].replace(0, np.nan)
    detail["P1_equiv"] = detail["Simulated_Revenue"] / detail["Equiv_Volume_Simulated"].replace(0, np.nan)

    # Volume effect: (Q1 - Q0) * (R0/Q0)
    g["Pbar0"] = g["R0"] / g["Q0"].replace(0, np.nan)
    g["Volume_Effect"] = (g["Q1"] - g["Q0"]) * g["Pbar0"]

    # Mix effect: Q1 * sum( (s1 - s0) * P0 )
    mix_detail = detail.groupby(group_columns, as_index=False, observed=True).apply(
        lambda df: pd.Series({
            "Mix_Effect": df["Q1"].iloc[0] * ((df["s1"] - df["s0"]) * df["P0_equiv"]).sum()
        })
    ).reset_index(drop=True)
    g = g.merge(mix_detail, on=group_columns, how="left")

    # Price effect: sum( V1 * (P1 - P0) )
    price_detail = detail.groupby(group_columns, as_index=False, observed=True).apply(
        lambda df: pd.Series({
            "Price_Effect": (df["Equiv_Volume_Simulated"] * (df["P1_equiv"] - df["P0_equiv"])).sum()
        })
    ).reset_index(drop=True)
    g = g.merge(price_detail, on=group_columns, how="left")

    # Reconciliation check
    g["Decomposition_Sum"] = g["Volume_Effect"] + g["Mix_Effect"] + g["Price_Effect"]
    g["Revenue_Delta"] = g["R1"] - g["R0"]

    # Enforce reconciliation at the group level
    recon_error = g["Decomposition_Sum"] - g["Revenue_Delta"]
    if not np.allclose(recon_error, 0.0, rtol=1e-9, atol=1e-6):
        raise RuntimeError(
            f"Decomposition reconciliation failed: max error = {recon_error.abs().max():.6f}"
        )

    if drop_total:
        g = g.drop(columns=["_total_market"])

    return g


# =============================================================================
# Market Shares
# =============================================================================

def value_shares(
    scenario: pd.DataFrame,
    group_columns: list,
    parent_columns: list | None = None,
) -> pd.DataFrame:
    """
    Calculate revenue shares with explicit and consistent denominators.

    Parameters
    ----------
    scenario : pd.DataFrame
        Simulation output with Baseline_Revenue and Simulated_Revenue.
    group_columns : list
        Columns defining the share numerator groups.
    parent_columns : list, optional
        Columns defining the denominator (parent groups). If None, uses grand total.

    Returns
    -------
    pd.DataFrame with Baseline_Share_Pct, Simulated_Share_Pct, Share_Change_pp.
    """
    grouped = (
        scenario.groupby(group_columns, as_index=False, observed=True)
        .agg(
            Baseline_Revenue=("Baseline_Revenue", "sum"),
            Simulated_Revenue=("Simulated_Revenue", "sum"),
        )
    )

    if parent_columns:
        baseline_denominator = grouped.groupby(
            parent_columns, observed=True
        )["Baseline_Revenue"].transform("sum")
        simulated_denominator = grouped.groupby(
            parent_columns, observed=True
        )["Simulated_Revenue"].transform("sum")
    else:
        baseline_denominator = grouped["Baseline_Revenue"].sum()
        simulated_denominator = grouped["Simulated_Revenue"].sum()

    # Zero denominators mean undefined shares, not zero market shares.
    grouped["Baseline_Share_Pct"] = grouped["Baseline_Revenue"] / baseline_denominator * 100
    grouped["Simulated_Share_Pct"] = grouped["Simulated_Revenue"] / simulated_denominator * 100
    grouped["Share_Change_pp"] = grouped["Simulated_Share_Pct"] - grouped["Baseline_Share_Pct"]

    return grouped.replace([np.inf, -np.inf], np.nan)