"""
Market substitution simulation using a baseline-calibrated share-weighted logit model.

This module implements a same-period, baseline-anchored scenario engine where
price changes can move modeled packs among eligible SKUs within the same
category × retailer group. An outside option captures category expansion/contraction.

Key modeling choices:
- One configurable price-sensitivity parameter (k) and outside-share assumption (a)
- Eligible SKUs: observed positive-baseline-sales cells in the selected period
- Zero-anchor SKUs (no positive history) are excluded from substitution
- Cross-retailer and cross-category switching are not modeled in v1
- The outside option represents modeled purchase opportunities, not observed units/revenue
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd

from .data import CELL_KEYS, PRODUCT_KEYS
from .validation import (
    check_price, check_finite_quantities_revenue, validate_scenario_targets,
    reconcile_revenue,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class MarketConfig:
    """Configuration for the market substitution model."""
    # Price sensitivity parameter (positive, shared across all SKUs)
    # Default is illustrative; not estimated from this dataset
    k: float = 2.0
    
    # Outside share assumption (0 <= a < 1)
    # a = 0: fixed group packs (only reallocation)
    # a > 0: group packs can expand/contract
    # Default is illustrative; not estimated from this dataset
    outside_share: float = 0.15
    
    # Named sensitivity profiles for UI/CLI
    @classmethod
    def get_profile(cls, name: str) -> "MarketConfig":
        profiles = {
            "low": cls(k=1.0, outside_share=0.05),
            "central": cls(k=2.0, outside_share=0.15),
            "high": cls(k=3.5, outside_share=0.30),
        }
        if name not in profiles:
            raise ValueError(f"Unknown profile: {name}. Choose from: {list(profiles.keys())}")
        return profiles[name]


# =============================================================================
# Eligibility and Grouping
# =============================================================================

def identify_eligible_cells(
    baseline: pd.DataFrame,
    price_changes: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify eligible cells for substitution within each category × retailer group.
    
    Eligible = positive baseline sales in the selected period.
    Returns (eligible_mask, group_ids) where group_ids maps each row to its 
    category × retailer group index.
    """
    # Eligible: positive baseline unit sales
    eligible = baseline["Unit_Sales"].to_numpy(float) > 0
    
    # Group by category × retailer
    group_key = baseline["Category"].astype(str) + "|" + baseline["Retailer"].astype(str)
    group_ids = pd.Categorical(group_key).codes
    
    return eligible, group_ids


def build_group_structure(
    baseline: pd.DataFrame,
    eligible: np.ndarray,
    group_ids: np.ndarray,
) -> dict:
    """
    Build group-level structure for the logit calculation.
    
    Returns dict with:
    - n_groups: number of category×retailer groups
    - group_sizes: number of eligible SKUs per group
    - group_offsets: start index of each group in the flattened eligible array
    - group_map: maps each row to its group index (-1 if ineligible)
    """
    n_rows = len(baseline)
    group_map = np.full(n_rows, -1, dtype=int)
    group_map[eligible] = group_ids[eligible]
    
    n_groups = int(group_ids.max()) + 1 if n_rows > 0 else 0
    group_sizes = np.bincount(group_ids[eligible], minlength=n_groups)
    group_offsets = np.concatenate([[0], np.cumsum(group_sizes)[:-1]])
    
    return {
        "n_groups": n_groups,
        "group_sizes": group_sizes,
        "group_offsets": group_offsets,
        "group_map": group_map,
    }


# =============================================================================
# Core Logit Calculation
# =============================================================================

def _logit_step(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    k: float,
    a: float,
    group_map: np.ndarray,
    group_offsets: np.ndarray,
    group_sizes: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core logit share calculation for eligible cells.
    
    Returns (q1, w_j, Z) where:
    - q1: simulated quantities for eligible cells (zeros for ineligible)
    - w_j: unnormalized weights (s_j0 * exp(d_j)) for eligible cells
    - Z: normalization factor per group (w_out + sum(w_j))
    """
    n = len(p0)
    q1 = np.zeros(n, dtype=float)
    w_j = np.zeros(n, dtype=float)
    Z = np.zeros(n_groups, dtype=float)
    
    # Log price changes for eligible cells
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -k * np.log(p1 / p0)
    d[~np.isfinite(d)] = 0.0
    
    # Process each group
    for g in range(n_groups):
        start = group_offsets[g]
        size = group_sizes[g]
        if size == 0:
            continue
        
        idx = slice(start, start + size)
        mask = group_map == g
        
        # Baseline shares within group
        q0_group = q0[mask]
        Q0 = q0_group.sum()
        
        if Q0 <= 0:
            # No baseline sales in this group - leave at zero
            continue
        
        s0 = q0_group / Q0
        s_out = a
        
        # Weights
        w = s0 * np.exp(d[mask])
        w_sum = w.sum()
        
        # Normalization
        Z_g = s_out + w_sum
        Z[g] = Z_g
        
        # Simulated quantities
        M = Q0 / (1 - a) if a < 1 else Q0  # Total modeled opportunities
        q1_group = M * w / Z_g
        
        q1[mask] = q1_group
        w_j[mask] = w
    
    return q1, w_j, Z


def simulate_market_scenario(
    baseline: pd.DataFrame,
    price_changes: dict,
    config: MarketConfig,
) -> pd.DataFrame:
    """
    Simulate a market substitution scenario.
    
    Applies price changes and computes substitution within each 
    category × retailer group using a share-weighted logit model.
    
    Parameters
    ----------
    baseline : DataFrame with baseline period data
    price_changes : dict mapping (Category, Retailer, Brand, SKU) -> new absolute price
    config : MarketConfig with k and outside_share parameters
    
    Returns
    -------
    DataFrame with baseline/simulated prices, quantities, revenue, and group metadata
    """
    # Validate inputs
    if baseline.empty or baseline.duplicated(CELL_KEYS).any():
        raise ValueError("Baseline must contain unique, nonempty SKU-retailer cells.")
    validate_scenario_targets(price_changes, baseline)
    
    # Check price sensitivity is positive
    if config.k <= 0:
        raise ValueError("Price sensitivity k must be positive.")
    if not (0 <= config.outside_share < 1):
        raise ValueError("Outside share must be in [0, 1).")
    
    # Apply price changes
    result = baseline.copy()
    result["Simulated_Price"] = result["Price"].astype(float)
    for key, new_price in price_changes.items():
        mask = np.ones(len(result), dtype=bool)
        for column, value in zip(CELL_KEYS, key):
            mask &= result[column].eq(value).to_numpy()
        result.loc[mask, "Simulated_Price"] = new_price
    result["Simulated_Price"] = pd.to_numeric(result["Simulated_Price"], errors="coerce")
    check_price(result["Simulated_Price"].to_numpy(float), "scenario")
    
    # Identify eligible cells and group structure
    p0 = result["Price"].to_numpy(float)
    p1 = result["Simulated_Price"].to_numpy(float)
    q0 = result["Unit_Sales"].to_numpy(float)
    
    eligible, group_ids = identify_eligible_cells(baseline, price_changes)
    groups = build_group_structure(baseline, eligible, group_ids)
    
    # Run logit calculation
    q1, w_j, Z = _logit_step(
        p0, p1, q0, config.k, config.outside_share,
        groups["group_map"], groups["group_offsets"], 
        groups["group_sizes"], groups["n_groups"]
    )
    
    # Ineligible cells keep baseline quantity
    q1[~eligible] = q0[~eligible]
    
    # Compute revenue
    r1 = p1 * q1
    
    check_finite_quantities_revenue(q1, r1)
    
    # Build output
    result["Baseline_Unit_Sales"] = q0
    result["Baseline_Price"] = p0
    result["Baseline_Revenue"] = result["Sales_Revenue"]
    result["Simulated_Unit_Sales"] = q1
    result["Simulated_Revenue"] = r1
    result["Price_Change_Pct"] = 100 * (p1 / p0 - 1)
    result["Revenue_Delta"] = r1 - result["Baseline_Revenue"]
    
    # Cell-level price/volume effects (accounting decomposition)
    result["Price_Effect"] = (p1 - p0) * q1
    result["Volume_Effect"] = (q1 - q0) * p0
    reconcile_revenue(result["Revenue_Delta"], result["Price_Effect"], result["Volume_Effect"])
    
    # Market mode metadata
    result["Market_Mode"] = True
    result["Market_k"] = config.k
    result["Market_Outside_Share"] = config.outside_share
    result["Eligible"] = eligible
    result["Group_ID"] = group_ids
    
    # Group-level outside quantities
    outside_q = np.zeros(len(result), dtype=float)
    for g in range(groups["n_groups"]):
        if groups["group_sizes"][g] > 0:
            start = groups["group_offsets"][g]
            size = groups["group_sizes"][g]
            mask = groups["group_map"] == g
            q0_group = q0[mask]
            Q0 = q0_group.sum()
            M = Q0 / (1 - config.outside_share) if config.outside_share < 1 else Q0
            Z_g = Z[g]
            outside_q_group = M * config.outside_share / Z_g
            outside_q[mask] = outside_q_group
    result["Outside_Quantity"] = outside_q
    
    return result


# =============================================================================
# Group-Level Aggregation
# =============================================================================

def aggregate_to_groups(
    scenario: pd.DataFrame,
    group_columns: list = None,
) -> pd.DataFrame:
    """
    Aggregate market scenario results to category × retailer groups.
    
    Includes captured quantity, outside quantity, and revenue.
    """
    if group_columns is None:
        group_columns = ["Category", "Retailer"]
    
    g = scenario.groupby(group_columns, as_index=False, observed=True).agg(
        Baseline_Revenue=("Baseline_Revenue", "sum"),
        Simulated_Revenue=("Simulated_Revenue", "sum"),
        Baseline_Unit_Sales=("Baseline_Unit_Sales", "sum"),
        Simulated_Unit_Sales=("Simulated_Unit_Sales", "sum"),
        Outside_Quantity=("Outside_Quantity", "sum"),
        Market_k=("Market_k", "first"),
        Market_Outside_Share=("Market_Outside_Share", "first"),
    )
    
    g["Revenue_Delta"] = g["Simulated_Revenue"] - g["Baseline_Revenue"]
    g["Unit_Sales_Delta"] = g["Simulated_Unit_Sales"] - g["Baseline_Unit_Sales"]
    g["Captured_Quantity"] = g["Simulated_Unit_Sales"]
    g["Total_Opportunities"] = g["Captured_Quantity"] + g["Outside_Quantity"]
    
    return g


# =============================================================================
# Gainers/Losers Report
# =============================================================================

def get_gainers_losers(
    scenario: pd.DataFrame,
    min_delta: float = 1.0,
) -> pd.DataFrame:
    """
    Identify net gainers and losers in the market scenario.
    
    Returns SKUs with meaningful quantity changes, showing the source
    of change (direct price edit vs. substitution spillover).
    """
    result = scenario.copy()
    result["Quantity_Delta"] = result["Simulated_Unit_Sales"] - result["Baseline_Unit_Sales"]
    result["Revenue_Delta"] = result["Simulated_Revenue"] - result["Baseline_Revenue"]
    
    # Mark edited cells
    result["Price_Edited"] = result["Price_Change_Pct"] != 0
    
    # Filter to meaningful changes
    meaningful = result[np.abs(result["Quantity_Delta"]) >= min_delta].copy()
    
    if len(meaningful) == 0:
        return pd.DataFrame()
    
    # Sort: gainers first (by quantity delta desc), then losers
    meaningful = meaningful.sort_values("Quantity_Delta", ascending=False)
    
    return meaningful[[
        "Category", "Retailer", "Brand", "SKU", "Pack_Size",
        "Baseline_Unit_Sales", "Simulated_Unit_Sales", "Quantity_Delta",
        "Baseline_Revenue", "Simulated_Revenue", "Revenue_Delta",
        "Price_Change_Pct", "Price_Edited", "Eligible",
    ]]


# =============================================================================
# Market Revenue Bridge (Accounting)
# =============================================================================

def market_revenue_bridge(scenario: pd.DataFrame) -> dict:
    """
    Build a revenue bridge from baseline to scenario at market level.
    
    Decomposes total revenue change into:
    - Direct price effects on edited cells
    - Direct volume effects on edited cells  
    - Substitution volume effects on unedited eligible cells
    - Outside quantity effects (category expansion/contraction)
    
    This is an accounting allocation, not a causal decomposition.
    """
    r0 = scenario["Baseline_Revenue"].sum()
    r1 = scenario["Simulated_Revenue"].sum()
    
    # Edited cells
    edited_mask = scenario["Price_Change_Pct"] != 0
    edited_price_effect = (scenario.loc[edited_mask, "Price_Effect"]).sum()
    edited_volume_effect = (scenario.loc[edited_mask, "Volume_Effect"]).sum()
    
    # Unedited but eligible cells (substitution effects)
    unedited_eligible = (~edited_mask) & scenario["Eligible"]
    sub_price_effect = (scenario.loc[unedited_eligible, "Price_Effect"]).sum()
    sub_volume_effect = (scenario.loc[unedited_eligible, "Volume_Effect"]).sum()
    
    # Ineligible cells (zero anchor, etc.)
    ineligible = ~scenario["Eligible"]
    ineligible_effect = (scenario.loc[ineligible, "Revenue_Delta"]).sum()
    
    # Verify reconciliation
    total = edited_price_effect + edited_volume_effect + sub_price_effect + sub_volume_effect + ineligible_effect
    if not np.isclose(total, r1 - r0, rtol=1e-9, atol=1e-6):
        raise RuntimeError(f"Bridge reconciliation failed: {total:.6f} != {r1 - r0:.6f}")
    
    return {
        "Baseline_Revenue": r0,
        "Edited_Price_Effect": edited_price_effect,
        "Edited_Volume_Effect": edited_volume_effect,
        "Substitution_Volume_Effect": sub_volume_effect,
        "Substitution_Price_Effect": sub_price_effect,
        "Ineligible_Cells_Effect": ineligible_effect,
        "Scenario_Revenue": r1,
        "Total_Change": r1 - r0,
    }