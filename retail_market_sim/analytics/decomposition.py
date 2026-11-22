"""
Exact historical revenue decomposition with entry/exit bridging.

Extends the core three-factor decomposition with:
- Common-cell decomposition (volume/mix/price)
- Entry/exit/unpriceable bridge to actual totals
- Zero accounting residual guarantee
- Clear labeling: descriptive, not causal
"""
from typing import Optional, Literal
import numpy as np
import pandas as pd

from ..data import CELL_KEYS, PRODUCT_KEYS
from .aggregates import enrich_with_derived


# =============================================================================
# Cell Classification
# =============================================================================

def classify_cells(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    cell_keys: list[str] = CELL_KEYS,
) -> pd.DataFrame:
    """
    Classify cells by their presence and priceability in both periods.
    
    Returns DataFrame with cell_key and classification:
    - 'common_priceable': positive sales + valid price in both periods
    - 'common_zero': zero sales in both periods
    - 'entry': not in baseline, positive in comparison
    - 'exit': positive in baseline, not in comparison
    - 'unpriceable_baseline': positive in baseline, no valid price
    - 'unpriceable_comparison': positive in comparison, no valid price
    - 'zero_to_positive': zero in baseline, positive in comparison
    - 'positive_to_zero': positive in baseline, zero in comparison
    """
    # Merge on cell keys
    merged = baseline[cell_keys + ["Unit_Sales", "Sales_Revenue", "Price", "Pack_Size"]].merge(
        comparison[cell_keys + ["Unit_Sales", "Sales_Revenue", "Price", "Pack_Size"]],
        on=cell_keys, how="outer", suffixes=("_0", "_1"), indicator=True
    )
    
    # Fill NaN for missing periods
    for col in ["Unit_Sales", "Sales_Revenue", "Price", "Pack_Size"]:
        merged[f"{col}_0"] = merged[f"{col}_0"].fillna(0)
        merged[f"{col}_1"] = merged[f"{col}_1"].fillna(0)
    
    # Determine priceability (positive sales and valid price)
    merged["priceable_0"] = (merged["Unit_Sales_0"] > 0) & (merged["Price_0"] > 0) & np.isfinite(merged["Price_0"])
    merged["priceable_1"] = (merged["Unit_Sales_1"] > 0) & (merged["Price_1"] > 0) & np.isfinite(merged["Price_1"])
    
    def classify(row):
        if row["_merge"] == "both":
            if row["priceable_0"] and row["priceable_1"]:
                return "common_priceable"
            elif row["Unit_Sales_0"] == 0 and row["Unit_Sales_1"] == 0:
                return "common_zero"
            elif row["priceable_0"] and row["Unit_Sales_1"] > 0 and not row["priceable_1"]:
                return "unpriceable_comparison"
            elif row["priceable_1"] and row["Unit_Sales_0"] > 0 and not row["priceable_0"]:
                return "unpriceable_baseline"
            elif row["Unit_Sales_0"] == 0 and row["Unit_Sales_1"] > 0:
                return "zero_to_positive"
            elif row["Unit_Sales_0"] > 0 and row["Unit_Sales_1"] == 0:
                return "positive_to_zero"
            else:
                return "other_common"
        elif row["_merge"] == "left_only":
            if row["priceable_0"]:
                return "exit"
            else:
                return "exit_unpriceable"
        elif row["_merge"] == "right_only":
            if row["priceable_1"]:
                return "entry"
            else:
                return "entry_unpriceable"
        return "unknown"
    
    merged["classification"] = merged.apply(classify, axis=1)
    
    return merged[CELL_KEYS + ["classification", "_merge"]].copy()


# =============================================================================
# Common-Cell Revenue Decomposition
# =============================================================================

def common_cell_revenue_decomposition(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    cell_keys: list[str] = CELL_KEYS,
) -> tuple[pd.DataFrame, dict]:
    """
    Exact revenue decomposition on common, priceable cells.
    
    Formula (ordered: Volume → Mix → Price):
    ΔRevenue = (Q₁ - Q₀) × P̄₀ + Q₁ × Σ[(w₁ - w₀) × P₀] + Σ[q₁ × (p₁ - p₀)]
    
    Where:
    - Q₀, Q₁ = total volume in baseline/comparison
    - w₀, w₁ = volume shares in baseline/comparison
    - P₀, P₁ = unit values (revenue/volume)
    - P̄₀ = baseline average unit value
    
    Returns (detail_df, summary_dict) where detail_df has per-cell components
    and summary_dict has aggregate totals.
    """
    # Filter to common, priceable cells
    baseline_c = baseline.copy()
    comparison_c = comparison.copy()
    
    # Add cell identifier
    for df in [baseline_c, comparison_c]:
        df["_cell_key"] = df[cell_keys].apply(tuple, axis=1)
    
    # Merge
    merged = baseline_c.merge(
        comparison_c,
        on=cell_keys + ["_cell_key"],
        how="inner",
        suffixes=("_0", "_1"),
        validate="one_to_one"
    )
    
    # Filter to priceable in both
    priceable = (
        (merged["Unit_Sales_0"] > 0) & (merged["Price_0"] > 0) &
        (merged["Unit_Sales_1"] > 0) & (merged["Price_1"] > 0)
    )
    common = merged[priceable].copy()
    
    if len(common) == 0:
        return pd.DataFrame(), {
            "common_cells": 0,
            "volume_effect": 0.0,
            "mix_effect": 0.0,
            "price_effect": 0.0,
            "total_effect": 0.0,
            "baseline_revenue": 0.0,
            "comparison_revenue": 0.0,
        }
    
    # Compute shares and prices
    q0 = common["Unit_Sales_0"].values
    q1 = common["Unit_Sales_1"].values
    r0 = common["Sales_Revenue_0"].values
    r1 = common["Sales_Revenue_1"].values
    
    p0 = r0 / q0
    p1 = r1 / q1
    
    Q0 = q0.sum()
    Q1 = q1.sum()
    
    w0 = q0 / Q0
    w1 = q1 / Q1
    
    Pbar0 = r0.sum() / Q0 if Q0 > 0 else 0
    
    # Volume effect: (Q1 - Q0) * Pbar0
    volume_effect = (Q1 - Q0) * Pbar0
    
    # Mix effect: Q1 * Σ[(w1 - w0) * P0]
    mix_effect = Q1 * np.sum((w1 - w0) * p0)
    
    # Price effect: Σ[q1 * (p1 - p0)]
    price_effect = np.sum(q1 * (p1 - p0))
    
    # Per-cell breakdown
    detail = common[cell_keys].copy()
    detail["q0"] = q0
    detail["q1"] = q1
    detail["r0"] = r0
    detail["r1"] = r1
    detail["p0"] = p0
    detail["p1"] = p1
    detail["w0"] = w0
    detail["w1"] = w1
    detail["volume_effect_cell"] = (q1 - q0) * Pbar0
    detail["mix_effect_cell"] = Q1 * (w1 - w0) * p0
    detail["price_effect_cell"] = q1 * (p1 - p0)
    detail["total_effect_cell"] = detail["volume_effect_cell"] + detail["mix_effect_cell"] + detail["price_effect_cell"]
    detail["revenue_delta_cell"] = r1 - r0
    
    # Verify reconciliation
    total_effect = volume_effect + mix_effect + price_effect
    actual_delta = r1.sum() - r0.sum()
    
    if not np.isclose(total_effect, actual_delta, rtol=1e-9, atol=1e-6):
        raise RuntimeError(f"Decomposition reconciliation failed: {total_effect:.6f} != {actual_delta:.6f}")
    
    summary = {
        "common_cells": len(common),
        "volume_effect": float(volume_effect),
        "mix_effect": float(mix_effect),
        "price_effect": float(price_effect),
        "total_effect": float(total_effect),
        "baseline_revenue": float(r0.sum()),
        "comparison_revenue": float(r1.sum()),
    }
    
    return detail, summary


# =============================================================================
# Bridge to Actual Totals (Entries/Exits/Unpriceable)
# =============================================================================

def bridge_to_actual_totals(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    cell_keys: list[str] = CELL_KEYS,
) -> pd.DataFrame:
    """
    Bridge from common-cell decomposition to actual market totals.
    
    Computes revenue contributions from:
    - Common priceable cells (from decomposition)
    - Common zero cells (zero revenue both periods)
    - Entry cells (new in comparison)
    - Exit cells (present in baseline only)
    - Unpriceable cells (positive sales but no valid price)
    - Zero-to-positive / positive-to-zero transitions
    
    Returns DataFrame with one row per classification and its revenue contribution.
    """
    # Classify all cells
    classified = classify_cells(baseline, comparison, cell_keys)
    
    # Merge with baseline and comparison data
    baseline_agg = baseline[cell_keys + ["Sales_Revenue"]].copy()
    baseline_agg = baseline_agg.rename(columns={"Sales_Revenue": "Revenue_0"})
    
    comparison_agg = comparison[cell_keys + ["Sales_Revenue"]].copy()
    comparison_agg = comparison_agg.rename(columns={"Sales_Revenue": "Revenue_1"})
    
    merged = classified.merge(baseline_agg, on=cell_keys, how="left")
    merged = merged.merge(comparison_agg, on=cell_keys, how="left")
    
    merged["Revenue_0"] = merged["Revenue_0"].fillna(0)
    merged["Revenue_1"] = merged["Revenue_1"].fillna(0)
    merged["Revenue_Delta"] = merged["Revenue_1"] - merged["Revenue_0"]
    
    # Summarize by classification
    bridge = merged.groupby("classification", observed=True).agg(
        cell_count=("classification", "size"),
        Revenue_0=("Revenue_0", "sum"),
        Revenue_1=("Revenue_1", "sum"),
        Revenue_Delta=("Revenue_Delta", "sum"),
    ).reset_index()
    
    # Order classifications logically
    order = [
        "common_priceable", "common_zero",
        "entry", "exit",
        "entry_unpriceable", "exit_unpriceable",
        "unpriceable_baseline", "unpriceable_comparison",
        "zero_to_positive", "positive_to_zero",
        "other_common"
    ]
    
    bridge["sort_order"] = bridge["classification"].map({c: i for i, c in enumerate(order)})
    bridge = bridge.sort_values("sort_order").drop(columns=["sort_order"])
    
    return bridge


def full_revenue_bridge(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    cell_keys: list[str] = CELL_KEYS,
) -> dict:
    """
    Complete revenue bridge from baseline to comparison.
    
    Combines common-cell decomposition with entry/exit bridge.
    Returns dict with all components and reconciliation check.
    """
    # Common cell decomposition
    detail, common_summary = common_cell_revenue_decomposition(baseline, comparison, cell_keys)
    
    # Full bridge
    bridge = bridge_to_actual_totals(baseline, comparison, cell_keys)
    
    # Common cells total from bridge
    common_bridge = bridge[bridge["classification"] == "common_priceable"]
    common_rev_0 = common_bridge["Revenue_0"].sum() if len(common_bridge) > 0 else 0
    common_rev_1 = common_bridge["Revenue_1"].sum() if len(common_bridge) > 0 else 0
    
    # Other classifications
    other = bridge[bridge["classification"] != "common_priceable"]
    other_rev_0 = other["Revenue_0"].sum()
    other_rev_1 = other["Revenue_1"].sum()
    
    # Totals
    baseline_total = baseline["Sales_Revenue"].sum()
    comparison_total = comparison["Sales_Revenue"].sum()
    
    # Verify: common + other = total
    check_0 = common_rev_0 + other_rev_0
    check_1 = common_rev_1 + other_rev_1
    
    if not np.isclose(check_0, baseline_total, rtol=1e-9, atol=1e-6):
        raise RuntimeError(f"Bridge check 0 failed: {check_0:.6f} != {baseline_total:.6f}")
    if not np.isclose(check_1, comparison_total, rtol=1e-9, atol=1e-6):
        raise RuntimeError(f"Bridge check 1 failed: {check_1:.6f} != {comparison_total:.6f}")
    
    # Build complete bridge components
    components = []
    
    # Common cells (decomposed)
    if common_summary["common_cells"] > 0:
        components.append({
            "label": "Common Priceable: Volume Effect",
            "value": common_summary["volume_effect"],
            "description": "Volume change at baseline average price",
            "provenance": "descriptive"
        })
        components.append({
            "label": "Common Priceable: Mix Effect",
            "value": common_summary["mix_effect"],
            "description": "Share shifts at baseline prices",
            "provenance": "descriptive"
        })
        components.append({
            "label": "Common Priceable: Price Effect",
            "value": common_summary["price_effect"],
            "description": "Price changes at comparison volumes",
            "provenance": "descriptive"
        })
    
    # Other classifications
    for _, row in other.iterrows():
        components.append({
            "label": f"{row['classification'].replace('_', ' ').title()}",
            "value": row["Revenue_Delta"],
            "description": f"{row['cell_count']} cells, {row['Revenue_0']:.0f} -> {row['Revenue_1']:.0f}",
            "provenance": "descriptive"
        })
    
    # Verification
    total_modeled = sum(c["value"] for c in components)
    actual_delta = comparison_total - baseline_total
    
    return {
        "baseline_revenue": baseline_total,
        "comparison_revenue": comparison_total,
        "total_change": actual_delta,
        "components": components,
        "reconciliation_check": np.isclose(total_modeled, actual_delta, rtol=1e-9, atol=1e-6),
        "detail": detail,
        "bridge": bridge,
    }


# =============================================================================
# Pack Size / Liter Decomposition (from core module)
# =============================================================================

def _pack_size_to_factor(pack_size) -> float:
    """Return validated litres per pack; invalid units are rejected, never defaulted."""
    value = pd.to_numeric(pd.Series([pack_size]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"Pack_Size must be a finite positive litres-per-pack value; got {pack_size!r}.")
    return float(value)


def _ensure_equiv_volumes(df: pd.DataFrame) -> pd.DataFrame:
    if "Equiv_Volume_Baseline" not in df.columns:
        df = df.copy()
        litres = df["Pack_Size"].map(_pack_size_to_factor)
        df["Equiv_Volume_Baseline"] = litres * df["Baseline_Unit_Sales"]
        df["Equiv_Volume_Simulated"] = litres * df["Simulated_Unit_Sales"]
    return df


def three_factor_liter_decomposition(
    scenario: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """
    Exact sequential Volume -> Mix -> Price attribution using litres.
    
    For scenario results (has Baseline/Simulated columns).
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
        raise RuntimeError(f"Liter decomposition reconciliation failed: max error = {error.abs().max():.6f}")
    
    if drop_total:
        g = g.drop(columns=["_total_market"])
    
    return g