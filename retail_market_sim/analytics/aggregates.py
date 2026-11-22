"""
Bottom-up aggregation engine for the Beverage Market Price Simulator.

All market views are calculated from the same lowest-level SKU×retailer cells
so that totals always reconcile exactly.
"""
from typing import Optional
import numpy as np
import pandas as pd

from ..data import CELL_KEYS, PRODUCT_KEYS


# =============================================================================
# Core Aggregation Functions
# =============================================================================

def compute_liters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add liter columns to a DataFrame with SKU-level data.
    
    Requires 'Unit_Sales' and 'Pack_Size' columns.
    Returns DataFrame with 'Liters' column added.
    """
    result = df.copy()
    result["Liters"] = result["Unit_Sales"] * result["Pack_Size"]
    return result


def compute_unit_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add unit value (sales/quantity) column.
    
    Only defined where Unit_Sales > 0.
    Returns DataFrame with 'Unit_Value' column added (NaN where Unit_Sales == 0).
    """
    result = df.copy()
    has_sales = result["Unit_Sales"] > 0
    result["Unit_Value"] = np.where(
        has_sales,
        result["Sales_Revenue"] / result["Unit_Sales"],
        np.nan
    )
    return result


def compute_value_per_liter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add value per liter column.
    
    Only defined where Liters > 0.
    Returns DataFrame with 'Value_Per_Liter' column added.
    """
    result = df.copy()
    result = compute_liters(result)
    has_liters = result["Liters"] > 0
    result["Value_Per_Liter"] = np.where(
        has_liters,
        result["Sales_Revenue"] / result["Liters"],
        np.nan
    )
    return result


def enrich_with_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived measures to a SKU×retailer DataFrame.
    
    Adds: Liters, Unit_Value, Value_Per_Liter
    """
    result = df.copy()
    result = compute_liters(result)
    result = compute_unit_value(result)
    result = compute_value_per_liter(result)
    return result


# =============================================================================
# Multi-Level Aggregation
# =============================================================================

def aggregate_to_level(
    df: pd.DataFrame,
    group_columns: list[str],
    value_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate a DataFrame to a specified grouping level.
    
    All totals are calculated by summing the same bottom-level cells,
    ensuring reconciliation across all levels.
    
    Parameters
    ----------
    df : DataFrame with at least Unit_Sales, Sales_Revenue, Pack_Size
    group_columns : Columns to group by (e.g., ["Category"], ["Category", "Brand"])
                     Use empty list [] for grand total
    value_columns : Optional list of additional columns to sum
    
    Returns
    -------
    Aggregated DataFrame with Units, Liters, Revenue, and derived measures
    """
    if value_columns is None:
        value_columns = []
    
    # Ensure derived columns exist
    df = enrich_with_derived(df)
    
    # Base aggregation columns
    agg_dict = {
        "Unit_Sales": "sum",
        "Sales_Revenue": "sum",
    }
    
    # We need Pack_Size to compute liters at the aggregate level
    # Use "first" since all rows in a group should have consistent Pack_Size per SKU
    agg_dict["Pack_Size"] = "first"
    
    # Add any extra value columns
    for col in value_columns:
        if col in df.columns:
            agg_dict[col] = "sum"
    
    # Handle empty group_columns (grand total) by using a temporary constant column
    if not group_columns:
        df = df.copy()
        df["_grand_total"] = 1
        group_cols = ["_grand_total"]
        drop_total = True
    else:
        group_cols = group_columns
        drop_total = False
    
    result = df.groupby(group_cols, as_index=False, observed=True).agg(agg_dict)
    
    if drop_total:
        result = result.drop(columns=["_grand_total"])
    
    # Recompute derived measures at aggregate level
    result = compute_liters(result)
    result = compute_unit_value(result)
    result = compute_value_per_liter(result)
    
    return result


def compute_all_aggregations(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute all standard aggregation levels from bottom-level data.
    
    Returns dict with keys:
    - 'total_market': Grand totals
    - 'category': By category
    - 'brand': By category + brand
    - 'sku': By category + brand + SKU
    - 'retailer': By retailer
    - 'category_retailer': By category + retailer
    - 'brand_retailer': By category + brand + retailer
    - 'sku_retailer': By SKU + retailer (cell level)
    - 'pack_size': By pack size
    - 'category_pack_size': By category + pack size
    """
    df = enrich_with_derived(df)
    
    aggregations = {
        "total_market": aggregate_to_level(df, []),
        "category": aggregate_to_level(df, ["Category"]),
        "brand": aggregate_to_level(df, ["Category", "Brand"]),
        "sku": aggregate_to_level(df, PRODUCT_KEYS),
        "retailer": aggregate_to_level(df, ["Retailer"]),
        "category_retailer": aggregate_to_level(df, ["Category", "Retailer"]),
        "brand_retailer": aggregate_to_level(df, ["Category", "Brand", "Retailer"]),
        "sku_retailer": aggregate_to_level(df, CELL_KEYS),
        "pack_size": aggregate_to_level(df, ["Pack_Size"]),
        "category_pack_size": aggregate_to_level(df, ["Category", "Pack_Size"]),
    }
    
    return aggregations


# =============================================================================
# Market Share Calculations
# =============================================================================

def compute_shares(
    df: pd.DataFrame,
    group_columns: list[str],
    parent_columns: Optional[list[str]] = None,
    value_column: str = "Sales_Revenue",
    share_suffix: str = "_Share_Pct",
) -> pd.DataFrame:
    """
    Compute market shares for grouped data.
    
    Parameters
    ----------
    df : Aggregated DataFrame with value_column
    group_columns : Columns defining the segments
    parent_columns : Columns defining the denominator (e.g., ["Category"] for brand shares within category)
    value_column : Column to compute shares from (default: Sales_Revenue)
    share_suffix : Suffix for share column name
    
    Returns
    -------
    DataFrame with Baseline_Share_Pct and Simulated_Share_Pct if both columns exist,
    or just Share_Pct for single-period data
    """
    result = df.copy()
    
    if parent_columns:
        # Conditional shares within parent
        baseline_denom = result.groupby(parent_columns, observed=True)[value_column].transform("sum")
        if f"Simulated_{value_column}" in result.columns:
            simulated_denom = result.groupby(parent_columns, observed=True)[f"Simulated_{value_column}"].transform("sum")
        else:
            simulated_denom = None
    else:
        # Total market shares
        baseline_denom = result[value_column].sum()
        if f"Simulated_{value_column}" in result.columns:
            simulated_denom = result[f"Simulated_{value_column}"].sum()
        else:
            simulated_denom = None
    
    # Compute shares
    result[f"Baseline{share_suffix}"] = (result[value_column] / baseline_denom * 100).replace([np.inf, -np.inf], np.nan)
    
    if simulated_denom is not None:
        result[f"Simulated{share_suffix}"] = (result[f"Simulated_{value_column}"] / simulated_denom * 100).replace([np.inf, -np.inf], np.nan)
        result[f"Share_Change_pp"] = result[f"Simulated{share_suffix}"] - result[f"Baseline{share_suffix}"]
    
    return result


def compute_all_shares(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute all standard share tables."""
    shares = {}
    
    # Determine which columns exist
    has_category = "Category" in df.columns
    has_brand = "Brand" in df.columns
    has_sku = "SKU" in df.columns
    has_retailer = "Retailer" in df.columns
    
    # Total market shares
    if has_category:
        shares["total_category"] = compute_shares(df, ["Category"])
    if has_category and has_brand:
        shares["total_brand"] = compute_shares(df, ["Category", "Brand"])
    if has_retailer:
        shares["total_retailer"] = compute_shares(df, ["Retailer"])
    if has_category and has_brand and has_sku:
        shares["total_sku"] = compute_shares(df, CELL_KEYS)
    
    # Conditional shares (only if parent columns exist)
    if has_category and has_brand:
        shares["brand_in_category"] = compute_shares(df, ["Category", "Brand"], parent_columns=["Category"])
    if has_category and has_retailer:
        shares["retailer_in_category"] = compute_shares(df, ["Category", "Retailer"], parent_columns=["Category"])
    if has_category and has_brand and has_sku:
        shares["sku_in_brand"] = compute_shares(df, ["Category", "Brand", "SKU"], parent_columns=["Category", "Brand"])
    if has_retailer and has_category and has_sku:
        shares["sku_in_retailer"] = compute_shares(df, ["Retailer", "Category", "SKU"], parent_columns=["Retailer"])
    if has_category and has_retailer and has_sku:
        shares["sku_in_category_retailer"] = compute_shares(
            df, ["Category", "Retailer", "SKU"], parent_columns=["Category", "Retailer"]
        )
    
    return shares


# =============================================================================
# Trend & Seasonality Analysis
# =============================================================================

def compute_period_over_period(
    df: pd.DataFrame,
    group_columns: list[str],
    periods: int = 1,
    value_column: str = "Sales_Revenue",
) -> pd.DataFrame:
    """
    Compute period-over-period changes (e.g., month-over-month, year-over-year).
    
    Returns DataFrame with absolute and percentage changes.
    """
    df = df.sort_values(group_columns + ["Period"]).copy()
    
    # Shift within each group
    shifted = df.groupby(group_columns, observed=True)[value_column].shift(periods)
    
    result = df.copy()
    result[f"{value_column}_prev"] = shifted
    result[f"{value_column}_abs_change"] = result[value_column] - result[f"{value_column}_prev"]
    result[f"{value_column}_pct_change"] = np.where(
        shifted != 0,
        (result[value_column] - shifted) / shifted * 100,
        np.nan
    )
    
    return result


def compute_rolling_averages(
    df: pd.DataFrame,
    group_columns: list[str],
    window: int = 3,
    value_column: str = "Sales_Revenue",
) -> pd.DataFrame:
    """Compute rolling averages within each group."""
    df = df.sort_values(group_columns + ["Period"]).copy()
    
    result = df.copy()
    result[f"{value_column}_rolling_{window}m"] = (
        df.groupby(group_columns, observed=True)[value_column]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    
    return result


def detect_trend_breaks(
    df: pd.DataFrame,
    group_columns: list[str],
    value_column: str = "Sales_Revenue",
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Detect potential trend breaks using rolling z-scores.
    
    Returns rows where the deviation from rolling mean exceeds threshold.
    """
    df = df.sort_values(group_columns + ["Period"]).copy()
    
    rolling_mean = df.groupby(group_columns, observed=True)[value_column].transform(
        lambda x: x.rolling(6, min_periods=3).mean()
    )
    rolling_std = df.groupby(group_columns, observed=True)[value_column].transform(
        lambda x: x.rolling(6, min_periods=3).std()
    )
    
    result = df.copy()
    result[f"{value_column}_zscore"] = np.where(
        rolling_std > 0,
        (result[value_column] - rolling_mean) / rolling_std,
        0
    )
    result["is_trend_break"] = np.abs(result[f"{value_column}_zscore"]) > threshold
    
    return result