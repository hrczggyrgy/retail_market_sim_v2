"""
Descriptive seasonal analysis for the Beverage Market Price Simulator.

Provides seasonality profiles, year-over-year comparisons, and seasonal strength metrics.
Does not claim causal seasonality - only describes observed patterns.
"""
from typing import Optional
import numpy as np
import pandas as pd

from ..data import CELL_KEYS, PRODUCT_KEYS
from .aggregates import enrich_with_derived


# =============================================================================
# Seasonal Profile Computation
# =============================================================================

def compute_seasonal_profile(
    df: pd.DataFrame,
    group_columns: list[str],
    value_column: str = "Sales_Revenue",
    period_column: str = "Period",
    season_column: str = "Season",
) -> pd.DataFrame:
    """
    Compute average seasonal profile for each group.
    
    Returns DataFrame with seasonal indices (average value per season / overall average).
    Season index of 1.0 = average; 1.2 = 20% above average in that season.
    
    Warning: A twelve-month pattern from a single year is not evidence of recurring seasonality.
    """
    df = df.copy()
    
    # Compute average per group per season
    seasonal_avg = df.groupby(group_columns + [season_column], observed=True)[value_column].mean().reset_index()
    seasonal_avg = seasonal_avg.rename(columns={value_column: f"{value_column}_seasonal_avg"})
    
    # Compute overall average per group
    overall_avg = df.groupby(group_columns, observed=True)[value_column].mean().reset_index()
    overall_avg = overall_avg.rename(columns={value_column: f"{value_column}_overall_avg"})
    
    # Merge and compute index
    profile = seasonal_avg.merge(overall_avg, on=group_columns, how="left", validate="many_to_one")
    profile[f"{value_column}_seasonal_index"] = np.where(
        profile[f"{value_column}_overall_avg"] != 0,
        profile[f"{value_column}_seasonal_avg"] / profile[f"{value_column}_overall_avg"],
        np.nan
    )
    
    # Add warning for insufficient years
    n_years = df[period_column].nunique() / 12.0
    if n_years < 2:
        # This is metadata - caller should display warning
        pass
    
    return profile


def compute_month_over_month(
    df: pd.DataFrame,
    group_columns: list[str],
    value_column: str = "Sales_Revenue",
    period_column: str = "Period",
) -> pd.DataFrame:
    """Compute month-over-month changes within each group."""
    return compute_period_over_period(
        df, group_columns, periods=1, value_column=value_column, period_column=period_column
    )


def compute_year_over_year(
    df: pd.DataFrame,
    group_columns: list[str],
    value_column: str = "Sales_Revenue",
    period_column: str = "Period",
) -> pd.DataFrame:
    """Compute year-over-year changes (12-month lag) within each group."""
    return compute_period_over_period(
        df, group_columns, periods=12, value_column=value_column, period_column=period_column
    )


def compute_period_over_period(
    df: pd.DataFrame,
    group_columns: list[str],
    periods: int = 1,
    value_column: str = "Sales_Revenue",
    period_column: str = "Period",
) -> pd.DataFrame:
    """Compute period-over-period changes within each group."""
    df = df.sort_values(group_columns + [period_column]).copy()
    
    shifted = df.groupby(group_columns, observed=True)[value_column].shift(periods)
    
    result = df.copy()
    result[f"{value_column}_prev_{periods}"] = shifted
    result[f"{value_column}_abs_change_{periods}"] = result[value_column] - shifted
    result[f"{value_column}_pct_change_{periods}"] = np.where(
        shifted != 0,
        (result[value_column] - shifted) / shifted * 100,
        np.nan
    )
    
    return result


# =============================================================================
# Seasonal Strength Metrics
# =============================================================================

def compute_seasonal_strength(
    df: pd.DataFrame,
    group_columns: list[str],
    value_column: str = "Sales_Revenue",
    season_column: str = "Season",
) -> pd.DataFrame:
    """
    Compute seasonal strength metrics for each group.
    
    Metrics:
    - peak_to_trough_ratio: max seasonal avg / min seasonal avg
    - amplitude: (max - min) / mean
    - seasonal_amplitude_pct: (max - min) / mean * 100
    - peak_month: month with highest average
    - trough_month: month with lowest average
    """
    # Get seasonal profile
    profile = compute_seasonal_profile(df, group_columns, value_column, season_column=season_column)
    
    # Aggregate metrics per group
    metrics = profile.groupby(group_columns, observed=True).agg(
        peak_month=(f"{value_column}_seasonal_avg", "idxmax"),
        trough_month=(f"{value_column}_seasonal_avg", "idxmin"),
        max_val=(f"{value_column}_seasonal_avg", "max"),
        min_val=(f"{value_column}_seasonal_avg", "min"),
        mean_val=(f"{value_column}_seasonal_avg", "mean"),
    ).reset_index()
    
    # Get actual month numbers
    season_map = profile.set_index(group_columns + [season_column])[f"{value_column}_seasonal_avg"]
    
    def get_peak_month(row):
        key = tuple(row[col] for col in group_columns)
        subset = season_map.loc[key]
        return int(subset.idxmax())
    
    def get_trough_month(row):
        key = tuple(row[col] for col in group_columns)
        subset = season_map.loc[key]
        return int(subset.idxmin())
    
    metrics["peak_month"] = metrics.apply(get_peak_month, axis=1)
    metrics["trough_month"] = metrics.apply(get_trough_month, axis=1)
    
    # Compute ratios
    metrics["peak_to_trough_ratio"] = np.where(
        metrics["min_val"] > 0,
        metrics["max_val"] / metrics["min_val"],
        np.nan
    )
    metrics["amplitude"] = np.where(
        metrics["mean_val"] > 0,
        (metrics["max_val"] - metrics["min_val"]) / metrics["mean_val"],
        np.nan
    )
    metrics["seasonal_amplitude_pct"] = metrics["amplitude"] * 100
    
    # Clean up
    metrics = metrics.drop(columns=["max_val", "min_val", "mean_val"])
    
    return metrics


def compute_seasonal_decomposition(
    df: pd.DataFrame,
    group_columns: list[str],
    value_column: str = "Sales_Revenue",
    period_column: str = "Period",
    season_column: str = "Season",
    trend_window: int = 12,
) -> pd.DataFrame:
    """
    Simple additive seasonal decomposition: Value = Trend + Seasonal + Residual.
    
    Uses rolling trend extraction (centered moving average).
    """
    df = df.sort_values(group_columns + [period_column]).copy()
    
    # Compute trend using centered moving average
    def get_trend(x):
        return x.rolling(trend_window, center=True, min_periods=1).mean()
    
    df["trend"] = df.groupby(group_columns, observed=True)[value_column].transform(get_trend)
    
    # Detrend
    df["detrended"] = df[value_column] - df["trend"]
    
    # Seasonal component (average detrended value per season)
    seasonal = df.groupby(group_columns + [season_column], observed=True)["detrended"].mean().reset_index()
    seasonal = seasonal.rename(columns={"detrended": "seasonal_component"})
    
    # Merge back
    result = df.merge(seasonal, on=group_columns + [season_column], how="left")
    
    # Residual
    result["residual"] = result["detrended"] - result["seasonal_component"]
    
    # Verify: Value = Trend + Seasonal + Residual (approximately)
    result["reconstructed"] = result["trend"] + result["seasonal_component"] + result["residual"]
    
    return result


# =============================================================================
# Year-over-Year Comparison
# =============================================================================

def compute_yoy_comparison(
    df: pd.DataFrame,
    group_columns: list[str],
    value_column: str = "Sales_Revenue",
    period_column: str = "Period",
) -> pd.DataFrame:
    """
    Compute year-over-year comparison with proper period alignment.
    
    Returns DataFrame with current year, prior year, absolute and percentage change.
    """
    df = df.copy()
    
    # Add year column
    df["year"] = df[period_column] // 12 + 1
    
    # Pivot to get current and prior year side by side
    # For each group, we need to match month 1 to month 13, month 2 to month 14, etc.
    
    # Create a key for matching: (group, month_in_year)
    df["month_in_year"] = df["year"].apply(lambda x: (x - 1) * 12)  # Not quite right
    # Better: use period directly for matching
    df["match_key"] = df[period_column] - 12  # Current period matches with period-12
    
    # Self-merge to get prior year
    prior = df[group_columns + ["match_key", value_column]].copy()
    prior = prior.rename(columns={value_column: f"{value_column}_prior_year"})
    
    result = df.merge(
        prior,
        left_on=group_columns + [period_column],
        right_on=group_columns + ["match_key"],
        how="left"
    )
    
    # Compute changes
    result[f"{value_column}_yoy_abs"] = result[value_column] - result[f"{value_column}_prior_year"]
    result[f"{value_column}_yoy_pct"] = np.where(
        result[f"{value_column}_prior_year"] != 0,
        (result[value_column] - result[f"{value_column}_prior_year"]) / result[f"{value_column}_prior_year"] * 100,
        np.nan
    )
    
    return result


# =============================================================================
# Seasonal Display Helpers
# =============================================================================

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

def format_seasonal_profile(profile: pd.DataFrame, season_column: str = "Season") -> pd.DataFrame:
    """Format seasonal profile for display with month names."""
    result = profile.copy()
    if season_column in result.columns:
        result["Month_Name"] = result[season_column].map(MONTH_NAMES)
    return result


def seasonal_insufficient_years_warning(n_years: float) -> str:
    """Generate warning text for insufficient years of data."""
    if n_years < 1:
        return "Less than one year of data; no seasonal analysis possible."
    elif n_years < 2:
        return "Only one year of data; seasonal patterns cannot be distinguished from trend or noise."
    elif n_years < 3:
        return "Only two years of data; seasonal patterns are weakly identified and may not recur."
    return ""