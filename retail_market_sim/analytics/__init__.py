"""
Historical analytical engine for the Beverage Market Price Simulator.

Provides bottom-up aggregation, seasonal analysis, and exact decompositions.
All totals reconcile to the same lowest-level SKU×retailer cells.
"""
from .aggregates import (
    enrich_with_derived,
    compute_liters,
    compute_unit_value,
    compute_value_per_liter,
    aggregate_to_level,
    compute_all_aggregations,
    compute_shares,
    compute_all_shares,
    compute_period_over_period,
    compute_rolling_averages,
    detect_trend_breaks,
)

from .seasonality import (
    compute_seasonal_profile,
    compute_month_over_month,
    compute_year_over_year,
    compute_seasonal_strength,
    compute_seasonal_decomposition,
    compute_yoy_comparison,
    format_seasonal_profile,
    seasonal_insufficient_years_warning,
    MONTH_NAMES,
)

from .decomposition import (
    classify_cells,
    common_cell_revenue_decomposition,
    bridge_to_actual_totals,
    full_revenue_bridge,
    three_factor_liter_decomposition,
)

__all__ = [
    # Aggregates
    "enrich_with_derived",
    "compute_liters",
    "compute_unit_value",
    "compute_value_per_liter",
    "aggregate_to_level",
    "compute_all_aggregations",
    "compute_shares",
    "compute_all_shares",
    "compute_period_over_period",
    "compute_rolling_averages",
    "detect_trend_breaks",
    # Seasonality
    "compute_seasonal_profile",
    "compute_month_over_month",
    "compute_year_over_year",
    "compute_seasonal_strength",
    "compute_seasonal_decomposition",
    "compute_yoy_comparison",
    "format_seasonal_profile",
    "seasonal_insufficient_years_warning",
    "MONTH_NAMES",
    # Decomposition
    "classify_cells",
    "common_cell_revenue_decomposition",
    "bridge_to_actual_totals",
    "full_revenue_bridge",
    "three_factor_liter_decomposition",
]