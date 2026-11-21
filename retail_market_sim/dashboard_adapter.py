"""Translate the dashboard's editable matrix into the core scenario contract."""
import numpy as np
import pandas as pd

from .data import CELL_KEYS, PRODUCT_KEYS
from .validation import check_price


def matrix_to_price_changes(
    baseline: pd.DataFrame,
    edited_matrix: pd.DataFrame,
    retailer_columns: dict[str, str],
) -> dict[tuple, float]:
    """Validate existing cells; ignore prices entered for absent listings."""
    required = set(PRODUCT_KEYS) | set(retailer_columns)
    missing = required - set(edited_matrix.columns)
    if missing:
        raise ValueError(f"Price matrix is missing columns: {sorted(missing)}")
    if baseline.duplicated(CELL_KEYS).any():
        raise ValueError("Baseline contains duplicate SKU-retailer cells.")
    if edited_matrix.duplicated(PRODUCT_KEYS).any():
        raise ValueError("Price matrix contains duplicate product rows.")
    edits = edited_matrix.melt(
        id_vars=PRODUCT_KEYS,
        value_vars=list(retailer_columns),
        var_name="_retailer_column",
        value_name="_edited_price",
    )
    edits["Retailer"] = edits["_retailer_column"].map(retailer_columns)
    matched = baseline[CELL_KEYS + ["Price"]].merge(
        edits[CELL_KEYS + ["_edited_price"]],
        on=CELL_KEYS, how="left", validate="one_to_one",
    )
    proposed = pd.to_numeric(matched["_edited_price"], errors="coerce").to_numpy(float)
    check_price(proposed, "price editor")
    changed = ~np.isclose(proposed, matched["Price"].to_numpy(float), rtol=1e-9, atol=1e-10)
    return {
        tuple(key): float(price)
        for key, price in zip(
            matched.loc[changed, CELL_KEYS].itertuples(index=False, name=None),
            proposed[changed],
        )
    }


def _compute_price_support_status(
    simulated_price: np.ndarray,
    baseline_price: np.ndarray,
    hist_min: np.ndarray,
    hist_max: np.ndarray,
    baseline_unit_sales: np.ndarray,
    elasticity: np.ndarray,
) -> pd.DataFrame:
    """Compute detailed price support status for each cell.
    
    Returns DataFrame with columns:
    - Support_Status: 'in_range', 'out_of_range', 'unknown'
    - Historical_Min_Price, Historical_Max_Price
    - Price_Change_Pct
    - Elasticity
    - Zero_Anchor: bool
    - Price_Ratio_Log: log(P1/P0) for numerical stability
    """
    n = len(simulated_price)
    support_status = np.full(n, "unknown", dtype=object)
    outside_observed = np.zeros(n, dtype=bool)
    zero_anchor = baseline_unit_sales == 0
    
    # Compute log price ratio for numerical stability
    with np.errstate(divide='ignore', invalid='ignore'):
        price_ratio_log = np.log(simulated_price / baseline_price)
    price_ratio_log[~np.isfinite(price_ratio_log)] = 0.0
    
    # For cells with historical data
    has_history = ~np.isnan(hist_min) & ~np.isnan(hist_max)
    support_status[has_history] = "in_range"
    
    # Check if outside historical range (with small tolerance)
    outside = has_history & (
        (simulated_price < hist_min - 1e-10) | 
        (simulated_price > hist_max + 1e-10)
    )
    support_status[outside] = "out_of_range"
    outside_observed[outside] = True
    
    # Zero anchor cells have unknown support (no historical prices)
    support_status[zero_anchor] = "unknown"
    
    # Price change percentage
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_change = 100 * (simulated_price / baseline_price - 1)
    pct_change[~np.isfinite(pct_change)] = 0.0
    
    return pd.DataFrame({
        "Support_Status": support_status,
        "Historical_Min_Price": hist_min,
        "Historical_Max_Price": hist_max,
        "Price_Change_Pct": pct_change,
        "Elasticity": elasticity,
        "Zero_Anchor": zero_anchor,
        "Price_Ratio_Log": price_ratio_log,
        "Outside_Observed_Range": outside_observed,
    })


def price_support(data: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    """Compare proposed prices with observed positive-sales cell-level support.
    
    Returns DataFrame with detailed support status for each cell.
    """
    # Get historical price bounds from positive-sales data
    support = (
        data.loc[data["Unit_Sales"] > 0]
        .groupby(CELL_KEYS, as_index=False, observed=True)
        .agg(Historical_Min_Price=("Price", "min"), Historical_Max_Price=("Price", "max"))
    )
    
    # Merge with scenario data
    result = scenario[CELL_KEYS + ["Simulated_Price", "Price_Change_Pct", "Elasticity_MAP", "Baseline_Price", "Baseline_Unit_Sales"]].merge(
        support, on=CELL_KEYS, how="left", validate="one_to_one",
    )
    
    # Compute detailed support status
    status_df = _compute_price_support_status(
        simulated_price=result["Simulated_Price"].to_numpy(float),
        baseline_price=result["Baseline_Price"].to_numpy(float),
        hist_min=result["Historical_Min_Price"].to_numpy(float),
        hist_max=result["Historical_Max_Price"].to_numpy(float),
        baseline_unit_sales=result["Baseline_Unit_Sales"].to_numpy(float),
        elasticity=result["Elasticity_MAP"].to_numpy(float),
    )
    
    # Combine with original columns
    result = pd.concat([result[CELL_KEYS], status_df], axis=1)
    
    return result
