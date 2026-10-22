"""
Data loading, validation, and aggregation.
"""
import hashlib
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

PRODUCT_KEYS = ["Category", "Brand", "SKU"]
CELL_KEYS = ["Category", "Retailer", "Brand", "SKU"]

REQUIRED_COLUMNS = [
    "Month",
    "Category",
    "Retailer",
    "Brand",
    "SKU",
    "Pack_Size",
    "Unit_Sales",
    "Sales_Revenue",
    "Price",
]


# =============================================================================
# Data Loading
# =============================================================================

def load_data(path: str, modified_ns: int | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load, validate, and aggregate monthly POS data.

    Parameters
    ----------
    path : str
        Path to CSV file.
    modified_ns : int, optional
        File modification time in nanoseconds, used for cache invalidation.

    Returns
    -------
    monthly : pd.DataFrame
        Aggregated monthly data with Period, Season, and accounting-consistent prices.
    notes : list[str]
        Data quality notes.
    """
    if modified_ns is not None:
        _ = modified_ns  # Used by Streamlit cache key

    raw = pd.read_csv(
        path,
        dtype={
            "Category": "string",
            "Retailer": "string",
            "Brand": "string",
            "SKU": "string",
            "Pack_Size": "string",
        },
    )
    raw.columns = raw.columns.str.strip()

    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if raw.empty:
        raise ValueError("The CSV contains no observations.")

    notes = []

    # Clean identifiers
    for column in CELL_KEYS + ["Pack_Size"]:
        raw[column] = raw[column].astype("string").str.strip()
        if raw[column].isna().any() or raw[column].eq("").any():
            raise ValueError(f"{column} contains missing or blank values.")

    # Coerce numeric columns
    for column in ["Unit_Sales", "Sales_Revenue", "Price"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
        if not np.isfinite(raw[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains missing or non-finite values.")

    if (raw["Unit_Sales"] < 0).any() or (raw["Sales_Revenue"] < 0).any():
        raise ValueError("Negative sales/returns require separate treatment before modeling.")

    if (raw["Price"] <= 0).any():
        raise ValueError("All supplied prices must be strictly positive.")

    if (raw["Unit_Sales"].eq(0) & raw["Sales_Revenue"].ne(0)).any():
        raise ValueError("Rows with zero units must have zero revenue.")

    # Period handling
    numeric_month = pd.to_numeric(raw["Month"], errors="coerce")
    is_period_number = (
        numeric_month.notna().all()
        and np.isfinite(numeric_month.to_numpy(dtype=float)).all()
        and np.allclose(numeric_month, np.round(numeric_month))
        and numeric_month.between(1, 1200).all()
    )

    if is_period_number:
        raw["Period"] = numeric_month.astype(int)
        raw["Season"] = ((raw["Period"] - 1) % 12).astype(int)
        period_labels = {
            int(v): f"Month {int(v)}" for v in sorted(raw["Period"].unique())
        }
        notes.append(
            "Month is a numeric period index. Seasonal position assumes "
            "periods repeat every 12 months; calendar month names are not inferred."
        )
    else:
        parsed = pd.to_datetime(raw["Month"].astype(str), errors="coerce")
        if parsed.isna().any():
            raise ValueError(
                "Month must contain positive integer periods or parseable dates."
            )
        month_ordinals = parsed.dt.year * 12 + parsed.dt.month
        raw["Period"] = (month_ordinals - int(month_ordinals.min()) + 1).astype(int)
        raw["Season"] = (parsed.dt.month - 1).astype(int)
        period_labels = (
            pd.DataFrame(
                {"Period": raw["Period"], "Label": parsed.dt.strftime("%Y-%m")}
            )
            .drop_duplicates("Period")
            .set_index("Period")["Label"]
            .to_dict()
        )

    # Pack size consistency
    pack_counts = raw.groupby(PRODUCT_KEYS, observed=True)["Pack_Size"].nunique()
    if (pack_counts > 1).any():
        raise ValueError(
            "Some Category/Brand/SKU identities have multiple Pack_Size values. "
            "Version resized SKUs or standardize pack descriptions first."
        )

    # Accounting-consistent prices
    positive_units = raw["Unit_Sales"] > 0
    implied_price = (
        raw.loc[positive_units, "Sales_Revenue"]
        / raw.loc[positive_units, "Unit_Sales"]
    )
    mismatch = ~np.isclose(
        raw.loc[positive_units, "Price"],
        implied_price,
        rtol=1e-4,
        atol=1e-6,
    )

    if mismatch.any():
        notes.append(
            f"{int(mismatch.sum()):,} source rows have supplied Price values "
            "different from Revenue / Units. Accounting-consistent prices "
            "are used throughout."
        )

    if (implied_price <= 0).any():
        raise ValueError(
            "Positive unit sales must have positive revenue for this log-price model."
        )

    # Aggregate duplicate monthly cells
    monthly = (
        raw.groupby(
            ["Period", "Season"] + CELL_KEYS,
            as_index=False,
            observed=True,
        )
        .agg(
            Unit_Sales=("Unit_Sales", "sum"),
            Sales_Revenue=("Sales_Revenue", "sum"),
            Supplied_Price=("Price", "mean"),
            Pack_Size=("Pack_Size", "first"),
        )
    )

    monthly["Price"] = monthly["Supplied_Price"].astype(float)
    has_units = monthly["Unit_Sales"] > 0
    monthly.loc[has_units, "Price"] = (
        monthly.loc[has_units, "Sales_Revenue"]
        / monthly.loc[has_units, "Unit_Sales"]
    )
    monthly = monthly.drop(columns="Supplied_Price")
    monthly["Month"] = monthly["Period"].map(period_labels)
    monthly = monthly.sort_values(["Period"] + CELL_KEYS).reset_index(drop=True)

    periods = sorted(monthly["Period"].unique())
    if len(periods) != 24:
        notes.append(
            f"The input contains {len(periods)} observed monthly periods, not 24."
        )

    if len(periods) < 12:
        notes.append(
            "Fewer than 12 months are available; annual seasonality is weakly identified."
        )

    if len(periods) > 1 and np.any(np.diff(periods) != 1):
        notes.append(
            "Some monthly periods are missing. Their sales are not imputed."
        )

    zero_count = int(monthly["Unit_Sales"].eq(0).sum())
    if zero_count:
        notes.append(
            f"{zero_count:,} zero-unit monthly cells are retained in reporting "
            "but excluded from the log-demand fit."
        )

    return monthly, notes


def tuple_index(frame: pd.DataFrame, columns: list) -> pd.MultiIndex:
    """Create collision-safe composite identities without concatenated strings."""
    return pd.MultiIndex.from_frame(frame[columns])


def get_data_hash(data: pd.DataFrame) -> str:
    """Compute deterministic hash of data for caching/versioning."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
    ).hexdigest()[:16]


# =============================================================================
# Validation
# =============================================================================

def validate_baseline(baseline: pd.DataFrame) -> None:
    """Validate that baseline data is suitable for simulation."""
    if baseline["Sales_Revenue"].sum() <= 0:
        raise ValueError("The latest month has no positive market revenue.")

    if (baseline["Price"] <= 0).any():
        raise ValueError("Baseline prices must be strictly positive.")

    if (baseline["Unit_Sales"] < 0).any():
        raise ValueError("Baseline unit sales cannot be negative.")