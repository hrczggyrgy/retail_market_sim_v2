"""POS ingestion, validation, aggregation, and deterministic data identity."""
import hashlib
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

PRODUCT_KEYS = ["Category", "Brand", "SKU"]
CELL_KEYS = ["Category", "Retailer", "Brand", "SKU"]
REQUIRED_COLUMNS = ["Month", "Category", "Retailer", "Brand", "SKU", "Pack_Size", "Unit_Sales", "Sales_Revenue", "Price"]


def load_data(path: str, modified_ns: int | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """Load and validate monthly POS data in litres per pack."""
    if modified_ns is not None:
        _ = modified_ns
    raw = pd.read_csv(path, dtype={c: "string" for c in CELL_KEYS + ["Pack_Size"]})
    raw.columns = raw.columns.str.strip()
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if raw.empty:
        raise ValueError("The CSV contains no observations.")
    notes: List[str] = []
    for column in CELL_KEYS + ["Pack_Size"]:
        raw[column] = raw[column].astype("string").str.strip()
        if raw[column].isna().any() or raw[column].eq("").any():
            raise ValueError(f"{column} contains missing or blank values.")
    for column in ["Unit_Sales", "Sales_Revenue", "Price", "Pack_Size"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    for column in ["Unit_Sales", "Sales_Revenue", "Price", "Pack_Size"]:
        values = raw[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains missing or non-finite values.")
    if (raw["Unit_Sales"] < 0).any() or (raw["Sales_Revenue"] < 0).any():
        raise ValueError("Negative sales or returns require separate treatment.")
    if (raw["Price"] <= 0).any():
        raise ValueError("All supplied prices must be strictly positive.")
    if (raw["Pack_Size"] <= 0).any():
        raise ValueError("Pack_Size must be strictly positive litres per pack.")
    if (raw["Unit_Sales"].eq(0) & raw["Sales_Revenue"].ne(0)).any():
        raise ValueError("Rows with zero units must have zero revenue.")
    numeric_month = pd.to_numeric(raw["Month"], errors="coerce")
    is_period = numeric_month.notna().all() and np.isfinite(numeric_month.to_numpy(float)).all() and np.allclose(numeric_month, np.round(numeric_month)) and numeric_month.between(1, 1200).all()
    if is_period:
        raw["Period"] = numeric_month.astype(int)
        raw["Season"] = ((raw["Period"] - 1) % 12).astype(int)
        labels = {int(v): f"Month {int(v)}" for v in sorted(raw["Period"].unique())}
        notes.append("Month is a numeric period index; calendar months are not inferred.")
    else:
        parsed = pd.to_datetime(raw["Month"].astype(str), errors="coerce")
        if parsed.isna().any():
            raise ValueError("Month must contain positive integer periods or parseable dates.")
        ordinals = parsed.dt.year * 12 + parsed.dt.month
        raw["Period"] = (ordinals - int(ordinals.min()) + 1).astype(int)
        raw["Season"] = (parsed.dt.month - 1).astype(int)
        labels = pd.DataFrame({"Period": raw["Period"], "Label": parsed.dt.strftime("%Y-%m")}).drop_duplicates("Period").set_index("Period")["Label"].to_dict()
    pack_counts = raw.groupby(PRODUCT_KEYS, observed=True)["Pack_Size"].nunique()
    if (pack_counts > 1).any():
        raise ValueError("Some Category/Brand/SKU identities have multiple Pack_Size values.")
    positive = raw["Unit_Sales"] > 0
    implied = raw.loc[positive, "Sales_Revenue"] / raw.loc[positive, "Unit_Sales"]
    mismatch = ~np.isclose(raw.loc[positive, "Price"], implied, rtol=1e-4, atol=1e-6)
    if mismatch.any():
        notes.append(f"{int(mismatch.sum()):,} rows have supplied Price values different from Revenue / Units. Accounting-consistent prices are used.")
    if (implied <= 0).any():
        raise ValueError("Positive unit sales must have positive revenue for this log-price model.")
    monthly = raw.groupby(["Period", "Season"] + CELL_KEYS, as_index=False, observed=True).agg(
        Unit_Sales=("Unit_Sales", "sum"), Sales_Revenue=("Sales_Revenue", "sum"), Supplied_Price=("Price", "mean"), Pack_Size=("Pack_Size", "first"))
    monthly["Price"] = monthly["Supplied_Price"].astype(float)
    has_units = monthly["Unit_Sales"] > 0
    monthly.loc[has_units, "Price"] = monthly.loc[has_units, "Sales_Revenue"] / monthly.loc[has_units, "Unit_Sales"]
    monthly = monthly.drop(columns="Supplied_Price")
    monthly["Month"] = monthly["Period"].map(labels)
    monthly = monthly.sort_values(["Period"] + CELL_KEYS).reset_index(drop=True)
    periods = sorted(monthly["Period"].unique())
    if len(periods) != 24:
        notes.append(f"The input contains {len(periods)} observed monthly periods, not 24.")
    if len(periods) < 12:
        notes.append("Fewer than 12 months are available; annual seasonality is weakly identified.")
    if len(periods) > 1 and np.any(np.diff(periods) != 1):
        notes.append("Some monthly periods are missing. Their sales are not imputed.")
    zero_count = int(monthly["Unit_Sales"].eq(0).sum())
    if zero_count:
        notes.append(f"{zero_count:,} zero-unit monthly cells are retained in reporting but excluded from the log-demand fit.")
    return monthly, notes


def tuple_index(frame: pd.DataFrame, columns: list) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[columns])


def get_data_hash(data: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()).hexdigest()[:16]


def validate_baseline(baseline: pd.DataFrame) -> None:
    if baseline.empty:
        raise ValueError("The baseline snapshot is empty.")
    if baseline["Sales_Revenue"].sum() <= 0:
        raise ValueError("The latest month has no positive market revenue.")
    if (baseline["Price"] <= 0).any():
        raise ValueError("Baseline prices must be strictly positive.")
    if (baseline["Unit_Sales"] < 0).any():
        raise ValueError("Baseline unit sales cannot be negative.")
    if (baseline["Pack_Size"] <= 0).any():
        raise ValueError("Baseline pack sizes must be strictly positive litres per pack.")
