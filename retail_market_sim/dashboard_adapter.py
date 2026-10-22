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


def price_support(data: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    """Compare proposed prices with observed positive-sales cell-level support."""
    support = (
        data.loc[data["Unit_Sales"] > 0]
        .groupby(CELL_KEYS, as_index=False, observed=True)
        .agg(Historical_Min_Price=("Price", "min"), Historical_Max_Price=("Price", "max"))
    )
    result = scenario[CELL_KEYS + ["Simulated_Price", "Price_Change_Pct"]].merge(
        support, on=CELL_KEYS, how="left", validate="one_to_one",
    )
    result["Support_Unknown"] = result["Historical_Min_Price"].isna()
    result["Outside_Observed_Range"] = (
        (result["Simulated_Price"] < result["Historical_Min_Price"] - 1e-10)
        | (result["Simulated_Price"] > result["Historical_Max_Price"] + 1e-10)
    )
    return result
