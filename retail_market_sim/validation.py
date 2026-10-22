"""
Validation utilities for scenarios and model outputs.
"""
import numpy as np
import pandas as pd

from .data import PRODUCT_KEYS, CELL_KEYS, tuple_index


def check_price(prices: np.ndarray, context: str = "scenario") -> None:
    """Validate scenario prices."""
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(
            f"Every existing SKU-retailer cell needs a finite price above zero "
            f"in {context}."
        )


def check_elasticity(elasticities: np.ndarray) -> None:
    """Validate elasticity estimates."""
    if not np.isfinite(elasticities).all():
        raise ValueError("Some SKU elasticity estimates are missing.")


def check_finite_response(log_response: np.ndarray, max_abs: float = 50.0) -> None:
    """Check for numerically explosive responses."""
    if not np.isfinite(log_response).all() or (np.abs(log_response) > max_abs).any():
        raise ValueError(
            "The scenario implies an extreme demand response. "
            "Use prices closer to baseline and inspect SKU elasticities."
        )


def check_finite_quantities_revenue(q: np.ndarray, r: np.ndarray) -> None:
    """Check simulated quantities and revenue are finite."""
    if not np.isfinite(q).all() or not np.isfinite(r).all():
        raise ValueError("The scenario produces non-finite quantities or revenue.")


def reconcile_revenue(delta: np.ndarray, pe: np.ndarray, ve: np.ndarray,
                      rtol: float = 1e-9, atol: float = 1e-6) -> None:
    """Verify price-volume effects reconcile to revenue delta."""
    if not np.allclose(delta, pe + ve, rtol=rtol, atol=atol):
        raise RuntimeError("Price-volume effects do not reconcile to revenue.")


def validate_scenario_targets(
    price_changes: dict,
    baseline: pd.DataFrame,
) -> None:
    """Validate that all scenario targets exist in baseline."""
    for (cat, ret, brand, sku) in price_changes.keys():
        mask = (
            (baseline["Category"] == cat) &
            (baseline["Retailer"] == ret) &
            (baseline["Brand"] == brand) &
            (baseline["SKU"] == sku)
        )
        if not mask.any():
            raise ValueError(
                f"Scenario target not found in baseline: ({cat}, {ret}, {brand}, {sku})"
            )
        if mask.sum() > 1:
            raise ValueError(
                f"Multiple matching rows for scenario target: ({cat}, {ret}, {brand}, {sku})"
            )


def check_product_mapping(indices: np.ndarray, context: str = "mapping") -> None:
    """Check that product index lookups didn't return -1 (unknown)."""
    if (indices == -1).any():
        raise ValueError(f"Some products not found in fitted model during {context}.")