"""Anchored price scenarios using MAP estimates or joint posterior draws."""
import numpy as np
import pandas as pd

from .data import CELL_KEYS, PRODUCT_KEYS, tuple_index
from .validation import (
    check_elasticity, check_finite_quantities_revenue, check_finite_response,
    check_price, check_product_mapping, reconcile_revenue, validate_scenario_targets,
)


def _validate_baseline(baseline: pd.DataFrame) -> None:
    if baseline.empty or baseline.duplicated(CELL_KEYS).any():
        raise ValueError("Baseline must contain unique, nonempty SKU-retailer cells.")
    check_price(baseline["Price"].to_numpy(float), "baseline")
    q = baseline["Unit_Sales"].to_numpy(float)
    r = baseline["Sales_Revenue"].to_numpy(float)
    check_finite_quantities_revenue(q, r)
    if (q < 0).any() or (r < 0).any():
        raise ValueError("Baseline sales and revenue must be nonnegative.")
    if not np.allclose(baseline["Price"].to_numpy(float) * q, r, rtol=1e-9, atol=1e-6):
        raise ValueError("Baseline prices, quantities and revenue do not reconcile.")


def _apply_price_changes(baseline: pd.DataFrame, price_changes: dict) -> pd.DataFrame:
    _validate_baseline(baseline)
    validate_scenario_targets(price_changes, baseline)
    result = baseline.copy()
    result["Simulated_Price"] = result["Price"].astype(float)
    for key, new_price in price_changes.items():
        mask = np.ones(len(result), dtype=bool)
        for column, value in zip(CELL_KEYS, key):
            mask &= result[column].eq(value).to_numpy()
        result.loc[mask, "Simulated_Price"] = new_price
    result["Simulated_Price"] = pd.to_numeric(result["Simulated_Price"], errors="coerce")
    check_price(result["Simulated_Price"].to_numpy(float), "scenario")
    return result


def simulate_scenario(
    baseline: pd.DataFrame, estimates: pd.DataFrame, price_changes: dict,
) -> pd.DataFrame:
    """Apply absolute prices to observed Q0; this is not a future-sales forecast."""
    result = baseline.merge(
        estimates[PRODUCT_KEYS + ["Elasticity_MAP"]],
        on=PRODUCT_KEYS, how="left", validate="many_to_one",
    )
    result = _apply_price_changes(result, price_changes)
    p0 = result["Price"].to_numpy(float)
    p1 = result["Simulated_Price"].to_numpy(float)
    q0 = result["Unit_Sales"].to_numpy(float)
    beta = result["Elasticity_MAP"].to_numpy(float)
    check_elasticity(beta)
    response = beta * np.log(p1 / p0)
    check_finite_response(response)
    q1 = q0 * np.exp(response)
    r1 = p1 * q1
    check_finite_quantities_revenue(q1, r1)
    result["Baseline_Unit_Sales"] = q0
    result["Baseline_Price"] = p0
    result["Baseline_Revenue"] = result["Sales_Revenue"]
    result["Simulated_Unit_Sales"] = q1
    result["Simulated_Revenue"] = r1
    result["Price_Change_Pct"] = 100 * (p1 / p0 - 1)
    result["Revenue_Delta"] = r1 - result["Baseline_Revenue"]
    result["Price_Effect"] = (p1 - p0) * q1
    result["Volume_Effect"] = (q1 - q0) * p0
    reconcile_revenue(result["Revenue_Delta"], result["Price_Effect"], result["Volume_Effect"])
    return result


def summarize_revenue_draws(revenue_draws: np.ndarray) -> dict:
    """Aggregate cells within each joint draw before calculating market uncertainty."""
    draws = np.asarray(revenue_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[0] < 2 or draws.shape[1] < 1:
        raise ValueError("Revenue draws require at least two draws and one cell.")
    if not np.isfinite(draws).all():
        raise ValueError("Revenue draws must be finite.")
    market = draws.sum(axis=1)
    return {
        "market_draws": market,
        "mean": float(market.mean()),
        "std": float(market.std(ddof=1)),
        "q025": float(np.quantile(market, 0.025)),
        "q975": float(np.quantile(market, 0.975)),
    }


def simulate_scenario_posterior(
    trace, baseline: pd.DataFrame, products: pd.DataFrame, pairs: pd.DataFrame,
    sku_idx: np.ndarray, pair_idx: np.ndarray, category_idx: np.ndarray,
    retailer_idx: np.ndarray, product_category_idx: np.ndarray,
    price_changes: dict, n_draws: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Propagate elasticity uncertainty, conditional on observed baseline demand.

    The encoding arguments are retained for CLI compatibility; only the product
    lookup is needed for an anchored response. No new observation noise is drawn.
    """
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)) or n_draws < 2:
        raise ValueError("n_draws must be an integer of at least two.")
    rng = np.random.default_rng(42) if rng is None else rng
    result = _apply_price_changes(baseline, price_changes)
    indices = tuple_index(products, PRODUCT_KEYS).get_indexer(tuple_index(result, PRODUCT_KEYS))
    check_product_mapping(indices, "posterior scenario")
    beta_da = trace.posterior["beta"].transpose("chain", "draw", "product")
    beta = np.asarray(beta_da.values, dtype=float).reshape(-1, len(products))
    if len(beta) < 2 or not np.isfinite(beta).all():
        raise ValueError("Posterior must contain at least two finite elasticity draws.")
    if n_draws < len(beta):
        beta = beta[rng.choice(len(beta), n_draws, replace=False)]
    p0 = result["Price"].to_numpy(float)
    p1 = result["Simulated_Price"].to_numpy(float)
    q0 = result["Unit_Sales"].to_numpy(float)
    response = beta[:, indices] * np.log(p1 / p0)[None, :]
    check_finite_response(response)
    quantities = q0[None, :] * np.exp(response)
    revenue = quantities * p1[None, :]
    check_finite_quantities_revenue(quantities, revenue)
    summary = summarize_revenue_draws(revenue)
    result["Simulated_Revenue_Mean"] = revenue.mean(axis=0)
    result["Simulated_Revenue_Std"] = revenue.std(axis=0, ddof=1)
    result["Simulated_Revenue_Q025"] = np.quantile(revenue, 0.025, axis=0)
    result["Simulated_Revenue_Q975"] = np.quantile(revenue, 0.975, axis=0)
    result["Baseline_Revenue"] = result["Sales_Revenue"]
    result["Revenue_Delta_Mean"] = result["Simulated_Revenue_Mean"] - result["Baseline_Revenue"]
    return result, summary["market_draws"]


def build_named_scenario(name: str, baseline: pd.DataFrame) -> dict:
    if name != "nexus_cola_20pct":
        raise ValueError(f"Unknown scenario name: {name}")
    key = ("Carbonated Soft Drinks", "Nexus_Retail_Group", "ColaClassic", "Cola_2L_Bottle")
    mask = np.ones(len(baseline), dtype=bool)
    for column, value in zip(CELL_KEYS, key):
        mask &= baseline[column].eq(value).to_numpy()
    if mask.sum() != 1:
        raise ValueError("Named scenario requires exactly one matching target cell.")
    price = float(baseline.loc[mask, "Price"].iloc[0])
    check_price(np.array([price]), "named scenario")
    return {key: 0.8 * price}
