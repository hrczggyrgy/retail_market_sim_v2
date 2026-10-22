"""
Scenario simulation with point estimates and posterior draws.
"""
import numpy as np
import pandas as pd

from .data import PRODUCT_KEYS, CELL_KEYS, tuple_index
from .decomposition import _pack_size_to_factor
from .decomposition import three_factor_decomposition


# =============================================================================
# Validation
# =============================================================================

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


# =============================================================================
# Scenario Application
# =============================================================================

def _apply_price_changes(
    baseline: pd.DataFrame,
    price_changes: dict,
) -> pd.DataFrame:
    """
    Apply price changes to baseline.

    Parameters
    ----------
    baseline : pd.DataFrame
        Baseline data for the simulation period.
    price_changes : dict
        Mapping of (Category, Retailer, Brand, SKU) -> new absolute price.

    Returns
    -------
    pd.DataFrame with Simulated_Price column initialized and updated.
    """
    result = baseline.copy()
    # Initialize simulated price to baseline (handles empty price_changes)
    result["Simulated_Price"] = result["Price"].copy()

    for (cat, ret, brand, sku), new_price in price_changes.items():
        mask = (
            (result["Category"] == cat) &
            (result["Retailer"] == ret) &
            (result["Brand"] == brand) &
            (result["SKU"] == sku)
        )
        if not mask.any():
            raise ValueError(
                f"Scenario target not found in baseline: "
                f"({cat}, {ret}, {brand}, {sku})"
            )
        result.loc[mask, "Simulated_Price"] = new_price

    return result


def _compute_response(
    result: pd.DataFrame,
    estimates: pd.DataFrame,
) -> pd.DataFrame:
    """Compute quantity and revenue response using point elasticity."""
    p1 = pd.to_numeric(result["Simulated_Price"], errors="coerce").to_numpy(dtype=float)
    p0 = result["Price"].to_numpy(dtype=float)
    q0 = result["Unit_Sales"].to_numpy(dtype=float)

    check_price(p1, "point scenario")
    check_elasticity(result["Elasticity_MAP"].to_numpy(dtype=float))

    beta = result["Elasticity_MAP"].to_numpy(dtype=float)
    log_response = beta * np.log(p1 / p0)
    check_finite_response(log_response)

    q1 = q0 * np.exp(log_response)
    r1 = p1 * q1

    check_finite_quantities_revenue(q1, r1)

    result["Baseline_Unit_Sales"] = q0
    result["Baseline_Price"] = p0
    result["Baseline_Revenue"] = result["Sales_Revenue"]
    result["Simulated_Unit_Sales"] = q1
    result["Simulated_Revenue"] = r1
    result["Price_Change_Pct"] = 100.0 * (p1 / p0 - 1.0)
    result["Revenue_Delta"] = r1 - result["Baseline_Revenue"]
    result["Price_Effect"] = (p1 - p0) * q1
    result["Volume_Effect"] = (q1 - q0) * p0

    reconcile_revenue(result["Revenue_Delta"], result["Price_Effect"], result["Volume_Effect"])

    return result


def simulate_scenario(
    baseline: pd.DataFrame,
    estimates: pd.DataFrame,
    price_changes: dict,
) -> pd.DataFrame:
    """
    Apply edited prices and compute expected response at MAP point.

    Expected response: Q1 = Q0 * exp(beta * log(P1 / P0))

    Anchoring to observed Q0 preserves the observed baseline, including its
    seasonal level. This is a same-month counterfactual, not a next-month forecast.
    """
    result = baseline.merge(
        estimates[PRODUCT_KEYS + ["Elasticity_MAP"]],
        on=PRODUCT_KEYS,
        how="left",
        validate="many_to_one",
    )

    result = _apply_price_changes(result, price_changes)
    result = _compute_response(result, estimates)

    return result


# =============================================================================
# Posterior Simulation
# =============================================================================

def simulate_scenario_posterior(
    trace,
    baseline: pd.DataFrame,
    products: pd.DataFrame,
    pairs: pd.DataFrame,
    sku_idx: np.ndarray,
    pair_idx: np.ndarray,
    category_idx: np.ndarray,
    retailer_idx: np.ndarray,
    product_category_idx: np.ndarray,
    price_changes: dict,
    n_draws: int = 500,
    rng: np.random.Generator | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Simulate scenario using posterior draws for uncertainty quantification.

    For each posterior draw d:
        Q1^(d) = Q0 * exp(beta^(d) * log(P1/P0))
    Revenue^(d) = P1 * Q1^(d)

    Aggregates within each draw before computing market-level summaries,
    preserving covariance between cells.

    Returns
    -------
    result : pd.DataFrame
        Cell-level posterior summaries (mean, std, quantiles).
    market_revenue_draws : np.ndarray
        (n_draws,) array of total market revenue per draw.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = baseline.copy()
    p0 = result["Price"].to_numpy(dtype=float)
    q0 = result["Unit_Sales"].to_numpy(dtype=float)

    result = _apply_price_changes(result, price_changes)
    p1 = result["Simulated_Price"].to_numpy(dtype=float)
    check_price(p1, "posterior scenario")

    # Map baseline rows to product indices
    product_lookup = tuple_index(products, PRODUCT_KEYS)
    pair_lookup = tuple_index(pairs, CELL_KEYS)
    base_sku_idx = product_lookup.get_indexer(tuple_index(result, PRODUCT_KEYS))
    base_pair_idx = pair_lookup.get_indexer(tuple_index(result, CELL_KEYS))

    # Check for unknown products
    if (base_sku_idx == -1).any():
        raise ValueError("Some baseline products not found in fitted model products.")

    # Extract posterior draws for beta (elasticity)
    beta_draws = trace.posterior["beta"].values  # (chain, draw, product)
    beta_draws = beta_draws.reshape(-1, beta_draws.shape[-1])  # (total_draws, product)
    if n_draws < len(beta_draws):
        idx = rng.choice(len(beta_draws), n_draws, replace=False)
        beta_draws = beta_draws[idx]

    # For each draw, compute revenue at cell level
    revenue_draws = []
    for b in beta_draws:
        beta_vals = b[base_sku_idx]
        log_resp = beta_vals * np.log(p1 / p0)
        check_finite_response(log_resp)
        q1_draw = q0 * np.exp(log_resp)
        r1_draw = p1 * q1_draw
        revenue_draws.append(r1_draw)

    revenue_draws = np.array(revenue_draws)  # (draws, n_obs)

    # Aggregate to market level WITHIN each draw (preserves covariance)
    market_revenue_draws = revenue_draws.sum(axis=1)  # (draws,)

    # Cell-level summaries
    result["Simulated_Revenue_Mean"] = revenue_draws.mean(axis=0)
    result["Simulated_Revenue_Std"] = revenue_draws.std(axis=0, ddof=1)
    result["Simulated_Revenue_Q025"] = np.percentile(revenue_draws, 2.5, axis=0)
    result["Simulated_Revenue_Q975"] = np.percentile(revenue_draws, 97.5, axis=0)

    result["Baseline_Revenue"] = result["Sales_Revenue"]
    result["Revenue_Delta_Mean"] = result["Simulated_Revenue_Mean"] - result["Baseline_Revenue"]

    return result, market_revenue_draws


def build_named_scenario(name: str, baseline: pd.DataFrame) -> dict:
    """
    Build a named scenario from baseline data.

    Currently supports:
    - "nexus_cola_20pct": 20% discount on Cola_2L_Bottle at Nexus_Retail_Group

    Returns
    -------
    dict: price_changes mapping for simulation
    """
    if name == "nexus_cola_20pct":
        mask = (
            (baseline["Category"] == "Carbonated Soft Drinks") &
            (baseline["Retailer"] == "Nexus_Retail_Group") &
            (baseline["Brand"] == "ColaClassic") &
            (baseline["SKU"] == "Cola_2L_Bottle")
        )
        if not mask.any():
            raise ValueError("Target SKU not found in baseline for nexus_cola_20pct")
        if mask.sum() > 1:
            raise ValueError("Multiple matching rows for nexus_cola_20pct target")
        baseline_price = float(baseline.loc[mask, "Price"].values[0])
        new_price = 0.80 * baseline_price  # 20% discount
        return {
            ("Carbonated Soft Drinks", "Nexus_Retail_Group", "ColaClassic", "Cola_2L_Bottle"): new_price,
        }
    else:
        raise ValueError(f"Unknown scenario name: {name}")