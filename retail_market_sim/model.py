"""
Bayesian model specification and fitting (MAP and MCMC).
"""
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import pymc as pm

from .data import PRODUCT_KEYS, CELL_KEYS, tuple_index

RANDOM_SEED = 42


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for the hierarchical demand model."""
    prior_elasticity: float = -1.5
    max_evaluations: int = 6000
    # Prior scales (fixed to avoid variance-collapse pathologies in MAP)
    alpha_sigma: float = 2.0
    beta_category_sigma: float = 0.75
    beta_offset_sigma: float = 1.0
    beta_offset_scale: float = 0.5
    retailer_sigma: float = 0.5
    pair_sigma: float = 0.5
    season_sigma: float = 0.3
    trend_sigma: float = 0.3
    sigma_sigma: float = 0.75

    # MCMC settings
    mcmc_draws: int = 1000
    mcmc_tune: int = 1000
    mcmc_chains: int = 4
    mcmc_target_accept: float = 0.9


# =============================================================================
# Model Construction
# =============================================================================

def _prepare_training_data(data: pd.DataFrame) -> dict:
    """Prepare arrays and indices for model fitting."""
    train = data.loc[data["Unit_Sales"] > 0].copy().reset_index(drop=True)
    if len(train) < 12:
        raise ValueError("At least 12 positive-unit observations are required.")

    products = (
        data[PRODUCT_KEYS]
        .drop_duplicates()
        .sort_values(PRODUCT_KEYS)
        .reset_index(drop=True)
    )
    pairs = (
        data[CELL_KEYS]
        .drop_duplicates()
        .sort_values(CELL_KEYS)
        .reset_index(drop=True)
    )
    categories = sorted(data["Category"].unique().tolist())
    retailers = sorted(data["Retailer"].unique().tolist())

    product_lookup = tuple_index(products, PRODUCT_KEYS)
    pair_lookup = tuple_index(pairs, CELL_KEYS)

    sku_idx = product_lookup.get_indexer(tuple_index(train, PRODUCT_KEYS))
    pair_idx = pair_lookup.get_indexer(tuple_index(train, CELL_KEYS))

    category_idx = pd.Categorical(train["Category"], categories=categories).codes.astype("int32")
    retailer_idx = pd.Categorical(train["Retailer"], categories=retailers).codes.astype("int32")
    product_category_idx = pd.Categorical(products["Category"], categories=categories).codes.astype("int32")

    season_idx = train["Season"].to_numpy(dtype="int32")
    log_q = np.log(train["Unit_Sales"].to_numpy(dtype=float))
    log_p = np.log(train["Price"].to_numpy(dtype=float))

    # Center prices within each SKU-retailer pair
    pair_price_means = pd.Series(log_p).groupby(pair_idx).transform("mean").to_numpy()
    x_price = log_p - pair_price_means

    # Time trend in years
    time_years = (train["Period"].to_numpy(dtype=float) - train["Period"].mean()) / 12.0

    coords = {
        "product": np.arange(len(products)),
        "pair": np.arange(len(pairs)),
        "retailer": retailers,
        "category": categories,
        "season": np.arange(12),
        "observation": np.arange(len(train)),
    }

    return {
        "train": train,
        "products": products,
        "pairs": pairs,
        "categories": categories,
        "retailers": retailers,
        "product_lookup": product_lookup,
        "pair_lookup": pair_lookup,
        "sku_idx": sku_idx,
        "pair_idx": pair_idx,
        "category_idx": category_idx,
        "retailer_idx": retailer_idx,
        "product_category_idx": product_category_idx,
        "season_idx": season_idx,
        "log_q": log_q,
        "log_p": log_p,
        "x_price": x_price,
        "time_years": time_years,
        "coords": coords,
    }


def _build_model(prepared: dict, config: ModelConfig) -> pm.Model:
    """Build the PyMC model (shared between MAP and MCMC)."""
    train = prepared["train"]
    products = prepared["products"]
    categories = prepared["categories"]
    retailers = prepared["retailers"]

    sku_idx = prepared["sku_idx"]
    pair_idx = prepared["pair_idx"]
    category_idx = prepared["category_idx"]
    retailer_idx = prepared["retailer_idx"]
    product_category_idx = prepared["product_category_idx"]
    season_idx = prepared["season_idx"]
    x_price = prepared["x_price"]
    time_years = prepared["time_years"]
    log_q = prepared["log_q"]
    coords = prepared["coords"]

    with pm.Model(coords=coords) as model:
        alpha = pm.Normal(
            "alpha",
            mu=float(log_q.mean()),
            sigma=config.alpha_sigma,
            dims="product",
        )

        beta_category = pm.Normal(
            "beta_category",
            mu=config.prior_elasticity,
            sigma=config.beta_category_sigma,
            dims="category",
        )
        beta_offset = pm.Normal(
            "beta_offset",
            mu=0.0,
            sigma=config.beta_offset_sigma,
            dims="product",
        )
        beta = pm.Deterministic(
            "beta",
            beta_category[product_category_idx] + config.beta_offset_scale * beta_offset,
            dims="product",
        )

        retailer_raw = pm.Normal(
            "retailer_raw",
            mu=0.0,
            sigma=config.retailer_sigma,
            dims="retailer",
        )
        retailer_effect = retailer_raw - pm.math.sum(retailer_raw) / len(retailers)

        pair_effect = pm.Normal(
            "pair_effect",
            mu=0.0,
            sigma=config.pair_sigma,
            dims="pair",
        )

        season_raw = pm.Normal(
            "season_raw",
            mu=0.0,
            sigma=config.season_sigma,
            dims=("category", "season"),
        )
        season_effect = season_raw - pm.math.sum(season_raw, axis=1, keepdims=True) / 12.0

        trend = pm.Normal(
            "trend",
            mu=0.0,
            sigma=config.trend_sigma,
            dims="category",
        )
        sigma = pm.HalfNormal("sigma", sigma=config.sigma_sigma)

        mu = (
            alpha[sku_idx]
            + retailer_effect[retailer_idx]
            + pair_effect[pair_idx]
            + beta[sku_idx] * x_price
            + season_effect[category_idx, season_idx]
            + trend[category_idx] * time_years
        )

        pm.Normal(
            "log_unit_sales",
            mu=mu,
            sigma=sigma,
            observed=log_q,
            dims="observation",
        )

    return model


# =============================================================================
# MAP Fitting
# =============================================================================

def fit_model_map(
    data: pd.DataFrame,
    config: ModelConfig | None = None,
) -> Dict[str, Any]:
    """
    Fit the hierarchical log-log demand model using MAP estimation.

    Returns
    -------
    dict with keys:
        estimates: DataFrame with elasticities and support diagnostics
        converged: bool
        message: optimizer message
        iterations: int
        training_rows: int
        positive_elasticities: int
        weak_support_count: int
        point: raw MAP point
        coords: model coordinates
        products, pairs, categories, retailers: metadata
    """
    if config is None:
        config = ModelConfig()

    prepared = _prepare_training_data(data)

    model = _build_model(prepared, config)

    with model:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            point, optimization = pm.find_MAP(
                method="L-BFGS-B",
                maxeval=config.max_evaluations,
                progressbar=False,
                return_raw=True,
                seed=RANDOM_SEED,
            )

    # Reconstruct elasticities from MAP point
    beta_category_map = np.asarray(point["beta_category"], dtype=float)
    beta_offset_map = np.asarray(point["beta_offset"], dtype=float)
    beta_map = beta_category_map[prepared["product_category_idx"]] + config.beta_offset_scale * beta_offset_map

    if not np.isfinite(beta_map).all():
        raise RuntimeError("The optimizer returned non-finite elasticities.")

    # Support diagnostics
    support = (
        prepared["train"]
        .assign(_log_price=prepared["log_p"], _within_log_price=prepared["x_price"])
        .groupby(PRODUCT_KEYS, observed=True)
        .agg(
            Positive_Observations=("Unit_Sales", "size"),
            Observed_Months=("Period", "nunique"),
            Distinct_Prices=("Price", "nunique"),
            Within_Pair_LogPrice_SD=(
                "_within_log_price",
                lambda v: float(np.std(v, ddof=0)),
            ),
        )
        .reset_index()
    )

    estimates = prepared["products"].copy()
    estimates["Elasticity_MAP"] = beta_map
    estimates = estimates.merge(
        support, on=PRODUCT_KEYS, how="left", validate="one_to_one"
    )
    estimates["Positive_Observations"] = estimates["Positive_Observations"].fillna(0).astype(int)
    estimates["Weak_Price_Support"] = (
        estimates["Positive_Observations"].lt(12)
        | estimates["Within_Pair_LogPrice_SD"].fillna(0).lt(0.02)
    )

    return {
        "estimates": estimates,
        "converged": bool(optimization.success),
        "message": str(optimization.message),
        "iterations": int(getattr(optimization, "nit", 0)),
        "training_rows": len(prepared["train"]),
        "positive_elasticities": int((beta_map > 0).sum()),
        "weak_support_count": int(estimates["Weak_Price_Support"].sum()),
        "point": point,
        "coords": prepared["coords"],
        "products": prepared["products"],
        "pairs": prepared["pairs"],
        "categories": prepared["categories"],
        "retailers": prepared["retailers"],
    }


# =============================================================================
# MCMC Fitting
# =============================================================================

def fit_model_mcmc(
    data: pd.DataFrame,
    config: ModelConfig | None = None,
) -> Tuple:
    """
    Fit the same model with MCMC for full posterior inference.

    Returns
    -------
    trace, products, pairs, categories, retailers, sku_idx, pair_idx,
    category_idx, retailer_idx, product_category_idx
    """
    if config is None:
        config = ModelConfig()

    prepared = _prepare_training_data(data)

    model = _build_model(prepared, config)

    with model:
        trace = pm.sample(
            draws=config.mcmc_draws,
            tune=config.mcmc_tune,
            chains=config.mcmc_chains,
            target_accept=config.mcmc_target_accept,
            random_seed=RANDOM_SEED,
        )

    return (
        trace,
        prepared["products"],
        prepared["pairs"],
        prepared["categories"],
        prepared["retailers"],
        prepared["sku_idx"],
        prepared["pair_idx"],
        prepared["category_idx"],
        prepared["retailer_idx"],
        prepared["product_category_idx"],
    )