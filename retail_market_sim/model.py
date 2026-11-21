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
from .types import FitResultDict

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
) -> FitResultDict:
    """
    Fit the hierarchical log-log demand model using MAP estimation.

    Returns
    -------
    FitResultDict with keys:
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

@dataclass
class MCMCDiagnostics:
    """MCMC convergence diagnostics from ArviZ."""
    divergences: int
    max_rhat: float
    mean_rhat: float
    min_bulk_ess: float
    min_tail_ess: float
    max_rhat_params: list[str]  # Parameters with highest R-hat
    min_ess_params: list[str]  # Parameters with lowest ESS
    passed_gates: bool
    gate_criteria: dict[str, Any]


def _run_mcmc_diagnostics(trace, var_names: list[str] | None = None) -> MCMCDiagnostics:
    """Run ArviZ diagnostics on MCMC trace.
    
    Fails closed: any missing or non-finite required diagnostic causes gate failure.
    """
    import arviz as az
    
    # Get posterior sample
    post = az.convert_to_inference_data(trace)
    
    # Divergences
    divergences = int(post.posterior_predictive.get("diverging", 0).sum()) if hasattr(post, 'posterior_predictive') else 0
    # Also check sample stats
    if "diverging" in post.sample_stats:
        divergences = int(post.sample_stats["diverging"].sum())
    
    # R-hat - fail if any required diagnostic is missing or non-finite
    rhat = az.rhat(post, var_names=var_names)
    rhat_values = []
    rhat_non_finite = False
    for var in rhat.data_vars:
        vals = rhat[var].values.flatten()
        if not np.all(np.isfinite(vals)):
            rhat_non_finite = True
            # Find which draws are non-finite
            non_finite_mask = ~np.isfinite(vals)
            warnings.warn(f"R-hat for {var} has {non_finite_mask.sum()} non-finite values")
        rhat_vals = vals[np.isfinite(vals)]
        if len(rhat_vals) > 0:
            rhat_values.extend(rhat_vals.tolist())
    
    if not rhat_values:
        max_rhat = float('inf')
        mean_rhat = float('inf')
        rhat_non_finite = True
    else:
        max_rhat = float(np.max(rhat_values))
        mean_rhat = float(np.mean(rhat_values))
    
    # Find parameters with highest R-hat
    rhat_by_param = {}
    for var in rhat.data_vars:
        vals = rhat[var].values.flatten()
        rhat_vals = vals[np.isfinite(vals)]
        if len(rhat_vals) > 0:
            rhat_by_param[var] = float(np.max(rhat_vals))
    
    max_rhat_params = sorted(rhat_by_param, key=rhat_by_param.get, reverse=True)[:5]
    
    # ESS - fail if any required diagnostic is missing or non-finite
    ess = az.ess(post, var_names=var_names, method="bulk")
    tail_ess = az.ess(post, var_names=var_names, method="tail")
    
    bulk_ess_vals = []
    tail_ess_vals = []
    ess_by_param = {}
    ess_non_finite = False
    
    for var in ess.data_vars:
        bulk_vals = ess[var].values.flatten()
        tail_vals = tail_ess[var].values.flatten()
        
        if not np.all(np.isfinite(bulk_vals)):
            ess_non_finite = True
            warnings.warn(f"Bulk ESS for {var} has non-finite values")
        if not np.all(np.isfinite(tail_vals)):
            ess_non_finite = True
            warnings.warn(f"Tail ESS for {var} has non-finite values")
        
        bulk_finite = bulk_vals[np.isfinite(bulk_vals)]
        tail_finite = tail_vals[np.isfinite(tail_vals)]
        if len(bulk_finite) > 0:
            bulk_ess_vals.extend(bulk_finite.tolist())
            ess_by_param[f"{var}_bulk"] = float(np.min(bulk_finite))
        if len(tail_finite) > 0:
            tail_ess_vals.extend(tail_finite.tolist())
            ess_by_param[f"{var}_tail"] = float(np.min(tail_finite))
    
    min_bulk_ess = float(np.min(bulk_ess_vals)) if bulk_ess_vals else 0.0
    min_tail_ess = float(np.min(tail_ess_vals)) if tail_ess_vals else 0.0
    
    min_ess_params = sorted(ess_by_param, key=ess_by_param.get)[:5]
    
    # Gate criteria (configurable thresholds)
    gate_criteria = {
        "max_divergences": 0,
        "max_rhat": 1.01,
        "min_bulk_ess": 400,
        "min_tail_ess": 400,
    }
    
    # Fail closed: any non-finite diagnostic means gate failure
    passed_gates = (
        divergences <= gate_criteria["max_divergences"] and
        not rhat_non_finite and
        not ess_non_finite and
        max_rhat <= gate_criteria["max_rhat"] and
        min_bulk_ess >= gate_criteria["min_bulk_ess"] and
        min_tail_ess >= gate_criteria["min_tail_ess"]
    )
    
    return MCMCDiagnostics(
        divergences=divergences,
        max_rhat=max_rhat,
        mean_rhat=mean_rhat,
        min_bulk_ess=min_bulk_ess,
        min_tail_ess=min_tail_ess,
        max_rhat_params=max_rhat_params,
        min_ess_params=min_ess_params,
        passed_gates=passed_gates,
        gate_criteria=gate_criteria,
    )


# =============================================================================
# MCMC Fitting
# =============================================================================

def fit_model_mcmc(
    data: pd.DataFrame,
    config: ModelConfig | None = None,
    require_diagnostics: bool = True,
) -> Tuple:
    """
    Fit the same model with MCMC for full posterior inference.

    Returns
    -------
    trace, products, pairs, categories, retailers, sku_idx, pair_idx,
    category_idx, retailer_idx, product_category_idx, diagnostics
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

    # Run diagnostics
    var_names = ["beta", "alpha", "sigma", "beta_category", "beta_offset", 
                 "retailer_raw", "pair_effect", "season_raw", "trend"]
    diagnostics = _run_mcmc_diagnostics(trace, var_names=var_names)
    
    if require_diagnostics and not diagnostics.passed_gates:
        raise RuntimeError(
            f"MCMC diagnostics gates not passed: divergences={diagnostics.divergences}, "
            f"max_rhat={diagnostics.max_rhat:.4f}, min_bulk_ess={diagnostics.min_bulk_ess:.0f}, "
            f"min_tail_ess={diagnostics.min_tail_ess:.0f}. "
            f"Posterior intervals withheld. Gate criteria: {diagnostics.gate_criteria}"
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
        diagnostics,
    )