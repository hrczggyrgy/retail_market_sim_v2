"""
Caching utilities for the dashboard.
Provides immutable scenario result caching with MultiIndex-based edit lookup.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from .data import CELL_KEYS, PRODUCT_KEYS, get_data_hash
from .dashboard_adapter import matrix_to_price_changes, price_support
from .decomposition import three_factor_decomposition, three_factor_decomposition_report, value_shares
from .simulation import simulate_scenario


@dataclass(frozen=True)
class ScenarioCacheKey:
    """Immutable cache key for scenario results."""
    data_hash: str
    config_hash: str
    edits_hash: str  # Hash of normalized price changes

    def __str__(self) -> str:
        return f"{self.data_hash}:{self.config_hash}:{self.edits_hash}"


def _normalize_edits(price_changes: dict[tuple, float]) -> list[tuple]:
    """Normalize price changes to a canonical sorted tuple for hashing."""
    return tuple(sorted((k, float(v)) for k, v in price_changes.items()))


def _hash_edits(price_changes: dict[tuple, float]) -> str:
    """Compute stable hash of price changes."""
    normalized = _normalize_edits(price_changes)
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()[:16]


def _hash_config(config: Any) -> str:
    """Compute stable hash of model config."""
    config_dict = {
        "prior_elasticity": config.prior_elasticity,
        "max_evaluations": config.max_evaluations,
        "alpha_sigma": config.alpha_sigma,
        "beta_category_sigma": config.beta_category_sigma,
        "beta_offset_sigma": config.beta_offset_sigma,
        "beta_offset_scale": config.beta_offset_scale,
        "retailer_sigma": config.retailer_sigma,
        "pair_sigma": config.pair_sigma,
        "season_sigma": config.season_sigma,
        "trend_sigma": config.trend_sigma,
        "sigma_sigma": config.sigma_sigma,
    }
    return hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:16]


def _build_edits_multiindex(baseline: pd.DataFrame, columns: dict[str, str]) -> pd.MultiIndex:
    """Build MultiIndex for fast edit lookup."""
    # Create index from baseline CELL_KEYS
    return pd.MultiIndex.from_frame(baseline[CELL_KEYS])


def _apply_edits_vectorized(
    baseline: pd.DataFrame,
    price_changes: dict[tuple, float],
    edits_index: pd.MultiIndex,
) -> pd.DataFrame:
    """Apply price changes using vectorized MultiIndex lookup."""
    if not price_changes:
        result = baseline.copy()
        result["Simulated_Price"] = result["Price"].astype(float)
        return result

    # Convert price_changes to DataFrame for vectorized merge
    edits_df = pd.DataFrame(
        [(k, v) for k, v in price_changes.items()],
        columns=["_cell_key", "_new_price"],
    )
    edits_df[CELL_KEYS] = pd.DataFrame(edits_df["_cell_key"].tolist(), index=edits_df.index)
    
    # Merge with baseline
    result = baseline.merge(
        edits_df[CELL_KEYS + ["_new_price"]],
        on=CELL_KEYS, how="left", validate="one_to_one",
    )
    result["Simulated_Price"] = result["Price"].astype(float)
    mask = result["_new_price"].notna()
    result.loc[mask, "Simulated_Price"] = result.loc[mask, "_new_price"]
    result = result.drop(columns=["_new_price"])
    return result


@dataclass
class ScenarioResultBundle:
    """Complete scenario result bundle for caching."""
    scenario: pd.DataFrame
    support: pd.DataFrame
    market_decomp: pd.DataFrame
    category_decomp: pd.DataFrame
    retailer_decomp: pd.DataFrame
    brand_decomp: pd.DataFrame


def compute_scenario_bundle(
    baseline: pd.DataFrame,
    fitted_estimates: pd.DataFrame,
    price_changes: dict[tuple, float],
    data: pd.DataFrame,
) -> ScenarioResultBundle:
    """Compute full scenario bundle (simulation + support + decompositions)."""
    # Apply edits and simulate
    scenario = simulate_scenario(baseline, fitted_estimates, price_changes)
    
    # Compute price support
    support = price_support(data, scenario)
    
    # Compute decompositions
    market_decomp = three_factor_decomposition(scenario, [])
    category_decomp = three_factor_decomposition_report(scenario, ["Category"])
    retailer_decomp = three_factor_decomposition_report(scenario, ["Retailer"])
    brand_decomp = three_factor_decomposition_report(scenario, ["Category", "Brand"])
    
    return ScenarioResultBundle(
        scenario=scenario,
        support=support,
        market_decomp=market_decomp,
        category_decomp=category_decomp,
        retailer_decomp=retailer_decomp,
        brand_decomp=brand_decomp,
    )


# Streamlit cache for scenario bundles - use a simpler approach with hash-based keys
import streamlit as st

# Module-level cache dict (persists across runs within a session)
_scenario_cache: dict[str, ScenarioResultBundle] = {}


def get_cached_scenario_bundle(
    baseline: pd.DataFrame,
    fitted_estimates: pd.DataFrame,
    price_changes: dict[tuple, float],
    data: pd.DataFrame,
    data_hash: str,
    config_hash: str,
) -> ScenarioResultBundle:
    """Get or compute cached scenario bundle using in-memory dict."""
    edits_hash = _hash_edits(price_changes)
    cache_key = f"{data_hash}:{config_hash}:{edits_hash}"
    
    if cache_key in _scenario_cache:
        return _scenario_cache[cache_key]
    
    # Compute bundle
    bundle = compute_scenario_bundle(baseline, fitted_estimates, price_changes, data)
    
    # Cache it
    _scenario_cache[cache_key] = bundle
    
    # Limit cache size
    if len(_scenario_cache) > 16:
        # Remove oldest entry (first inserted)
        oldest_key = next(iter(_scenario_cache))
        del _scenario_cache[oldest_key]
    
    return bundle