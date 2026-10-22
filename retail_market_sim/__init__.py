"""
Retail Market Simulation - Core analytical functions.
"""
from .data import load_data, tuple_index, validate_baseline, get_data_hash
from .model import fit_model_map, fit_model_mcmc, ModelConfig, RANDOM_SEED
from .simulation import simulate_scenario, simulate_scenario_posterior, build_named_scenario
from .decomposition import three_factor_decomposition, _pack_size_to_factor, value_shares
from .validation import check_price, check_elasticity, check_finite_response

__all__ = [
    "load_data",
    "tuple_index",
    "validate_baseline",
    "get_data_hash",
    "fit_model_map",
    "fit_model_mcmc",
    "ModelConfig",
    "RANDOM_SEED",
    "simulate_scenario",
    "simulate_scenario_posterior",
    "build_named_scenario",
    "three_factor_decomposition",
    "_pack_size_to_factor",
    "value_shares",
    "check_price",
    "check_elasticity",
    "check_finite_response",
]