"""
Retail Market Simulation - Core analytical functions.
"""
from .data import load_data, tuple_index, validate_baseline, get_data_hash
from .model import fit_model_map, fit_model_mcmc, ModelConfig, RANDOM_SEED
from .simulation import simulate_scenario, simulate_scenario_posterior, build_named_scenario
from .decomposition import (three_factor_decomposition, three_factor_decomposition_report,
                            _pack_size_to_factor, value_shares)
from .validation import check_price, check_elasticity, check_finite_response
from .types import (
    FitEstimates, FitResult, ScenarioResult, DecompositionResult,
    ValueSharesResult, PosteriorSummary, FitResultDict, ScenarioResultDict,
)
from .dashboard_adapter import matrix_to_price_changes, price_support
from .dashboard_cache import (
    compute_scenario_bundle, get_cached_scenario_bundle,
    ScenarioResultBundle, _hash_config, _hash_edits,
)
from .evaluation import (
    evaluate_rolling, EvaluationResult, FoldResult,
    evaluation_to_dataframe, print_evaluation_summary,
)

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
    "three_factor_decomposition_report",
    "_pack_size_to_factor",
    "value_shares",
    "check_price",
    "check_elasticity",
    "check_finite_response",
    "FitEstimates",
    "FitResult",
    "ScenarioResult",
    "DecompositionResult",
    "ValueSharesResult",
    "PosteriorSummary",
    "FitResultDict",
    "ScenarioResultDict",
    "matrix_to_price_changes",
    "price_support",
    "compute_scenario_bundle",
    "get_cached_scenario_bundle",
    "ScenarioResultBundle",
    "_hash_config",
    "_hash_edits",
    "evaluate_rolling",
    "EvaluationResult",
    "FoldResult",
    "evaluation_to_dataframe",
    "print_evaluation_summary",
]