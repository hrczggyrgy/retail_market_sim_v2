"""
Type definitions for fit results and scenario payloads.
Provides strict contracts at the analytical/UI boundary.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FitEstimates:
    """Elasticity estimates with support diagnostics from MAP fitting."""
    data: pd.DataFrame  # Columns: Category, Brand, SKU, Elasticity_MAP, Positive_Observations, Observed_Months, Distinct_Prices, Within_Pair_LogPrice_SD, Weak_Price_Support

    def __post_init__(self) -> None:
        required = {"Category", "Brand", "SKU", "Elasticity_MAP", "Weak_Price_Support"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"FitEstimates missing required columns: {missing}")
        if not np.isfinite(self.data["Elasticity_MAP"]).all():
            raise ValueError("Elasticity_MAP contains non-finite values")


@dataclass(frozen=True)
class FitResult:
    """Complete result from MAP model fitting."""
    estimates: FitEstimates
    converged: bool
    message: str
    iterations: int
    training_rows: int
    positive_elasticities: int
    weak_support_count: int
    point: dict[str, Any]
    coords: dict[str, np.ndarray]
    products: pd.DataFrame
    pairs: pd.DataFrame
    categories: list[str]
    retailers: list[str]


@dataclass(frozen=True)
class ScenarioResult:
    """Result from scenario simulation with baseline and simulated values."""
    data: pd.DataFrame  # Contains all baseline/simulated columns

    def __post_init__(self) -> None:
        required = {
            "Category", "Retailer", "Brand", "SKU", "Pack_Size",
            "Baseline_Unit_Sales", "Baseline_Price", "Baseline_Revenue",
            "Simulated_Unit_Sales", "Simulated_Price", "Simulated_Revenue",
            "Price_Change_Pct", "Revenue_Delta", "Elasticity_MAP"
        }
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"ScenarioResult missing required columns: {missing}")


@dataclass(frozen=True)
class DecompositionResult:
    """Result from three-factor decomposition."""
    data: pd.DataFrame  # Grouped decomposition with Volume_Effect, Mix_Effect, Price_Effect, etc.


@dataclass(frozen=True)
class ValueSharesResult:
    """Result from value shares calculation."""
    data: pd.DataFrame  # Grouped shares with Baseline_Share_Pct, Simulated_Share_Pct, Share_Change_pp


@dataclass(frozen=True)
class PosteriorSummary:
    """Summary of posterior scenario analysis."""
    mean_revenue: float
    std_revenue: float
    q025_revenue: float
    q975_revenue: float
    mean_delta: float
    probability_positive_delta: float


# Type aliases for common dict-based payloads (legacy compatibility)
FitResultDict = dict[str, Any]
ScenarioResultDict = dict[str, Any]