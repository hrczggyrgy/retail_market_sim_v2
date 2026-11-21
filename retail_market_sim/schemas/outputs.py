"""
Output schemas for the Beverage Market Price Simulator.

Defines the canonical response types for model fitting, diagnostics, and scenarios.
"""
from datetime import datetime
from typing import Annotated, Literal
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np


# Provenance type alias
ProvenanceType = Literal["observed", "descriptive", "modeled", "assumed_flow"]


# =============================================================================
# Model Fit Outputs
# =============================================================================

class ElasticityEstimate(BaseModel):
    """Elasticity estimate for a single SKU."""
    category: str
    brand: str
    sku: str
    elasticity_map: float = Field(alias="Elasticity_MAP")
    elasticity_mean: float | None = None
    elasticity_std: float | None = None
    elasticity_q025: float | None = None
    elasticity_q975: float | None = None
    positive_observations: int = Field(alias="Positive_Observations")
    observed_months: int = Field(alias="Observed_Months")
    distinct_prices: int = Field(alias="Distinct_Prices")
    within_pair_logprice_sd: float = Field(alias="Within_Pair_LogPrice_SD")
    weak_price_support: bool = Field(alias="Weak_Price_Support")


class FitEstimates(BaseModel):
    """Complete elasticity estimates with diagnostics."""
    estimates: list[ElasticityEstimate]
    converged: bool
    message: str
    iterations: int
    training_rows: int
    positive_elasticities: int
    weak_support_count: int
    data_version: str
    model_version: str
    generated_at: str


class MCMCDiagnostics(BaseModel):
    """MCMC convergence diagnostics from ArviZ."""
    divergences: int
    max_rhat: float
    mean_rhat: float
    min_bulk_ess: float
    min_tail_ess: float
    max_rhat_params: list[str] = Field(default_factory=list)
    min_ess_params: list[str] = Field(default_factory=list)
    passed_gates: bool
    gate_criteria: dict[str, float] = Field(default_factory=dict)


class FitResult(BaseModel):
    """Complete model fit result with diagnostics."""
    estimates: FitEstimates
    diagnostics: MCMCDiagnostics | None = None
    model_configuration: dict = Field(default_factory=dict)
    data_hash: str
    model_version: str
    generated_at: str


# =============================================================================
# Diagnostics Outputs
# =============================================================================

class PriorPredictiveCheck(BaseModel):
    """Prior predictive check results."""
    category_totals: dict[str, dict]  # mean, std, q025, q975 vs observed
    sku_shares: dict[str, dict]
    seasonal_profiles: dict[str, dict]
    plausibility_passed: bool
    notes: list[str] = Field(default_factory=list)


class PosteriorPredictiveCheck(BaseModel):
    """Posterior predictive check results."""
    category_totals: dict[str, dict]
    sku_shares: dict[str, dict]
    sparse_cells: dict[str, dict]
    seasonal_profiles: dict[str, dict]
    passed: bool
    notes: list[str] = Field(default_factory=list)


class BacktestResult(BaseModel):
    """Single backtest fold result."""
    test_month: str
    mae_packs: float
    wapec: float
    pred_coverage: float
    coverage_in_range: float
    coverage_breakdown: dict[str, float]
    seasonal_naive_mae: float
    last_observed_mae: float
    # Matched-row comparisons
    matched_seasonal_n: int
    matched_model_mae_vs_seasonal: float
    matched_seasonal_mae: float
    matched_model_minus_seasonal_mae: float
    matched_last_n: int
    matched_model_mae_vs_last: float
    matched_last_mae: float
    matched_model_minus_last_mae: float


class BacktestSummary(BaseModel):
    """Aggregated backtest results."""
    folds: list[BacktestResult]
    mean_mae_packs: float
    mean_wapec: float
    mean_pred_coverage: float
    mean_coverage_in_range: float
    overall_row_coverage: float
    coverage_breakdown: dict[str, float]
    mean_seasonal_naive_mae: float
    mean_last_observed_mae: float
    # Matched-row aggregates
    mean_matched_seasonal_n: float
    mean_matched_model_mae_vs_seasonal: float
    mean_matched_seasonal_mae: float
    mean_matched_model_minus_seasonal_mae: float
    mean_matched_last_n: float
    mean_matched_model_mae_vs_last: float
    mean_matched_last_mae: float
    mean_matched_model_minus_last_mae: float
    prior_sensitivity: dict | None = None


class PriceIdentificationReport(BaseModel):
    """Report on price effect identification quality."""
    relative_price_variation: dict[str, float]  # by category/SKU
    retailer_price_differences: dict[str, float]  # same SKU different retailers
    nearby_price_correlations: dict[str, float]
    volume_covered_by_priceable_core_pct: float
    extrapolation_risk: dict[str, float]
    prior_sensitivity: dict[str, float]
    fit_quality_badge: Literal["strong", "moderate", "weak"]
    price_identification_badge: Literal["strong", "moderate", "weak", "insufficient"]
    notes: list[str] = Field(default_factory=list)


class DiagnosticsReport(BaseModel):
    """Complete model diagnostics report."""
    model_id: str
    model_version: str
    data_hash: str
    generated_at: str
    prior_predictive: PriorPredictiveCheck
    posterior_predictive: PosteriorPredictiveCheck
    backtest: BacktestSummary
    price_identification: PriceIdentificationReport
    mcmc_diagnostics: MCMCDiagnostics | None = None
    release_gates_passed: bool
    gate_failures: list[str] = Field(default_factory=list)


# =============================================================================
# Scenario Outputs
# =============================================================================

class MetricSummary(BaseModel):
    """Summary statistics for a metric across posterior draws."""
    unit: Literal["units", "liters", "revenue", "share_pct"]
    provenance: ProvenanceType
    baseline: float
    scenario: float
    absolute_delta: float
    pct_delta: float
    q025: float | None = None
    q975: float | None = None
    std: float | None = None


class SegmentResult(BaseModel):
    """Results for a single segment."""
    segment_type: Literal["sku", "brand", "category", "retailer", "pack_size", "total_market"]
    segment_key: dict[str, str]
    metrics: dict[str, MetricSummary]
    price_change_pct: float | None = None
    provenance: ProvenanceType = "modeled"


class RevenueBridgeComponent(BaseModel):
    """Single component in revenue bridge."""
    label: str
    value: float
    description: str
    provenance: ProvenanceType


class RevenueBridge(BaseModel):
    """Revenue bridge from baseline to scenario."""
    baseline_revenue: float
    components: list[RevenueBridgeComponent]
    scenario_revenue: float
    total_change: float
    provenance: ProvenanceType = "descriptive"


class FlowMatrixCell(BaseModel):
    """Single cell in liter flow matrix."""
    from_segment: str
    to_segment: str
    liters: float
    allocation_rule: str


class FlowMatrix(BaseModel):
    """Model-implied liter flow matrix (assumed, not observed)."""
    cells: list[FlowMatrixCell]
    allocation_rule: str
    allocation_rule_id: str
    outside_market_source_sink_liters: float = 0.0
    provenance: ProvenanceType = "assumed_flow"
    warning: str = "This matrix represents model-implied allocation under assumed substitution rules, not observed customer movements."


class OutOfSupportFlag(BaseModel):
    """Flag for out-of-support conditions."""
    flag_type: Literal["extrapolation", "zero_anchor", "weak_support", "missing_history", "prior_driven"]
    affected_cells: list[dict]
    message: str
    severity: Literal["warning", "info"]


class ScenarioResponse(BaseModel):
    """Complete scenario response."""
    scenario_id: str
    model_id: str
    data_version: str
    model_version: str
    price_index_basis: str
    baseline_month: str
    calibration_profile: Literal["lower", "central", "higher"]
    eligible_market_coverage_pct: float
    out_of_support_flags: list[OutOfSupportFlag] = Field(default_factory=list)
    provenance_type: ProvenanceType = "modeled"
    unit: Literal["units", "liters", "revenue", "share_pct"]
    posterior_summary: dict[str, MetricSummary] = Field(default_factory=dict)
    segment_results: list[SegmentResult] = Field(default_factory=list)
    revenue_bridge: RevenueBridge | None = None
    flow_matrix: FlowMatrix | None = None
    generated_at: str
    seed: int


# =============================================================================
# Artifact Manifest
# =============================================================================

class RunManifest(BaseModel):
    """Manifest of artifacts produced by a run."""
    run_id: str
    data_hash: str
    raw_input_digest: str
    model_version: str
    inference_method: Literal["MAP", "posterior", "market_substitution"]
    environment: dict
    artifacts: list[str]
    generated_at: str


# =============================================================================
# Helper Functions
# =============================================================================

def build_metric_summary(
    unit: Literal["units", "liters", "revenue", "share_pct"],
    provenance: ProvenanceType,
    baseline: float,
    scenario: float,
    draws: np.ndarray | None = None
) -> MetricSummary:
    """Build a MetricSummary from baseline, scenario, and optional posterior draws."""
    abs_delta = scenario - baseline
    pct_delta = (abs_delta / baseline * 100) if baseline != 0 else 0.0
    
    result = MetricSummary(
        unit=unit,
        provenance=provenance,
        baseline=baseline,
        scenario=scenario,
        absolute_delta=abs_delta,
        pct_delta=pct_delta,
    )
    
    if draws is not None and len(draws) > 0:
        result.q025 = float(np.quantile(draws, 0.025))
        result.q975 = float(np.quantile(draws, 0.975))
        result.std = float(np.std(draws, ddof=1))
    
    return result


def build_revenue_bridge(
    scenario_df: pd.DataFrame,
    baseline_revenue: float,
    scenario_revenue: float
) -> RevenueBridge:
    """Build revenue bridge from scenario DataFrame."""
    # This is a placeholder - actual implementation depends on scenario type
    components = [
        RevenueBridgeComponent(
            label="Baseline Revenue",
            value=baseline_revenue,
            description="Observed baseline revenue",
            provenance="observed"
        ),
        RevenueBridgeComponent(
            label="Scenario Revenue",
            value=scenario_revenue,
            description="Scenario revenue",
            provenance="modeled"
        ),
    ]
    
    return RevenueBridge(
        baseline_revenue=baseline_revenue,
        components=components,
        scenario_revenue=scenario_revenue,
        total_change=scenario_revenue - baseline_revenue,
        provenance="descriptive"
    )