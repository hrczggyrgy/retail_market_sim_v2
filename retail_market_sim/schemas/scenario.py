"""
Scenario request/response schemas for the Beverage Market Price Simulator.

Defines the contract for price scenario inputs and results.
"""
from datetime import date
from decimal import Decimal
from typing import Annotated, Literal
from pydantic import BaseModel, Field, field_validator
import pandas as pd


# =============================================================================
# Provenance Type
# =============================================================================

ProvenanceType = Literal["observed", "descriptive", "modeled", "assumed_flow"]


# =============================================================================
# Scenario Input Schemas
# =============================================================================

class PriceChange(BaseModel):
    """Single SKU–retailer price change."""
    sku_id: Annotated[str, Field(min_length=1)]
    retailer_id: Annotated[str, Field(min_length=1)]
    
    # Exactly one of pct_change or target_unit_value must be provided
    pct_change: float | None = Field(
        default=None, ge=-0.99, le=10.0,
        description="Percentage change in unit value (e.g., 0.08 for +8%, -0.05 for -5%)"
    )
    target_unit_value: float | None = Field(
        default=None, gt=0,
        description="Target absolute unit value (sales/quantity) for the scenario"
    )
    
    @field_validator("pct_change", "target_unit_value", mode="after")
    @classmethod
    def check_exactly_one(cls, v, info):
        # This will be validated at the scenario level
        return v


class ScenarioRequest(BaseModel):
    """Complete scenario request."""
    model_id: Annotated[str, Field(min_length=1, description="Fitted model identifier")]
    baseline_month: Annotated[str, Field(min_length=1, description="Baseline month identifier (e.g., '2024-12' or 'Month 24')")]
    
    price_changes: Annotated[
        list[PriceChange],
        Field(min_length=1, description="At least one price change required")
    ]
    
    response_profile: Literal["lower", "central", "higher"] = Field(
        default="central", description="Price responsiveness calibration profile"
    )
    
    baseline_mode: Literal["anchored", "fitted"] = Field(
        default="anchored", description="Anchored reproduces observed baseline exactly; fitted uses model predictions"
    )
    
    @field_validator("price_changes", mode="after")
    @classmethod
    def validate_price_changes(cls, v: list[PriceChange]) -> list[PriceChange]:
        for pc in v:
            has_pct = pc.pct_change is not None
            has_target = pc.target_unit_value is not None
            if has_pct == has_target:  # XOR - exactly one must be provided
                raise ValueError(
                    f"Each price change must have exactly one of pct_change or target_unit_value. "
                    f"Got pct_change={pc.pct_change}, target_unit_value={pc.target_unit_value}"
                )
            if has_pct and not (-0.99 < pc.pct_change <= 10.0):
                raise ValueError(f"pct_change must be in (-0.99, 10.0], got {pc.pct_change}")
            if has_target and pc.target_unit_value <= 0:
                raise ValueError(f"target_unit_value must be > 0, got {pc.target_unit_value}")
        return v


class ResolvedPriceChange(BaseModel):
    """Price change after resolution against baseline data."""
    sku_id: str
    retailer_id: str
    category_id: str | None = None
    brand_id: str | None = None
    pack_liters: float | None = None
    baseline_unit_value: float | None = None  # sales/quantity at baseline
    baseline_price: float | None = None       # alias for baseline_unit_value
    proposed_unit_value: float
    proposed_price: float                     # alias for proposed_unit_value
    pct_change: float
    baseline_quantity: int | None = None
    baseline_revenue: float | None = None
    historical_unit_value_min: float | None = None
    historical_unit_value_max: float | None = None
    is_extrapolation: bool = False
    is_zero_anchor: bool = False  # no positive baseline sales
    support_status: Literal["in_range", "out_of_range", "unknown", "zero_anchor"] = "unknown"


class ResolvedScenarioRequest(BaseModel):
    """Scenario request with all price changes resolved against baseline data."""
    model_id: str
    baseline_month: str
    price_changes: list[ResolvedPriceChange]
    response_profile: Literal["lower", "central", "higher"]
    baseline_mode: Literal["anchored", "fitted"]
    eligible_market_coverage_pct: float = 0.0
    warnings: list[str] = Field(default_factory=list)


# =============================================================================
# Scenario Output Schemas
# =============================================================================


class MetricSummary(BaseModel):
    """Summary statistics for a metric across posterior draws."""
    unit: Literal["units", "liters", "revenue", "share_pct"]
    provenance: ProvenanceType
    baseline: float
    scenario: float
    absolute_delta: float
    pct_delta: float
    # Posterior uncertainty (if available)
    q025: float | None = None
    q975: float | None = None
    std: float | None = None


class SegmentResult(BaseModel):
    """Results for a single segment (SKU, brand, category, retailer, etc.)."""
    segment_type: Literal["sku", "brand", "category", "retailer", "pack_size", "total_market"]
    segment_key: dict[str, str]  # e.g., {"category": "CSD", "brand": "ColaClassic"}
    metrics: dict[str, MetricSummary]  # keyed by "units", "liters", "revenue", "share_pct"
    price_change_pct: float | None = None  # for edited SKU×retailer cells
    provenance: ProvenanceType = "modeled"


class RevenueBridgeComponent(BaseModel):
    """Single component in a revenue bridge decomposition."""
    label: str
    value: float
    description: str
    provenance: ProvenanceType


class RevenueBridge(BaseModel):
    """Complete revenue bridge from baseline to scenario."""
    baseline_revenue: float
    components: list[RevenueBridgeComponent]
    scenario_revenue: float
    total_change: float
    provenance: ProvenanceType = "descriptive"


class FlowMatrixCell(BaseModel):
    """Single cell in the liter flow matrix."""
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
    out_of_support_flags: list[str] = Field(default_factory=list)
    provenance_type: ProvenanceType = "modeled"
    unit: Literal["units", "liters", "revenue", "share_pct"]
    posterior_summary: dict[str, MetricSummary] = Field(default_factory=dict)
    segment_results: list[SegmentResult] = Field(default_factory=list)
    revenue_bridge: RevenueBridge | None = None
    flow_matrix: FlowMatrix | None = None
    generated_at: str  # ISO timestamp
    seed: int


# =============================================================================
# Scenario Resolution Functions
# =============================================================================

def resolve_scenario_request(
    request: ScenarioRequest,
    baseline_data: pd.DataFrame,
    sku_master: pd.DataFrame
) -> ResolvedScenarioRequest:
    """
    Resolve a scenario request against baseline data.
    
    Validates that all SKU–retailer cells exist in baseline, computes proposed prices,
    checks historical support, and identifies extrapolation/zero-anchor cases.
    """
    from .inputs import validate_sales_monthly, validate_sku_master
    
    # Parse baseline month - handle both "YYYY-MM" and "Month N" formats
    baseline_month_str = request.baseline_month.strip()
    target_period = None
    target_month_label = None
    
    if baseline_month_str.lower().startswith("month "):
        # Handle "Month N" format
        try:
            target_period = int(baseline_month_str.split()[1])
            target_month_label = f"Month {target_period}"
        except (IndexError, ValueError):
            raise ValueError(f"Invalid 'Month N' format: {request.baseline_month}")
    else:
        # Try parsing as YYYY-MM
        try:
            period_obj = pd.Period(baseline_month_str, freq="M")
            target_period = period_obj.ordinal  # This might not match our data periods
            target_month_label = period_obj.strftime("%Y-%m")
        except Exception:
            raise ValueError(f"Unable to parse baseline month: {request.baseline_month}. Use 'YYYY-MM' or 'Month N' format.")
    
    # Filter baseline to the selected month
    # Our data uses "Period" column for integer periods and "Month" column for "Month N" labels
    if target_period is not None:
        # Try to match by Period column first (integer periods)
        if "Period" in baseline_data.columns:
            baseline_month_data = baseline_data[baseline_data["Period"] == target_period].copy()
        elif "Month" in baseline_data.columns:
            # Fall back to Month column
            baseline_month_data = baseline_data[baseline_data["Month"] == target_month_label].copy()
        else:
            raise ValueError("Baseline data missing 'Period' or 'Month' column")
    else:
        # Use datetime match
        if baseline_dt is not None:
            baseline_month_data = baseline_data[baseline_data["Month"] == baseline_dt].copy()
        else:
            raise ValueError("Could not determine baseline month filter")
    
    if baseline_month_data.empty:
        raise ValueError(f"No data found for baseline month {request.baseline_month}")
    
    # Build SKU master lookup
    sku_lookup = sku_master.set_index("sku_id") if not sku_master.empty else pd.DataFrame()
    
    resolved_changes = []
    warnings = []
    eligible_volume = 0.0
    total_volume = 0.0
    
    for pc in request.price_changes:
        # Find the cell in baseline
        mask = (
            (baseline_month_data["SKU"] == pc.sku_id) & 
            (baseline_month_data["Retailer"] == pc.retailer_id)
        )
        matches = baseline_month_data[mask]
        
        if matches.empty:
            raise ValueError(
                f"SKU {pc.sku_id} at retailer {pc.retailer_id} not found in baseline month {request.baseline_month}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple matches for SKU {pc.sku_id} at retailer {pc.retailer_id} in baseline"
            )
        
        row = matches.iloc[0]
        baseline_uv = row["Sales_Revenue"] / row["Unit_Sales"] if row["Unit_Sales"] > 0 else None
        baseline_qty = int(row["Unit_Sales"])
        baseline_rev = float(row["Sales_Revenue"])
        pack_liters = float(row["Pack_Size"])
        
        # Get SKU metadata
        cat = row.get("Category")
        brand = row.get("Brand")
        
        # Compute proposed unit value
        if pc.pct_change is not None:
            if baseline_uv is None:
                raise ValueError(
                    f"Cannot apply percentage change to SKU {pc.sku_id} at {pc.retailer_id}: "
                    "no baseline unit value (zero baseline sales)"
                )
            proposed_uv = baseline_uv * (1 + pc.pct_change)
            pct = pc.pct_change
        else:
            proposed_uv = pc.target_unit_value
            pct = (proposed_uv / baseline_uv - 1) if baseline_uv and baseline_uv > 0 else None
        
        # Check historical support
        # Get historical unit value range for this SKU–retailer
        hist_data = baseline_data[
            (baseline_data["SKU"] == pc.sku_id) & 
            (baseline_data["Retailer"] == pc.retailer_id) &
            (baseline_data["Unit_Sales"] > 0)
        ]
        
        is_zero_anchor = baseline_qty == 0
        is_extrapolation = False
        support_status = "zero_anchor" if is_zero_anchor else "unknown"
        hist_min = hist_max = None
        
        if not hist_data.empty:
            hist_uv = hist_data["Sales_Revenue"] / hist_data["Unit_Sales"]
            hist_min = float(hist_uv.min())
            hist_max = float(hist_uv.max())
            
            if proposed_uv < hist_min * 0.999 or proposed_uv > hist_max * 1.001:
                is_extrapolation = True
                support_status = "out_of_range"
                warnings.append(
                    f"SKU {pc.sku_id} at {pc.retailer_id}: proposed unit value {proposed_uv:.4f} "
                    f"outside historical range [{hist_min:.4f}, {hist_max:.4f}]"
                )
            else:
                support_status = "in_range"
        elif not is_zero_anchor:
            support_status = "unknown"
            warnings.append(
                f"SKU {pc.sku_id} at {pc.retailer_id}: no historical positive-sales data for price validation"
            )
        
        resolved_changes.append(ResolvedPriceChange(
            sku_id=pc.sku_id,
            retailer_id=pc.retailer_id,
            category_id=cat,
            brand_id=brand,
            pack_liters=pack_liters,
            baseline_unit_value=baseline_uv,
            baseline_price=baseline_uv,
            proposed_unit_value=proposed_uv,
            proposed_price=proposed_uv,
            pct_change=pct if pct is not None else 0.0,
            baseline_quantity=baseline_qty,
            baseline_revenue=baseline_rev,
            historical_unit_value_min=hist_min,
            historical_unit_value_max=hist_max,
            is_extrapolation=is_extrapolation,
            is_zero_anchor=is_zero_anchor,
            support_status=support_status,
        ))
        
        # Track eligible market coverage
        if baseline_qty > 0:
            eligible_volume += baseline_qty * pack_liters
        total_volume += baseline_qty * pack_liters
    
    eligible_pct = (eligible_volume / total_volume * 100) if total_volume > 0 else 0.0
    
    # Add extrapolation warnings
    extrapolation_count = sum(1 for c in resolved_changes if c.is_extrapolation)
    if extrapolation_count > 0:
        warnings.append(
            f"{extrapolation_count} price change(s) exceed historical unit value range. "
            "Results are model extrapolations without data support."
        )
    
    zero_anchor_count = sum(1 for c in resolved_changes if c.is_zero_anchor)
    if zero_anchor_count > 0:
        warnings.append(
            f"{zero_anchor_count} edited cell(s) have zero baseline sales (zero anchor). "
            "No elasticity can be estimated; quantity response assumed zero."
        )
    
    return ResolvedScenarioRequest(
        model_id=request.model_id,
        baseline_month=request.baseline_month,
        price_changes=resolved_changes,
        response_profile=request.response_profile,
        baseline_mode=request.baseline_mode,
        eligible_market_coverage_pct=eligible_pct,
        warnings=warnings,
    )