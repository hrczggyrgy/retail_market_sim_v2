"""
Input data schemas and validation for the Beverage Market Price Simulator.

Defines the canonical data contracts for SKU master data and monthly sales data.
All validation is strict; problematic records are reported, never silently dropped.
"""
from datetime import date
from decimal import Decimal
from typing import Annotated, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd


# =============================================================================
# SKU Master Schema
# =============================================================================

class SKUMasterRow(BaseModel):
    """Single SKU master record."""
    sku_id: Annotated[str, Field(min_length=1, description="Unique, stable SKU identifier")]
    category_id: Annotated[str, Field(min_length=1, description="Primary category (one per SKU)")]
    brand_id: Annotated[str, Field(min_length=1, description="Brand identifier (may span categories)")]
    pack_liters: Annotated[float, Field(gt=0, description="Package size in liters, strictly positive")]
    sku_name: str | None = Field(default=None, description="Optional human-readable label")
    attributes: dict | None = Field(default=None, description="Optional future attributes")


class SKUMaster(BaseModel):
    """SKU master data container with validation."""
    skus: list[SKUMasterRow]
    
    @field_validator("skus", mode="after")
    @classmethod
    def check_unique_sku_ids(cls, v: list[SKUMasterRow]) -> list[SKUMasterRow]:
        ids = [s.sku_id for s in v]
        if len(ids) != len(set(ids)):
            dupes = [x for x in ids if ids.count(x) > 1]
            raise ValueError(f"Duplicate sku_ids found: {set(dupes)}")
        return v
    
    @field_validator("skus", mode="after")
    @classmethod
    def check_pack_liters_consistency(cls, v: list[SKUMasterRow]) -> list[SKUMasterRow]:
        # A SKU should have consistent pack_liters (enforced by unique sku_id)
        # But check for any suspicious values
        for s in v:
            if s.pack_liters <= 0:
                raise ValueError(f"SKU {s.sku_id}: pack_liters must be > 0, got {s.pack_liters}")
        return v


# =============================================================================
# Monthly Sales Schema
# =============================================================================

class SalesMonthlyRow(BaseModel):
    """Single monthly sales record for SKU × retailer."""
    month: Annotated[date, Field(description="First day of calendar month (normalized)")]
    sku_id: Annotated[str, Field(min_length=1, description="Foreign key to SKU master")]
    retailer_id: Annotated[str, Field(min_length=1, description="Stable retailer identifier")]
    quantity: Annotated[int, Field(ge=0, description="Units sold (nonnegative, net-sales convention)")]
    sales: Annotated[Decimal, Field(ge=0, description="Revenue in consistent currency")]


class SalesMonthly(BaseModel):
    """Monthly sales data container with validation."""
    records: list[SalesMonthlyRow]
    
    @field_validator("records", mode="after")
    @classmethod
    def check_natural_key(cls, v: list[SalesMonthlyRow]) -> list[SalesMonthlyRow]:
        """Natural key is (month, sku_id, retailer_id). Report duplicates, don't collapse silently."""
        seen = {}
        for i, r in enumerate(v):
            key = (r.month, r.sku_id, r.retailer_id)
            if key in seen:
                raise ValueError(
                    f"Duplicate natural key at records {seen[key]} and {i}: "
                    f"month={r.month}, sku_id={r.sku_id}, retailer_id={r.retailer_id}. "
                    "Do not automatically collapse duplicates without explicit import rule."
                )
            seen[key] = i
        return v


# =============================================================================
# Data Quality Report
# =============================================================================

class QualityIssue(BaseModel):
    """Single data quality finding."""
    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    affected_rows: list[int] | None = None
    details: dict | None = None


class DataQualityReport(BaseModel):
    """Complete data quality report from import/validation."""
    total_rows: int
    total_skus: int
    total_retailers: int
    total_months: int
    date_range: tuple[date, date] | None = None
    issues: list[QualityIssue] = Field(default_factory=list)
    
    # Coverage metrics
    positive_sales_rows: int = 0
    zero_sales_rows: int = 0
    missing_unit_value_rows: int = 0
    model_eligible_rows: int = 0
    model_eligible_volume_pct: float = 0.0
    
    # Anomalies
    outlier_unit_values: int = 0
    sharp_coding_changes: int = 0
    sparse_cells: int = 0
    
    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_sku_master(df: pd.DataFrame) -> tuple[SKUMaster, DataQualityReport]:
    """
    Validate SKU master DataFrame against schema.
    Returns validated model and quality report.
    """
    issues = []
    
    # Check required columns
    required = {"sku_id", "category_id", "brand_id", "pack_liters"}
    missing = required - set(df.columns)
    if missing:
        issues.append(QualityIssue(
            severity="error", code="MISSING_COLUMNS",
            message=f"SKU master missing required columns: {missing}"
        ))
        return SKUMaster(skus=[]), DataQualityReport(
            total_rows=0, total_skus=0, total_retailers=0, total_months=0, issues=issues
        )
    
    # Check for missing/blank values
    for col in ["sku_id", "category_id", "brand_id", "pack_liters"]:
        null_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if null_mask.any():
            issues.append(QualityIssue(
                severity="error", code="NULL_VALUES",
                message=f"Column '{col}' contains {null_mask.sum()} missing/blank values",
                affected_rows=df.index[null_mask].tolist()
            ))
    
    # Check pack_liters > 0
    if "pack_liters" in df.columns:
        nonpos = pd.to_numeric(df["pack_liters"], errors="coerce") <= 0
        if nonpos.any():
            issues.append(QualityIssue(
                severity="error", code="NONPOSITIVE_PACK_LITERS",
                message=f"{nonpos.sum()} SKUs have non-positive pack_liters",
                affected_rows=df.index[nonpos].tolist()
            ))
    
    # Check duplicate sku_ids
    dup_mask = df.duplicated(subset=["sku_id"], keep=False)
    if dup_mask.any():
        issues.append(QualityIssue(
            severity="error", code="DUPLICATE_SKU_IDS",
            message=f"{dup_mask.sum()} rows have duplicate sku_ids",
            affected_rows=df.index[dup_mask].tolist()
        ))
    
    # Build validated objects (only if no errors)
    skus = []
    if not any(i.severity == "error" for i in issues):
        for _, row in df.iterrows():
            skus.append(SKUMasterRow(
                sku_id=str(row["sku_id"]).strip(),
                category_id=str(row["category_id"]).strip(),
                brand_id=str(row["brand_id"]).strip(),
                pack_liters=float(row["pack_liters"]),
                sku_name=str(row["sku_name"]).strip() if "sku_name" in df.columns and pd.notna(row["sku_name"]) else None,
                attributes=row["attributes"] if "attributes" in df.columns and pd.notna(row["attributes"]) else None,
            ))
    
    report = DataQualityReport(
        total_rows=len(df),
        total_skus=df["sku_id"].nunique() if "sku_id" in df.columns else 0,
        total_retailers=0,
        total_months=0,
        issues=issues
    )
    
    return SKUMaster(skus=skus), report


def validate_sales_monthly(df: pd.DataFrame, sku_master: SKUMaster | None = None) -> tuple[SalesMonthly, DataQualityReport]:
    """
    Validate monthly sales DataFrame against schema.
    Returns validated model and quality report.
    """
    issues = []
    
    # Check required columns
    required = {"month", "sku_id", "retailer_id", "quantity", "sales"}
    missing = required - set(df.columns)
    if missing:
        issues.append(QualityIssue(
            severity="error", code="MISSING_COLUMNS",
            message=f"Sales monthly missing required columns: {missing}"
        ))
        return SalesMonthly(records=[]), DataQualityReport(
            total_rows=0, total_skus=0, total_retailers=0, total_months=0, issues=issues
        )
    
    # Normalize month to first day of month
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    null_months = df["month"].isna()
    if null_months.any():
        issues.append(QualityIssue(
            severity="error", code="INVALID_MONTH",
            message=f"{null_months.sum()} rows have unparseable month values",
            affected_rows=df.index[null_months].tolist()
        ))
    
    # Normalize to first of month
    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
    
    # Check quantity >= 0
    neg_qty = df["quantity"] < 0
    if neg_qty.any():
        issues.append(QualityIssue(
            severity="error", code="NEGATIVE_QUANTITY",
            message=f"{neg_qty.sum()} rows have negative quantity",
            affected_rows=df.index[neg_qty].tolist()
        ))
    
    # Check sales >= 0
    neg_sales = df["sales"] < 0
    if neg_sales.any():
        issues.append(QualityIssue(
            severity="error", code="NEGATIVE_SALES",
            message=f"{neg_sales.sum()} rows have negative sales",
            affected_rows=df.index[neg_sales].tolist()
        ))
    
    # Check quantity=0 implies sales=0
    zero_qty_nonzero_sales = (df["quantity"] == 0) & (df["sales"] != 0)
    if zero_qty_nonzero_sales.any():
        issues.append(QualityIssue(
            severity="error", code="ZERO_QTY_NONZERO_SALES",
            message=f"{zero_qty_nonzero_sales.sum()} rows have zero quantity but non-zero sales",
            affected_rows=df.index[zero_qty_nonzero_sales].tolist()
        ))
    
    # Check positive sales have positive revenue for log-price models
    pos_qty = df["quantity"] > 0
    if pos_qty.any():
        implied_price = df.loc[pos_qty, "sales"] / df.loc[pos_qty, "quantity"]
        nonpos_implied = implied_price <= 0
        if nonpos_implied.any():
            idx = df.index[pos_qty][nonpos_implied]
            issues.append(QualityIssue(
                severity="error", code="NONPOSITIVE_IMPLIED_PRICE",
                message=f"{nonpos_implied.sum()} positive-quantity rows have non-positive implied price",
                affected_rows=idx.tolist()
            ))
    
    # Check natural key uniqueness
    dup_mask = df.duplicated(subset=["month", "sku_id", "retailer_id"], keep=False)
    if dup_mask.any():
        issues.append(QualityIssue(
            severity="error", code="DUPLICATE_NATURAL_KEY",
            message=f"{dup_mask.sum()} rows have duplicate (month, sku_id, retailer_id). "
                    "Do not automatically collapse without explicit import rule.",
            affected_rows=df.index[dup_mask].tolist()
        ))
    
    # Cross-reference with SKU master if provided
    if sku_master is not None:
        valid_skus = {s.sku_id for s in sku_master.skus}
        invalid_sku = ~df["sku_id"].isin(valid_skus)
        if invalid_sku.any():
            issues.append(QualityIssue(
                severity="error", code="INVALID_SKU_REFERENCE",
                message=f"{invalid_sku.sum()} rows reference SKUs not in master data",
                affected_rows=df.index[invalid_sku].tolist()
            ))
    
    # Build validated objects
    records = []
    if not any(i.severity == "error" for i in issues):
        for _, row in df.iterrows():
            records.append(SalesMonthlyRow(
                month=row["month"].date() if hasattr(row["month"], "date") else row["month"],
                sku_id=str(row["sku_id"]).strip(),
                retailer_id=str(row["retailer_id"]).strip(),
                quantity=int(row["quantity"]),
                sales=Decimal(str(row["sales"])),
            ))
    
    # Compute coverage metrics
    total_rows = len(df)
    pos_sales = int((df["quantity"] > 0).sum()) if "quantity" in df.columns else 0
    zero_sales = int((df["quantity"] == 0).sum()) if "quantity" in df.columns else 0
    
    # Unit value coverage
    missing_uv = 0
    if "quantity" in df.columns and "sales" in df.columns:
        missing_uv = int(((df["quantity"] > 0) & (df["sales"].isna() | (df["sales"] == 0))).sum())
    
    report = DataQualityReport(
        total_rows=total_rows,
        total_skus=df["sku_id"].nunique() if "sku_id" in df.columns else 0,
        total_retailers=df["retailer_id"].nunique() if "retailer_id" in df.columns else 0,
        total_months=df["month"].nunique() if "month" in df.columns else 0,
        date_range=(
            df["month"].min().date(),
            df["month"].max().date()
        ) if "month" in df.columns and df["month"].notna().any() else None,
        issues=issues,
        positive_sales_rows=pos_sales,
        zero_sales_rows=zero_sales,
        missing_unit_value_rows=missing_uv,
    )
    
    return SalesMonthly(records=records), report