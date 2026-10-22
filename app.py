"""
app.py — Bayesian Retail Price Simulation Dashboard

Install:
    pip install streamlit pandas numpy pymc plotly

Run:
    streamlit run app.py

Place sample_data.csv in the same directory as this file.

Repository references reviewed for this implementation:
    retail_market_model.md
    sample_data.csv

Important implementation decisions:
    1. Numeric Month values such as 1,...,24 are period numbers, NOT timestamps.
    2. Baseline prices are reconciled to Sales_Revenue / Unit_Sales.
    3. The PyMC model estimates partially pooled SKU elasticities with retailer,
       SKU-retailer, category-seasonal, and category-trend controls.
    4. MAP is a Bayesian point estimate, not a posterior distribution.
    5. Scenarios are anchored to observed latest-month quantities so unchanged
       prices reproduce observed baseline revenue exactly.
    6. Cached PyMC objects are never mutated during interactive simulations.
    7. Market-wide results aggregate own-price responses; cross-price effects
       and competitor reactions are NOT estimated by this application.
"""

from pathlib import Path
import hashlib
import json
import logging
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymc as pm
import streamlit as st


# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="Bayesian Retail Price Simulator",
    page_icon="🛒",
    layout="wide",
)

DATA_PATH = Path(__file__).resolve().parent / "sample_data.csv"
MODEL_VERSION = "retail-map-v1"
RANDOM_SEED = 42

PRODUCT_KEYS = ["Category", "Brand", "SKU"]
CELL_KEYS = ["Category", "Retailer", "Brand", "SKU"]

REQUIRED_COLUMNS = [
    "Month",
    "Category",
    "Retailer",
    "Brand",
    "SKU",
    "Pack_Size",
    "Unit_Sales",
    "Sales_Revenue",
    "Price",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Data loading, validation, and aggregation
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data(path: str, modified_ns: int) -> tuple:
    """
    Load, validate, and aggregate monthly POS data.

    modified_ns participates in the cache key so changing the local CSV
    invalidates the cached dataset.

    Zero-sales rows are retained for reporting, but excluded later from the
    log-demand likelihood. No arbitrary log(Q + 1) transformation is used.
    """
    del modified_ns

    raw = pd.read_csv(
        path,
        dtype={
            "Category": "string",
            "Retailer": "string",
            "Brand": "string",
            "SKU": "string",
            "Pack_Size": "string",
        },
    )
    raw.columns = raw.columns.str.strip()

    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if raw.empty:
        raise ValueError("The CSV contains no observations.")

    notes = []

    # Preserve identifiers and pack descriptions; Pack_Size may contain text
    # such as '500 g', '1.5 L', or '6 pc'.
    for column in CELL_KEYS + ["Pack_Size"]:
        raw[column] = raw[column].astype("string").str.strip()
        if raw[column].isna().any() or raw[column].eq("").any():
            raise ValueError(f"{column} contains missing or blank values.")

    for column in ["Unit_Sales", "Sales_Revenue", "Price"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
        if not np.isfinite(raw[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains missing or non-finite values.")

    if (raw["Unit_Sales"] < 0).any() or (raw["Sales_Revenue"] < 0).any():
        raise ValueError(
            "Negative sales/returns require separate treatment before modeling."
        )

    if (raw["Price"] <= 0).any():
        raise ValueError("All supplied prices must be strictly positive.")

    if (
        raw["Unit_Sales"].eq(0)
        & raw["Sales_Revenue"].ne(0)
    ).any():
        raise ValueError("Rows with zero units must have zero revenue.")

    # The repository sample uses Month = 1, 2, ..., 24.
    # Date-formatted inputs are supported without interpreting integers as
    # nanoseconds since the Unix epoch.
    numeric_month = pd.to_numeric(raw["Month"], errors="coerce")
    is_period_number = (
        numeric_month.notna().all()
        and np.isfinite(numeric_month.to_numpy(dtype=float)).all()
        and np.allclose(numeric_month, np.round(numeric_month))
        and numeric_month.between(1, 1200).all()
    )

    if is_period_number:
        raw["Period"] = numeric_month.astype(int)
        raw["Season"] = ((raw["Period"] - 1) % 12).astype(int)

        period_labels = {
            int(value): f"Month {int(value)}"
            for value in sorted(raw["Period"].unique())
        }
        notes.append(
            "Month is a numeric period index. Seasonal position assumes "
            "periods repeat every 12 months; calendar month names are not inferred."
        )
    else:
        parsed = pd.to_datetime(
            raw["Month"].astype(str),
            errors="coerce",
        )
        if parsed.isna().any():
            raise ValueError(
                "Month must contain positive integer periods or parseable dates."
            )

        # Month ordinals preserve missing calendar months rather than compressing
        # them into consecutive observations.
        month_ordinals = parsed.dt.year * 12 + parsed.dt.month
        raw["Period"] = (
            month_ordinals - int(month_ordinals.min()) + 1
        ).astype(int)
        raw["Season"] = (parsed.dt.month - 1).astype(int)

        period_labels = (
            pd.DataFrame(
                {
                    "Period": raw["Period"],
                    "Label": parsed.dt.strftime("%Y-%m"),
                }
            )
            .drop_duplicates("Period")
            .set_index("Period")["Label"]
            .to_dict()
        )

    # A stable SKU identity must also have a stable pack definition. Reused SKU
    # identifiers for resized products should be versioned upstream.
    pack_counts = raw.groupby(PRODUCT_KEYS, observed=True)["Pack_Size"].nunique()
    if (pack_counts > 1).any():
        raise ValueError(
            "Some Category/Brand/SKU identities have multiple Pack_Size values. "
            "Version resized SKUs or standardize pack descriptions first."
        )

    # Explain any discrepancy between supplied Price and the accounting price.
    positive_units = raw["Unit_Sales"] > 0
    implied_price = (
        raw.loc[positive_units, "Sales_Revenue"]
        / raw.loc[positive_units, "Unit_Sales"]
    )
    mismatch = ~np.isclose(
        raw.loc[positive_units, "Price"],
        implied_price,
        rtol=1e-4,
        atol=1e-6,
    )

    if mismatch.any():
        notes.append(
            f"{int(mismatch.sum()):,} source rows have supplied Price values "
            "different from Revenue / Units. Accounting-consistent prices "
            "are used throughout."
        )

    if (implied_price <= 0).any():
        raise ValueError(
            "Positive unit sales must have positive revenue for this log-price model."
        )

    # Aggregate duplicate monthly cells by SUMMING revenue and units.
    # Never average SKU prices to construct an aggregate selling price.
    monthly = (
        raw.groupby(
            ["Period", "Season"] + CELL_KEYS,
            as_index=False,
            observed=True,
        )
        .agg(
            Unit_Sales=("Unit_Sales", "sum"),
            Sales_Revenue=("Sales_Revenue", "sum"),
            Supplied_Price=("Price", "mean"),
            Pack_Size=("Pack_Size", "first"),
        )
    )

    monthly["Price"] = monthly["Supplied_Price"].astype(float)
    has_units = monthly["Unit_Sales"] > 0
    monthly.loc[has_units, "Price"] = (
        monthly.loc[has_units, "Sales_Revenue"]
        / monthly.loc[has_units, "Unit_Sales"]
    )
    monthly = monthly.drop(columns="Supplied_Price")
    monthly["Month"] = monthly["Period"].map(period_labels)
    monthly = monthly.sort_values(
        ["Period"] + CELL_KEYS
    ).reset_index(drop=True)

    periods = sorted(monthly["Period"].unique())
    if len(periods) != 24:
        notes.append(
            f"The input contains {len(periods)} observed monthly periods, not 24."
        )

    if len(periods) < 12:
        notes.append(
            "Fewer than 12 months are available; annual seasonality is weakly identified."
        )

    if len(periods) > 1 and np.any(np.diff(periods) != 1):
        notes.append(
            "Some monthly periods are missing. Their sales are not imputed."
        )

    zero_count = int(monthly["Unit_Sales"].eq(0).sum())
    if zero_count:
        notes.append(
            f"{zero_count:,} zero-unit monthly cells are retained in reporting "
            "but excluded from the log-demand fit."
        )

    return monthly, notes


def tuple_index(frame: pd.DataFrame, columns: list) -> pd.MultiIndex:
    """Create collision-safe composite identities without concatenated strings."""
    return pd.MultiIndex.from_frame(frame[columns])


# =============================================================================
# Bayesian model fitting
# =============================================================================

@st.cache_resource(show_spinner=False, max_entries=2)
def fit_bayesian_model(
    data: pd.DataFrame,
    prior_elasticity: float,
    max_evaluations: int,
    model_version: str,
) -> dict:
    """
    Fit a regularized hierarchical PyMC log-log demand model using MAP.

    Model:
        log Q_i =
            alpha_SKU[i]
            + retailer_effect[r_i]
            + pair_effect[SKU_i, r_i]
            + beta_SKU[i] * centered_log_price_i
            + season_effect[category_i, season_i]
            + trend_effect[category_i] * centered_time_i
            + error_i

    SKU elasticities partially pool toward category-level elasticities.
    Fixed conditional prior scales avoid variance-collapse pathologies that
    can occur in joint MAP optimization of hierarchical scale parameters.

    Price is centered within SKU-retailer cells, emphasizing within-cell
    historical variation instead of comparing unrelated SKU price levels.

    Thread safety:
        The PyMC model is local to this function and is not returned or mutated.
        The cached result contains only fitted parameters and diagnostics.
    """
    del model_version

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

    category_idx = pd.Categorical(
        train["Category"], categories=categories
    ).codes.astype("int32")
    retailer_idx = pd.Categorical(
        train["Retailer"], categories=retailers
    ).codes.astype("int32")
    product_category_idx = pd.Categorical(
        products["Category"], categories=categories
    ).codes.astype("int32")

    season_idx = train["Season"].to_numpy(dtype="int32")
    log_q = np.log(train["Unit_Sales"].to_numpy(dtype=float))
    log_p = np.log(train["Price"].to_numpy(dtype=float))

    # Center prices within each SKU-retailer pair.
    pair_price_means = (
        pd.Series(log_p)
        .groupby(pair_idx)
        .transform("mean")
        .to_numpy()
    )
    x_price = log_p - pair_price_means

    # One unit of the trend predictor corresponds to one year.
    time_years = (
        train["Period"].to_numpy(dtype=float)
        - train["Period"].mean()
    ) / 12.0

    coords = {
        "product": np.arange(len(products)),
        "pair": np.arange(len(pairs)),
        "retailer": retailers,
        "category": categories,
        "season": np.arange(12),
        "observation": np.arange(len(train)),
    }

    with pm.Model(coords=coords):
        # Proper priors keep sparse SKU estimates regularized.
        alpha = pm.Normal(
            "alpha",
            mu=float(log_q.mean()),
            sigma=2.0,
            dims="product",
        )

        beta_category = pm.Normal(
            "beta_category",
            mu=prior_elasticity,
            sigma=0.75,
            dims="category",
        )
        beta_offset = pm.Normal(
            "beta_offset",
            mu=0.0,
            sigma=1.0,
            dims="product",
        )
        beta = pm.Deterministic(
            "beta",
            beta_category[product_category_idx] + 0.5 * beta_offset,
            dims="product",
        )

        retailer_raw = pm.Normal(
            "retailer_raw",
            mu=0.0,
            sigma=0.5,
            dims="retailer",
        )
        retailer_effect = (
            retailer_raw - pm.math.sum(retailer_raw) / len(retailers)
        )

        # SKU-retailer heterogeneity helps avoid treating persistent differences
        # in a retailer's SKU performance as price responses.
        pair_effect = pm.Normal(
            "pair_effect",
            mu=0.0,
            sigma=0.5,
            dims="pair",
        )

        season_raw = pm.Normal(
            "season_raw",
            mu=0.0,
            sigma=0.3,
            dims=("category", "season"),
        )
        season_effect = (
            season_raw
            - pm.math.sum(season_raw, axis=1, keepdims=True) / 12.0
        )

        trend = pm.Normal(
            "trend",
            mu=0.0,
            sigma=0.3,
            dims="category",
        )
        sigma = pm.HalfNormal("sigma", sigma=0.75)

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

        # MAP is intentionally used instead of MCMC for interactive startup.
        # Compilation and optimization are cached across scenario edits.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            point, optimization = pm.find_MAP(
                method="L-BFGS-B",
                maxeval=int(max_evaluations),
                progressbar=False,
                return_raw=True,
                seed=RANDOM_SEED,
            )

    # Reconstruct deterministic elasticities from unconstrained MAP parameters.
    beta_map = (
        np.asarray(point["beta_category"], dtype=float)[product_category_idx]
        + 0.5 * np.asarray(point["beta_offset"], dtype=float)
    )
    if not np.isfinite(beta_map).all():
        raise RuntimeError("The optimizer returned non-finite elasticities.")

    beta_map.setflags(write=False)

    # Report data support rather than presenting every MAP estimate as equally
    # well identified.
    support = (
        train.assign(_log_price=log_p, _within_log_price=x_price)
        .groupby(PRODUCT_KEYS, observed=True)
        .agg(
            Positive_Observations=("Unit_Sales", "size"),
            Observed_Months=("Period", "nunique"),
            Distinct_Prices=("Price", "nunique"),
            Within_Pair_LogPrice_SD=(
                "_within_log_price",
                lambda values: float(np.std(values, ddof=0)),
            ),
        )
        .reset_index()
    )

    estimates = products.copy()
    estimates["Elasticity_MAP"] = beta_map
    estimates = estimates.merge(
        support,
        on=PRODUCT_KEYS,
        how="left",
        validate="one_to_one",
    )
    estimates["Positive_Observations"] = (
        estimates["Positive_Observations"].fillna(0).astype(int)
    )
    estimates["Weak_Price_Support"] = (
        estimates["Positive_Observations"].lt(12)
        | estimates["Within_Pair_LogPrice_SD"].fillna(0).lt(0.02)
    )

    return {
        "estimates": estimates,
        "converged": bool(optimization.success),
        "message": str(optimization.message),
        "iterations": int(getattr(optimization, "nit", 0)),
        "training_rows": len(train),
        "positive_elasticities": int((beta_map > 0).sum()),
        "weak_support_count": int(estimates["Weak_Price_Support"].sum()),
    }


# =============================================================================
# Simulation and share calculations
# =============================================================================

def simulate_market(
    baseline: pd.DataFrame,
    edited_matrix: pd.DataFrame,
    retailer_columns: dict,
    estimates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply edited latest-month prices to all observed SKU-retailer cells.

    Expected-response calculation at the MAP point:
        Q1 = Q0 * exp(beta * log(P1 / P0))

    For a log-normal demand model with unchanged residual variance, the
    log-normal correction cancels in this expected-response ratio.

    Anchoring to observed Q0 preserves the observed latest-month baseline,
    including its seasonal level. This is a same-month counterfactual,
    not a next-month forecast.
    """
    new_prices = edited_matrix.melt(
        id_vars=PRODUCT_KEYS,
        value_vars=list(retailer_columns),
        var_name="_RetailerColumn",
        value_name="Simulated_Price",
    )
    new_prices["Retailer"] = new_prices["_RetailerColumn"].map(
        retailer_columns
    )
    new_prices = new_prices.drop(columns="_RetailerColumn")

    # Left joining to baseline prevents creating nonexistent SKU-retailer cells.
    result = baseline.merge(
        new_prices,
        on=CELL_KEYS,
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        estimates[PRODUCT_KEYS + ["Elasticity_MAP"]],
        on=PRODUCT_KEYS,
        how="left",
        validate="many_to_one",
    )

    p1 = pd.to_numeric(
        result["Simulated_Price"], errors="coerce"
    ).to_numpy(dtype=float)
    p0 = result["Price"].to_numpy(dtype=float)

    if not np.isfinite(p1).all() or (p1 <= 0).any():
        raise ValueError(
            "Every existing SKU-retailer cell needs a finite price above zero."
        )

    beta = result["Elasticity_MAP"].to_numpy(dtype=float)
    if not np.isfinite(beta).all():
        raise ValueError("Some SKU elasticity estimates are missing.")

    log_response = beta * np.log(p1 / p0)

    # Reject numerically explosive scenarios instead of silently clipping them.
    if not np.isfinite(log_response).all() or (
        np.abs(log_response) > 50
    ).any():
        raise ValueError(
            "The scenario implies an extreme demand response. "
            "Use prices closer to baseline and inspect SKU elasticities."
        )

    q0 = result["Unit_Sales"].to_numpy(dtype=float)
    q1 = q0 * np.exp(log_response)
    r1 = p1 * q1

    if not np.isfinite(q1).all() or not np.isfinite(r1).all():
        raise ValueError("The scenario produces non-finite quantities or revenue.")

    result["Baseline_Unit_Sales"] = q0
    result["Baseline_Price"] = p0
    result["Baseline_Revenue"] = result["Sales_Revenue"]
    result["Simulated_Price"] = p1
    result["Simulated_Unit_Sales"] = q1
    result["Simulated_Revenue"] = r1
    result["Price_Change_Pct"] = 100.0 * (p1 / p0 - 1.0)
    result["Revenue_Delta"] = r1 - result["Baseline_Revenue"]

    # Exact SKU-retailer revenue bridge (two-factor, cell-level).
    # Summing these cell-level effects preserves exact market reconciliation.
    result["Price_Effect"] = (p1 - p0) * q1
    result["Volume_Effect"] = (q1 - q0) * p0

    if not np.allclose(
        result["Revenue_Delta"],
        result["Price_Effect"] + result["Volume_Effect"],
        rtol=1e-9,
        atol=1e-6,
    ):
        raise RuntimeError("Price-volume effects do not reconcile to revenue.")

    # Compute equivalized volume for three-factor decomposition at aggregated levels.
    # Pack_Size is a string like '500 g', '1.5 L', '6 pc'; extract numeric factor.
    result["Equiv_Volume_Baseline"] = result.apply(
        lambda row: _pack_size_to_factor(row["Pack_Size"]) * row["Baseline_Unit_Sales"], axis=1
    )
    result["Equiv_Volume_Simulated"] = result.apply(
        lambda row: _pack_size_to_factor(row["Pack_Size"]) * row["Simulated_Unit_Sales"], axis=1
    )
    result["Price_Per_Equiv"] = result["Baseline_Revenue"] / result["Equiv_Volume_Baseline"].replace(0, np.nan)
    result["Simulated_Price_Per_Equiv"] = result["Simulated_Revenue"] / result["Equiv_Volume_Simulated"].replace(0, np.nan)

    return result


def _pack_size_to_factor(pack_size: str) -> float:
    """Extract numeric pack-size factor from string (e.g., '500 g' -> 0.5, '1.5 L' -> 1.5, '6 pc' -> 6)."""
    if pd.isna(pack_size):
        return 1.0
    s = str(pack_size).strip().lower()
    # Try to extract leading number
    import re
    m = re.match(r"([\d.]+)", s)
    if not m:
        return 1.0
    val = float(m.group(1))
    # Heuristic: if unit suggests kg/L and value > 10, assume grams/mL -> convert to kg/L
    if any(u in s for u in ["g", "ml", "mg"]) and val > 10:
        val = val / 1000.0
    return val


def three_factor_decomposition(
    scenario: pd.DataFrame,
    group_columns: list,
) -> pd.DataFrame:
    """
    Compute Volume / Mix / Price decomposition at an aggregated level.

    Uses the sequential convention (Volume -> Mix -> Price) with equivalized volumes:
      VE = (Q1 - Q0) * Pbar0
      ME = Q1 * sum_s (s1_s - s0_s) * P0_s
      PE = sum_s V1_s * (P1_s - P0_s)

    where Q = sum V (equivalized volume), s = V/Q (share), P = revenue per equiv volume.
    """
    # Add equivalized volumes if not present
    if "Equiv_Volume_Baseline" not in scenario.columns:
        scenario = scenario.copy()
        scenario["Equiv_Volume_Baseline"] = scenario.apply(
            lambda row: _pack_size_to_factor(row["Pack_Size"]) * row["Baseline_Unit_Sales"], axis=1
        )
        scenario["Equiv_Volume_Simulated"] = scenario.apply(
            lambda row: _pack_size_to_factor(row["Pack_Size"]) * row["Simulated_Unit_Sales"], axis=1
        )

    # Handle empty group_columns (total market)
    if not group_columns:
        scenario = scenario.copy()
        scenario["_total_market"] = 1
        group_columns = ["_total_market"]
        drop_total = True
    else:
        drop_total = False

    g = scenario.groupby(group_columns, as_index=False, observed=True).agg(
        Q0=("Equiv_Volume_Baseline", "sum"),
        Q1=("Equiv_Volume_Simulated", "sum"),
        R0=("Baseline_Revenue", "sum"),
        R1=("Simulated_Revenue", "sum"),
    )

    # Need SKU-level details within each group for Mix and Price
    detail = scenario.copy()

    # Merge Q1 into detail
    detail = detail.merge(g[group_columns + ["Q1"]], on=group_columns, how="left")

    detail["s0"] = detail["Equiv_Volume_Baseline"] / detail.groupby(group_columns)["Equiv_Volume_Baseline"].transform("sum")
    detail["s1"] = detail["Equiv_Volume_Simulated"] / detail.groupby(group_columns)["Equiv_Volume_Simulated"].transform("sum")
    detail["P0_equiv"] = detail["Baseline_Revenue"] / detail["Equiv_Volume_Baseline"].replace(0, np.nan)
    detail["P1_equiv"] = detail["Simulated_Revenue"] / detail["Equiv_Volume_Simulated"].replace(0, np.nan)

    # Volume effect: (Q1 - Q0) * (R0/Q0)
    g["Pbar0"] = g["R0"] / g["Q0"].replace(0, np.nan)
    g["Volume_Effect"] = (g["Q1"] - g["Q0"]) * g["Pbar0"]

    # Mix effect: Q1 * sum( (s1 - s0) * P0 )
    mix_detail = detail.groupby(group_columns, as_index=False, observed=True).apply(
        lambda df: pd.Series({
            "Mix_Effect": df["Q1"].iloc[0] * ((df["s1"] - df["s0"]) * df["P0_equiv"]).sum()
        })
    ).reset_index(drop=True)
    g = g.merge(mix_detail, on=group_columns, how="left")

    # Price effect: sum( V1 * (P1 - P0) )
    price_detail = detail.groupby(group_columns, as_index=False, observed=True).apply(
        lambda df: pd.Series({
            "Price_Effect": (df["Equiv_Volume_Simulated"] * (df["P1_equiv"] - df["P0_equiv"])).sum()
        })
    ).reset_index(drop=True)
    g = g.merge(price_detail, on=group_columns, how="left")

    # Reconciliation check
    g["Decomposition_Sum"] = g["Volume_Effect"] + g["Mix_Effect"] + g["Price_Effect"]
    g["Revenue_Delta"] = g["R1"] - g["R0"]

    if drop_total:
        g = g.drop(columns=["_total_market"])

    return g


def value_shares(
    scenario: pd.DataFrame,
    group_columns: list,
    parent_columns: list | None = None,
) -> pd.DataFrame:
    """Calculate revenue shares with explicit and consistent denominators."""
    grouped = (
        scenario.groupby(group_columns, as_index=False, observed=True)
        .agg(
            Baseline_Revenue=("Baseline_Revenue", "sum"),
            Simulated_Revenue=("Simulated_Revenue", "sum"),
        )
    )

    if parent_columns:
        baseline_denominator = grouped.groupby(
            parent_columns, observed=True
        )["Baseline_Revenue"].transform("sum")
        simulated_denominator = grouped.groupby(
            parent_columns, observed=True
        )["Simulated_Revenue"].transform("sum")
    else:
        baseline_denominator = grouped["Baseline_Revenue"].sum()
        simulated_denominator = grouped["Simulated_Revenue"].sum()

    # Zero denominators mean undefined shares, not zero market shares.
    grouped["Baseline_Share_Pct"] = (
        grouped["Baseline_Revenue"] / baseline_denominator * 100
    )
    grouped["Simulated_Share_Pct"] = (
        grouped["Simulated_Revenue"] / simulated_denominator * 100
    )
    grouped["Share_Change_pp"] = (
        grouped["Simulated_Share_Pct"] - grouped["Baseline_Share_Pct"]
    )
    return grouped.replace([np.inf, -np.inf], np.nan)


def share_chart(table: pd.DataFrame, label_column: str, title: str) -> go.Figure:
    """Side-by-side baseline and simulated value-share bars."""
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Baseline",
            x=table[label_column],
            y=table["Baseline_Share_Pct"],
            marker_color="#64748B",
            hovertemplate="%{x}<br>Value share: %{y:.2f}%<extra>Baseline</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            name="Simulated",
            x=table[label_column],
            y=table["Simulated_Share_Pct"],
            marker_color="#2563EB",
            hovertemplate="%{x}<br>Value share: %{y:.2f}%<extra>Simulated</extra>",
        )
    )
    figure.update_layout(
        title=title,
        barmode="group",
        yaxis_title="Value share (%)",
        xaxis_title="",
        legend_title_text="Scenario",
        margin=dict(t=60, b=50),
    )
    return figure


# =============================================================================
# Application
# =============================================================================

def main() -> None:
    st.title("Bayesian Retail Price Simulation")
    st.caption(
        "Edit latest-month SKU-retailer prices to simulate own-price demand "
        "responses across the observed market."
    )

    if not DATA_PATH.exists():
        st.error("Place sample_data.csv in the same directory as app.py.")
        st.stop()

    try:
        data, data_notes = load_data(
            str(DATA_PATH),
            DATA_PATH.stat().st_mtime_ns,
        )
    except Exception as exc:
        st.error(f"Unable to load POS data: {exc}")
        st.stop()

    latest_period = int(data["Period"].max())
    latest_label = data.loc[
        data["Period"].eq(latest_period), "Month"
    ].iloc[0]

    # The latest GLOBAL month defines the market snapshot.
    # We deliberately do not carry forward stale rows from older months.
    baseline = (
        data.loc[data["Period"].eq(latest_period)]
        .copy()
        .reset_index(drop=True)
    )

    if baseline["Sales_Revenue"].sum() <= 0:
        st.error("The latest month has no positive market revenue.")
        st.stop()

    st.sidebar.header("Bayesian model")
    st.sidebar.caption(
        "MAP estimation is cached. Editing prices does not refit the model."
    )

    prior_elasticity = st.sidebar.number_input(
        "Prior mean own-price elasticity",
        min_value=-5.0,
        max_value=-0.1,
        value=-1.5,
        step=0.1,
        help=(
            "A regularizing prior, not a fixed scenario elasticity. "
            "Sparse SKU estimates may remain close to this prior."
        ),
    )
    max_evaluations = st.sidebar.selectbox(
        "Maximum MAP evaluations",
        options=[3000, 6000, 12000],
        index=1,
    )
    st.sidebar.info(
        "First fit may take time for PyMC compilation. "
        "The dashboard uses MAP point estimates; it does not display "
        "posterior credible intervals."
    )

    with st.spinner("Compiling and fitting the cached PyMC demand model…"):
        try:
            fitted = fit_bayesian_model(
                data,
                float(prior_elasticity),
                int(max_evaluations),
                MODEL_VERSION,
            )
        except Exception as exc:
            logger.exception("Bayesian model fitting failed")
            st.error(f"Bayesian fitting failed: {exc}")
            st.info(
                "Check the data and PyMC installation. "
                "No fallback elasticity has been silently substituted."
            )
            st.stop()

    estimates = fitted["estimates"]

    if not fitted["converged"]:
        st.warning(
            "MAP optimization did not report convergence. Results are provisional. "
            f"Optimizer: {fitted['message']}. Try increasing the evaluation limit."
        )

    if fitted["positive_elasticities"]:
        st.warning(
            f"{fitted['positive_elasticities']} SKU MAP elasticities are positive. "
            "These are retained transparently, not forced negative. "
            "Inspect confounding and price variation before using their scenarios."
        )

    if fitted["weak_support_count"]:
        st.caption(
            f"{fitted['weak_support_count']} SKU estimates have limited "
            "within-retailer price variation or few observations and may "
            "be strongly influenced by the prior."
        )

    # -------------------------------------------------------------------------
    # Editable wide price matrix
    # -------------------------------------------------------------------------
    st.subheader(f"Price matrix — {latest_label}")
    st.caption(
        "Rows identify products; columns identify retailers. "
        "Blank cells indicate no observation in the latest month and cannot "
        "create new listings. All reported market totals use this full snapshot."
    )

    retailers = sorted(baseline["Retailer"].unique().tolist())
    retailer_to_column = {
        retailer: f"Price | {retailer}" for retailer in retailers
    }
    column_to_retailer = {
        column: retailer for retailer, column in retailer_to_column.items()
    }

    matrix_source = baseline.assign(
        _PriceColumn=baseline["Retailer"].map(retailer_to_column)
    )
    price_matrix = (
        matrix_source.pivot(
            index=PRODUCT_KEYS,
            columns="_PriceColumn",
            values="Price",
        )
        .reindex(columns=list(column_to_retailer))
        .reset_index()
    )
    price_matrix.columns.name = None

    # A data-specific widget key prevents stale edits when the file changes.
    data_signature = hashlib.sha256(
        pd.util.hash_pandas_object(
            data, index=True
        ).to_numpy().tobytes()
    ).hexdigest()[:16]

    reset_key = f"matrix_reset_{data_signature}"
    if reset_key not in st.session_state:
        st.session_state[reset_key] = 0

    if st.button("Reset all prices to baseline"):
        st.session_state[reset_key] += 1

    editor_key = (
        f"price_editor_{data_signature}_{st.session_state[reset_key]}"
    )

    editor_config = {
        column: st.column_config.NumberColumn(
            label=retailer,
            min_value=0.000001,
            format="%.6f",
            help="Enter a positive price per sold pack/unit.",
        )
        for column, retailer in column_to_retailer.items()
    }

    edited_matrix = st.data_editor(
        price_matrix,
        key=editor_key,
        disabled=PRODUCT_KEYS,
        column_config=editor_config,
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
    )

    # Edits to absent cells are ignored by the baseline left join. Tell the user
    # explicitly rather than treating these cells as simulated distribution.
    absent_cells = price_matrix[list(column_to_retailer)].isna()
    filled_absent_cells = (
        edited_matrix[list(column_to_retailer)].notna() & absent_cells
    )
    if filled_absent_cells.to_numpy().any():
        st.warning(
            "Prices entered in originally blank cells are ignored. "
            "New listings require a separate baseline demand assumption."
        )

    try:
        scenario = simulate_market(
            baseline,
            edited_matrix,
            column_to_retailer,
            estimates,
        )
    except Exception as exc:
        st.error(f"Invalid scenario: {exc}")
        st.stop()

    changed = ~np.isclose(
        scenario["Simulated_Price"],
        scenario["Baseline_Price"],
        rtol=1e-9,
        atol=1e-10,
    )
    st.caption(
        f"{int(changed.sum()):,} of {len(scenario):,} "
        "latest-month SKU-retailer prices changed."
    )

    if scenario["Price_Change_Pct"].abs().gt(50).any():
        st.warning(
            "Some price changes exceed ±50%. Constant-elasticity extrapolation "
            "this far from baseline may be unreliable."
        )

    # -------------------------------------------------------------------------
    # Market-wide KPIs
    # -------------------------------------------------------------------------
    r0 = float(scenario["Baseline_Revenue"].sum())
    r1 = float(scenario["Simulated_Revenue"].sum())
    delta = r1 - r0
    delta_pct = delta / r0 * 100
    q0 = float(scenario["Baseline_Unit_Sales"].sum())
    q1 = float(scenario["Simulated_Unit_Sales"].sum())

    st.subheader("Market-wide impact")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline revenue", f"{r0:,.2f}")
    k2.metric(
        "Simulated revenue",
        f"{r1:,.2f}",
        delta=f"{delta:+,.2f} ({delta_pct:+.2f}%)",
    )
    k3.metric("Absolute revenue change", f"{delta:+,.2f}")
    k4.metric("Revenue change (%)", f"{delta_pct:+.2f}%")

    u1, u2 = st.columns(2)
    u1.metric("Baseline unit sales", f"{q0:,.0f}")
    u2.metric(
        "Simulated unit sales",
        f"{q1:,.1f}",
        delta=f"{q1 - q0:+,.1f}",
    )
    st.caption(
        "Revenue is shown in the dataset's currency units. Unit sales count "
        "sold packs/items, not a combined kg/L measure. "
        "Simulated quantities are expectations and need not be integers."
    )

    # -------------------------------------------------------------------------
    # Three-factor decomposition (Volume -> Mix -> Price) at market level
    # -------------------------------------------------------------------------
    market_decomp = three_factor_decomposition(scenario, [])
    ve = float(market_decomp["Volume_Effect"].iloc[0])
    me = float(market_decomp["Mix_Effect"].iloc[0])
    pe = float(market_decomp["Price_Effect"].iloc[0])
    recon_error = float(market_decomp["Revenue_Delta"].iloc[0] - market_decomp["Decomposition_Sum"].iloc[0])

    waterfall = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=[
                "Baseline revenue",
                "Volume effect",
                "Mix effect",
                "Price effect",
                "Simulated revenue",
            ],
            y=[r0, ve, me, pe, 0],
            text=[
                f"{r0:,.2f}",
                f"{ve:+,.2f}",
                f"{me:+,.2f}",
                f"{pe:+,.2f}",
                f"{r1:,.2f}",
            ],
            textposition="outside",
            connector={"line": {"color": "#94A3B8"}},
            increasing={"marker": {"color": "#16A34A"}},
            decreasing={"marker": {"color": "#DC2626"}},
            totals={"marker": {"color": "#2563EB"}},
        )
    )
    waterfall.update_layout(
        title="Revenue bridge (Volume → Mix → Price) using equivalized volumes",
        yaxis_title="Revenue",
        showlegend=False,
        margin=dict(t=70, b=40),
    )
    st.plotly_chart(waterfall, use_container_width=True)
    st.caption(
        f"Volume: {ve:+,.2f} | Mix: {me:+,.2f} | Price: {pe:+,.2f} | "
        f"Reconciliation residual: {recon_error:.8f}"
    )

    # Also show two-factor cell-level bridge for reference (exact by construction)
    with st.expander("Two-factor cell-level bridge (exact for matched SKU–retailer cells)"):
        cell_pe = float(scenario["Price_Effect"].sum())
        cell_ve = float(scenario["Volume_Effect"].sum())
        st.write(
            f"Price effect (Σ(P₁−P₀)Q₁): {cell_pe:+,.2f} | "
            f"Volume effect (Σ(Q₁−Q₀)P₀): {cell_ve:+,.2f} | "
            f"Sum: {cell_pe + cell_ve:+,.2f} | ΔRevenue: {delta:+,.2f}"
        )

    # -------------------------------------------------------------------------
    # Decomposition by Category / Retailer / Brand
    # -------------------------------------------------------------------------
    st.subheader("Decomposition by segment")
    dec_tab_cat, dec_tab_ret, dec_tab_brand = st.tabs(["By Category", "By Retailer", "By Brand"])

    cat_decomp = three_factor_decomposition(scenario, ["Category"])
    ret_decomp = three_factor_decomposition(scenario, ["Retailer"])
    brand_decomp = three_factor_decomposition(scenario, ["Category", "Brand"])
    brand_decomp["Portfolio"] = brand_decomp["Category"] + " / " + brand_decomp["Brand"]

    for label, df, key_col in [
        ("Category", cat_decomp, "Category"),
        ("Retailer", ret_decomp, "Retailer"),
        ("Brand", brand_decomp, "Portfolio"),
    ]:
        with eval(f"dec_tab_{label.lower()[:3]}"):
            st.dataframe(
                df[[key_col, "R0", "R1", "Revenue_Delta", "Volume_Effect", "Mix_Effect", "Price_Effect", "Decomposition_Sum"]],
                hide_index=True,
                use_container_width=True,
            )
            # Stacked bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Volume", x=df[key_col], y=df["Volume_Effect"], marker_color="#64748B"))
            fig.add_trace(go.Bar(name="Mix", x=df[key_col], y=df["Mix_Effect"], marker_color="#F59E0B"))
            fig.add_trace(go.Bar(name="Price", x=df[key_col], y=df["Price_Effect"], marker_color="#2563EB"))
            fig.update_layout(barmode="relative", title=f"{label} decomposition", yaxis_title="Revenue effect", margin=dict(t=50, b=50))
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # Market shares: total-market views and category-conditional views
    # -------------------------------------------------------------------------
    st.subheader("Value-share shifts")
    st.caption(
        "Overall retailer and category shares use total observed market revenue. "
        "Overall brand shares identify each Category/Brand portfolio separately. "
        "Conditional views recalculate shares within the selected category."
    )

    retailer_shares = value_shares(scenario, ["Retailer"])
    category_shares = value_shares(scenario, ["Category"])
    brand_shares = value_shares(scenario, ["Category", "Brand"])
    brand_shares["Portfolio"] = (
        brand_shares["Category"] + " / " + brand_shares["Brand"]
    )

    tab_retailer, tab_brand, tab_category, tab_conditional = st.tabs(
        ["Retailers", "Brands", "Categories", "Within-category shares"]
    )

    with tab_retailer:
        st.plotly_chart(
            share_chart(
                retailer_shares,
                "Retailer",
                "Retailer shares of total market revenue",
            ),
            use_container_width=True,
        )
        st.dataframe(
            retailer_shares,
            hide_index=True,
            use_container_width=True,
        )

    with tab_brand:
        st.plotly_chart(
            share_chart(
                brand_shares,
                "Portfolio",
                "Category-brand portfolios: share of total market revenue",
            ),
            use_container_width=True,
        )
        st.dataframe(
            brand_shares,
            hide_index=True,
            use_container_width=True,
        )

    with tab_category:
        st.plotly_chart(
            share_chart(
                category_shares,
                "Category",
                "Category shares of total market revenue",
            ),
            use_container_width=True,
        )
        st.dataframe(
            category_shares,
            hide_index=True,
            use_container_width=True,
        )

    with tab_conditional:
        selected_category = st.selectbox(
            "Category for conditional share analysis",
            sorted(scenario["Category"].unique()),
        )

        retailer_category = value_shares(
            scenario,
            ["Category", "Retailer"],
            parent_columns=["Category"],
        )
        brand_category = value_shares(
            scenario,
            ["Category", "Brand"],
            parent_columns=["Category"],
        )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                share_chart(
                    retailer_category.loc[
                        retailer_category["Category"].eq(selected_category)
                    ],
                    "Retailer",
                    "Retailer shares within category",
                ),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                share_chart(
                    brand_category.loc[
                        brand_category["Category"].eq(selected_category)
                    ],
                    "Brand",
                    "Brand shares within category, all retailers",
                ),
                use_container_width=True,
            )

        selected_retailer = st.selectbox(
            "Retailer for brand-within-retailer shares",
            sorted(scenario["Retailer"].unique()),
        )
        brand_retailer_category = value_shares(
            scenario,
            ["Category", "Retailer", "Brand"],
            parent_columns=["Category", "Retailer"],
        )
        conditional_table = brand_retailer_category.loc[
            brand_retailer_category["Category"].eq(selected_category)
            & brand_retailer_category["Retailer"].eq(selected_retailer)
        ]
        st.dataframe(
            conditional_table,
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Share_Change_pp is measured in percentage points.")

    # -------------------------------------------------------------------------
    # Scenario detail and downloads
    # -------------------------------------------------------------------------
    st.subheader("SKU-retailer simulation details")

    display_columns = [
        "Month",
        "Category",
        "Retailer",
        "Brand",
        "SKU",
        "Pack_Size",
        "Elasticity_MAP",
        "Baseline_Price",
        "Simulated_Price",
        "Price_Change_Pct",
        "Baseline_Unit_Sales",
        "Simulated_Unit_Sales",
        "Baseline_Revenue",
        "Simulated_Revenue",
        "Revenue_Delta",
        "Price_Effect",
        "Volume_Effect",
    ]

    st.dataframe(
        scenario[display_columns],
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Download scenario CSV",
        data=scenario[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="retail_price_scenario.csv",
        mime="text/csv",
    )

    # -------------------------------------------------------------------------
    # Diagnostics and transparent model specification
    # -------------------------------------------------------------------------
    with st.expander("Estimated SKU elasticities and fit diagnostics"):
        st.write(
            {
                "Fit method": "PyMC maximum a posteriori (MAP)",
                "Converged": fitted["converged"],
                "Optimizer message": fitted["message"],
                "Optimizer iterations": fitted["iterations"],
                "Positive-unit training observations": fitted["training_rows"],
                "Observed periods": int(data["Period"].nunique()),
                "Latest-month cells": len(baseline),
            }
        )
        st.dataframe(
            estimates,
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Weak_Price_Support flags fewer than 12 positive observations "
            "or within-pair log-price standard deviation below 0.02. "
            "These are screening thresholds, not statistical significance tests."
        )

        settings = {
            "model_version": MODEL_VERSION,
            "prior_elasticity": float(prior_elasticity),
            "maximum_MAP_evaluations": int(max_evaluations),
            "baseline_period": latest_period,
            "baseline_label": str(latest_label),
            "data_signature": data_signature,
            "uncertainty_intervals": False,
        }
        st.download_button(
            "Download model settings",
            data=json.dumps(settings, indent=2).encode("utf-8"),
            file_name="retail_model_settings.json",
            mime="application/json",
        )

    with st.expander("Data checks"):
        if data_notes:
            for note in data_notes:
                st.write(f"- {note}")

        historical_pairs = data[CELL_KEYS].drop_duplicates()
        absent_latest = len(historical_pairs) - len(baseline)
        st.write(
            f"{absent_latest:,} historically observed SKU-retailer identities "
            "are absent from the latest snapshot. They are not backfilled."
        )
        st.write(
            "Pack_Size is preserved as metadata. Without an explicit, "
            "validated measurement dimension, kg, litres, and pieces are "
            "not summed into a single physical-volume KPI."
        )

    with st.expander("Economic model, accounting, and limitations"):
        st.latex(
            r"\ln Q_i = \alpha_{s(i)} + \mu_{r(i)} + a_{s(i),r(i)}"
            r" + \beta_{s(i)}\left(\ln P_i"
            r" - \overline{\ln P}_{s(i),r(i)}\right)"
            r" + \lambda_{c(i),m(i)} + \tau_{c(i)}T_i + u_i"
        )
        st.latex(
            r"\beta_s \mid \bar{\beta}_{c(s)}"
            r"\sim \mathcal{N}\!\left(\bar{\beta}_{c(s)},\,0.5^2\right)"
        )
        st.latex(
            r"Q_1 = Q_0\exp\!\left("
            r"\widehat{\beta}_{s,\mathrm{MAP}}\ln(P_1/P_0)\right),"
            r"\qquad R_1=P_1Q_1"
        )
        st.latex(
            r"\Delta R = \sum_i (P_{1,i}-P_{0,i})Q_{1,i}"
            r" + \sum_i (Q_{1,i}-Q_{0,i})P_{0,i}"
        )

        st.markdown(
            """
            - Retailer, SKU-retailer, seasonal, and trend controls enter model
              estimation. During a same-month scenario, they are held fixed.
            - Seasonal effects are pooled within category for stability with
              only two annual cycles.
            - The calculation is anchored to observed quantities. An unchanged
              price gives unchanged units and revenue, avoiding a baseline
              mismatch caused by model fitting error.
            - MAP is a point estimate under the specified priors. It is not
              posterior sampling and supplies no credible or predictive interval.
            - Price coefficients may be confounded by unobserved promotions,
              stockouts, distribution changes, and demand shocks. They are not
              automatically causal elasticities.
            - Unedited products retain their baseline quantities and revenue.
              Their shares may still change because the denominator changes.
            - No substitution, cannibalization, halo, retailer switching,
              inventory constraints, or competitor reactions are simulated.
              Market-wide totals are sums of independent own-price responses.
            - A zero-unit baseline cell remains at zero units under this
              multiplicative scenario. Launches and latent demand require a
              separate model.
            - Constant-elasticity responses can become unrealistic for large
              price changes. This dashboard is a scenario tool, not a guarantee.
            - The exact SKU-level price and volume effects already sum to the
              revenue change. Adding a residual “mix effect” to these same
              formulas would double-count: that residual is algebraically zero.
            - Aggregate price, when needed, is Revenue / Units: a
              quantity-weighted arithmetic mean, not a revenue-weighted
              arithmetic mean of prices.
            """
        )


if __name__ == "__main__":
    main()