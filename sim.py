"""
Standalone simulation script using the same hierarchical model as app.py.
Uses MAP for speed; posterior sampling available via --sample flag.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False

# =============================================================================
# Configuration (matching app.py)
# =============================================================================
DATA_PATH = Path(__file__).resolve().parent / "sample_data.csv"
MODEL_VERSION = "retail-map-v1"
RANDOM_SEED = 42

PRODUCT_KEYS = ["Category", "Brand", "SKU"]
CELL_KEYS = ["Category", "Retailer", "Brand", "SKU"]

REQUIRED_COLUMNS = [
    "Month", "Category", "Retailer", "Brand", "SKU",
    "Pack_Size", "Unit_Sales", "Sales_Revenue", "Price",
]


# =============================================================================
# Data loading (matching app.py)
# =============================================================================
def load_data(path: str) -> tuple[pd.DataFrame, list[str]]:
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

    for column in CELL_KEYS + ["Pack_Size"]:
        raw[column] = raw[column].astype("string").str.strip()
        if raw[column].isna().any() or raw[column].eq("").any():
            raise ValueError(f"{column} contains missing or blank values.")

    for column in ["Unit_Sales", "Sales_Revenue", "Price"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
        if not np.isfinite(raw[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains missing or non-finite values.")

    if (raw["Unit_Sales"] < 0).any() or (raw["Sales_Revenue"] < 0).any():
        raise ValueError("Negative sales/returns require separate treatment before modeling.")

    if (raw["Price"] <= 0).any():
        raise ValueError("All supplied prices must be strictly positive.")

    if (raw["Unit_Sales"].eq(0) & raw["Sales_Revenue"].ne(0)).any():
        raise ValueError("Rows with zero units must have zero revenue.")

    # Period handling
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
            int(v): f"Month {int(v)}" for v in sorted(raw["Period"].unique())
        }
        notes.append("Month is a numeric period index. Seasonal position assumes periods repeat every 12 months.")
    else:
        parsed = pd.to_datetime(raw["Month"].astype(str), errors="coerce")
        if parsed.isna().any():
            raise ValueError("Month must contain positive integer periods or parseable dates.")
        month_ordinals = parsed.dt.year * 12 + parsed.dt.month
        raw["Period"] = (month_ordinals - int(month_ordinals.min()) + 1).astype(int)
        raw["Season"] = (parsed.dt.month - 1).astype(int)
        period_labels = (
            pd.DataFrame({"Period": raw["Period"], "Label": parsed.dt.strftime("%Y-%m")})
            .drop_duplicates("Period").set_index("Period")["Label"].to_dict()
        )

    # Pack size consistency
    pack_counts = raw.groupby(PRODUCT_KEYS, observed=True)["Pack_Size"].nunique()
    if (pack_counts > 1).any():
        raise ValueError("Some Category/Brand/SKU identities have multiple Pack_Size values.")

    # Accounting-consistent prices
    positive_units = raw["Unit_Sales"] > 0
    implied_price = raw.loc[positive_units, "Sales_Revenue"] / raw.loc[positive_units, "Unit_Sales"]
    mismatch = ~np.isclose(raw.loc[positive_units, "Price"], implied_price, rtol=1e-4, atol=1e-6)
    if mismatch.any():
        notes.append(f"{int(mismatch.sum()):,} rows have supplied Price != Revenue/Units. Accounting prices used.")

    if (implied_price <= 0).any():
        raise ValueError("Positive unit sales must have positive revenue for this log-price model.")

    # Aggregate duplicate monthly cells
    monthly = (
        raw.groupby(["Period", "Season"] + CELL_KEYS, as_index=False, observed=True)
        .agg(
            Unit_Sales=("Unit_Sales", "sum"),
            Sales_Revenue=("Sales_Revenue", "sum"),
            Supplied_Price=("Price", "mean"),
            Pack_Size=("Pack_Size", "first"),
        )
    )
    monthly["Price"] = monthly["Supplied_Price"].astype(float)
    has_units = monthly["Unit_Sales"] > 0
    monthly.loc[has_units, "Price"] = monthly.loc[has_units, "Sales_Revenue"] / monthly.loc[has_units, "Unit_Sales"]
    monthly = monthly.drop(columns="Supplied_Price")
    monthly["Month"] = monthly["Period"].map(period_labels)
    monthly = monthly.sort_values(["Period"] + CELL_KEYS).reset_index(drop=True)

    return monthly, notes


def tuple_index(frame: pd.DataFrame, columns: list) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[columns])


# =============================================================================
# Model fitting (matching app.py exactly)
# =============================================================================
def fit_model(
    data: pd.DataFrame,
    prior_elasticity: float = -1.5,
    max_evaluations: int = 6000,
) -> dict:
    """Fit the hierarchical log-log demand model using MAP (matching app.py)."""
    train = data.loc[data["Unit_Sales"] > 0].copy().reset_index(drop=True)
    if len(train) < 12:
        raise ValueError("At least 12 positive-unit observations are required.")

    products = data[PRODUCT_KEYS].drop_duplicates().sort_values(PRODUCT_KEYS).reset_index(drop=True)
    pairs = data[CELL_KEYS].drop_duplicates().sort_values(CELL_KEYS).reset_index(drop=True)
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

    with pm.Model(coords=coords):
        alpha = pm.Normal("alpha", mu=float(log_q.mean()), sigma=2.0, dims="product")

        beta_category = pm.Normal("beta_category", mu=prior_elasticity, sigma=0.75, dims="category")
        beta_offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, dims="product")
        beta = pm.Deterministic("beta", beta_category[product_category_idx] + 0.5 * beta_offset, dims="product")

        retailer_raw = pm.Normal("retailer_raw", mu=0.0, sigma=0.5, dims="retailer")
        retailer_effect = retailer_raw - pm.math.sum(retailer_raw) / len(retailers)

        pair_effect = pm.Normal("pair_effect", mu=0.0, sigma=0.5, dims="pair")

        season_raw = pm.Normal("season_raw", mu=0.0, sigma=0.3, dims=("category", "season"))
        season_effect = season_raw - pm.math.sum(season_raw, axis=1, keepdims=True) / 12.0

        trend = pm.Normal("trend", mu=0.0, sigma=0.3, dims="category")
        sigma = pm.HalfNormal("sigma", sigma=0.75)

        mu = (
            alpha[sku_idx]
            + retailer_effect[retailer_idx]
            + pair_effect[pair_idx]
            + beta[sku_idx] * x_price
            + season_effect[category_idx, season_idx]
            + trend[category_idx] * time_years
        )

        pm.Normal("log_unit_sales", mu=mu, sigma=sigma, observed=log_q, dims="observation")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            point, optimization = pm.find_MAP(
                method="L-BFGS-B",
                maxeval=int(max_evaluations),
                progressbar=False,
                return_raw=True,
                seed=RANDOM_SEED,
            )

    beta_map = (
        np.asarray(point["beta_category"], dtype=float)[product_category_idx]
        + 0.5 * np.asarray(point["beta_offset"], dtype=float)
    )
    if not np.isfinite(beta_map).all():
        raise RuntimeError("Optimizer returned non-finite elasticities.")

    support = (
        train.assign(_log_price=log_p, _within_log_price=x_price)
        .groupby(PRODUCT_KEYS, observed=True)
        .agg(
            Positive_Observations=("Unit_Sales", "size"),
            Observed_Months=("Period", "nunique"),
            Distinct_Prices=("Price", "nunique"),
            Within_Pair_LogPrice_SD=("_within_log_price", lambda v: float(np.std(v, ddof=0))),
        )
        .reset_index()
    )

    estimates = products.copy()
    estimates["Elasticity_MAP"] = beta_map
    estimates = estimates.merge(support, on=PRODUCT_KEYS, how="left", validate="one_to_one")
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
        "training_rows": len(train),
        "positive_elasticities": int((beta_map > 0).sum()),
        "weak_support_count": int(estimates["Weak_Price_Support"].sum()),
        "point": point,
        "coords": coords,
        "products": products,
        "pairs": pairs,
        "categories": categories,
        "retailers": retailers,
    }


# =============================================================================
# Posterior sampling (optional, for uncertainty)
# =============================================================================
def fit_model_mcmc(
    data: pd.DataFrame,
    prior_elasticity: float = -1.5,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
) -> tuple:
    """Fit the same model with MCMC for full posterior inference."""
    train = data.loc[data["Unit_Sales"] > 0].copy().reset_index(drop=True)

    products = data[PRODUCT_KEYS].drop_duplicates().sort_values(PRODUCT_KEYS).reset_index(drop=True)
    pairs = data[CELL_KEYS].drop_duplicates().sort_values(CELL_KEYS).reset_index(drop=True)
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

    pair_price_means = pd.Series(log_p).groupby(pair_idx).transform("mean").to_numpy()
    x_price = log_p - pair_price_means
    time_years = (train["Period"].to_numpy(dtype=float) - train["Period"].mean()) / 12.0

    coords = {
        "product": np.arange(len(products)),
        "pair": np.arange(len(pairs)),
        "retailer": retailers,
        "category": categories,
        "season": np.arange(12),
        "observation": np.arange(len(train)),
    }

    with pm.Model(coords=coords):
        alpha = pm.Normal("alpha", mu=float(log_q.mean()), sigma=2.0, dims="product")
        beta_category = pm.Normal("beta_category", mu=prior_elasticity, sigma=0.75, dims="category")
        beta_offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, dims="product")
        beta = pm.Deterministic("beta", beta_category[product_category_idx] + 0.5 * beta_offset, dims="product")
        retailer_raw = pm.Normal("retailer_raw", mu=0.0, sigma=0.5, dims="retailer")
        retailer_effect = retailer_raw - pm.math.sum(retailer_raw) / len(retailers)
        pair_effect = pm.Normal("pair_effect", mu=0.0, sigma=0.5, dims="pair")
        season_raw = pm.Normal("season_raw", mu=0.0, sigma=0.3, dims=("category", "season"))
        season_effect = season_raw - pm.math.sum(season_raw, axis=1, keepdims=True) / 12.0
        trend = pm.Normal("trend", mu=0.0, sigma=0.3, dims="category")
        sigma = pm.HalfNormal("sigma", sigma=0.75)

        mu = (
            alpha[sku_idx] + retailer_effect[retailer_idx] + pair_effect[pair_idx]
            + beta[sku_idx] * x_price + season_effect[category_idx, season_idx]
            + trend[category_idx] * time_years
        )
        pm.Normal("log_unit_sales", mu=mu, sigma=sigma, observed=log_q, dims="observation")

        trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.9, random_seed=RANDOM_SEED)

    return trace, products, pairs, categories, retailers, sku_idx, pair_idx, category_idx, retailer_idx, product_category_idx


# =============================================================================
# Scenario simulation (matching app.py logic)
# =============================================================================
def simulate_scenario(
    baseline: pd.DataFrame,
    estimates: pd.DataFrame,
    price_changes: dict[tuple, float],  # {(cat, ret, brand, sku): new_price}
) -> pd.DataFrame:
    """
    Apply price changes to baseline and compute expected response.

    Anchored to observed Q0: Q1 = Q0 * exp(beta * log(P1/P0))
    """
    result = baseline.copy()
    result = result.merge(estimates[PRODUCT_KEYS + ["Elasticity_MAP"]], on=PRODUCT_KEYS, how="left", validate="many_to_one")

    # Apply price changes
    for (cat, ret, brand, sku), new_price in price_changes.items():
        mask = (
            (result["Category"] == cat) &
            (result["Retailer"] == ret) &
            (result["Brand"] == brand) &
            (result["SKU"] == sku)
        )
        result.loc[mask, "Simulated_Price"] = new_price

    # Where no change specified, keep baseline price
    result["Simulated_Price"] = result["Simulated_Price"].fillna(result["Price"])

    p1 = result["Simulated_Price"].to_numpy(dtype=float)
    p0 = result["Price"].to_numpy(dtype=float)
    q0 = result["Unit_Sales"].to_numpy(dtype=float)
    beta = result["Elasticity_MAP"].to_numpy(dtype=float)

    if not np.isfinite(beta).all():
        raise ValueError("Some SKU elasticity estimates are missing.")

    log_response = beta * np.log(p1 / p0)
    if not np.isfinite(log_response).all() or (np.abs(log_response) > 50).any():
        raise ValueError("Scenario implies extreme demand response.")

    q1 = q0 * np.exp(log_response)
    r1 = p1 * q1

    result["Baseline_Unit_Sales"] = q0
    result["Baseline_Price"] = p0
    result["Baseline_Revenue"] = result["Sales_Revenue"]
    result["Simulated_Unit_Sales"] = q1
    result["Simulated_Revenue"] = r1
    result["Price_Change_Pct"] = 100.0 * (p1 / p0 - 1.0)
    result["Revenue_Delta"] = r1 - result["Baseline_Revenue"]
    result["Price_Effect"] = (p1 - p0) * q1
    result["Volume_Effect"] = (q1 - q0) * p0

    return result


def simulate_scenario_posterior(
    trace,
    baseline: pd.DataFrame,
    products: pd.DataFrame,
    pairs: pd.DataFrame,
    categories: list,
    retailers: list,
    sku_idx: np.ndarray,
    pair_idx: np.ndarray,
    category_idx: np.ndarray,
    retailer_idx: np.ndarray,
    product_category_idx: np.ndarray,
    price_changes: dict[tuple, float],
    n_draws: int = 500,
) -> pd.DataFrame:
    """
    Simulate scenario using posterior draws for uncertainty quantification.

    For each posterior draw d:
      Q1^(d) = Q0 * exp(beta^(d) * log(P1/P0))
    Revenue^(d) = P1 * Q1^(d)
    """
    result = baseline.copy()
    p0 = result["Price"].to_numpy(dtype=float)
    q0 = result["Unit_Sales"].to_numpy(dtype=float)

    # Apply price changes
    for (cat, ret, brand, sku), new_price in price_changes.items():
        mask = (
            (result["Category"] == cat) &
            (result["Retailer"] == ret) &
            (result["Brand"] == brand) &
            (result["SKU"] == sku)
        )
        result.loc[mask, "Simulated_Price"] = new_price
    result["Simulated_Price"] = result["Simulated_Price"].fillna(result["Price"])
    p1 = result["Simulated_Price"].to_numpy(dtype=float)

    # Map baseline rows to product/pair indices
    product_lookup = tuple_index(products, PRODUCT_KEYS)
    pair_lookup = tuple_index(pairs, CELL_KEYS)
    base_sku_idx = product_lookup.get_indexer(tuple_index(result, PRODUCT_KEYS))
    base_pair_idx = pair_lookup.get_indexer(tuple_index(result, CELL_KEYS))

    # Extract posterior draws for beta (elasticity)
    beta_draws = trace.posterior["beta"].values  # (chain, draw, product)
    beta_draws = beta_draws.reshape(-1, beta_draws.shape[-1])  # (total_draws, product)
    if n_draws < len(beta_draws):
        idx = np.random.choice(len(beta_draws), n_draws, replace=False)
        beta_draws = beta_draws[idx]

    # For each draw, compute revenue
    revenue_draws = []
    for b in beta_draws:
        beta_vals = b[base_sku_idx]
        log_resp = beta_vals * np.log(p1 / p0)
        q1_draw = q0 * np.exp(log_resp)
        r1_draw = p1 * q1_draw
        revenue_draws.append(r1_draw)

    revenue_draws = np.array(revenue_draws)  # (draws, n_obs)
    result["Simulated_Revenue_Mean"] = revenue_draws.mean(axis=0)
    result["Simulated_Revenue_Std"] = revenue_draws.std(axis=0)
    result["Simulated_Revenue_Q025"] = np.percentile(revenue_draws, 2.5, axis=0)
    result["Simulated_Revenue_Q975"] = np.percentile(revenue_draws, 97.5, axis=0)

    result["Baseline_Revenue"] = result["Sales_Revenue"]
    result["Revenue_Delta_Mean"] = result["Simulated_Revenue_Mean"] - result["Baseline_Revenue"]

    return result


# =============================================================================
# Three-factor decomposition (matching app.py)
# =============================================================================
def _pack_size_to_factor(pack_size: str) -> float:
    if pd.isna(pack_size):
        return 1.0
    s = str(pack_size).strip().lower()
    import re
    m = re.match(r"([\d.]+)", s)
    if not m:
        return 1.0
    val = float(m.group(1))
    if any(u in s for u in ["g", "ml", "mg"]) and val > 10:
        val = val / 1000.0
    return val


def three_factor_decomposition(scenario: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    """Compute Volume / Mix / Price decomposition at aggregated level."""
    # Add equivalized volumes if not present
    if "Equiv_Volume_Baseline" not in scenario.columns:
        scenario = scenario.copy()
        scenario["Equiv_Volume_Baseline"] = scenario.apply(
            lambda r: _pack_size_to_factor(r["Pack_Size"]) * r["Baseline_Unit_Sales"], axis=1
        )
        scenario["Equiv_Volume_Simulated"] = scenario.apply(
            lambda r: _pack_size_to_factor(r["Pack_Size"]) * r["Simulated_Unit_Sales"], axis=1
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


# =============================================================================
# Main entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Retail price simulation")
    parser.add_argument("--scenario", choices=["nexus_cola_20pct", "custom"], default="nexus_cola_20pct")
    parser.add_argument("--sample", action="store_true", help="Run MCMC sampling for posterior")
    parser.add_argument("--prior-elasticity", type=float, default=-1.5)
    parser.add_argument("--max-evals", type=int, default=6000)
    args = parser.parse_args()

    print(f"Loading data from {DATA_PATH}...")
    data, notes = load_data(str(DATA_PATH))
    for note in notes:
        print(f"  Note: {note}")

    latest_period = int(data["Period"].max())
    latest_label = data.loc[data["Period"] == latest_period, "Month"].iloc[0]
    baseline = data[data["Period"] == latest_period].copy().reset_index(drop=True)

    print(f"\nFitting model (MAP) on {len(data)} observations...")
    fitted = fit_model(data, prior_elasticity=args.prior_elasticity, max_evaluations=args.max_evals)
    estimates = fitted["estimates"]

    print(f"  Converged: {fitted['converged']} (iters: {fitted['iterations']})")
    print(f"  Positive elasticities: {fitted['positive_elasticities']}")
    print(f"  Weak support: {fitted['weak_support_count']}")

    # Define scenario
    if args.scenario == "nexus_cola_20pct":
        price_changes = {
            ("Carbonated Soft Drinks", "Nexus_Retail_Group", "ColaClassic", "Cola_2L_Bottle"): 0.80,
        }
    else:
        price_changes = {}

    # Apply scenario
    scenario = simulate_scenario(baseline, estimates, price_changes)

    # Results
    r0 = scenario["Baseline_Revenue"].sum()
    r1 = scenario["Simulated_Revenue"].sum()
    q0 = scenario["Baseline_Unit_Sales"].sum()
    q1 = scenario["Simulated_Unit_Sales"].sum()

    print(f"\n--- Scenario: {args.scenario} ---")
    print(f"Baseline revenue:  ${r0:,.2f}")
    print(f"Simulated revenue: ${r1:,.2f}")
    print(f"Revenue delta:     ${r1 - r0:+,.2f} ({(r1-r0)/r0*100:+.2f}%)")
    print(f"Baseline units:    {q0:,.0f}")
    print(f"Simulated units:   {q1:,.1f}")

    # Three-factor decomposition
    decomp = three_factor_decomposition(scenario, [])
    print(f"\nThree-factor decomposition (Volume -> Mix -> Price):")
    print(f"  Volume: {decomp['Volume_Effect'].iloc[0]:+,.2f}")
    print(f"  Mix:    {decomp['Mix_Effect'].iloc[0]:+,.2f}")
    print(f"  Price:  {decomp['Price_Effect'].iloc[0]:+,.2f}")
    print(f"  Sum:    {decomp['Decomposition_Sum'].iloc[0]:+,.2f} (ΔRevenue: {decomp['Revenue_Delta'].iloc[0]:+,.2f})")

    # Detailed SKU results for changed items
    changed = scenario[scenario["Price_Change_Pct"] != 0]
    if len(changed) > 0:
        print("\nChanged SKU-retailer cells:")
        for _, row in changed.iterrows():
            print(f"  {row['Category']}/{row['Retailer']}/{row['Brand']}/{row['SKU']}: "
                  f"P {row['Baseline_Price']:.2f} -> {row['Simulated_Price']:.2f} "
                  f"({row['Price_Change_Pct']:+.1f}%), "
                  f"Q {row['Baseline_Unit_Sales']:.0f} -> {row['Simulated_Unit_Sales']:.1f}, "
                  f"R {row['Baseline_Revenue']:.2f} -> {row['Simulated_Revenue']:.2f}, "
                  f"Elasticity: {row['Elasticity_MAP']:.3f}")

    # Posterior sampling if requested
    if args.sample:
        if not HAS_ARVIZ:
            print("\nArviZ not installed; skipping MCMC.")
        else:
            print("\nRunning MCMC for posterior uncertainty...")
            trace, products, pairs, categories, retailers, sku_idx, pair_idx, cat_idx, ret_idx, prod_cat_idx = fit_model_mcmc(data)
            print("  Sampling posterior predictive for scenario...")
            post_scenario = simulate_scenario_posterior(
                trace, baseline, products, pairs, categories, retailers,
                sku_idx, pair_idx, cat_idx, ret_idx, prod_cat_idx, price_changes
            )
            r1_post = post_scenario["Simulated_Revenue_Mean"].sum()
            r1_std = np.sqrt((post_scenario["Simulated_Revenue_Std"]**2).sum())
            print(f"  Posterior mean revenue: ${r1_post:,.2f} ± ${r1_std:,.2f}")
            print(f"  95% interval: ${post_scenario['Simulated_Revenue_Q025'].sum():,.2f} - ${post_scenario['Simulated_Revenue_Q975'].sum():,.2f}")

    # Save outputs
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    scenario.to_csv(out_dir / "scenario_detail.csv", index=False)
    estimates.to_csv(out_dir / "elasticity_estimates.csv", index=False)
    decomp.to_csv(out_dir / "market_decomposition.csv", index=False)

    settings = {
        "model_version": MODEL_VERSION,
        "prior_elasticity": args.prior_elasticity,
        "max_evaluations": args.max_evals,
        "baseline_period": latest_period,
        "baseline_label": str(latest_label),
        "converged": fitted["converged"],
        "iterations": fitted["iterations"],
    }
    with open(out_dir / "model_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()