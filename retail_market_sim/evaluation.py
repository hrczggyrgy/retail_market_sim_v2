"""
Out-of-time evaluation with rolling training prefixes.
Evaluates predictive skill without conflating it with causal elasticity validation.
"""
import warnings
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from .data import PRODUCT_KEYS, CELL_KEYS, get_data_hash, tuple_index
from .model import ModelConfig, fit_model_map, _prepare_training_data, _build_model, RANDOM_SEED
from .simulation import simulate_scenario
from .validation import check_elasticity, check_finite_response, check_finite_quantities_revenue


@dataclass
class FoldResult:
    """Result for a single evaluation fold."""
    train_periods: list[int]
    test_period: int
    train_periods_count: int
    test_rows: int
    test_positive_rows: int
    # Row counts for coverage reporting
    attempted_rows: int  # All test rows (attempted predictions)
    scored_rows: int     # Rows with finite model predictions (supported_positive)
    # Elasticity estimates from this fold
    elasticities: pd.DataFrame  # Category, Brand, SKU, Elasticity_MAP
    # Predictions on test set
    predictions: pd.DataFrame  # Test rows with predicted vs actual
    # Metrics
    mae_packs: float
    wapec: float  # Weighted Absolute Percentage Error (when denom > 0)
    pred_coverage: float  # Fraction of scored rows with finite predictions
    coverage_in_range: float  # Fraction of test prices within historical range
    # Coverage breakdown by type
    coverage_supported_positive: float
    coverage_unseen_sku: float
    coverage_unseen_pair: float
    coverage_no_positive_history: float
    coverage_zero_sales: float
    # Baseline comparison (standalone MAE against actual on their own valid rows)
    seasonal_naive_mae: float
    last_observed_mae: float
    # Matched-row comparisons: on rows where actual, model, AND comparator are all finite
    # Seasonal
    matched_seasonal_n: int
    matched_model_mae_vs_seasonal: float  # Model MAE on matched rows
    matched_seasonal_mae: float  # Seasonal MAE on matched rows
    matched_model_minus_seasonal_mae: float  # Model MAE - Seasonal MAE (disagreement)
    # Last observed
    matched_last_n: int
    matched_model_mae_vs_last: float  # Model MAE on matched rows
    matched_last_mae: float  # Last observed MAE on matched rows
    matched_model_minus_last_mae: float  # Model MAE - Last MAE (disagreement)


@dataclass
class EvaluationResult:
    """Complete evaluation result across all folds."""
    folds: list[FoldResult]
    data_hash: str
    config: ModelConfig
    # Aggregate metrics
    mean_mae_packs: float
    mean_wapec: float
    mean_pred_coverage: float
    mean_coverage_in_range: float
    overall_row_coverage: float
    # Coverage breakdown aggregates
    mean_coverage_supported_positive: float
    mean_coverage_unseen_sku: float
    mean_coverage_unseen_pair: float
    mean_coverage_no_positive_history: float
    mean_coverage_zero_sales: float
    mean_seasonal_naive_mae: float
    mean_last_observed_mae: float
    # Matched-row aggregates (seasonal)
    mean_matched_seasonal_n: float
    mean_matched_model_mae_vs_seasonal: float
    mean_matched_seasonal_mae: float
    mean_matched_model_minus_seasonal_mae: float
    # Matched-row aggregates (last observed)
    mean_matched_last_n: float
    mean_matched_model_mae_vs_last: float
    mean_matched_last_mae: float
    mean_matched_model_minus_last_mae: float
    # Prior sensitivity
    prior_sensitivity: dict[str, Any] | None = None


def _prepare_test_data(data: pd.DataFrame, test_period: int) -> pd.DataFrame:
    """Prepare test data for a specific period."""
    return data.loc[data["Period"] == test_period].copy().reset_index(drop=True)


def _make_predictions(
    test_data: pd.DataFrame,
    fitted_estimates: pd.DataFrame,
    fitted_point: dict,
    prepared_train: dict,
    config: ModelConfig,
) -> pd.DataFrame:
    """Make predictions on test data using fitted model.
    
    This is a RETROSPECTIVE evaluation using observed test-period prices for centering.
    It does not represent a genuine prospective forecast where future prices are unknown.
    """
    # Get elasticity estimates
    estimates_df = fitted_estimates.data if hasattr(fitted_estimates, "data") else fitted_estimates
    
    # Merge test data with elasticity estimates
    test_with_est = test_data.merge(
        estimates_df[PRODUCT_KEYS + ["Elasticity_MAP"]],
        on=PRODUCT_KEYS, how="left", validate="many_to_one",
    )
    
    # Track missing elasticities (new SKUs not in training)
    missing_elast = test_with_est["Elasticity_MAP"].isna()
    if missing_elast.any():
        missing_skus = test_with_est.loc[missing_elast, PRODUCT_KEYS].drop_duplicates()
        warnings.warn(f"Test data contains {len(missing_skus)} SKUs not in training data; predictions will be NaN for these.")
    
    beta = test_with_est["Elasticity_MAP"].to_numpy(float)
    # Only check elasticity for non-missing
    if not missing_elast.all():
        check_elasticity(beta[~missing_elast])
    
    # For test predictions, we need the price centering from training
    # Use the pair_price_means from training - store one mean per unique pair
    pair_lookup = prepared_train["pair_lookup"]
    pair_price_means = prepared_train.get("pair_price_means")
    
    if pair_price_means is None:
        # Recompute from training data - one mean per unique pair
        train = prepared_train["train"]
        log_p = np.log(train["Price"].to_numpy(float))
        pair_idx = prepared_train["pair_idx"]
        # Compute mean per unique pair, not per row
        pair_means_series = pd.Series(log_p, index=pair_idx).groupby(level=0).mean()
        # Store as array indexed by pair_id (0 to n_pairs-1)
        n_pairs = len(prepared_train["pairs"])
        pair_price_means = np.zeros(n_pairs)
        pair_price_means[pair_means_series.index] = pair_means_series.values
        # Cache for reuse
        prepared_train["pair_price_means"] = pair_price_means
    
    # Get pair indices for test data
    test_pair_idx = pair_lookup.get_indexer(
        tuple_index(test_with_est, CELL_KEYS)
    )
    
    # Check for unknown pairs (new SKU-retailer combinations)
    unknown_pairs = test_pair_idx == -1
    if unknown_pairs.any():
        unknown_cells = test_with_est.loc[unknown_pairs, CELL_KEYS].drop_duplicates()
        warnings.warn(f"Test data contains {len(unknown_cells)} unknown SKU-retailer pairs; predictions will be NaN for these.")
    
    # Check for pairs with no positive training history (zero-only training)
    # These have pair_price_means = 0 (placeholder), which centers on log(1) = 0
    # We should track these separately and not score them
    zero_history_pairs = np.zeros(len(test_pair_idx), dtype=bool)
    if len(pair_price_means) > 0:
        # Pairs with mean = 0 and no positive training observations
        # We can detect this by checking if the pair had positive sales in training
        train_positive_pairs = set(prepared_train["train"].loc[prepared_train["train"]["Unit_Sales"] > 0, CELL_KEYS].apply(tuple, axis=1))
        test_pairs = test_with_est[CELL_KEYS].apply(tuple, axis=1)
        zero_history_pairs = ~test_pairs.isin(train_positive_pairs).to_numpy()
        if zero_history_pairs.any():
            warnings.warn(f"Test data contains {zero_history_pairs.sum()} pairs with no positive training history; centering uses placeholder.")
    
    # Get pair price means for test data (NaN for unknown/zero-history pairs to avoid placeholder centering)
    test_pair_price_means = np.full(len(test_pair_idx), np.nan)
    valid_pair_mask = ~unknown_pairs & ~zero_history_pairs
    if valid_pair_mask.any():
        test_pair_price_means[valid_pair_mask] = pair_price_means[test_pair_idx[valid_pair_mask]]
    
    # Compute log price and centered price
    p0 = test_with_est["Price"].to_numpy(float)
    log_p0 = np.log(p0)
    x_price = np.full_like(log_p0, np.nan)
    x_price[valid_pair_mask] = log_p0[valid_pair_mask] - test_pair_price_means[valid_pair_mask]
    
    # Get other indices
    sku_idx = prepared_train["product_lookup"].get_indexer(
        tuple_index(test_with_est, PRODUCT_KEYS)
    )
    
    # Track unknown SKUs
    unknown_skus = sku_idx == -1
    if unknown_skus.any():
        warnings.warn(f"Test data contains {unknown_skus.sum()} unknown SKUs; predictions will be NaN for these.")
    
    # Season and trend
    categories = prepared_train["categories"]
    test_season_idx = test_with_est["Season"].to_numpy(int)
    test_cat_idx = pd.Categorical(test_with_est["Category"], categories=categories).codes.astype("int32")
    
    # Time trend
    time_years = (test_with_est["Period"].to_numpy(float) - prepared_train["train"]["Period"].mean()) / 12.0
    
    # Extract fitted parameters
    alpha = np.asarray(fitted_point["alpha"], dtype=float)
    beta_category = np.asarray(fitted_point["beta_category"], dtype=float)
    beta_offset = np.asarray(fitted_point["beta_offset"], dtype=float)
    # beta_category is indexed by category, beta_offset by product (SKU)
    # Compute per-row elasticity: category elasticity + product offset
    beta_map = np.full(len(test_with_est), np.nan)
    valid_sku = ~unknown_skus & ~missing_elast
    if valid_sku.any():
        beta_map[valid_sku] = beta_category[test_cat_idx[valid_sku]] + config.beta_offset_scale * beta_offset[sku_idx[valid_sku]]
    retailer_effect = np.asarray(fitted_point["retailer_raw"], dtype=float)
    retailer_effect = retailer_effect - retailer_effect.mean()
    pair_effect = np.asarray(fitted_point["pair_effect"], dtype=float)
    season_raw = np.asarray(fitted_point["season_raw"], dtype=float)
    season_effect = season_raw - season_raw.sum(axis=1, keepdims=True) / 12.0
    trend = np.asarray(fitted_point["trend"], dtype=float)
    
    # Compute predictions (only for rows with all required components)
    mu = np.full(len(test_with_est), np.nan)
    # Valid rows: have SKU, elasticity, pair price mean, and retailer
    retailer_codes = pd.Categorical(test_with_est["Retailer"], categories=prepared_train["retailers"]).codes
    valid_retailer = retailer_codes != -1
    valid_all = valid_sku & valid_pair_mask & valid_retailer & ~missing_elast
    
    if valid_all.any():
        mu[valid_all] = (
            alpha[sku_idx[valid_all]]
            + retailer_effect[retailer_codes[valid_all]]
            + pair_effect[test_pair_idx[valid_all]]
            + beta_map[valid_all] * x_price[valid_all]
            + season_effect[test_cat_idx[valid_all], test_season_idx[valid_all]]
            + trend[test_cat_idx[valid_all]] * time_years[valid_all]
        )
    
    q_pred = np.exp(mu)
    q_actual = test_with_est["Unit_Sales"].to_numpy(float)
    
    # Set NaN predictions for all invalid rows
    q_pred = np.where(valid_all, q_pred, np.nan)
    
    # Build predictions DataFrame with coverage tracking
    pred_df = test_with_est[CELL_KEYS + ["Period", "Unit_Sales", "Price"]].copy()
    pred_df["Predicted_Unit_Sales"] = q_pred
    pred_df["Actual_Unit_Sales"] = q_actual
    pred_df["Prediction_Error"] = q_pred - q_actual
    pred_df["Abs_Error"] = np.abs(q_pred - q_actual)
    pred_df["Elasticity"] = beta_map
    # Coverage flags
    pred_df["Coverage_Type"] = "unknown"
    pred_df.loc[valid_all, "Coverage_Type"] = "supported_positive"
    pred_df.loc[missing_elast, "Coverage_Type"] = "unseen_sku"
    pred_df.loc[unknown_pairs & ~missing_elast, "Coverage_Type"] = "unseen_pair"
    pred_df.loc[zero_history_pairs & ~missing_elast & ~unknown_pairs, "Coverage_Type"] = "no_positive_history"
    pred_df.loc[(test_with_est["Unit_Sales"] == 0) & valid_all, "Coverage_Type"] = "zero_sales"
    
    return pred_df


def _compute_baselines(
    test_data: pd.DataFrame,
    data: pd.DataFrame,
    test_period: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute seasonal naive and last-observed baselines."""
    # Seasonal naive: use same month from previous year (period - 12)
    seasonal_period = test_period - 12
    seasonal_data = data.loc[data["Period"] == seasonal_period].copy()
    seasonal_pred = test_data.merge(
        seasonal_data[CELL_KEYS + ["Unit_Sales"]],
        on=CELL_KEYS, how="left", suffixes=("", "_seasonal")
    )
    seasonal_pred = seasonal_pred.rename(columns={"Unit_Sales_seasonal": "Seasonal_Pred"})
    
    # Last observed: use most recent period before test
    last_period = test_period - 1
    last_data = data.loc[data["Period"] == last_period].copy()
    last_pred = test_data.merge(
        last_data[CELL_KEYS + ["Unit_Sales"]],
        on=CELL_KEYS, how="left", suffixes=("", "_last")
    )
    last_pred = last_pred.rename(columns={"Unit_Sales_last": "Last_Pred"})
    
    return seasonal_pred, last_pred


def evaluate_rolling(
    data: pd.DataFrame,
    config: ModelConfig | None = None,
    min_train_periods: int = 12,
    test_periods: list[int] | None = None,
    prior_grid: list[float] | None = None,
) -> EvaluationResult:
    """
    Evaluate model using rolling training prefixes.
    
    Parameters
    ----------
    data : Full panel data
    config : Model configuration
    min_train_periods : Minimum periods required for training
    test_periods : List of periods to use as test (default: all periods after min_train_periods)
    prior_grid : List of prior elasticities for sensitivity analysis
    
    Returns
    -------
    EvaluationResult with fold-level and aggregate metrics
    """
    if config is None:
        config = ModelConfig()
    
    periods = sorted(data["Period"].unique())
    
    if test_periods is None:
        # Use all periods after min_train_periods as test
        test_periods = [p for p in periods if p > min_train_periods]
    
    folds = []
    
    for test_period in test_periods:
        # Training data: all periods up to test_period - 1
        train_periods = [p for p in periods if p < test_period]
        
        if len(train_periods) < min_train_periods:
            continue
        
        train_data = data.loc[data["Period"].isin(train_periods)].copy()
        test_data = _prepare_test_data(data, test_period)
        
        # Fit model on training data
        fitted = fit_model_map(train_data, config)
        if not fitted["converged"]:
            raise RuntimeError(f"MAP optimization failed for fold {test_period}: {fitted['message']}")
        
        # Prepare training data for predictions
        # _make_predictions will compute pair_price_means correctly (one mean per unique pair)
        prepared_train = _prepare_training_data(train_data)
        
        # Make predictions
        predictions = _make_predictions(
            test_data, fitted["estimates"], fitted["point"], prepared_train, config
        )
        
        # Compute baselines
        seasonal_pred, last_pred = _compute_baselines(test_data, data, test_period)
        
        # Merge predictions with baselines
        pred_merged = predictions.merge(
            seasonal_pred[CELL_KEYS + ["Seasonal_Pred"]], on=CELL_KEYS, how="left"
        ).merge(
            last_pred[CELL_KEYS + ["Last_Pred"]], on=CELL_KEYS, how="left"
        )
        
        # Compute metrics - use finite prediction mask for both numerator and denominator
        mask = pred_merged["Actual_Unit_Sales"] >= 0  # All valid rows
        actual = pred_merged.loc[mask, "Actual_Unit_Sales"]
        pred = pred_merged.loc[mask, "Predicted_Unit_Sales"]
        seasonal = pred_merged.loc[mask, "Seasonal_Pred"]
        last = pred_merged.loc[mask, "Last_Pred"]
        
        # Finite prediction mask - use np.isfinite to exclude inf as well as NaN
        finite_pred = np.isfinite(pred)
        actual_finite = actual[finite_pred]
        pred_finite = pred[finite_pred]
        
        # Prediction coverage - fraction of scored rows with finite predictions
        pred_coverage = finite_pred.mean() if len(finite_pred) > 0 else 0.0
        
        # MAE in packs (only on finite predictions)
        mae = np.mean(np.abs(pred_finite - actual_finite)) if len(pred_finite) > 0 else np.nan
        
        # WAPE - use same finite mask for numerator and denominator
        denom = np.sum(actual_finite)
        if denom > 0:
            wapec = np.sum(np.abs(pred_finite - actual_finite)) / denom
        else:
            wapec = np.nan
        
        # Seasonal naive MAE - only on rows where seasonal prediction exists (standalone)
        seasonal_valid = np.isfinite(seasonal)
        seasonal_mae = np.mean(np.abs(seasonal[seasonal_valid] - actual[seasonal_valid])) if seasonal_valid.any() else np.nan
        
        # Last observed MAE - only on rows where last prediction exists (standalone)
        last_valid = np.isfinite(last)
        last_mae = np.mean(np.abs(last[last_valid] - actual[last_valid])) if last_valid.any() else np.nan
        
        # Matched-row comparisons: rows where actual, model, AND comparator are ALL finite
        # Seasonal
        matched_seasonal = finite_pred & seasonal_valid & np.isfinite(actual)
        matched_seasonal_n = int(matched_seasonal.sum())
        if matched_seasonal_n > 0:
            matched_model_mae_vs_seasonal = float(np.mean(np.abs(pred[matched_seasonal] - actual[matched_seasonal])))
            matched_seasonal_mae = float(np.mean(np.abs(seasonal[matched_seasonal] - actual[matched_seasonal])))
            matched_model_minus_seasonal_mae = matched_model_mae_vs_seasonal - matched_seasonal_mae
        else:
            matched_model_mae_vs_seasonal = np.nan
            matched_seasonal_mae = np.nan
            matched_model_minus_seasonal_mae = np.nan
        
        # Last observed
        matched_last = finite_pred & last_valid & np.isfinite(actual)
        matched_last_n = int(matched_last.sum())
        if matched_last_n > 0:
            matched_model_mae_vs_last = float(np.mean(np.abs(pred[matched_last] - actual[matched_last])))
            matched_last_mae = float(np.mean(np.abs(last[matched_last] - actual[matched_last])))
            matched_model_minus_last_mae = matched_model_mae_vs_last - matched_last_mae
        else:
            matched_model_mae_vs_last = np.nan
            matched_last_mae = np.nan
            matched_model_minus_last_mae = np.nan
        
        # Coverage: fraction of test prices within historical range
        historical = train_data.loc[train_data["Unit_Sales"] > 0].groupby(CELL_KEYS)["Price"].agg(["min", "max"]).reset_index()
        test_with_hist = test_data.merge(historical, on=CELL_KEYS, how="left")
        # Only consider rows with historical data for coverage
        has_hist = test_with_hist["min"].notna() & test_with_hist["max"].notna()
        if has_hist.any():
            in_range = (
                (test_with_hist.loc[has_hist, "Price"] >= test_with_hist.loc[has_hist, "min"]) & 
                (test_with_hist.loc[has_hist, "Price"] <= test_with_hist.loc[has_hist, "max"])
            ).mean()
        else:
            in_range = 0.0
        
        # Coverage breakdown by type (from predictions)
        total_rows = len(pred_merged)
        coverage_supported_positive = (pred_merged["Coverage_Type"] == "supported_positive").mean() if total_rows > 0 else 0.0
        coverage_unseen_sku = (pred_merged["Coverage_Type"] == "unseen_sku").mean() if total_rows > 0 else 0.0
        coverage_unseen_pair = (pred_merged["Coverage_Type"] == "unseen_pair").mean() if total_rows > 0 else 0.0
        coverage_no_positive_history = (pred_merged["Coverage_Type"] == "no_positive_history").mean() if total_rows > 0 else 0.0
        coverage_zero_sales = (pred_merged["Coverage_Type"] == "zero_sales").mean() if total_rows > 0 else 0.0
        
        # scored_rows = rows with finite model predictions (same mask used for MAE)
        scored_rows = int(finite_pred.sum())
        
        # Extract elasticities
        estimates_df = fitted["estimates"].data if hasattr(fitted["estimates"], "data") else fitted["estimates"]
        elasticities = estimates_df[PRODUCT_KEYS + ["Elasticity_MAP"]].copy()
        
        folds.append(FoldResult(
            train_periods=train_periods,
            test_period=test_period,
            train_periods_count=len(train_periods),
            test_rows=len(test_data),
            test_positive_rows=int((test_data["Unit_Sales"] > 0).sum()),
            attempted_rows=len(test_data),
            scored_rows=scored_rows,
            elasticities=elasticities,
            predictions=pred_merged,
            mae_packs=mae,
            wapec=wapec,
            pred_coverage=pred_coverage,
            coverage_in_range=in_range,
            coverage_supported_positive=coverage_supported_positive,
            coverage_unseen_sku=coverage_unseen_sku,
            coverage_unseen_pair=coverage_unseen_pair,
            coverage_no_positive_history=coverage_no_positive_history,
            coverage_zero_sales=coverage_zero_sales,
            seasonal_naive_mae=seasonal_mae,
            last_observed_mae=last_mae,
            matched_seasonal_n=matched_seasonal_n,
            matched_model_mae_vs_seasonal=matched_model_mae_vs_seasonal,
            matched_seasonal_mae=matched_seasonal_mae,
            matched_model_minus_seasonal_mae=matched_model_minus_seasonal_mae,
            matched_last_n=matched_last_n,
            matched_model_mae_vs_last=matched_model_mae_vs_last,
            matched_last_mae=matched_last_mae,
            matched_model_minus_last_mae=matched_model_minus_last_mae,
        ))
    
    # Aggregate metrics
    valid_folds = [f for f in folds if not np.isnan(f.mae_packs)]
    # For coverage and pred_coverage, use all folds with attempted rows
    attempted_folds = [f for f in folds if f.attempted_rows > 0]
    # For coverage breakdown, use all folds (including zero-scoreable)
    all_folds = folds if folds else []
    
    # Coverage breakdown aggregates - over ALL folds, not just valid ones
    mean_coverage_supported_positive = np.mean([f.coverage_supported_positive for f in all_folds]) if all_folds else np.nan
    mean_coverage_unseen_sku = np.mean([f.coverage_unseen_sku for f in all_folds]) if all_folds else np.nan
    mean_coverage_unseen_pair = np.mean([f.coverage_unseen_pair for f in all_folds]) if all_folds else np.nan
    mean_coverage_no_positive_history = np.mean([f.coverage_no_positive_history for f in all_folds]) if all_folds else np.nan
    mean_coverage_zero_sales = np.mean([f.coverage_zero_sales for f in all_folds]) if all_folds else np.nan
    
    # Mean prediction coverage over all folds with attempted rows
    mean_pred_coverage = np.mean([f.pred_coverage for f in attempted_folds]) if attempted_folds else np.nan
    # Overall row coverage: total scored rows / total attempted rows across all folds
    total_scored = sum(f.scored_rows for f in folds)
    total_attempted = sum(f.attempted_rows for f in folds)
    overall_row_coverage = total_scored / total_attempted if total_attempted > 0 else np.nan
    
    # Matched-row aggregates (seasonal) - only folds with matched rows
    seasonal_folds = [f for f in valid_folds if not np.isnan(f.matched_model_minus_seasonal_mae)]
    mean_matched_seasonal_n = np.mean([f.matched_seasonal_n for f in seasonal_folds]) if seasonal_folds else np.nan
    mean_matched_model_mae_vs_seasonal = np.nanmean([f.matched_model_mae_vs_seasonal for f in seasonal_folds]) if seasonal_folds else np.nan
    mean_matched_seasonal_mae = np.nanmean([f.matched_seasonal_mae for f in seasonal_folds]) if seasonal_folds else np.nan
    mean_matched_model_minus_seasonal_mae = np.nanmean([f.matched_model_minus_seasonal_mae for f in seasonal_folds]) if seasonal_folds else np.nan
    
    # Matched-row aggregates (last observed) - only folds with matched rows
    last_folds = [f for f in valid_folds if not np.isnan(f.matched_model_minus_last_mae)]
    mean_matched_last_n = np.mean([f.matched_last_n for f in last_folds]) if last_folds else np.nan
    mean_matched_model_mae_vs_last = np.nanmean([f.matched_model_mae_vs_last for f in last_folds]) if last_folds else np.nan
    mean_matched_last_mae = np.nanmean([f.matched_last_mae for f in last_folds]) if last_folds else np.nan
    mean_matched_model_minus_last_mae = np.nanmean([f.matched_model_minus_last_mae for f in last_folds]) if last_folds else np.nan
    
    result = EvaluationResult(
        folds=folds,
        data_hash=get_data_hash(data),
        config=config,
        mean_mae_packs=np.mean([f.mae_packs for f in valid_folds]) if valid_folds else np.nan,
        mean_wapec=np.nanmean([f.wapec for f in valid_folds]) if valid_folds else np.nan,
        mean_pred_coverage=mean_pred_coverage,
        mean_coverage_in_range=np.mean([f.coverage_in_range for f in valid_folds]) if valid_folds else np.nan,
        overall_row_coverage=overall_row_coverage,
        mean_coverage_supported_positive=mean_coverage_supported_positive,
        mean_coverage_unseen_sku=mean_coverage_unseen_sku,
        mean_coverage_unseen_pair=mean_coverage_unseen_pair,
        mean_coverage_no_positive_history=mean_coverage_no_positive_history,
        mean_coverage_zero_sales=mean_coverage_zero_sales,
        mean_seasonal_naive_mae=np.nanmean([f.seasonal_naive_mae for f in valid_folds]) if valid_folds else np.nan,
        mean_last_observed_mae=np.nanmean([f.last_observed_mae for f in valid_folds]) if valid_folds else np.nan,
        mean_matched_seasonal_n=mean_matched_seasonal_n,
        mean_matched_model_mae_vs_seasonal=mean_matched_model_mae_vs_seasonal,
        mean_matched_seasonal_mae=mean_matched_seasonal_mae,
        mean_matched_model_minus_seasonal_mae=mean_matched_model_minus_seasonal_mae,
        mean_matched_last_n=mean_matched_last_n,
        mean_matched_model_mae_vs_last=mean_matched_model_mae_vs_last,
        mean_matched_last_mae=mean_matched_last_mae,
        mean_matched_model_minus_last_mae=mean_matched_model_minus_last_mae,
    )
    
    # Prior sensitivity analysis
    if prior_grid is not None:
        result.prior_sensitivity = _prior_sensitivity_analysis(data, config, prior_grid, periods, min_train_periods, test_periods)
    
    return result


def _prior_sensitivity_analysis(
    data: pd.DataFrame,
    base_config: ModelConfig,
    prior_grid: list[float],
    periods: list[int],
    min_train_periods: int,
    test_periods: list[int],
) -> dict[str, Any]:
    """Analyze sensitivity of elasticities and predictions to prior choice."""
    sensitivity_results = {}
    
    for prior in prior_grid:
        config = ModelConfig(
            prior_elasticity=prior,
            max_evaluations=base_config.max_evaluations,
            alpha_sigma=base_config.alpha_sigma,
            beta_category_sigma=base_config.beta_category_sigma,
            beta_offset_sigma=base_config.beta_offset_sigma,
            beta_offset_scale=base_config.beta_offset_scale,
            retailer_sigma=base_config.retailer_sigma,
            pair_sigma=base_config.pair_sigma,
            season_sigma=base_config.season_sigma,
            trend_sigma=base_config.trend_sigma,
            sigma_sigma=base_config.sigma_sigma,
        )
        
        # Run evaluation for this prior (using last test period only for speed)
        if test_periods:
            last_test = test_periods[-1]
            train_periods = [p for p in periods if p < last_test]
            
            if len(train_periods) >= min_train_periods:
                train_data = data.loc[data["Period"].isin(train_periods)].copy()
                
                fitted = fit_model_map(train_data, config)
                if fitted["converged"]:
                    estimates_df = fitted["estimates"].data if hasattr(fitted["estimates"], "data") else fitted["estimates"]
                    sensitivity_results[f"prior_{prior}"] = {
                        "elasticities": estimates_df[PRODUCT_KEYS + ["Elasticity_MAP"]].to_dict("records"),
                        "mean_elasticity": estimates_df["Elasticity_MAP"].mean(),
                        "std_elasticity": estimates_df["Elasticity_MAP"].std(),
                        "converged": True,
                    }
                else:
                    sensitivity_results[f"prior_{prior}"] = {"converged": False, "message": fitted["message"]}
    
    return sensitivity_results


def evaluation_to_dataframe(result: EvaluationResult) -> dict[str, pd.DataFrame]:
    """Convert EvaluationResult to DataFrames for export."""
    # Fold summary
    fold_data = []
    for f in result.folds:
        fold_data.append({
            "test_period": f.test_period,
            "train_periods_count": f.train_periods_count,
            "test_rows": f.test_rows,
            "test_positive_rows": f.test_positive_rows,
            "attempted_rows": f.attempted_rows,
            "scored_rows": f.scored_rows,
            "mae_packs": f.mae_packs,
            "wapec": f.wapec,
            "pred_coverage": f.pred_coverage,
            "coverage_in_range": f.coverage_in_range,
            # Coverage breakdown
            "coverage_supported_positive": f.coverage_supported_positive,
            "coverage_unseen_sku": f.coverage_unseen_sku,
            "coverage_unseen_pair": f.coverage_unseen_pair,
            "coverage_no_positive_history": f.coverage_no_positive_history,
            "coverage_zero_sales": f.coverage_zero_sales,
            "seasonal_naive_mae": f.seasonal_naive_mae,
            "last_observed_mae": f.last_observed_mae,
            # Matched-row (seasonal)
            "matched_seasonal_n": f.matched_seasonal_n,
            "matched_model_mae_vs_seasonal": f.matched_model_mae_vs_seasonal,
            "matched_seasonal_mae": f.matched_seasonal_mae,
            "matched_model_minus_seasonal_mae": f.matched_model_minus_seasonal_mae,
            # Matched-row (last observed)
            "matched_last_n": f.matched_last_n,
            "matched_model_mae_vs_last": f.matched_model_mae_vs_last,
            "matched_last_mae": f.matched_last_mae,
            "matched_model_minus_last_mae": f.matched_model_minus_last_mae,
        })
    fold_df = pd.DataFrame(fold_data)
    
    # All predictions
    all_preds = []
    for f in result.folds:
        if len(f.predictions) > 0:
            pred_copy = f.predictions.copy()
            pred_copy["test_period"] = f.test_period
            all_preds.append(pred_copy)
    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    
    # Elasticity comparison across folds
    elast_data = []
    for f in result.folds:
        elast_copy = f.elasticities.copy()
        elast_copy["test_period"] = f.test_period
        elast_data.append(elast_copy)
    elast_df = pd.concat(elast_data, ignore_index=True) if elast_data else pd.DataFrame()
    
    # Prior sensitivity
    prior_df = pd.DataFrame()
    if result.prior_sensitivity:
        prior_rows = []
        for key, val in result.prior_sensitivity.items():
            if val.get("converged"):
                prior_rows.append({
                    "prior": key,
                    "mean_elasticity": val["mean_elasticity"],
                    "std_elasticity": val["std_elasticity"],
                })
        prior_df = pd.DataFrame(prior_rows) if prior_rows else pd.DataFrame()
    
    return {
        "folds": fold_df,
        "predictions": pred_df,
        "elasticities": elast_df,
        "prior_sensitivity": prior_df,
        "aggregate": pd.DataFrame([{
            "overall_row_coverage": result.overall_row_coverage,
            "mean_pred_coverage": result.mean_pred_coverage,
            "mean_coverage_supported_positive": result.mean_coverage_supported_positive,
            "mean_coverage_unseen_sku": result.mean_coverage_unseen_sku,
            "mean_coverage_unseen_pair": result.mean_coverage_unseen_pair,
            "mean_coverage_no_positive_history": result.mean_coverage_no_positive_history,
            "mean_coverage_zero_sales": result.mean_coverage_zero_sales,
        }]),
    }


def print_evaluation_summary(result: EvaluationResult) -> None:
    """Print human-readable evaluation summary."""
    print("\n" + "="*60)
    print("OUT-OF-TIME EVALUATION SUMMARY (RETROSPECTIVE)")
    print("="*60)
    print(f"Data hash: {result.data_hash}")
    print(f"Folds evaluated: {len(result.folds)}")
    print()
    
    print("Fold-level results:")
    for f in result.folds:
        print(f"  Period {f.test_period} (train: {f.train_periods_count} periods, test rows: {f.test_rows}):")
        print(f"    MAE (packs): {f.mae_packs:.2f}")
        print(f"    WAPE: {f.wapec:.4f}" if not np.isnan(f.wapec) else "    WAPE: N/A")
        print(f"    Prediction coverage: {f.pred_coverage:.2%}")
        print(f"    Coverage in historical range: {f.coverage_in_range:.2%}")
        print(f"    Attempted rows: {f.attempted_rows}, Scored rows (supported): {f.scored_rows}")
        # Coverage breakdown
        print(f"    Coverage breakdown:")
        print(f"      Supported positive: {f.coverage_supported_positive:.2%}")
        print(f"      Unseen SKU: {f.coverage_unseen_sku:.2%}")
        print(f"      Unseen SKU-Retailer pair: {f.coverage_unseen_pair:.2%}")
        print(f"      No positive training history: {f.coverage_no_positive_history:.2%}")
        print(f"      Zero sales: {f.coverage_zero_sales:.2%}")
        print(f"    Seasonal naive MAE (standalone): {f.seasonal_naive_mae:.2f}" if not np.isnan(f.seasonal_naive_mae) else "    Seasonal naive MAE (standalone): N/A")
        print(f"    Last observed MAE (standalone): {f.last_observed_mae:.2f}" if not np.isnan(f.last_observed_mae) else "    Last observed MAE (standalone): N/A")
        # Matched-row (seasonal)
        if not np.isnan(f.matched_model_minus_seasonal_mae):
            print(f"    Matched rows (seasonal): n={f.matched_seasonal_n}")
            print(f"      Model MAE: {f.matched_model_mae_vs_seasonal:.2f}")
            print(f"      Seasonal MAE: {f.matched_seasonal_mae:.2f}")
            print(f"      Model - Seasonal MAE (matched MAE difference): {f.matched_model_minus_seasonal_mae:+.2f}")
        else:
            print(f"    Matched rows (seasonal): N/A (no overlapping finite rows)")
        # Matched-row (last observed)
        if not np.isnan(f.matched_model_minus_last_mae):
            print(f"    Matched rows (last): n={f.matched_last_n}")
            print(f"      Model MAE: {f.matched_model_mae_vs_last:.2f}")
            print(f"      Last MAE: {f.matched_last_mae:.2f}")
            print(f"      Model - Last MAE (matched MAE difference): {f.matched_model_minus_last_mae:+.2f}")
        else:
            print(f"    Matched rows (last): N/A (no overlapping finite rows)")
    
    print()
    print("Aggregate metrics:")
    print(f"  Mean MAE (packs): {result.mean_mae_packs:.2f}" if not np.isnan(result.mean_mae_packs) else "  Mean MAE (packs): N/A")
    print(f"  Mean WAPE: {result.mean_wapec:.4f}" if not np.isnan(result.mean_wapec) else "  Mean WAPE: N/A")
    print(f"  Mean prediction coverage (per-fold avg): {result.mean_pred_coverage:.2%}")
    print(f"  Overall row coverage (total scored/total attempted): {result.overall_row_coverage:.2%}")
    print(f"  Mean coverage in range: {result.mean_coverage_in_range:.2%}")
    print(f"  Mean coverage breakdown:")
    print(f"    Supported positive: {result.mean_coverage_supported_positive:.2%}" if not np.isnan(result.mean_coverage_supported_positive) else "    Supported positive: N/A")
    print(f"    Unseen SKU: {result.mean_coverage_unseen_sku:.2%}" if not np.isnan(result.mean_coverage_unseen_sku) else "    Unseen SKU: N/A")
    print(f"    Unseen SKU-Retailer pair: {result.mean_coverage_unseen_pair:.2%}" if not np.isnan(result.mean_coverage_unseen_pair) else "    Unseen SKU-Retailer pair: N/A")
    print(f"    No positive training history: {result.mean_coverage_no_positive_history:.2%}" if not np.isnan(result.mean_coverage_no_positive_history) else "    No positive training history: N/A")
    print(f"    Zero sales: {result.mean_coverage_zero_sales:.2%}" if not np.isnan(result.mean_coverage_zero_sales) else "    Zero sales: N/A")
    print(f"  Mean seasonal naive MAE (standalone): {result.mean_seasonal_naive_mae:.2f}" if not np.isnan(result.mean_seasonal_naive_mae) else "  Mean seasonal naive MAE (standalone): N/A")
    print(f"  Mean last observed MAE (standalone): {result.mean_last_observed_mae:.2f}" if not np.isnan(result.mean_last_observed_mae) else "  Mean last observed MAE (standalone): N/A")
    print(f"  Mean matched n (seasonal): {result.mean_matched_seasonal_n:.1f}" if not np.isnan(result.mean_matched_seasonal_n) else "  Mean matched n (seasonal): N/A")
    print(f"  Mean Model MAE on matched (seasonal): {result.mean_matched_model_mae_vs_seasonal:.2f}" if not np.isnan(result.mean_matched_model_mae_vs_seasonal) else "  Mean Model MAE on matched (seasonal): N/A")
    print(f"  Mean Seasonal MAE on matched: {result.mean_matched_seasonal_mae:.2f}" if not np.isnan(result.mean_matched_seasonal_mae) else "  Mean Seasonal MAE on matched: N/A")
    print(f"  Mean Model - Seasonal MAE (matched MAE difference): {result.mean_matched_model_minus_seasonal_mae:+.2f}" if not np.isnan(result.mean_matched_model_minus_seasonal_mae) else "  Mean Model - Seasonal MAE (matched MAE difference): N/A")
    print(f"  Mean matched n (last): {result.mean_matched_last_n:.1f}" if not np.isnan(result.mean_matched_last_n) else "  Mean matched n (last): N/A")
    print(f"  Mean Model MAE on matched (last): {result.mean_matched_model_mae_vs_last:.2f}" if not np.isnan(result.mean_matched_model_mae_vs_last) else "  Mean Model MAE on matched (last): N/A")
    print(f"  Mean Last MAE on matched: {result.mean_matched_last_mae:.2f}" if not np.isnan(result.mean_matched_last_mae) else "  Mean Last MAE on matched: N/A")
    print(f"  Mean Model - Last MAE (matched MAE difference): {result.mean_matched_model_minus_last_mae:+.2f}" if not np.isnan(result.mean_matched_model_minus_last_mae) else "  Mean Model - Last MAE (matched MAE difference): N/A")
    
    if result.prior_sensitivity:
        print()
        print("Prior sensitivity:")
        for key, val in result.prior_sensitivity.items():
            if val.get("converged"):
                print(f"  {key}: mean_elasticity={val['mean_elasticity']:.3f}, std={val['std_elasticity']:.3f}")
            else:
                print(f"  {key}: FAILED - {val.get('message', 'unknown')}")
    
    print()
    print("NOTE: This is a RETROSPECTIVE conditional evaluation using observed test-period prices.")
    print("      It does not measure genuine prospective forecast skill.")
    print("      Predictive validation does not prove causal elasticity.")
    print("      Price identification, omitted variable bias, and endogenous pricing")
    print("      may invalidate causal interpretation of these metrics.")
    print("      Price inputs are realized holdout unit values (revenue/units), not decision-time posted prices.")
    print("      Matched-row metrics: n = rows where actual, model, AND comparator are all finite.")
    print("      Model - Comparator MAE (matched MAE difference) measures disagreement,")
    print("      not which is more accurate. Negative = model had lower error on those rows.")
    print("      exp(mu) is a conditional median, not a mean.")
    print("="*60)