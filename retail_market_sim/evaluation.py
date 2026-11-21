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
    # Elasticity estimates from this fold
    elasticities: pd.DataFrame  # Category, Brand, SKU, Elasticity_MAP
    # Predictions on test set
    predictions: pd.DataFrame  # Test rows with predicted vs actual
    # Metrics
    mae_packs: float
    wapec: float  # Weighted Absolute Percentage Error (when denom > 0)
    pred_coverage: float  # Fraction of scored rows with finite predictions
    coverage_in_range: float  # Fraction of test prices within historical range
    # Baseline comparison
    seasonal_naive_mae: float
    last_observed_mae: float


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
    mean_seasonal_naive_mae: float
    mean_last_observed_mae: float
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
    """Make predictions on test data using fitted model."""
    # Get elasticity estimates
    estimates_df = fitted_estimates.data if hasattr(fitted_estimates, "data") else fitted_estimates
    
    # Merge test data with elasticity estimates
    test_with_est = test_data.merge(
        estimates_df[PRODUCT_KEYS + ["Elasticity_MAP"]],
        on=PRODUCT_KEYS, how="left", validate="many_to_one",
    )
    
    # Check for missing elasticities
    missing_elast = test_with_est["Elasticity_MAP"].isna()
    if missing_elast.any():
        missing_skus = test_with_est.loc[missing_elast, PRODUCT_KEYS].drop_duplicates()
        raise ValueError(f"Missing elasticity estimates for SKUs in test set: {missing_skus.to_dict('records')}")
    
    beta = test_with_est["Elasticity_MAP"].to_numpy(float)
    check_elasticity(beta)
    
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
    
    # Check for unknown pairs - mark them for NaN predictions rather than aborting
    unknown_pairs = test_pair_idx == -1
    if unknown_pairs.any():
        unknown_cells = test_with_est.loc[unknown_pairs, CELL_KEYS].drop_duplicates()
        warnings.warn(f"Test data contains {len(unknown_cells)} unknown SKU-retailer pairs; predictions will be NaN for these.")
    
    # Get pair price means for test data (use 0 for unknown pairs to avoid index error)
    test_pair_price_means = np.where(unknown_pairs, 0.0, pair_price_means[np.where(unknown_pairs, 0, test_pair_idx)])
    
    # Compute log price and centered price
    p0 = test_with_est["Price"].to_numpy(float)
    log_p0 = np.log(p0)
    x_price = log_p0 - test_pair_price_means
    
    # Get other indices
    sku_idx = prepared_train["product_lookup"].get_indexer(
        tuple_index(test_with_est, PRODUCT_KEYS)
    )
    
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
    beta_map = beta_category[test_cat_idx] + config.beta_offset_scale * beta_offset[sku_idx]
    retailer_effect = np.asarray(fitted_point["retailer_raw"], dtype=float)
    retailer_effect = retailer_effect - retailer_effect.mean()
    pair_effect = np.asarray(fitted_point["pair_effect"], dtype=float)
    season_raw = np.asarray(fitted_point["season_raw"], dtype=float)
    season_effect = season_raw - season_raw.sum(axis=1, keepdims=True) / 12.0
    trend = np.asarray(fitted_point["trend"], dtype=float)
    
    # Compute predictions
    mu = (
        alpha[sku_idx]
        + retailer_effect[pd.Categorical(test_with_est["Retailer"], categories=prepared_train["retailers"]).codes]
        + pair_effect[test_pair_idx]
        + beta_map * x_price  # beta_map is already per-row
        + season_effect[test_cat_idx, test_season_idx]
        + trend[test_cat_idx] * time_years
    )
    
    q_pred = np.exp(mu)
    q_actual = test_with_est["Unit_Sales"].to_numpy(float)
    
    # Set NaN predictions for unknown pairs
    q_pred = np.where(unknown_pairs, np.nan, q_pred)
    
    # Build predictions DataFrame
    pred_df = test_with_est[CELL_KEYS + ["Period", "Unit_Sales", "Price"]].copy()
    pred_df["Predicted_Unit_Sales"] = q_pred
    pred_df["Actual_Unit_Sales"] = q_actual
    pred_df["Prediction_Error"] = q_pred - q_actual
    pred_df["Abs_Error"] = np.abs(q_pred - q_actual)
    pred_df["Elasticity"] = beta_map  # per-row elasticity
    
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
        prepared_train = _prepare_training_data(train_data)
        # Store pair_price_means for later use
        log_p = np.log(prepared_train["train"]["Price"].to_numpy(float))
        pair_idx = prepared_train["pair_idx"]
        pair_price_means = pd.Series(log_p).groupby(pair_idx).transform("mean").to_numpy()
        prepared_train["pair_price_means"] = pair_price_means
        
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
        
        # Finite prediction mask - use same mask for both numerator and denominator
        finite_pred = pred.notna()
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
        
        # Seasonal naive MAE - only on rows where seasonal prediction exists
        seasonal_valid = seasonal.notna()
        seasonal_mae = np.mean(np.abs(seasonal[seasonal_valid] - actual[seasonal_valid])) if seasonal_valid.any() else np.nan
        
        # Last observed MAE - only on rows where last prediction exists
        last_valid = last.notna()
        last_mae = np.mean(np.abs(last[last_valid] - actual[last_valid])) if last_valid.any() else np.nan
        
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
        
        # Extract elasticities
        estimates_df = fitted["estimates"].data if hasattr(fitted["estimates"], "data") else fitted["estimates"]
        elasticities = estimates_df[PRODUCT_KEYS + ["Elasticity_MAP"]].copy()
        
        folds.append(FoldResult(
            train_periods=train_periods,
            test_period=test_period,
            train_periods_count=len(train_periods),
            test_rows=len(test_data),
            test_positive_rows=int((test_data["Unit_Sales"] > 0).sum()),
            elasticities=elasticities,
            predictions=pred_merged,
            mae_packs=mae,
            wapec=wapec,
            pred_coverage=pred_coverage,
            coverage_in_range=in_range,
            seasonal_naive_mae=seasonal_mae,
            last_observed_mae=last_mae,
        ))
    
    # Aggregate metrics
    valid_folds = [f for f in folds if not np.isnan(f.mae_packs)]
    
    result = EvaluationResult(
        folds=folds,
        data_hash=get_data_hash(data),
        config=config,
        mean_mae_packs=np.mean([f.mae_packs for f in valid_folds]) if valid_folds else np.nan,
        mean_wapec=np.nanmean([f.wapec for f in valid_folds]) if valid_folds else np.nan,
        mean_pred_coverage=np.mean([f.pred_coverage for f in valid_folds]) if valid_folds else np.nan,
        mean_coverage_in_range=np.mean([f.coverage_in_range for f in valid_folds]) if valid_folds else np.nan,
        mean_seasonal_naive_mae=np.nanmean([f.seasonal_naive_mae for f in valid_folds]) if valid_folds else np.nan,
        mean_last_observed_mae=np.nanmean([f.last_observed_mae for f in valid_folds]) if valid_folds else np.nan,
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
            "mae_packs": f.mae_packs,
            "wapec": f.wapec,
            "pred_coverage": f.pred_coverage,
            "coverage_in_range": f.coverage_in_range,
            "seasonal_naive_mae": f.seasonal_naive_mae,
            "last_observed_mae": f.last_observed_mae,
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
    }


def print_evaluation_summary(result: EvaluationResult) -> None:
    """Print human-readable evaluation summary."""
    print("\n" + "="*60)
    print("OUT-OF-TIME EVALUATION SUMMARY")
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
        print(f"    Seasonal naive MAE: {f.seasonal_naive_mae:.2f}" if not np.isnan(f.seasonal_naive_mae) else "    Seasonal naive MAE: N/A")
        print(f"    Last observed MAE: {f.last_observed_mae:.2f}" if not np.isnan(f.last_observed_mae) else "    Last observed MAE: N/A")
    
    print()
    print("Aggregate metrics:")
    print(f"  Mean MAE (packs): {result.mean_mae_packs:.2f}" if not np.isnan(result.mean_mae_packs) else "  Mean MAE (packs): N/A")
    print(f"  Mean WAPE: {result.mean_wapec:.4f}" if not np.isnan(result.mean_wapec) else "  Mean WAPE: N/A")
    print(f"  Mean prediction coverage: {result.mean_pred_coverage:.2%}")
    print(f"  Mean coverage in range: {result.mean_coverage_in_range:.2%}")
    print(f"  Mean seasonal naive MAE: {result.mean_seasonal_naive_mae:.2f}" if not np.isnan(result.mean_seasonal_naive_mae) else "  Mean seasonal naive MAE: N/A")
    print(f"  Mean last observed MAE: {result.mean_last_observed_mae:.2f}" if not np.isnan(result.mean_last_observed_mae) else "  Mean last observed MAE: N/A")
    
    if result.prior_sensitivity:
        print()
        print("Prior sensitivity:")
        for key, val in result.prior_sensitivity.items():
            if val.get("converged"):
                print(f"  {key}: mean_elasticity={val['mean_elasticity']:.3f}, std={val['std_elasticity']:.3f}")
            else:
                print(f"  {key}: FAILED - {val.get('message', 'unknown')}")
    
    print()
    print("NOTE: Predictive validation does not prove causal elasticity.")
    print("      Price identification, omitted variable bias, and endogenous pricing")
    print("      may invalidate causal interpretation of these metrics.")
    print("="*60)