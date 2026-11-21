"""CLI simulator using the shared package. Run: python sim.py --scenario nexus_cola_20pct"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

import pymc
import arviz

from retail_market_sim import (
    build_named_scenario, evaluate_rolling, fit_model_map, fit_model_mcmc, get_data_hash, load_data,
    simulate_scenario, simulate_scenario_posterior, three_factor_decomposition, RANDOM_SEED, ModelConfig,
    evaluation_to_dataframe, print_evaluation_summary,
)

DATA_PATH = Path(__file__).resolve().parent / "sample_data.csv"
MODEL_VERSION = "retail-map-v2"
OUTPUT_ROOT = Path("outputs")
RUNS_DIR = OUTPUT_ROOT / "runs"


def _get_git_revision() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _get_environment_info() -> dict:
    """Capture environment versions for reproducibility."""
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pymc_version": pymc.__version__,
        "arviz_version": arviz.__version__,
        "git_revision": _get_git_revision(),
        "random_seed": RANDOM_SEED,
    }


def _create_run_dir() -> Path:
    """Create a unique run directory under outputs/runs/."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]  # millisecond precision
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_artifacts(run_dir: Path, scenario: pd.DataFrame, estimates: pd.DataFrame,
                     decomp: pd.DataFrame, settings: dict, is_posterior: bool = False,
                     posterior_summary: pd.DataFrame | None = None) -> list[str]:
    """Write all artifacts to the run directory and return list of artifact filenames."""
    artifacts = []
    
    # Core artifacts (always produced)
    scenario_path = run_dir / "scenario_detail.csv"
    scenario.to_csv(scenario_path, index=False)
    artifacts.append("scenario_detail.csv")
    
    estimates_path = run_dir / "elasticity_estimates.csv"
    estimates.to_csv(estimates_path, index=False)
    artifacts.append("elasticity_estimates.csv")
    
    decomp_path = run_dir / "market_decomposition.csv"
    decomp.to_csv(decomp_path, index=False)
    artifacts.append("market_decomposition.csv")
    
    settings_path = run_dir / "model_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2))
    artifacts.append("model_settings.json")
    
    # Posterior artifacts (only if MCMC run succeeded)
    if is_posterior and posterior_summary is not None:
        post_path = run_dir / "posterior_summary.csv"
        posterior_summary.to_csv(post_path, index=False)
        artifacts.append("posterior_summary.csv")
        # Update settings with posterior note
        settings["posterior_summary.csv"] = "elasticity-posterior scenario summary; not a future-sales predictive interval"
        settings_path.write_text(json.dumps(settings, indent=2))
    
    return artifacts


def _load_changes_file(path: Path, baseline: pd.DataFrame) -> dict[tuple, float]:
    """Load price changes from a CSV or JSON file exported by the dashboard."""
    import json
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        # Expected format: [{"cell": [...], "price": ...}, ...]
        changes = {}
        for item in data.get("price_changes", []):
            cell = tuple(item["cell"])
            price = float(item["price"])
            # Validate cell exists in baseline
            mask = (
                (baseline["Category"] == cell[0]) &
                (baseline["Retailer"] == cell[1]) &
                (baseline["Brand"] == cell[2]) &
                (baseline["SKU"] == cell[3])
            )
            if not mask.any():
                raise ValueError(f"Cell not found in baseline: {cell}")
            if mask.sum() > 1:
                raise ValueError(f"Multiple matches for cell: {cell}")
            changes[cell] = price
        return changes
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        # Expected columns: Category, Retailer, Brand, SKU, Price
        required_cols = {"Category", "Retailer", "Brand", "SKU", "Price"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        changes = {}
        for _, row in df.iterrows():
            cell = (row["Category"], row["Retailer"], row["Brand"], row["SKU"])
            price = float(row["Price"])
            # Validate cell exists in baseline
            mask = (
                (baseline["Category"] == cell[0]) &
                (baseline["Retailer"] == cell[1]) &
                (baseline["Brand"] == cell[2]) &
                (baseline["SKU"] == cell[3])
            )
            if not mask.any():
                raise ValueError(f"Cell not found in baseline: {cell}")
            if mask.sum() > 1:
                raise ValueError(f"Multiple matches for cell: {cell}")
            changes[cell] = price
        return changes
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .csv")


def _run_evaluation_mode(args, data: pd.DataFrame, config: ModelConfig) -> None:
    """Run out-of-time evaluation mode."""
    print("Running out-of-time evaluation...")
    
    # Determine test periods
    periods = sorted(data["Period"].unique())
    if args.eval_test_periods:
        test_periods = [int(p) for p in args.eval_test_periods.split(",")]
    else:
        test_periods = [p for p in periods if p > args.eval_min_train]
    
    # Prior grid for sensitivity
    prior_grid = None
    if args.eval_prior_grid:
        prior_grid = [float(p) for p in args.eval_prior_grid.split(",")]
    
    result = evaluate_rolling(
        data=data,
        config=config,
        min_train_periods=args.eval_min_train,
        test_periods=test_periods,
        prior_grid=prior_grid,
    )
    
    # Print summary
    print_evaluation_summary(result)
    
    # Create run directory
    run_dir = _create_run_dir()
    run_id = run_dir.name
    
    # Export results
    dfs = evaluation_to_dataframe(result)
    
    # Write fold summary
    if not dfs["folds"].empty:
        dfs["folds"].to_csv(run_dir / "evaluation_folds.csv", index=False)
    
    # Write predictions
    if not dfs["predictions"].empty:
        dfs["predictions"].to_csv(run_dir / "evaluation_predictions.csv", index=False)
    
    # Write elasticities
    if not dfs["elasticities"].empty:
        dfs["elasticities"].to_csv(run_dir / "evaluation_elasticities.csv", index=False)
    
    # Write prior sensitivity
    if not dfs["prior_sensitivity"].empty:
        dfs["prior_sensitivity"].to_csv(run_dir / "evaluation_prior_sensitivity.csv", index=False)
    
    # Settings
    generated_at = datetime.now(timezone.utc).isoformat()
    settings = {
        "model_version": MODEL_VERSION,
        "data_hash": get_data_hash(data),
        "config": vars(config),
        "mode": "evaluation",
        "min_train_periods": args.eval_min_train,
        "test_periods": test_periods,
        "prior_grid": prior_grid,
        "generated_at": generated_at,
        "environment": _get_environment_info(),
    }
    (run_dir / "evaluation_settings.json").write_text(json.dumps(settings, indent=2))
    
    # Manifest
    manifest = {
        "run_id": run_id,
        "data_hash": settings["data_hash"],
        "model_version": MODEL_VERSION,
        "mode": "evaluation",
        "artifacts": sorted(p.name for p in run_dir.iterdir() if p.is_file() and p.name != "evaluation_settings.json"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    
    print(f"Evaluation outputs saved to {run_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail price simulation")
    parser.add_argument("--scenario", choices=["nexus_cola_20pct", "custom"], default="nexus_cola_20pct")
    parser.add_argument("--changes", type=Path, help="Load price changes from exported JSON/CSV file")
    parser.add_argument("--sample", action="store_true", help="Run MCMC posterior analysis")
    parser.add_argument("--prior-elasticity", type=float, default=-1.5)
    parser.add_argument("--max-evals", type=int, default=6000)
    
    # Evaluation mode arguments
    parser.add_argument("--eval", action="store_true", help="Run out-of-time evaluation with rolling training prefixes")
    parser.add_argument("--eval-min-train", type=int, default=12, help="Minimum training periods for evaluation (default: 12)")
    parser.add_argument("--eval-test-periods", type=str, help="Comma-separated test periods (default: all after min_train)")
    parser.add_argument("--eval-prior-grid", type=str, help="Comma-separated prior elasticities for sensitivity (e.g., -0.75,-1.5,-2.25)")
    
    args = parser.parse_args()
    
    data, notes = load_data(str(DATA_PATH))
    latest = int(data["Period"].max())
    baseline = data[data["Period"].eq(latest)].copy().reset_index(drop=True)
    config = ModelConfig(prior_elasticity=args.prior_elasticity, max_evaluations=args.max_evals)
    
    # Handle evaluation mode
    if args.eval:
        _run_evaluation_mode(args, data, config)
        return
    
    # Create run directory for this execution
    run_dir = _create_run_dir()
    run_id = run_dir.name
    
    # Compute raw input digest for reproducibility
    raw_input = DATA_PATH.read_bytes()
    raw_input_digest = hashlib.sha256(raw_input).hexdigest()[:16]
    
    try:
        # MAP fit
        fitted = fit_model_map(data, config)
        if not fitted["converged"]:
            raise RuntimeError(f"MAP optimization failed: {fitted['message']}")
        
        # Determine price changes
        if args.changes:
            price_changes = _load_changes_file(args.changes, baseline)
            scenario_name = "replay"
        elif args.scenario != "custom":
            price_changes = build_named_scenario(args.scenario, baseline)
            scenario_name = args.scenario
        else:
            price_changes = {}
            scenario_name = "custom"
        scenario = simulate_scenario(baseline, fitted["estimates"], price_changes)
        decomp = three_factor_decomposition(scenario, [])
        
        # Settings for MAP run
        generated_at = datetime.now(timezone.utc).isoformat()
        settings = {
            "model_version": MODEL_VERSION,
            "data_hash": get_data_hash(data),
            "raw_input_digest": raw_input_digest,
            "config": vars(config),
            "inference_method": "MAP",
            "baseline_period": latest,
            "converged": fitted["converged"],
            "iterations": fitted["iterations"],
            "scenario": scenario_name,
            "price_changes": [{"cell": list(k), "price": v} for k, v in price_changes.items()],
            "generated_at": generated_at,
            "environment": _get_environment_info(),
        }
        if args.changes:
            settings["changes_file"] = str(args.changes)
        
        # Write MAP artifacts
        artifacts = _write_artifacts(run_dir, scenario, fitted["estimates"], decomp, settings)
        
        # Posterior run if requested
        if args.sample:
            try:
                trace, products, pairs, categories, retailers, sku_idx, pair_idx, cat_idx, ret_idx, prod_cat_idx, diagnostics = fit_model_mcmc(data, config)
            except RuntimeError as e:
                # Diagnostics failed - clean up run directory and exit without creating a second run
                print(f"ERROR: MCMC diagnostics failed - {e}")
                print("Posterior intervals withheld. Previous run (if any) is unaffected.")
                # Clean up the run directory we created for this failed run
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                return
            
            # Save diagnostics
            diag_path = run_dir / "mcmc_diagnostics.json"
            diag_data = {
                "divergences": diagnostics.divergences,
                "max_rhat": diagnostics.max_rhat,
                "mean_rhat": diagnostics.mean_rhat,
                "min_bulk_ess": diagnostics.min_bulk_ess,
                "min_tail_ess": diagnostics.min_tail_ess,
                "max_rhat_params": diagnostics.max_rhat_params,
                "min_ess_params": diagnostics.min_ess_params,
                "passed_gates": diagnostics.passed_gates,
                "gate_criteria": diagnostics.gate_criteria,
            }
            diag_path.write_text(json.dumps(diag_data, indent=2))
            
            rng = np.random.default_rng(RANDOM_SEED)
            post, market = simulate_scenario_posterior(trace, baseline, products, pairs, sku_idx, pair_idx, cat_idx, ret_idx, prod_cat_idx, price_changes, n_draws=500, rng=rng)
            delta = market - baseline["Sales_Revenue"].sum()
            posterior_summary = pd.DataFrame([{
                "mean_revenue": market.mean(),
                "std_revenue": market.std(ddof=1),
                "q025_revenue": np.quantile(market, .025),
                "q975_revenue": np.quantile(market, .975),
                "mean_delta": delta.mean(),
                "probability_positive_delta": float((delta > 0).mean())
            }])
            
            # Add diagnostics to posterior summary
            posterior_summary["mcmc_divergences"] = diagnostics.divergences
            posterior_summary["mcmc_max_rhat"] = diagnostics.max_rhat
            posterior_summary["mcmc_min_bulk_ess"] = diagnostics.min_bulk_ess
            posterior_summary["mcmc_min_tail_ess"] = diagnostics.min_tail_ess
            posterior_summary["mcmc_passed_gates"] = diagnostics.passed_gates
            
            # Update settings for posterior
            settings["inference_method"] = "posterior"
            settings["mcmc_diagnostics"] = diag_data
            # Re-write artifacts with posterior summary
            artifacts = _write_artifacts(run_dir, scenario, fitted["estimates"], decomp, settings, 
                                         is_posterior=True, posterior_summary=posterior_summary)
        
        # Write manifest with only successfully produced artifacts
        manifest = {
            "run_id": run_id,
            "data_hash": settings["data_hash"],
            "raw_input_digest": settings["raw_input_digest"],
            "model_version": MODEL_VERSION,
            "inference_method": settings["inference_method"],
            "environment": settings["environment"],
            "artifacts": sorted(artifacts),
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        
        print(f"Outputs saved to {run_dir}/")
        
    except Exception as e:
        # Clean up on failure - remove the run directory
        if run_dir.exists():
            shutil.rmtree(run_dir)
        raise


if __name__ == "__main__":
    main()
