"""CLI simulator using the shared package. Run: python sim.py --scenario nexus_cola_20pct"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from retail_market_sim import (
    build_named_scenario, fit_model_map, fit_model_mcmc, get_data_hash, load_data,
    simulate_scenario, simulate_scenario_posterior, three_factor_decomposition, RANDOM_SEED, ModelConfig,
)

DATA_PATH = Path(__file__).resolve().parent / "sample_data.csv"
MODEL_VERSION = "retail-map-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail price simulation")
    parser.add_argument("--scenario", choices=["nexus_cola_20pct", "custom"], default="nexus_cola_20pct")
    parser.add_argument("--sample", action="store_true", help="Run MCMC posterior analysis")
    parser.add_argument("--prior-elasticity", type=float, default=-1.5)
    parser.add_argument("--max-evals", type=int, default=6000)
    args = parser.parse_args()
    data, notes = load_data(str(DATA_PATH))
    latest = int(data["Period"].max())
    baseline = data[data["Period"].eq(latest)].copy().reset_index(drop=True)
    config = ModelConfig(prior_elasticity=args.prior_elasticity, max_evaluations=args.max_evals)
    fitted = fit_model_map(data, config)
    if not fitted["converged"]:
        raise RuntimeError(f"MAP optimization failed: {fitted['message']}")
    price_changes = build_named_scenario(args.scenario, baseline) if args.scenario != "custom" else {}
    scenario = simulate_scenario(baseline, fitted["estimates"], price_changes)
    decomp = three_factor_decomposition(scenario, [])
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    scenario.to_csv(out / "scenario_detail.csv", index=False)
    fitted["estimates"].to_csv(out / "elasticity_estimates.csv", index=False)
    decomp.to_csv(out / "market_decomposition.csv", index=False)
    settings = {
        "model_version": MODEL_VERSION, "data_hash": get_data_hash(data),
        "config": vars(config), "inference_method": "MAP", "baseline_period": latest,
        "converged": fitted["converged"], "iterations": fitted["iterations"],
        "price_changes": [{"cell": list(k), "price": v} for k, v in price_changes.items()],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "model_settings.json").write_text(json.dumps(settings, indent=2))
    if args.sample:
        trace, products, pairs, categories, retailers, sku_idx, pair_idx, cat_idx, ret_idx, prod_cat_idx = fit_model_mcmc(data, config)
        rng = np.random.default_rng(RANDOM_SEED)
        post, market = simulate_scenario_posterior(trace, baseline, products, pairs, sku_idx, pair_idx, cat_idx, ret_idx, prod_cat_idx, price_changes, n_draws=500, rng=rng)
        delta = market - baseline["Sales_Revenue"].sum()
        summary = pd.DataFrame([{"mean_revenue": market.mean(), "std_revenue": market.std(ddof=1), "q025_revenue": np.quantile(market, .025), "q975_revenue": np.quantile(market, .975), "mean_delta": delta.mean(), "probability_positive_delta": float((delta > 0).mean())}])
        summary.to_csv(out / "posterior_summary.csv", index=False)
        settings["posterior_summary.csv"] = "elasticity-posterior scenario summary; not a future-sales predictive interval"
        (out / "model_settings.json").write_text(json.dumps(settings, indent=2))
    manifest = {"run_id": settings["generated_at"], "data_hash": settings["data_hash"], "model_version": MODEL_VERSION,
                "inference_method": "posterior" if args.sample else "MAP", "artifacts": sorted(p.name for p in out.iterdir() if p.is_file())}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Outputs saved to {out}/")


if __name__ == "__main__":
    main()
