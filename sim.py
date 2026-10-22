"""
Standalone simulation script using the shared retail_market_sim package.

Usage:
    python sim.py --scenario nexus_cola_20pct
    python sim.py --scenario nexus_cola_20pct --sample
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retail_market_sim import (
    load_data,
    fit_model_map,
    fit_model_mcmc,
    ModelConfig,
    simulate_scenario,
    build_named_scenario,
    simulate_scenario_posterior,
    three_factor_decomposition,
    RANDOM_SEED,
)

DATA_PATH = Path(__file__).resolve().parent / "sample_data.csv"
MODEL_VERSION = "retail-map-v1"


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

    config = ModelConfig(prior_elasticity=args.prior_elasticity, max_evaluations=args.max_evals)

    print(f"\nFitting model (MAP) on {len(data)} observations...")
    fitted = fit_model_map(data, config)
    estimates = fitted["estimates"]

    print(f"  Converged: {fitted['converged']} (iters: {fitted['iterations']})")
    print(f"  Positive elasticities: {fitted['positive_elasticities']}")
    print(f"  Weak support: {fitted['weak_support_count']}")

    # Build scenario
    if args.scenario == "nexus_cola_20pct":
        price_changes = build_named_scenario("nexus_cola_20pct", baseline)
        # Print scenario info
        for key, new_price in price_changes.items():
            cat, ret, brand, sku = key
            mask = (
                (baseline["Category"] == cat) &
                (baseline["Retailer"] == ret) &
                (baseline["Brand"] == brand) &
                (baseline["SKU"] == sku)
            )
            baseline_price = float(baseline.loc[mask, "Price"].values[0])
            print(f"Scenario: 20% discount on {sku} at {ret}")
            print(f"  Baseline price: ${baseline_price:.2f} -> New price: ${new_price:.2f}")
    else:
        price_changes = {}

    # Apply scenario (MAP point estimate)
    scenario = simulate_scenario(baseline, estimates, price_changes)

    # Results
    r0 = scenario["Baseline_Revenue"].sum()
    r1 = scenario["Simulated_Revenue"].sum()
    q0 = scenario["Baseline_Unit_Sales"].sum()
    q1 = scenario["Simulated_Unit_Sales"].sum()

    print(f"\n--- Scenario: {args.scenario} (MAP) ---")
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
        print("\nChanged SKU–retailer cells:")
        for _, row in changed.iterrows():
            print(f"  {row['Category']}/{row['Retailer']}/{row['Brand']}/{row['SKU']}: "
                  f"P {row['Baseline_Price']:.2f} -> {row['Simulated_Price']:.2f} "
                  f"({row['Price_Change_Pct']:+.1f}%), "
                  f"Q {row['Baseline_Unit_Sales']:.0f} -> {row['Simulated_Unit_Sales']:.1f}, "
                  f"R {row['Baseline_Revenue']:.2f} -> {row['Simulated_Revenue']:.2f}, "
                  f"Elasticity: {row['Elasticity_MAP']:.3f}")

    # Posterior sampling if requested
    if args.sample:
        try:
            import arviz as az
            HAS_ARVIZ = True
        except ImportError:
            HAS_ARVIZ = False

        if not HAS_ARVIZ:
            print("\nArviZ not installed; skipping MCMC.")
        else:
            print("\nRunning MCMC for posterior uncertainty...")
            trace, products, pairs, categories, retailers, sku_idx, pair_idx, cat_idx, ret_idx, prod_cat_idx = fit_model_mcmc(
                data, config
            )
            print("  Sampling posterior predictive for scenario...")
            rng = np.random.default_rng(RANDOM_SEED)
            post_scenario, market_revenue_draws = simulate_scenario_posterior(
                trace, baseline, products, pairs, sku_idx, pair_idx,
                cat_idx, ret_idx, prod_cat_idx, price_changes,
                n_draws=500, rng=rng
            )
            baseline_revenue = baseline["Sales_Revenue"].sum()
            delta_draws = market_revenue_draws - baseline_revenue
            prob_positive = (delta_draws > 0).mean()
            print(f"  Posterior mean revenue: ${market_revenue_draws.mean():,.2f}")
            print(f"  Posterior std:          ${market_revenue_draws.std(ddof=1):,.2f}")
            print(f"  95% interval:           ${np.quantile(market_revenue_draws, 0.025):,.2f} - ${np.quantile(market_revenue_draws, 0.975):,.2f}")
            print(f"  P(ΔRevenue > 0):        {prob_positive:.2%}")

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