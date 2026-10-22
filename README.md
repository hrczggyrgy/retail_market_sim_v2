# Retail Market Simulation

Explore how SKU-retailer price changes could affect revenue, demand, and value share in an observed POS dataset. This project combines a hierarchical Bayesian demand model with an interactive scenario dashboard and explicit accounting and modeling limitations.

## The decision

**Should a category manager discount Cola 2L Bottle at Nexus Retail Group, hold its price, or test a modest increase?**

The built-in example estimates that a 20% discount increases modeled revenue for the targeted SKU-retailer cell by about +58.8%, while increasing total observed-market revenue by about +2.27%. This is a model-implied scenario, not observed commercial uplift. The estimated elasticity is approximately -3.07; validate it with prior sensitivity, out-of-time evaluation, and a price test before acting.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
streamlit run app.py
python sim.py --scenario nexus_cola_20pct
python sim.py --scenario nexus_cola_20pct --sample
```

The dashboard caches data loading and MAP fitting. Price edits do not refit the model. The CLI `--sample` option runs MCMC and exports an elasticity-posterior scenario summary.

## What is implemented

| Capability | Status | Notes |
|---|---|---|
| Hierarchical own-price elasticity | Implemented | Partial pooling from category to SKU; within-cell price centering |
| MAP dashboard | Implemented | Cached fit; no uncertainty intervals |
| MCMC posterior scenario | Implemented | Joint elasticity draws; anchored response |
| Price-volume-mix attribution | Implemented | Sequential Volume -> Mix -> Price in litres |
| Value shares | Implemented | Total and category-conditional revenue shares |
| Scenario exports | Implemented | CSV, settings, and run manifest |
| Cross-price / cannibalization | Planned | Requires strong assumptions or additional evidence |
| Profit optimization | Planned | Requires unit costs |
| Future forecasting | Planned | Current scenarios anchor to an observed month |

## Case study

```bash
python sim.py --scenario nexus_cola_20pct
```

| Measure | Baseline | Scenario | Change |
|---|---:|---:|---:|
| Target price | 3.02 | 2.42 | -20% |
| Target units | 6,988 | 13,872 | +98.5% |
| Target revenue | 21,104 | 33,515 | +58.8% |
| Observed-market revenue | 547,323 | 559,734 | +2.27% |

The target-cell result is much larger than the observed-market result because only one cell is edited. Other cells retain baseline demand, although their revenue shares can change because the denominator changes.

## Evidence and limitations

- POS elasticities are associations under observed retail conditions, not guaranteed causal effects.
- The scenario anchors to observed Month 24 quantities; it does not forecast Month 25.
- Substitution, cannibalization, promotions, availability, and competitor reactions are not modeled.
- A large modeled demand response does not establish price extrapolation or prior sensitivity; use historical support and sensitivity analysis.
- The dashboard uses MAP point estimates. `--sample` produces elasticity-posterior uncertainty, not a future-sales predictive interval.
- Pack sizes must be numeric litres per pack; kg, pieces, grams, and millilitres are rejected rather than mixed into a single physical-volume measure.

## Tests and CI

```bash
pytest tests -v
```

CI runs package import, Ruff undefined-name checks, and tests on Python 3.10 and 3.12. The test suite covers unchanged-price reconciliation, cell and aggregate revenue bridges, shares, zero/unit elasticity, nonzero mix, named discount construction, posterior aggregation, dashboard caching, and CLI artifacts.

## Reproducibility

The CLI writes `scenario_detail.csv`, `elasticity_estimates.csv`, `market_decomposition.csv`, `model_settings.json`, and `run_manifest.json`. With `--sample`, it also writes `posterior_summary.csv`. The manifest records the run timestamp, data hash, model version, inference method, and artifact list.

## License

MIT License. See `LICENSE`.
