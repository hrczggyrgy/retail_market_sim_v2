# Implementation Plan: Beverage Market Price Simulator

Based on the detailed project plan at `/home/lptop/Documents/coding/bayesian-modeling/pymc_price_model/beverage_market_price_simulator_detailed_project_plan.md`

## Current State Summary

**Already Working (✓):**
- Data ingestion & validation (`data.py`)
- Historical 3-factor decomposition with exact reconciliation (`decomposition.py`)
- Hierarchical own-price Bayesian model with MAP/MCMC (`model.py`)
- Independent scenario simulation with posterior draws (`simulation.py`)
- Market substitution logit model with outside option (`market.py`)
- CLI with independent + market modes (`sim.py`)
- Streamlit dashboard with mode selector (`dashboard.py`)
- Rolling out-of-time evaluation (`evaluation.py`)
- 48 passing tests

**Major Gaps per Project Plan:**

---

## Phase 0: Project Setup & Data Contracts

### 0.1 Create formal data schemas (Pydantic)
- [ ] `schemas/inputs.py`: SKU master, sales monthly, validation configs
- [ ] `schemas/scenario.py`: Resolved price edits, scenario request/response
- [ ] `schemas/outputs.py`: Metric + interval + provenance types

### 0.2 Synthetic data generator
- [ ] `scripts/make_synthetic_data.py`: Known seasonal behavior + price responses for smoke tests
- [ ] Add synthetic data to `data/examples/`

### 0.3 Lockfile
- [ ] Generate `uv.lock` (or `requirements.lock`) - single lockfile only

---

## Phase 1: Historical Dashboard (Milestone 1)

### 1.1 Bottom-up aggregation engine (`analytics/aggregates.py`)
- [ ] Units, liters, revenue by all segment combinations
- [ ] Conditional shares (brand within category, retailer within brand, etc.)
- [ ] All totals reconcile to same bottom-level cells

### 1.2 Seasonal analysis (`analytics/seasonality.py`)
- [ ] Descriptive seasonal profiles with year-count warning
- [ ] Only model/visualize seasonality supported by data years

### 1.3 Exact historical waterfalls (`analytics/decomposition.py`) - extend existing
- [ ] Common-cell revenue decomposition (volume/mix/price)
- [ ] Entries/exits/unpriceable bridge to actual totals
- [ ] Zero accounting residual test

### 1.4 Dashboard pages
- [ ] **Page A - Market Overview**: Monthly units/liters/revenue, YoY, segment shares, drill-downs
- [ ] **Page B - What Changed?**: Period comparison, additive contributions, waterfalls for units/liters/revenue
- [ ] Label: "Accounting decomposition of observed data; does not establish why prices, product mix, or sales changed."

---

## Phase 2: Bayesian Baseline Model (Milestone 2)

### 2.1 Category-total model (`model/demand.py`)
- [ ] NegativeBinomial for category total units
- [ ] Fixed-base-weight geometric price index (pre-computed, frozen for scenarios)
- [ ] Trend + seasonal + shared time effects
- [ ] Hierarchical pooling for category-level parameters

### 2.2 Conditional allocation model (`model/demand.py`)
- [ ] DirichletMultinomial for SKU×retailer shares within category
- [ ] Utility: intercept + seasonality - beta × log(unit_value) + limited peer features
- [ ] Partially pooled beta across category/brand/package/retailer
- [ ] Sparse substitution neighborhoods (same SKU other retailer, same brand adjacent size, nearby competitor)

### 2.3 Fixed-base-weight indices (`model/features.py`)
- [ ] Geometric category price index with frozen weights
- [ ] Eligible SKU basket defined before simulation
- [ ] Weights don't change when predicted demand changes

### 2.4 Calibration profiles (`model/priors.py`)
- [ ] Lower/Central/Higher price responsiveness as versioned prior settings
- [ ] Cross-category only where identified, else labeled assumptions
- [ ] Historically supported price-change range with warnings

### 2.5 Model fitting & artifacts (`model/fit.py`)
- [ ] MAP + MCMC with `pm.Data` for scenario covariates
- [ ] Prior predictive checks (`pm.sample_prior_predictive`)
- [ ] Posterior artifact persistence (trace, metadata, config)
- [ ] Reproducible seed and model versioning

---

## Phase 3: Relative-Price Effects (Milestone 3)

### 3.1 Price identification diagnostics (`model/diagnostics.py`)
- [ ] Relative price variation report (retailer-specific price differences)
- [ ] Correlation of nearby prices
- [ ] Proportion of volume covered by priceable core
- [ ] Extent of model extrapolation
- [ ] Prior sensitivity analysis

### 3.2 Rolling backtests (`model/diagnostics.py`)
- [ ] Chronological holdouts: units, liters, revenue, shares at all levels
- [ ] Relative-price holdout tests: sign/magnitude where prices moved differently
- [ ] Separate fit quality vs price-identification badges

### 3.3 Warning system
- [ ] UI flags when posterior effect is mostly prior-driven
- [ ] Extrapolation warnings beyond supported price ranges

---

## Phase 4: Scenario Engine (Milestone 4)

### 4.1 Paired posterior draws (`simulator/counterfactual.py`)
- [ ] Baseline and scenario using SAME posterior draws
- [ ] `pm.Data` / `pm.set_data` workflow for price updates
- [ ] Verify all deterministic expected-value variables recompute
- [ ] Expected demand for comparison; simulate noise only if requested

### 4.2 Exact baseline anchoring (`simulator/anchor.py`)
- [ ] `Q_c*(θ) = Q_c0 × mu_c*(θ)/mu_c0(θ)`
- [ ] `s_j*(θ) = [s_j0 × pi_j*(θ)/pi_j0(θ)] / Σ_h[s_h0 × pi_h*(θ)/pi_h0(θ)]`
- [ ] `q_j*(θ) = Q_c*(θ) × s_j*(θ)`
- [ ] Unchanged prices → exact observed baseline
- [ ] Show both historical actual and unanchored fit in diagnostics

### 4.3 Market substitution with fitted parameters (`simulator/pricing.py`)
- [ ] Replace assumption-based (k, outside_share) with fitted elasticities
- [ ] Limited peer-price relationships from model
- [ ] Outside option for market expansion/contraction

### 4.4 Attribution & flows (`simulator/attribution.py`, `simulator/flows.py`)
- [ ] SKU revenue: `Δrevenue_j = (p_j*-p_j0)×q_j0 + p_j*×(q_j*-q_j0)`
- [ ] Higher-level: Shapley or fixed-order volume/mix/price
- [ ] Liter flow allocation with documented proximity matrix
- [ ] Outside-market source/sink for unbalanced gains/losses
- [ ] Report allocation rule + sensitivity alternative

### 4.5 Scenario reconciliation tests
- [ ] No-change → zero deltas (anchored mode)
- [ ] Units/liters/revenue sum correctly at all levels
- [ ] Same baseline + seed = same posterior summary
- [ ] Unsupported products/extrapolations trigger warnings

---

## Phase 5: API & Storage (Milestone 5)

### 5.1 Storage layer (`storage/repository.py`, `storage/artifacts.py`)
- [ ] Dataset/model/scenario metadata persistence
- [ ] Inference output and chart-ready caches
- [ ] Versioned artifacts with reproducible seeds

### 5.2 FastAPI service (`api/app.py`)
- [ ] `POST /datasets/validate` - quality report without promotion
- [ ] `POST /datasets` - persist approved versioned dataset
- [ ] `GET /datasets/{id}/coverage` - observation & eligibility coverage
- [ ] `POST /models/fit` - dataset ID + calibration → queued fit ID
- [ ] `GET /models/{id}/diagnostics` - posterior, backtest, identification
- [ ] `POST /scenarios` - model ID, baseline month, resolved price changes → scenario ID
- [ ] `GET /scenarios/{id}` - draw summaries, segment breakdowns, warnings
- [ ] `GET /scenarios/{id}/flows` - modeled liter flows + allocation rule

### 5.3 Background execution
- [ ] Non-blocking model fitting (queue + worker)
- [ ] Dashboard polls for fit completion

---

## Phase 6: Dashboard Polish (Milestone 5 continued)

### 6.1 Page C - Price Scenario Editor
- [ ] Select baseline month
- [ ] Edit prices by SKU×retailer (pct or target unit value)
- [ ] Bulk operations (brand/category/pack/retailer) with resolved cell preview
- [ ] Baseline value, proposed value, historical range, coverage warnings
- [ ] Save/version scenario definitions
- [ ] Compare 2-3 scenarios on same baseline/model

### 6.2 Page D - Scenario Results
- [ ] All aggregation levels: baseline, scenario, delta, pct, uncertainty
- [ ] Units, liters, revenue, share
- [ ] Market-size waterfall + within-market reallocation
- [ ] Slope charts for segment share
- [ ] Sensitivity curves for price steps
- [ ] Credible-interval plots for largest segments
- [ ] "Net category change" and "net observed-market change" alongside SKU gains

### 6.3 Page E - Switching View
- [ ] Model-implied loss/gain matrix in liters
- [ ] Own brand/other size, competitor brand, other retailer, other category, outside market
- [ ] Explicit allocation convention documented
- [ ] Unit changes separate from liter changes (2L ≠ 0.5L)
- [ ] Title: "Model-implied allocation, not measured transitions"

### 6.4 Page F - Model Health
- [ ] Estimation sample, excluded/ambiguous cells
- [ ] Observed relative price variation
- [ ] Prior sensitivity, posterior diagnostics
- [ ] PPC, backtest results
- [ ] Warning when price change extrapolates beyond observed
- [ ] Separate parameter uncertainty from omitted variable uncertainty

### 6.5 Provenance labels everywhere
- [ ] `observed` / `descriptive` / `modeled` / `assumed_flow` badges
- [ ] Dataset version, model version, baseline month, calibration profile
- [ ] Eligible-market coverage, key warnings alongside results

---

## Phase 7: Tests & Validation

### 7.1 Unit tests (`tests/unit/`)
- [ ] Aggregation, liter conversion, price indices
- [ ] Historical decomposition exact reconciliation
- [ ] No-change anchoring invariants
- [ ] Posterior-draw pairing
- [ ] Scenario reconciliation

### 7.2 Integration tests (`tests/integration/`)
- [ ] Import → fit → diagnostics → scenario → API response

### 7.3 Synthetic recovery (`tests/synthetic/`)
- [ ] Known seasonal + price response DGP
- [ ] Tolerances appropriate for stochastic model

### 7.4 Edge case tests
- [ ] Zero sales, missing rows, product entry/exit
- [ ] Nonexistent SKU-retailer, invalid prices
- [ ] Unsupported scenario changes

### 7.5 UI tests (`tests/ui/`)
- [ ] Provenance labels visible
- [ ] Important warnings surfaced

---

## Phase 8: Documentation & Release

### 8.1 Documentation (`docs/`)
- [ ] `product_spec.md` - personas, claims, out-of-scope
- [ ] `data_contract.md` - schema, quality checks, units
- [ ] `modeling.md` - likelihood, priors, identification
- [ ] `scenario_semantics.md` - baseline, anchoring, flows, attribution
- [ ] `api.md` - request/response examples
- [ ] `validation.md` - backtests, release gates

### 8.2 README
- [ ] Clean environment startup commands
- [ ] Pin PyMC + companion versions in lockfile

### 8.3 Release gates
- [ ] Prior predictive checks pass
- [ ] Sampling diagnostics pass (R-hat, ESS, divergences)
- [ ] PPC: category totals, SKU shares, sparse cells, seasonal
- [ ] Rolling backtests meet thresholds
- [ ] Relative-price holdout tests visible
- [ ] Sensitivity checks: priors, core eligibility, outliers
- [ ] If gates fail: serve historical analytics, disable/downgrade scenarios

---

## Priority Order (Vertical Slices)

| Slice | Deliverable | Depends On |
|-------|-------------|------------|
| 0 | Schemas + synthetic data + lockfile | - |
| 1 | Historical dashboard (Pages A, B) | 0 |
| 2 | Category-total + conditional model + fit | 0, 1 |
| 3 | Price identification + backtests | 2 |
| 4 | Scenario engine (paired draws, anchoring, flows) | 2, 3 |
| 5 | API + storage + background execution | 2, 4 |
| 6 | Dashboard Pages C, D, E, F | 4, 5 |
| 7 | Test suite + validation | All |
| 8 | Documentation + release | All |

---

## Key Technical Decisions Needed

1. **PyMC version pinning**: Test against specific version, document API usage
2. **DirichletMultinomial zero-count handling**: Omit composition obs for zero-category months
3. **Fixed-base-weight index**: Pre-compute weights from baseline period, freeze for scenarios
4. **Outside option**: Model as separate "outside market" category or as logit outside share
5. **Anchoring mode**: Default on or opt-in? (Plan suggests opt-in for decision support)
6. **Flow allocation**: Proximity matrix - same brand/size first, then category, then outside
7. **Storage backend**: Start with local filesystem, design interface for later DB/S3

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| PyMC API changes | Pin version, test `pm.Data`/`pm.set_data` recompute behavior |
| DirichletMultinomial with zeros | Document zero-category handling, test |
| Price identification weak | Explicit badges, sensitivity profiles, prior-driven warnings |
| Scenario recompute failures | Test all deterministic vars recompute after `pm.set_data` |
| Dashboard blocking on fit | Background queue + polling UI |
| Synthetic recovery too strict | Use appropriate tolerances for stochastic model |

---

## Acceptance Criteria Checklist

- [ ] README commands start app from clean env with pinned deps
- [ ] User loads synthetic data, inspects historical, fits model, changes 2+ SKU prices, sees segment results + uncertainty
- [ ] No-change anchored scenario → zero deltas within numerical tolerance
- [ ] Bottom-level units/liters/revenue reconcile to all aggregates; historical decompositions reconcile
- [ ] UI/API distinguish `observed` / `descriptive` / `modeled` / `assumed_flow`
- [ ] Diagnostics and out-of-support warnings visible (not buried in logs)
- [ ] Automated tests pass, no unimplemented placeholders on core path

---

## Next Immediate Action

Start with **Phase 0**: Create formal Pydantic schemas, synthetic data generator, and lockfile. This provides the foundation for all subsequent work.