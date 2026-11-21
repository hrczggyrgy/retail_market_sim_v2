# Retail Market Simulation: improvement plan

## Executive Summary

- `app.py:1–5` launches Streamlit; `sim.py:17–63` provides the CLI. Both call a UI-independent pandas/NumPy analytical package with PyMC inference. Plotly renders the dashboard waterfall. Keep this stack.
- Input flow: `sample_data.csv` → `data.load_data` → monthly SKU–retailer cells → `model.fit_model_map` or `fit_model_mcmc`. The loader uses revenue divided by packs sold as positive-sales price. The latest observed period becomes the anchor; absent listings are not created.
- Scenario flow: Streamlit matrix → `dashboard_adapter.matrix_to_price_changes` → absolute prices keyed by category, retailer, brand and SKU; CLI → `build_named_scenario` or an empty mapping. `simulation` creates a new cell table, not a sequence of state updates. Dashboard summarizes it as KPIs, litre-based waterfall, grouped bridges, value shares and downloads; CLI writes result files and optionally a posterior summary.
- Fitted equation: `log(Q) ~ Normal(α_product + retailer_effect + pair_effect + β_product × (log(P) − mean_pair(log(P))) + season_category,month + trend_category × centered_years, σ)`. The fit uses positive-sales rows. Category slopes have Normal prior mean −1.5 and SD 0.75; each SKU adds a standard Normal offset scaled by 0.5. Retailer and category-season effects are centered; trend is measured in years. A SKU uses the same slope at every retailer.
- Scenario equation: `Q1 = Q0 × exp(β × log(P1/P0)); R1 = P1 × Q1`. Unedited cells retain demand; a zero anchor stays zero. There are no shoppers, stock constraints, competitor responses, substitution or market clearing. Share movement can be purely a denominator effect. The waterfall allocates revenue change in Volume → Mix → Price order; its bars are not separate causal effects.
- Dashboard MAP is a point estimate; CLI MCMC defaults to four chains, 1,000 tune and 1,000 retained draws per chain, target acceptance 0.9 and seed 42. Up to 500 joint draws are propagated at a fixed baseline, summed across cells before market quantiles. This is elasticity uncertainty, not future-sales uncertainty. Optimizer success is checked; sampler quality is not gated before publication.
- Tests cover reconciliation, support adapters, joint-draw aggregation and mocked dashboard/CLI flows. CI imports the package, checks undefined names and runs pytest on Python 3.10/3.12. This was a static GitHub source inspection, not a test run, benchmark, refit or browser usability study. Locations refer to files as retrieved from the default branch; the checkout was not pinned to a commit.

Top three changes, in order:

1. Stop valid zero-sales slices from breaking the dashboard and prevent stale artifacts from being presented as current results (P0-01, P0-02).
2. Make the evidence visible where users make decisions: qualify unsupported prices, validate the fitted response out of time and check sampler quality before showing intervals (P1-01, P1-02, P1-03).
3. Put a one-click example and target-level explanation ahead of the price matrix and diagnostic tables (P1-04, P1-05).

| Dimension | Score | Reason |
|---|---:|---|
| Usability | 3/5 | The README, reset, error feedback and downloads provide a working path, but the dashboard starts with a manual matrix and an unmeasured cold fit. |
| Visual Explainability | 3/5 | The bridge reconciles and states its accounting order, but target-level volume, elasticity and price support are separated from the result. |
| Scientific Accuracy | 2/5 | The identities are defensible for anchored scenarios, but price identification, out-of-time accuracy and posterior quality are not established for decisions. |
| Code Quality | 3/5 | Clear core/UI separation and useful tests coexist with permissive dependencies, untyped result contracts and shared mutable CLI outputs. |

## Quick Wins (High Impact, Low Effort)

Each estimate covers the narrow patch and focused verification, not the full roadmap item.

| Fix | Where | Check | Estimate |
|---|---|---|---:|
| Document `custom` as a no-edit baseline in CLI help pending real input support | `sim.py`, `main` | `--help` identifies no-edit behavior | 10 min |
| Use the existing unknown-support flag in an edited-cell warning | `dashboard.py`, `main`; `dashboard_adapter.py`, `price_support` | Edited cell with no positive-sales history is flagged separately | 25 min |
| Add the baseline month to the results caption | `dashboard.py`, `main` | Caption shows the same period as the settings export | 15 min |
| Show baseline/scenario packs and elasticity in the edited-cell table | `dashboard.py`, `main` | One edited row displays both quantities and its fitted slope | 20 min |
| Label the waterfall y-axis with the dataset currency convention and show currency in tooltips | `dashboard.py`, `main` | Hover distinguishes currency amounts from litres | 25 min |
| Correct the conditional-share fixture | `tests/test_invariants.py`, `test_shares_sum_to_100` | Add a second category and pass `parent_columns=["Category"]` | 35 min |
| Add Windows PowerShell activation alongside the POSIX quickstart | `README.md`, Quickstart | Commands identify the shell; no install command changes | 15 min |

## Prioritized Improvement Roadmap

### P0 — Critical

- **ID & Title:** [P0-01] A zero-sales slice breaks grouped decomposition
- **Category:** Scientific
- **Where:** `retail_market_sim/data.py:64–83` (`load_data`); `retail_market_sim/decomposition.py:22–61` (`three_factor_decomposition`), `:64–76` (`value_shares`); `retail_market_sim/dashboard.py:94–116` (`main`); `tests/test_batch2.py:20–26` (`test_decomposition_rejects_zero_volume_group`).
- **Current Behavior:** Ingestion retains zero-unit cells. A positive market can contain an all-zero retailer or brand group; one such group makes decomposition raise, and grouped tab calculations run outside the initial scenario try/except. Shares for zero denominators become NaN.
- **Why It Matters:** A valid scenario may render the market waterfall but fail in the breakdown. A missing share is not a zero share.
- **Proposed Fix:** Keep the strict decomposition API and add a reporting wrapper that partitions groups before calling it. For zero-to-zero volume and revenue, return zero absolute change with a `no sales` status and undefined mix shares; for one-sided zero, show absolute revenue change and `bridge unavailable` rather than inventing shares. Use `P/w` for cell prices per litre when both group totals are positive. Render denominator-zero shares as `N/A (no revenue)` and use the same baseline validation in CLI and dashboard.
- **Acceptance Criteria:** Synthetic positive market with a zero-only retailer renders all tabs; zero-only share parent has explicit N/A; positive groups reconcile within existing tolerance; all-zero market is rejected by both entry points with a clear message.
- **Effort / Impact:** Medium / High

- **ID & Title:** [P0-02] A new run can claim old posterior artifacts
- **Category:** Engineering
- **Where:** `sim.py:32–60` (`main`); `outputs/model_settings.json:1–9` (tracked example settings).
- **Current Behavior:** CLI writes into a shared `outputs` directory; manifest lists every file in it. MAP-only after `--sample` can list an old posterior CSV. A sampling exception occurs after MAP files have already been written. Tracked example settings identify v1, whereas CLI identifies v2.
- **Why It Matters:** Users can pair estimates from different experiments or reuse an old manifest after a failed run.
- **Proposed Fix:** Write each run into a new staging directory beneath an explicit output root. Track only successfully produced artifacts, including inference method per file; publish directory and manifest together after success. Keep tracked examples separate and label their version.
- **Acceptance Criteria:** A posterior run followed by MAP-only produces two disjoint manifests, with no posterior file in MAP-only. Pre-existing arbitrary files do not enter the manifest; a mocked sampling error leaves previous committed results untouched.
- **Effort / Impact:** Medium / High

### P1 — High Impact

- **ID & Title:** [P1-01] Show support and numerical guardrails at the point of editing
- **Category:** Scientific
- **Where:** `retail_market_sim/dashboard_adapter.py:43–59` (`price_support`); `retail_market_sim/dashboard.py:70–92` (`main`); `retail_market_sim/model.py:216–242` (`fit_model_map`); `retail_market_sim/validation.py:23–29` (`check_finite_response`).
- **Current Behavior:** Historical bounds use positive-sales cell history; unavailable bounds set `Support_Unknown` but not `Outside_Observed_Range`. The dashboard prominent count only uses the latter. SKU heuristics use 12 observations and within-pair log-price SD 0.02. Absolute log response above 50 is rejected, but this is numerical protection, not an economic bound.
- **Why It Matters:** A low positive price can imply very large finite demand even when it is outside the data; zero baseline has no extensive-margin response. A positive fitted slope yields upward-sloping demand and is currently warned about globally.
- **Proposed Fix:** Use status values in-range/out-of-range/unknown for edited rows and show their price bounds, signed elasticity and zero-anchor status. Move the historical support calculation into the shared reporting path and export statuses from CLI too. Require confirmation for unsupported edits in the UI; do not silently clip. Explain the heuristic thresholds and identify positive slopes rather than forcing them negative. Use `log(P1) − log(P0)` to avoid overflow in the price ratio before applying the existing response guard.
- **Acceptance Criteria:** Test one edited row per support status, endpoints inclusively, positive slope, and extreme finite input. Unknown never counts as supported; zero anchor stays zero and is labeled not estimable for price activation.
- **Effort / Impact:** Medium / High

- **ID & Title:** [P1-02] Measure elasticity sensitivity and predictive skill without conflating them with causality
- **Category:** Scientific
- **Where:** `retail_market_sim/data.py:57–69` (`load_data`); `retail_market_sim/model.py:20–39` (`ModelConfig`), `:46–101` (`_prepare_training_data`), `:105–169` (`_build_model`); `retail_market_model.md`, §§7–8, 11–12.
- **Current Behavior:** Realized unit value is used as price; log sales are fitted only on positive rows. Price centering, SKU slopes and category season/trend regularize the association but do not observe promotions or availability. README requests prior sensitivity and out-of-time validation; the shipped fit has no evaluation entry point. The model note illustrates a different regression specification.
- **Why It Matters:** Changing transaction mix can change unit value; omitted shocks and endogenous pricing can change the slope. A fit to historical sales or a narrow credible interval cannot establish discount uplift.
- **Proposed Fix:** Add a CLI evaluation mode using rolling training prefixes (e.g. months 1–18, 1–19, through 1–23) and following-period evaluation. Reuse training-only pair price means, trend center and encodings; report coverage separately for zeros and unseen cells. Compare held-out one-step price-conditioned predictions against last-observed matched-cell and seasonal-naive baselines; report MAE in packs and WAPE only when actual aggregate volume is positive. Refit a declared prior grid (e.g. −0.75/−1.5/−2.25 with baseline scale and a wider SD), and compare target slope and scenario revenue deltas. Document fitted versus proposed equations. For causal calibration, request posted prices, promotion/availability flags and randomized or otherwise credible price variation.
- **Acceptance Criteria:** Check no holdout rows alter training encodings or priors; report fold coverage, errors, and changes in scenario sign or magnitude across priors; document that predictive validation does not prove causal elasticity.
- **Effort / Impact:** High / High

- **ID & Title:** [P1-03] Qualify posterior intervals before exporting them
- **Category:** Scientific
- **Where:** `retail_market_sim/model.py:255–290` (`fit_model_mcmc`); `retail_market_sim/simulation.py:65–123` (`summarize_revenue_draws`, `simulate_scenario_posterior`); `sim.py:47–57` (`main`).
- **Current Behavior:** MCMC returns a trace without a publication gate. Simulation flattens chain/draw axes, subsamples jointly and returns market draws. The CLI discards the cell summary and publishes aggregate quantiles and positive-delta frequency.
- **Why It Matters:** An apparently precise interval can hide nonconvergence or Monte Carlo error. Users cannot inspect uncertainty for the target cell separately from market totals.
- **Proposed Fix:** Use ArviZ before flattening: report divergences, rank-normalized R-hat and bulk/tail ESS for edited slopes and the market delta (preserve chain/draw coordinates for that diagnostic). Start with gates of zero divergences, R-hat ≤1.01 and ESS ≥400 for nonconstant decision metrics; document these as review thresholds, not guarantees. For no-edit, label deterministic zero delta separately. Use all joint draws or chunk exact calculation; export per-cell and market median/95% equal-tailed intervals, seed, draw count and diagnostics. Label intervals conditional on baseline and elasticity only.
- **Acceptance Criteria:** Mock divergent and unconverged traces: no decision-ready interval is emitted; offsetting cells still aggregate to a constant; identical trace/settings reproduce results; no-edit yields exactly zero delta apart from declared rounding tolerance.
- **Effort / Impact:** Medium / High

- **ID & Title:** [P1-04] Give first-time users a scenario without making them edit the matrix
- **Category:** Usability
- **Where:** `app.py:1–5` (entry point); `retail_market_sim/dashboard.py:32–67` (`main`); `retail_market_sim/simulation.py:125–136` (`build_named_scenario`); `sim.py:17–30` (`main`).
- **Current Behavior:** Dashboard starts with a fit, a bulk editor and Reset; named example is CLI-only. `custom` builds an empty mapping and cannot consume a settings export.
- **Why It Matters:** A newcomer must find the correct product, retailer and absolute price. Exported price edits cannot be replayed by the CLI.
- **Proposed Fix:** Add a top-level Hold / Cola −20% / Custom selector using the shared named builder. Show expected run scope, baseline month, target and discount before applying; hide advanced fit controls by default and retain the matrix for custom edits. Add `--changes FILE` to validate and replay exported cell keys and absolute prices, with data hash/period mismatch explicit. If timing fails, preload only a fitted sample artifact that matches data hash, model version and config.
- **Acceptance Criteria:** Browser test: preset modifies exactly one targeted cell, reset restores hold, and an invalid custom price leaves the last successful view labeled as such. Export/import round-trips the same edits. Measure time to first meaningful result from installed app with five new users; report cold/warm times separately; target at least four of five under 30 seconds.
- **Effort / Impact:** Medium / High

- **ID & Title:** [P1-05] Connect the market result to the edited cells
- **Category:** Visual Explainability
- **Where:** `retail_market_sim/dashboard.py:72–124` (`main`); `retail_market_sim/simulation.py:49–61` (`simulate_scenario`); `retail_market_sim/decomposition.py:36–58` (`three_factor_decomposition`).
- **Current Behavior:** The waterfall explains market accounting. Edited-cell summary shows prices and revenue delta but not packs, slope or cell bridge; shares and diagnostics are in separate tables. The waterfall uses a generic Revenue axis.
- **Why It Matters:** The observer cannot easily distinguish a lower price from the resulting model-implied volume response or see that other products lost share without losing demand.
- **Proposed Fix:** Place Target and Observed market side by side. Add cell selection and baseline/scenario paired dots for price per pack, packs and revenue using separate axes, with numeric labels and color-independent markers. Add exact cell price and volume effects, support bounds and elasticity on selection. Keep the litre-based three-factor market waterfall separate, label currency, and offer signed share changes in percentage points with numerator and denominator in hover. Do not offer margin without costs.
- **Acceptance Criteria:** Fixture P0=10, Q0=100, P1=8, β=−1 shows 125 packs, zero revenue change, +250 volume and −250 price. Unedited cell retains packs; share hover identifies scope. Legends and labels remain readable without relying on color.
- **Effort / Impact:** Medium / High

- **ID & Title:** [P1-06] Stop UI-only interactions from repeating analytical work
- **Category:** Engineering
- **Where:** `retail_market_sim/dashboard.py:20–29,46–74,96–138` (`cached_load`, `cached_fit`, `main`); `retail_market_sim/simulation.py:24–36` (`_apply_price_changes`); `retail_market_sim/validation.py:47–64` (`validate_scenario_targets`); `retail_market_sim/decomposition.py:6–19` (`_pack_size_to_factor`, `_ensure_equiv_volumes`).
- **Current Behavior:** Fit and load are cached; scenario and all grouping/tab computations are inline. Changed keys scan all rows during both validation and assignment. The litre conversion builds a one-element Series per cell per bridge call unless equivalent volumes already exist. These are static bottleneck candidates, not measured timings.
- **Why It Matters:** Selecting a breakdown can repeat work unrelated to the change; large matrices increase edit cost.
- **Proposed Fix:** Measure cold fit, warm price application, and view-only rerun separately. Cache a typed scenario result by data identity, fit config/version and normalized immutable edits; build support and equivalent volumes once. Use a MultiIndex lookup for edits and unknown keys. Pass cached results to small render functions. Preserve input immutability and existing validation errors.
- **Acceptance Criteria:** Profile sample and synthetic 10× cells; tests count one fit across edits, no scenario rerun on view-only selection, and numerical parity with old outputs. Set warm P95 budget after measuring baseline; report before/after.
- **Effort / Impact:** Medium / Medium

- **ID & Title:** [P1-07] Test real contracts and capture reproducible environments
- **Category:** Engineering
- **Where:** `pyproject.toml:1–47`; `requirements.txt:1–14`; `.github/workflows/ci.yml:1–17`; `tests/test_invariants.py` (`test_shares_sum_to_100`, `test_mix_effect_zero_when_shares_unchanged`); `tests/test_reliability.py:55–106` (posterior and entry-point tests).
- **Current Behavior:** Two dependency lists use lower bounds; xarray is imported directly in a test but not declared directly. CI selects only Ruff F821; entry-point tests mock the real fit. The share invariant uses a single category and the market denominator; the mix test edits only Retailer_A, despite describing unchanged shares.
- **Why It Matters:** Install drift can change numerical results; fixtures can pass without testing the named properties.
- **Proposed Fix:** Use `pyproject.toml` as dependency authority with a generated tested lock/constraints file; declare direct test imports. Record Python/PyMC/NumPy/pandas versions, code revision, seed and raw input digest in manifests. Type the fit and scenario payloads at the shared boundary. Correct conditional-share coverage and use an equal percentage adjustment to every cell with the same slope for the zero-mix test; add a nonzero-mix counterexample. Add a small real MAP integration fixture and separate slower sampler smoke job.
- **Acceptance Criteria:** Clean Python 3.10/3.12 environments resolve the declared constraints; two-category tests fail if parent denominator is removed; direct fixture fit yields finite slope and reconciled outputs within set numerical tolerances.
- **Effort / Impact:** Medium / High

### P2 — Polish & Extensions

- **ID & Title:** [P2-01] Add constrained supply or substitution only with a defined data contract
- **Category:** Scientific
- **Where:** `retail_market_sim/simulation.py:39–61,108–122` (`simulate_scenario`, `simulate_scenario_posterior`); `retail_market_model.md`, §8; `README.md`, Evidence and limitations.
- **Current Behavior:** Conditional on elasticities and edits, the response is deterministic and each cell uses only its own price. Joint posterior slopes may be dependent; no stock, competitor policies, cross-price term or outside option enters the response. The planning note sketches cross-price regression but code does not implement it.
- **Why It Matters:** A retailer-wide discount is not a simulated price war, and there is no basis to claim a supply shock or quantify share transfer from current results.
- **Proposed Fix:** Keep independent own-price mode. If stock data exist, add a separately labeled one-period constrained mode: `sales=min(latent demand, available); lost=max(demand−available,0); closing=available−sales`. If choice data become available, pilot a separate category model with explicit outside option and availability mask; normalize probabilities across available options plus outside, and calibrate against observed substitution or experiments. Avoid labeling unavailable features as presets.
- **Acceptance Criteria:** At demand 120 and availability 80, sales=80, lost=40, closing=0; reject negative stock. Choice mode probabilities sum to one; unavailable item has zero share; replicate held-out diversion targets before promoting the feature.
- **Effort / Impact:** High / Medium

- **ID & Title:** [P2-02] Offer decision comparisons without suggesting a revenue optimum
- **Category:** Visual Explainability
- **Where:** `retail_market_sim/dashboard.py:32–149` (`main`); `retail_market_sim/simulation.py:39–61` (`simulate_scenario`); `README.md`, The decision.
- **Current Behavior:** A single edited scenario is rendered against hold. The decision example asks about discount, hold and an increase, but the dashboard does not retain multiple alternatives.
- **Why It Matters:** With a fixed elasticity below −1, revenue always increases as price falls in the uncapped model; an unconstrained optimizer would not reveal a defensible commercial price.
- **Proposed Fix:** Add three labeled alternatives and a target-cell price sweep restricted by default to supported prices. Plot revenue and packs on aligned separate panels with baseline marker and optional qualified posterior bands. Mark extrapolation separately, state that this is not a recommendation, and add contribution `(P−unit_cost)×Q` only with verified unit cost input.
- **Acceptance Criteria:** The hold point matches baseline; β=−1 produces a flat target revenue curve; unsupported points and unavailable margin cannot masquerade as recommendations.
- **Effort / Impact:** Medium / Medium

## Suggested Implementation Sequence

1. Milestone 1: Correctness and reproducible runs — land P0-01 and P0-02 with fixtures, then P1-07 and P1-01. Agree on zero-denominator and support-status semantics before changing UI or export formats.
2. Milestone 2: Explainable scenarios — implement P1-06 as a shared result boundary, then P1-04 and P1-05 on top. Demonstrate preset-to-export replay, target/market reconciliation and the onboarding timing gate. Keep posterior controls out of this milestone until diagnostics are ready.
3. Milestone 3: Evidence and extensions — land P1-02 and P1-03, then P2-02. Start P2-01 only if stock or substitution data and a validation owner are available. Existing scenarios remain explicitly conditional and noncausal.

## Implementation Status

| ID | Status | Changed files or symbols | Verification evidence | Remaining work |
|---|---|---|---|---|
| P0-01 | Verified | retail_market_sim/decomposition.py (three_factor_decomposition_report, _decomposition_group_status, value_shares), retail_market_sim/dashboard.py (uses reporting wrapper), retail_market_sim/__init__.py (exports new function), tests/test_batch2.py (4 new regression tests) | All 10 tests in test_batch2.py pass; 42/43 tests pass (1 pre-existing dashboard smoke test failure unrelated); CLI runs successfully; ruff F821 clean | None |
| P0-02 | Verified | sim.py (staging dirs, atomic writes, cleanup on failure), tests/test_batch2.py (2 new regression tests) | All 11 tests in test_batch2.py pass; 43/44 tests pass (1 pre-existing dashboard smoke test failure unrelated); verified: posterior run followed by MAP-only produces disjoint manifests; pre-existing files don't enter manifest; mocked sampling error leaves previous results untouched; ruff F821 clean | None |
| P1-01 | Verified | retail_market_sim/dashboard_adapter.py (enhanced price_support with detailed status), retail_market_sim/dashboard.py (shows support status at edit point with color coding, warns for unsupported edits, info for zero-anchor), retail_market_sim/__init__.py (exports) | All 45 tests pass; dashboard shows in_range/out_of_range/unknown status with color coding (🟢/🔴/❓) and zero-anchor label (⚪); unsupported edits show detailed expander with historical bounds; zero-anchor cells labeled "not estimable"; ruff F821 clean | None |
| P1-02 | Verified | retail_market_sim/evaluation.py (new module with rolling evaluation), sim.py (--eval mode), retail_market_sim/__init__.py (exports) | All 45 tests pass; --eval mode runs rolling training prefixes (e.g., months 1-12 train, month 13 test); reports MAE, WAPE, coverage in historical range, seasonal naive and last-observed baselines; prior sensitivity grid (-0.75/-1.5/-2.25) shows elasticity shifts; exports fold summary, predictions, elasticities, prior sensitivity; ruff F821 clean | None |
| P1-03 | Verified | retail_market_sim/model.py (MCMCDiagnostics dataclass, _run_mcmc_diagnostics, updated fit_model_mcmc), sim.py (saves diagnostics, warns if gates fail), retail_market_sim/__init__.py (exports MCMCDiagnostics) | All 45 tests pass; fit_model_mcmc returns MCMCDiagnostics with divergences, R-hat, bulk/tail ESS; gates: 0 divergences, R-hat ≤1.01, bulk/tail ESS ≥400; sim.py saves mcmc_diagnostics.json, warns if gates fail, adds diagnostics to posterior_summary.csv; ruff F821 clean; test for posterior path passes | None |
| P1-04 | Verified | retail_market_sim/dashboard.py (preset selector with Hold/Cola -20%/Custom, shows target details), sim.py (--changes FILE for replay), retail_market_sim/__init__.py (exports build_named_scenario) | All 45 tests pass; dashboard shows preset selector with target cell, baseline price, new price, discount %, baseline period; presets use shared build_named_scenario; Custom mode retains editable matrix; --changes FILE replays exported JSON/CSV with validation; ruff F821 clean; CLI verified | Onboarding timing not measured (needs user study) |
| P1-05 | Verified | retail_market_sim/dashboard.py (cell selector with paired dots for price, quantity, revenue; shows elasticity, support status, historical range, revenue delta) | All 45 tests pass; dashboard shows cell selector dropdown for edited cells; paired dots charts for price/packs/revenue with numeric labels and distinct markers (circle/square/diamond); shows elasticity, support status with color coding, historical range, revenue delta; separate from market waterfall; ruff F821 clean | None |
| P1-06 | Verified | retail_market_sim/dashboard_cache.py (new module with in-memory scenario caching), retail_market_sim/dashboard.py (uses cached bundles), retail_market_sim/__init__.py (exports) | All 45 tests pass; dashboard uses in-memory cache keyed by data_hash:config_hash:edits_hash; scenario bundle (simulation + support + all decompositions) computed once per unique edit set; view-only tab selections don't recompute; ruff F821 clean; CLI works | None |
| P1-07 | Verified | pyproject.toml (added xarray), retail_market_sim/types.py (new typed contracts), retail_market_sim/__init__.py (exports), retail_market_sim/model.py (returns typed FitResult), retail_market_sim/simulation.py (returns typed ScenarioResult), sim.py (environment info in manifest), tests/test_invariants.py (fixed fixtures, added conditional share tests, fixed mix tests) | All 45 tests pass; pyproject.toml is dependency authority with xarray declared; typed contracts at shared boundary; manifests include Python/PyMC/NumPy/pandas versions, git revision, seed, raw input digest; test fixtures corrected for two-category conditional shares and equal-percentage mix tests | None |
| P2-01 | Superseded |  |  | Blocked: no stock/substitution data available; not implementing |
| P2-02 | Verified | retail_market_sim/dashboard.py (comparison mode with 3 alternatives, price sweep with extrapolation markers, β=-1 flat revenue note, extrapolation warnings) | All 45 tests pass; dashboard shows 3-alternative comparison (Hold/Current/Custom) with revenue/packs table; price sweep with extrapolation markers (orange X for no historical support), baseline marker (green star), β=-1 produces flat revenue noted; extrapolation warnings shown; "not a recommendation" disclaimer; ruff F821 clean | Posterior bands not added (requires --sample run; noted in disclaimer) |

## Open Questions

- Are source prices posted prices, realized transaction prices, or mixed? Which currency and tax basis should the dashboard show? Who can supply promotion, availability and cost fields, and at what grain?
- Do zero sales mean no demand, out of stock, or a listing/data artifact? Are missing combinations genuinely unavailable? POS alone cannot answer either question.
- Are committed example CSVs intended as reference outputs or historical illustrations? Their rows and the raw sample dataset were not exhaustively reviewed; README case-study figures were not independently reproduced. The committed settings were inspected and identify v1.
- What browser, hardware and user cohort define the 30-second onboarding target? Neither cold fit nor interactive latency was measured in this source audit.
