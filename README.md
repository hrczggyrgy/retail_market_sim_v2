# Retail Market Simulation

> **Explore how SKU–retailer price changes could affect revenue, demand, and value share.**  
> This project combines a hierarchical Bayesian demand model with an interactive scenario dashboard, while making uncertainty and the limits of observational POS data explicit.

**🌐 Live demo:** [https://retailmarketsim.streamlit.app/](https://retailmarketsim.streamlit.app/)

---

## The Decision

**Should a category manager discount Cola 2L Bottle at Nexus Retail Group, hold its price, or test a modest increase?**

This repository provides a decision-support workflow to answer that question:

1. **Load** 24 months of POS data (units, revenue, price by SKU × retailer)
2. **Estimate** own-price elasticities with a partially-pooled hierarchical model
3. **Simulate** a price change (e.g., –20%) and see the implied revenue, volume, and share impact
4. **Decompose** the result into Volume / Mix / Price effects
5. **Inspect** evidence quality (price variation, convergence, extrapolation warnings)

> **Result for the built-in example:**  
> A 20% discount on Cola 2L Bottle at Nexus Retail Group (from $3.02 → $2.42) is estimated to **increase revenue by +2.27%** (+$12.4K), driven by a volume surge that more than offsets the price cut.  
> **Elasticity:** −3.07 (weak price variation; prior-sensitive).  
> **Limitation:** No substitution, competitor response, or promotion confounding is modeled.

![Dashboard screenshot](docs/dashboard.png)

---

## Quickstart

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run the interactive dashboard
streamlit run app.py

# Or run the CLI simulator
python sim.py --scenario nexus_cola_20pct
python sim.py --scenario nexus_cola_20pct --sample   # MCMC posterior (slower)
```

**Expected first-run behavior:**  
PyMC compiles the model on first fit (~10–30 s). Subsequent dashboard interactions use cached MAP estimates instantly.

---

## Project Structure

| File / Directory | Purpose |
|------------------|---------|
| `app.py` | Streamlit dashboard (MAP point estimates, price matrix editor, waterfall charts) |
| `sim.py` | CLI simulator (MAP + optional MCMC posterior, scripted scenarios) |
| `retail_market_sim/` | Shared analytical core (see below) |
| `retail_market_model.md` | Full mathematical specification (LaTeX) |
| `sample_data.csv` | 24-month synthetic beverage POS data (4 categories, 3 retailers, 14 SKUs) |
| `tests/test_invariants.py` | 18 analytical invariant tests (run with `pytest`) |
| `outputs/` | Generated scenario artifacts (git-ignored) |

### `retail_market_sim/` — shared core

| Module | Exports |
|--------|---------|
| `data.py` | `load_data()`, `validate_baseline()`, `tuple_index()`, `get_data_hash()` |
| `model.py` | `ModelConfig`, `fit_model_map()`, `fit_model_mcmc()`, `RANDOM_SEED` |
| `simulation.py` | `simulate_scenario()`, `simulate_scenario_posterior()`, `build_named_scenario()` |
| `decomposition.py` | `three_factor_decomposition()`, `value_shares()`, `_pack_size_to_factor()` |
| `validation.py` | `check_price()`, `check_elasticity()`, `check_finite_response()`, etc. |

---

## Implemented vs. Planned

| Capability | Status | Notes |
|------------|--------|-------|
| Own-price elasticity (hierarchical MAP) | ✅ Implemented | Partially pooled SKU→category, within-cell price centering |
| Own-price elasticity (MCMC posterior) | ✅ Implemented | `--sample` flag, 4 chains, draws preserved for aggregation |
| Price–Volume–Mix decomposition | ✅ Implemented | Sequential Volume → Mix → Price with equivalized volumes |
| Value-share shifts (total & conditional) | ✅ Implemented | Retailer, Category, Brand, Brand-within-Retailer |
| Interactive price editing | ✅ Implemented | Wide matrix, reset, change tracking, extrapolation warnings |
| Scenario export (CSV + settings JSON) | ✅ Implemented | Reproducible artifact bundle |
| Cross-price elasticity / cannibalization | 📋 Planned | Not identifiable from POS-only without strong assumptions |
| Promo / distribution / advertising effects | 📋 Planned | Not in POS data; requires external merge |
| Profit optimization (with costs) | 📋 Planned | Revenue-only currently; costs need external input |
| Predictive demand forecasting | 📋 Planned | Current model is same-month counterfactual, not next-month forecast |

---

## The Nexus Cola Case Study (Reproducible)

```bash
python sim.py --scenario nexus_cola_20pct
```

**Inputs (from `sample_data.csv`, Month 24):**
- SKU: Cola_2L_Bottle (Carbonated Soft Drinks / ColaClassic)
- Retailer: Nexus_Retail_Group
- Baseline price: $3.02
- Baseline monthly units: 6,988
- Baseline monthly revenue: $21,104

**Scenario:** 20% discount → new price $2.42

**MAP Results:**
| Metric | Baseline | Simulated | Change |
|--------|---------:|----------:|-------:|
| Price | 3.02 | 2.42 | –20.0% |
| Units | 6,988 | 13,872 | +98.5% |
| Revenue | 21,104 | 33,515 | +58.8% |
| **Market Revenue** | **547,323** | **559,734** | **+2.27%** |

**Three-Factor Decomposition (Market):**
| Effect | Value |
|--------|------:|
| Volume | +26,894 |
| Mix | –6,105 |
| Price | –8,379 |
| **Sum = ΔRevenue** | **+12,411** |

**Evidence Warnings:**
- Elasticity for this SKU is **prior-sensitive** (weak within-retailer price variation)
- Constant-elasticity extrapolation nearly **doubles quantity** — far outside observed price range
- **No substitution modeled** — competitor volumes held fixed; true market impact may differ

**Artifacts saved to `outputs/`:**
- `scenario_detail.csv` — cell-level results
- `elasticity_estimates.csv` — all SKU elasticities + support flags
- `market_decomposition.csv` — Volume/Mix/Price by segment
- `model_settings.json` — config, convergence, data hash

---

## Mathematical Foundation

The model estimates a hierarchical log-log demand system:

$$
\ln Q_{it} = \alpha_{s(i)} + \mu_{r(i)} + a_{s(i),r(i)} + \beta_{s(i)}\bigl(\ln P_{it} - \overline{\ln P}_{s(i),r(i)}\bigr) + \lambda_{c(i),m(i)} + \tau_{c(i)} T_t + u_{it}
$$

- $\beta_s \mid \bar\beta_{c(s)} \sim \mathcal{N}(\bar\beta_{c(s)}, 0.5^2)$ — partial pooling toward category mean
- Price centered **within SKU–retailer cell** → identifies from historical variation, not cross-SKU levels
- Seasonal (11 dummies) and linear trend at category level
- SKU–retailer pair effect $a_{s,r}$ absorbs persistent cross-retailer differences

**Scenario response** (anchored to observed baseline $Q_0$):

$$
Q_1 = Q_0 \exp\bigl(\hat\beta \ln(P_1/P_0)\bigr), \quad R_1 = P_1 Q_1
$$

**Decomposition** uses equivalized volumes (litres for beverages):

$$
\begin{aligned}
VE &= (Q_1 - Q_0)\bar P_0 \\
ME &= Q_1 \sum_s (s_{1s} - s_{0s}) P^V_{0s} \\
PE &= \sum_s V_{1s}(P^V_{1s} - P^V_{0s})
\end{aligned}
$$

---

## Limitations (Read Before Using)

| Limitation | Implication |
|------------|-------------|
| **Observational POS only** | Elasticities are *associations under observed retail conditions*, not guaranteed causal effects. Unobserved promos, stockouts, and demand shocks may coincide with price moves. |
| **Same-month counterfactual** | Anchors to observed Month 24 quantities. Does **not** forecast Month 25. |
| **No substitution** | Competitor volumes fixed. Cross-price effects excluded. Share changes are purely denominator effects. |
| **Constant elasticity** | Extrapolation far from observed prices (e.g., 2× quantity) is unreliable. |
| **Weak price variation** | Several SKUs flagged `Weak_Price_Support` → estimates shrink toward prior. |
| **MAP ≠ posterior** | Dashboard uses point estimates. `--sample` runs MCMC but dashboard doesn't display intervals yet. |

---

## Running Tests

```bash
pytest tests/ -v
# 18 tests covering:
# - Unchanged prices → baseline
# - Cell & aggregate price/volume reconciliation
# - Share normalization (sum to 100%)
# - Zero/unit elasticity edge cases
# - Three-factor decomposition reconciliation
# - Pure mix effect (non-zero hand-calculated)
# - Empty scenario, discount application
# - Posterior covariance preservation
```

---

## Reproducibility

- **Dependencies:** `numpy`, `pandas`, `pymc≥5.10`, `arviz`, `streamlit`, `plotly`
- **Environment:** `pip install -e ".[dev]"` (includes `pytest`, `ruff`, `mypy`)
- **Lockfile:** Generate with `pip freeze > requirements-lock.txt` after install
- **CI:** Add `.github/workflows/ci.yml` running `pytest`, `ruff`, `mypy`

---

## License

MIT License — see `LICENSE` file.