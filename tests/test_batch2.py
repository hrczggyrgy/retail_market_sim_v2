"""Tests for strict units, decomposition edge cases, and CLI provenance."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from retail_market_sim.decomposition import _pack_size_to_factor, three_factor_decomposition, three_factor_decomposition_report, value_shares
from retail_market_sim.simulation import build_named_scenario, simulate_scenario
from retail_market_sim.model import MCMCDiagnostics


@pytest.fixture
def cells():
    return pd.DataFrame({"Category": ["C", "C"], "Retailer": ["R", "R"], "Brand": ["B", "B"], "SKU": ["S1", "S2"],
                         "Pack_Size": ["1.0", "1.0"], "Unit_Sales": [100., 100.], "Sales_Revenue": [100., 300.],
                         "Price": [1., 3.], "Period": [1, 1], "Season": [0, 0], "Month": ["Month 1", "Month 1"]})


def test_pack_size_rejects_nonpositive_and_invalid():
    for bad in ("", "invalid", "0", "-1", None, np.nan, np.inf, "500 g"):
        with pytest.raises(ValueError, match="finite positive litres"):
            _pack_size_to_factor(bad)
    assert _pack_size_to_factor("2.0") == 2.0


def test_decomposition_rejects_zero_volume_group(cells):
    scenario = simulate_scenario(cells, cells[["Category", "Brand", "SKU"]].assign(Elasticity_MAP=[0., 0.]), {})
    scenario["Equiv_Volume_Baseline"] = 0.
    scenario["Equiv_Volume_Simulated"] = 0.
    with pytest.raises(ValueError, match="positive total equivalized volume"):
        three_factor_decomposition(scenario, [])


def test_decomposition_rejects_mixed_units(cells):
    scenario = simulate_scenario(cells, cells[["Category", "Brand", "SKU"]].assign(Elasticity_MAP=[0., 0.]), {})
    scenario.loc[1, "Pack_Size"] = "1 piece"
    with pytest.raises(ValueError, match="finite positive litres"):
        three_factor_decomposition(scenario, [])


def test_grouped_decomposition_has_no_stray_columns_and_reconciles(cells):
    scenario = simulate_scenario(cells, cells[["Category", "Brand", "SKU"]].assign(Elasticity_MAP=[-1.5, -1.0]),
                                 {("C", "R", "B", "S1"): 0.8})
    result = three_factor_decomposition(scenario, ["Category"])
    assert not any(c.startswith("index") for c in result.columns)
    np.testing.assert_allclose(result["Decomposition_Sum"], result["Revenue_Delta"], atol=1e-6)


def test_named_scenario_is_single_target_and_discount(cells):
    target = cells.iloc[[0]].copy()
    target.loc[:, ["Category", "Retailer", "Brand", "SKU"]] = ["Carbonated Soft Drinks", "Nexus_Retail_Group", "ColaClassic", "Cola_2L_Bottle"]
    changes = build_named_scenario("nexus_cola_20pct", target)
    assert list(changes.values()) == [0.8]
    scenario = simulate_scenario(target, target[["Category", "Brand", "SKU"]].assign(Elasticity_MAP=[-1.5]), changes)
    assert scenario.loc[0, "Price_Change_Pct"] == pytest.approx(-20.)


def test_cli_writes_provenance_and_posterior_artifacts(cells, tmp_path, monkeypatch):
    import sim
    from retail_market_sim.model import MCMCDiagnostics
    estimates = cells[["Category", "Brand", "SKU"]].copy()
    estimates["Elasticity_MAP"] = [-1.5, -1.5]
    fitted = {"estimates": estimates, "converged": True, "message": "test", "iterations": 1, "weak_support_count": 0, "positive_elasticities": 0}
    target = {("C", "R", "B", "S1"): 0.8}
    
    # Mock diagnostics
    mock_diagnostics = MCMCDiagnostics(
        divergences=0,
        max_rhat=1.001,
        mean_rhat=1.0005,
        min_bulk_ess=1000,
        min_tail_ess=1000,
        max_rhat_params=[],
        min_ess_params=[],
        passed_gates=True,
        gate_criteria={"max_divergences": 0, "max_rhat": 1.01, "min_bulk_ess": 400, "min_tail_ess": 400},
    )
    
    monkeypatch.chdir(tmp_path)
    with patch("sys.argv", ["sim.py", "--scenario", "nexus_cola_20pct", "--sample"]), \
            patch.object(sim, "load_data", return_value=(cells, [])), \
            patch.object(sim, "fit_model_map", return_value=fitted), \
            patch.object(sim, "build_named_scenario", return_value=target), \
            patch.object(sim, "fit_model_mcmc", return_value=(None, cells[["Category", "Brand", "SKU"]], cells, [], [], None, None, None, None, None, mock_diagnostics)), \
            patch.object(sim, "simulate_scenario_posterior", return_value=(cells.assign(Simulated_Revenue_Mean=0.), np.array([100., 200.]))):
        sim.main()
    out = tmp_path / "outputs"
    # Find the run directory
    runs_dir = out / "runs"
    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    manifest = (run_dir / "run_manifest.json").read_text()
    assert "posterior_summary.csv" in manifest and "scenario_detail.csv" in manifest
    assert (run_dir / "posterior_summary.csv").is_file()


# =============================================================================
# Regression tests for P0-01: Zero-sales subgroup handling
# =============================================================================

@pytest.fixture
def cells_with_zero_retailer():
    """Baseline with a zero-sales retailer group."""
    return pd.DataFrame({
        "Category": ["Beverages"] * 6,
        "Retailer": ["Retailer_A", "Retailer_A", "Retailer_B", "Retailer_B", "Retailer_C", "Retailer_C"],
        "Brand": ["Brand_X"] * 6,
        "SKU": ["SKU_1", "SKU_2", "SKU_1", "SKU_2", "SKU_1", "SKU_2"],
        "Pack_Size": ["1.0", "0.5", "1.0", "0.5", "1.0", "0.5"],
        "Unit_Sales": [100.0, 200.0, 150.0, 250.0, 0.0, 0.0],
        "Sales_Revenue": [1000.0, 1000.0, 1500.0, 1250.0, 0.0, 0.0],
        "Price": [10.0, 5.0, 10.0, 5.0, 10.0, 5.0],
        "Period": [24] * 6,
        "Season": [11] * 6,
        "Month": ["Month 24"] * 6,
    })


def make_scenario_ready(baseline: pd.DataFrame) -> pd.DataFrame:
    """Add required columns for decomposition functions."""
    scenario = baseline.copy()
    scenario["Baseline_Unit_Sales"] = scenario["Unit_Sales"]
    scenario["Baseline_Price"] = scenario["Price"]
    scenario["Baseline_Revenue"] = scenario["Sales_Revenue"]
    scenario["Simulated_Price"] = scenario["Price"]
    scenario["Simulated_Unit_Sales"] = scenario["Unit_Sales"]
    scenario["Simulated_Revenue"] = scenario["Sales_Revenue"]
    scenario["Price_Change_Pct"] = 0.0
    scenario["Revenue_Delta"] = 0.0
    scenario["Elasticity_MAP"] = -1.5
    return scenario


def test_report_decomposition_handles_zero_volume_group(cells_with_zero_retailer):
    """Reporting wrapper should handle zero-volume groups without raising."""
    scenario = make_scenario_ready(cells_with_zero_retailer)
    
    # Market-level should work (total market has positive volume)
    market_decomp = three_factor_decomposition_report(scenario, [])
    assert len(market_decomp) == 1
    assert market_decomp.iloc[0]["_group_status"] == "positive"
    assert market_decomp.iloc[0]["_bridge_status"] == "available"
    
    # Retailer-level should work with zero-volume Retailer_C
    ret_decomp = three_factor_decomposition_report(scenario, ["Retailer"])
    assert len(ret_decomp) == 3
    
    # Check statuses
    statuses = ret_decomp.set_index("Retailer")["_group_status"]
    assert statuses["Retailer_A"] == "positive"
    assert statuses["Retailer_B"] == "positive"
    assert statuses["Retailer_C"] == "zero_to_zero"
    
    # Check bridge statuses
    bridges = ret_decomp.set_index("Retailer")["_bridge_status"]
    assert bridges["Retailer_A"] == "available"
    assert bridges["Retailer_B"] == "available"
    assert bridges["Retailer_C"] == "no sales"
    
    # Zero-to-zero group should have zero effects
    c_row = ret_decomp[ret_decomp["Retailer"] == "Retailer_C"].iloc[0]
    assert c_row["Volume_Effect"] == 0.0
    assert c_row["Mix_Effect"] == 0.0
    assert c_row["Price_Effect"] == 0.0
    assert c_row["Decomposition_Sum"] == 0.0
    assert c_row["Revenue_Delta"] == 0.0


def test_report_decomposition_handles_one_sided_zero():
    """Reporting wrapper should handle one-sided zero (baseline zero, scenario positive)."""
    baseline = pd.DataFrame({
        "Category": ["Beverages"] * 4,
        "Retailer": ["Retailer_A", "Retailer_A", "Retailer_B", "Retailer_B"],
        "Brand": ["Brand_X"] * 4,
        "SKU": ["SKU_1", "SKU_2", "SKU_1", "SKU_2"],
        "Pack_Size": ["1.0", "0.5", "1.0", "0.5"],
        "Unit_Sales": [100.0, 200.0, 0.0, 0.0],  # Retailer_B has zero baseline
        "Sales_Revenue": [1000.0, 1000.0, 0.0, 0.0],
        "Price": [10.0, 5.0, 10.0, 5.0],
        "Period": [24] * 4,
        "Season": [11] * 4,
        "Month": ["Month 24"] * 4,
    })
    scenario = make_scenario_ready(baseline)
    # Simulate a price change that creates demand for Retailer_B
    # Two SKUs, each with 500 revenue = 1000 total
    scenario.loc[scenario["Retailer"] == "Retailer_B", "Simulated_Unit_Sales"] = 50.0
    scenario.loc[scenario["Retailer"] == "Retailer_B", "Simulated_Revenue"] = 500.0
    
    ret_decomp = three_factor_decomposition_report(scenario, ["Retailer"])
    assert len(ret_decomp) == 2
    
    # Retailer_B should be one_sided_zero
    b_row = ret_decomp[ret_decomp["Retailer"] == "Retailer_B"].iloc[0]
    assert b_row["_group_status"] == "one_sided_zero"
    assert b_row["_bridge_status"] == "bridge unavailable"
    assert np.isnan(b_row["Volume_Effect"])
    assert np.isnan(b_row["Mix_Effect"])
    assert np.isnan(b_row["Price_Effect"])
    assert np.isnan(b_row["Decomposition_Sum"])
    # But revenue delta should be correct (two SKUs * 500 = 1000)
    assert b_row["Revenue_Delta"] == 1000.0


def test_value_shares_handles_zero_denominator():
    """value_shares should return NaN for zero denominator (all-zero market)."""
    baseline = pd.DataFrame({
        "Category": ["Beverages"] * 2,
        "Retailer": ["Retailer_A", "Retailer_A"],
        "Brand": ["Brand_X", "Brand_X"],
        "SKU": ["SKU_1", "SKU_2"],
        "Pack_Size": ["1.0", "0.5"],
        "Unit_Sales": [0.0, 0.0],
        "Sales_Revenue": [0.0, 0.0],
        "Price": [10.0, 5.0],
        "Period": [24] * 2,
        "Season": [11] * 2,
        "Month": ["Month 24"] * 2,
    })
    scenario = make_scenario_ready(baseline)
    
    shares = value_shares(scenario, ["Retailer"])
    assert shares["Baseline_Share_Pct"].isna().all()
    assert shares["Simulated_Share_Pct"].isna().all()
    assert shares["Share_Change_pp"].isna().all()


def test_value_shares_zero_numerator_positive_denominator():
    """value_shares should return 0% for zero numerator with positive denominator."""
    baseline = pd.DataFrame({
        "Category": ["Beverages"] * 4,
        "Retailer": ["Retailer_A", "Retailer_A", "Retailer_B", "Retailer_B"],
        "Brand": ["Brand_X"] * 4,
        "SKU": ["SKU_1", "SKU_2", "SKU_1", "SKU_2"],
        "Pack_Size": ["1.0", "0.5", "1.0", "0.5"],
        "Unit_Sales": [100.0, 200.0, 0.0, 0.0],
        "Sales_Revenue": [1000.0, 1000.0, 0.0, 0.0],
        "Price": [10.0, 5.0, 10.0, 5.0],
        "Period": [24] * 4,
        "Season": [11] * 4,
        "Month": ["Month 24"] * 4,
    })
    scenario = make_scenario_ready(baseline)
    
    shares = value_shares(scenario, ["Retailer"])
    # Retailer_A: 2000/2000 = 100%, Retailer_B: 0/2000 = 0%
    a_row = shares[shares["Retailer"] == "Retailer_A"].iloc[0]
    b_row = shares[shares["Retailer"] == "Retailer_B"].iloc[0]
    assert a_row["Baseline_Share_Pct"] == pytest.approx(100.0)
    assert b_row["Baseline_Share_Pct"] == pytest.approx(0.0)
    assert not np.isnan(b_row["Baseline_Share_Pct"])  # 0% is valid, not NaN


def test_failed_sampling_run_cleans_up_and_leaves_prior_results(cells, tmp_path, monkeypatch):
    """A failed sampling run should not create a new run directory and not affect prior runs."""
    import sim
    estimates = cells[["Category", "Brand", "SKU"]].copy()
    estimates["Elasticity_MAP"] = [-1.5, -1.5]
    fitted = {"estimates": estimates, "converged": True, "message": "test", "iterations": 1, "weak_support_count": 0, "positive_elasticities": 0}
    target = {("C", "R", "B", "S1"): 0.8}
    monkeypatch.chdir(tmp_path)
    
    # First run: successful MAP only
    with patch("sys.argv", ["sim.py", "--scenario", "nexus_cola_20pct"]), \
            patch.object(sim, "load_data", return_value=(cells, [])), \
            patch.object(sim, "fit_model_map", return_value=fitted), \
            patch.object(sim, "build_named_scenario", return_value=target):
        sim.main()
    
    out = tmp_path / "outputs"
    runs_dir = out / "runs"
    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1
    first_run_dir = run_dirs[0]
    assert (first_run_dir / "scenario_detail.csv").is_file()
    assert (first_run_dir / "run_manifest.json").is_file()
    
    # Second run: sampling fails (fit_model_mcmc raises RuntimeError)
    # CLI should exit with non-zero status and clean up the failed run directory
    with patch("sys.argv", ["sim.py", "--scenario", "nexus_cola_20pct", "--sample"]), \
            patch.object(sim, "load_data", return_value=(cells, [])), \
            patch.object(sim, "fit_model_map", return_value=fitted), \
            patch.object(sim, "build_named_scenario", return_value=target), \
            patch.object(sim, "fit_model_mcmc", side_effect=RuntimeError("MCMC failed")):
        with pytest.raises(SystemExit) as exc_info:
            sim.main()
        assert exc_info.value.code == 1
    
    # First run directory should still exist and be intact
    assert first_run_dir.exists()
    assert (first_run_dir / "scenario_detail.csv").is_file()
    assert (first_run_dir / "run_manifest.json").is_file()
    
    # No new run directory should have been created (failed run cleaned up)
    run_dirs_after = list(runs_dir.iterdir())
    assert len(run_dirs_after) == 1
    assert run_dirs_after[0] == first_run_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
