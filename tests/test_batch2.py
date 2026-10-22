"""Tests for strict units, decomposition edge cases, and CLI provenance."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from retail_market_sim.decomposition import _pack_size_to_factor, three_factor_decomposition
from retail_market_sim.simulation import build_named_scenario, simulate_scenario


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
    estimates = cells[["Category", "Brand", "SKU"]].copy()
    estimates["Elasticity_MAP"] = [-1.5, -1.5]
    fitted = {"estimates": estimates, "converged": True, "message": "test", "iterations": 1, "weak_support_count": 0, "positive_elasticities": 0}
    target = {("C", "R", "B", "S1"): 0.8}
    monkeypatch.chdir(tmp_path)
    with patch("sys.argv", ["sim.py", "--scenario", "nexus_cola_20pct", "--sample"]), \
            patch.object(sim, "load_data", return_value=(cells, [])), \
            patch.object(sim, "fit_model_map", return_value=fitted), \
            patch.object(sim, "build_named_scenario", return_value=target), \
            patch.object(sim, "fit_model_mcmc", return_value=(None, cells[["Category", "Brand", "SKU"]], cells, [], [], None, None, None, None, None)), \
            patch.object(sim, "simulate_scenario_posterior", return_value=(cells.assign(Simulated_Revenue_Mean=0.), np.array([100., 200.]))):
        sim.main()
    out = tmp_path / "outputs"
    manifest = (out / "run_manifest.json").read_text()
    assert "posterior_summary.csv" in manifest and "scenario_detail.csv" in manifest
    assert (out / "posterior_summary.csv").is_file()
