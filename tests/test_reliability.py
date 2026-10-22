"""Regression tests of production adapters, joint draws, and dashboard wiring."""
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from retail_market_sim.dashboard_adapter import matrix_to_price_changes
from retail_market_sim.simulation import (
    build_named_scenario, simulate_scenario_posterior, summarize_revenue_draws,
)


@pytest.fixture
def cells():
    return pd.DataFrame({
        "Category": ["C", "C"], "Retailer": ["R", "R"], "Brand": ["B", "B"],
        "SKU": ["S1", "S2"], "Price": [10., 5.], "Unit_Sales": [100., 100.],
        "Sales_Revenue": [1000., 500.], "Pack_Size": ["1", "1"],
        "Period": [1, 1], "Season": [0, 0], "Month": ["Month 1", "Month 1"],
    })


def test_matrix_adapter_baseline_and_edit(cells):
    matrix = cells[["Category", "Brand", "SKU"]].copy()
    matrix["Price | R"] = [10., 5.]
    matrix["Price | Absent"] = [99., 99.]
    mapping = {"Price | R": "R", "Price | Absent": "Absent"}
    assert matrix_to_price_changes(cells, matrix, mapping) == {}
    matrix.loc[0, "Price | R"] = 8.
    assert matrix_to_price_changes(cells, matrix, mapping) == {("C", "R", "B", "S1"): 8.}
    matrix.loc[0, "Price | R"] = np.nan
    with pytest.raises(ValueError, match="finite price above zero"):
        matrix_to_price_changes(cells, matrix, mapping)


def test_named_builder_uses_baseline_price(cells):
    target = cells.iloc[[0]].copy()
    target.loc[:, ["Category", "Retailer", "Brand", "SKU"]] = [
        "Carbonated Soft Drinks", "Nexus_Retail_Group", "ColaClassic", "Cola_2L_Bottle",
    ]
    assert list(build_named_scenario("nexus_cola_20pct", target).values()) == [8.]


def test_production_aggregation_preserves_covariance():
    draws = np.array([[100., 100.], [200., 200.], [300., 300.]])
    assert summarize_revenue_draws(draws)["std"] == pytest.approx(2 * draws[:, 0].std(ddof=1))
    offsetting = np.array([[100., 200.], [200., 100.]])
    summary = summarize_revenue_draws(offsetting)
    assert summary["q025"] == summary["q975"] == 300.
    assert np.quantile(offsetting, 0.025, axis=0).sum() != summary["q025"]


def test_actual_posterior_path_and_guards(cells):
    products = cells[["Category", "Brand", "SKU"]].copy()
    beta = xr.DataArray(np.array([[[-1., -1.], [-2., -2.], [-3., -3.]]]), dims=("chain", "draw", "product"))
    trace = SimpleNamespace(posterior={"beta": beta})
    args = (trace, cells, products, cells, None, None, None, None, None)
    _, totals = simulate_scenario_posterior(*args, {}, n_draws=3)
    np.testing.assert_allclose(totals, 1500.)
    _, totals = simulate_scenario_posterior(*args, {("C", "R", "B", "S1"): 5.}, n_draws=3)
    np.testing.assert_allclose(totals, [1500., 2500., 4500.])
    for bad in (0, 1, -2, 1.5, True):
        with pytest.raises(ValueError, match="n_draws"):
            simulate_scenario_posterior(*args, {}, n_draws=bad)
    with pytest.raises(ValueError, match="finite price above zero"):
        simulate_scenario_posterior(*args, {("C", "R", "B", "S1"): 0.}, n_draws=3)
    with pytest.raises(ValueError, match="not found"):
        simulate_scenario_posterior(trace, cells, products.iloc[[0]], cells, None, None, None, None, None, {}, n_draws=3)


def test_dashboard_smoke_and_fit_cache(cells):
    from streamlit.testing.v1 import AppTest
    from retail_market_sim import dashboard
    estimates = cells[["Category", "Brand", "SKU"]].copy()
    estimates["Elasticity_MAP"] = [-1.5, -1.5]
    fitted = {"estimates": estimates, "converged": True, "message": "test fit", "iterations": 1,
              "weak_support_count": 0, "positive_elasticities": 0}
    dashboard.cached_fit.clear()
    entry = Path(__file__).resolve().parents[1] / "app.py"
    with patch.object(dashboard, "cached_load", return_value=(cells, [])), patch.object(
        dashboard, "fit_model_map", return_value=fitted,
    ) as fit:
        app = AppTest.from_file(str(entry)).run(timeout=20)
        assert not app.exception
        assert not app.error
        app.button[0].click().run(timeout=20)
        assert not app.exception
        assert not app.error
        assert fit.call_count == 1
    dashboard.cached_fit.clear()


def test_cli_entrypoint_writes_artifacts(cells, tmp_path, monkeypatch):
    import sim
    estimates = cells[["Category", "Brand", "SKU"]].copy()
    estimates["Elasticity_MAP"] = [-1.5, -1.5]
    fitted = {"estimates": estimates, "converged": True, "iterations": 1,
              "weak_support_count": 0, "positive_elasticities": 0}
    monkeypatch.chdir(tmp_path)
    with patch("sys.argv", ["sim.py", "--scenario", "custom"]), patch.object(
        sim, "load_data", return_value=(cells, []),
    ), patch.object(sim, "fit_model_map", return_value=fitted):
        sim.main()
    assert (tmp_path / "outputs" / "scenario_detail.csv").is_file()
