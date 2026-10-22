"""Tests for the dashboard chart builders (Plotly only; no Streamlit needed)."""
import numpy as np
import pandas as pd
import pytest

from retail_market_sim import charts


@pytest.fixture
def scenario():
    return pd.DataFrame({
        "Category": ["Cola", "Cola", "Juice", "Juice"],
        "Retailer": ["A", "B", "A", "B"],
        "Brand": ["X", "X", "Y", "Y"],
        "SKU": ["S1", "S1", "S2", "S2"],
        "Baseline_Revenue": [1000.0, 500.0, 300.0, 200.0],
        "Simulated_Revenue": [1300.0, 500.0, 270.0, 200.0],
        "Revenue_Delta": [300.0, 0.0, -30.0, 0.0],
    })


def test_price_response_matches_model_and_marks_points():
    fig = charts.price_response_curve(10.0, 100.0, -2.0, 8.0, observed_min=9.0, observed_max=11.0)
    prices, revenue, units = np.array(fig.data[0].x), np.array(fig.data[0].y), np.array(fig.data[1].y)
    at_baseline = np.argmin(np.abs(prices - 10.0))
    assert revenue[at_baseline] == pytest.approx(1000.0, rel=1e-9)
    assert units[at_baseline] == pytest.approx(100.0, rel=1e-9)
    scenario_trace = next(t for t in fig.data if t.name == "Scenario" and t.showlegend is not False)
    assert scenario_trace.x[0] == 8.0
    assert scenario_trace.y[0] == pytest.approx(8.0 * 100.0 * (0.8 ** -2.0), rel=1e-9)
    assert len(fig.layout.shapes) == 2


def test_price_response_unit_elasticity_is_flat_and_no_band_message():
    fig = charts.price_response_curve(10.0, 100.0, -1.0, 9.0)
    assert np.ptp(np.array(fig.data[0].y)) < 1e-6
    assert len(fig.layout.shapes) == 0
    assert any("No historical price range" in a.text for a in fig.layout.annotations)


def test_price_response_handles_extreme_proposals_without_overflow():
    fig = charts.price_response_curve(10.0, 100.0, -8.0, 1e6)
    assert np.isfinite(np.array(fig.data[0].y)).all()


@pytest.mark.parametrize("args", [(0, 100, -1, 8), (10, -1, -1, 8), (10, 100, np.nan, 8), (10, 100, -1, 0)])
def test_price_response_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        charts.price_response_curve(*args)


def test_revenue_by_group_totals_and_order(scenario):
    fig = charts.revenue_by_group_chart(scenario, "Category")
    assert list(fig.data[0].x) == ["Cola", "Juice"]
    assert list(fig.data[0].y) == [1500.0, 500.0]
    assert list(fig.data[1].y) == [1800.0, 470.0]
    assert fig.data[0].customdata[0][0] == pytest.approx(300.0)
    assert fig.data[0].customdata[0][1] == pytest.approx(20.0)


def test_movers_only_include_changed_cells_sorted_by_size(scenario):
    fig = charts.revenue_movers_chart(scenario)
    assert list(fig.data[0].x) == [-30.0, 300.0]
    assert list(fig.data[0].marker.color) == [charts.DECREASE, charts.INCREASE]
    assert len(charts.revenue_movers_chart(scenario, top_n=1).data[0].x) == 1


def test_movers_empty_state(scenario):
    unchanged = scenario.assign(Revenue_Delta=0.0)
    fig = charts.revenue_movers_chart(unchanged)
    assert len(fig.data) == 0 and "No revenue change" in fig.layout.annotations[0].text


def test_elasticity_chart_separates_weak_support_and_marks_unit_elasticity():
    estimates = pd.DataFrame({"Category": ["Cola", "Juice", "Water"], "SKU": ["S1", "S2", "S3"],
                              "Elasticity_MAP": [-3.0, -1.2, -0.4], "Weak_Price_Support": [True, False, False]})
    fig = charts.elasticity_chart(estimates)
    assert sorted(t.name for t in fig.data) == ["Adequate price variation", "Weak price support"]
    assert len(fig.layout.shapes) == 1
    without_flag = charts.elasticity_chart(estimates.drop(columns="Weak_Price_Support"))
    assert len(without_flag.data) == 1 and len(without_flag.data[0].x) == 3


def test_share_change_chart_and_empty_state():
    shares = pd.DataFrame({"Retailer": ["A", "B"], "Share_Change_pp": [1.5, -1.5]})
    fig = charts.share_change_chart(shares, ["Retailer"])
    assert list(fig.data[0].y) == ["B", "A"] and list(fig.data[0].x) == [-1.5, 1.5]
    flat = charts.share_change_chart(shares.assign(Share_Change_pp=0.0), ["Retailer"])
    assert len(flat.data) == 0


def test_revenue_bridge_totals_and_truncated_axis():
    effects = {"Volume_Effect": 26894.0, "Mix_Effect": -6105.0, "Price_Effect": -8379.0}
    fig = charts.revenue_bridge_chart(547323.0, effects)
    assert fig.data[0].y[-1] == pytest.approx(547323.0 + 26894.0 - 6105.0 - 8379.0)
    low, high = fig.layout.yaxis.range
    assert low > 400000 and high > 559734.0 and low < 547323.0


def test_every_figure_serialises(scenario):
    figures = [
        charts.price_response_curve(10.0, 100.0, -2.0, 8.0, 9.0, 11.0),
        charts.revenue_by_group_chart(scenario, "Retailer"),
        charts.revenue_movers_chart(scenario),
        charts.revenue_bridge_chart(1000.0, {"Volume_Effect": 1.0, "Mix_Effect": 0.0, "Price_Effect": -1.0}),
    ]
    for fig in figures:
        assert fig.to_json()
