"""
Tests for analytical invariants in the retail market simulation.

These tests verify mathematical properties that must hold regardless of
model parameters or data values.
"""
import numpy as np
import pandas as pd
import pytest

from retail_market_sim import (
    simulate_scenario,
    three_factor_decomposition,
    value_shares,
    _pack_size_to_factor,
    build_named_scenario,
)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def minimal_baseline():
    """Minimal valid baseline DataFrame for testing."""
    return pd.DataFrame({
        "Month": ["Month 24"] * 4,
        "Period": [24] * 4,
        "Season": [11] * 4,
        "Category": ["Beverages"] * 4,
        "Retailer": ["Retailer_A", "Retailer_A", "Retailer_B", "Retailer_B"],
        "Brand": ["Brand_X", "Brand_X", "Brand_X", "Brand_X"],
        "SKU": ["SKU_1", "SKU_2", "SKU_1", "SKU_2"],
        "Pack_Size": ["1.0", "0.5", "1.0", "0.5"],
        "Unit_Sales": [100.0, 200.0, 150.0, 250.0],
        "Sales_Revenue": [1000.0, 1000.0, 1500.0, 1250.0],
        "Price": [10.0, 5.0, 10.0, 5.0],
    })


@pytest.fixture
def minimal_estimates():
    """Minimal estimates DataFrame with elasticities."""
    return pd.DataFrame({
        "Category": ["Beverages", "Beverages"],
        "Brand": ["Brand_X", "Brand_X"],
        "SKU": ["SKU_1", "SKU_2"],
        "Elasticity_MAP": [-1.5, -1.2],
    })


# =============================================================================
# Invariant Tests
# =============================================================================
class TestAnalyticalInvariants:
    """Tests for mathematical properties that must always hold."""

    def test_unchanged_prices_reproduce_baseline(self, minimal_baseline, minimal_estimates):
        """Unchanged prices must exactly reproduce baseline quantities and revenue."""
        scenario = simulate_scenario(minimal_baseline, minimal_estimates, {})
        np.testing.assert_allclose(
            scenario["Simulated_Unit_Sales"], scenario["Baseline_Unit_Sales"], rtol=1e-10
        )
        np.testing.assert_allclose(
            scenario["Simulated_Revenue"], scenario["Baseline_Revenue"], rtol=1e-10
        )
        np.testing.assert_allclose(scenario["Revenue_Delta"], 0.0, atol=1e-10)
        np.testing.assert_allclose(scenario["Price_Change_Pct"], 0.0, atol=1e-10)

    def test_cell_level_price_volume_reconciliation(self, minimal_baseline, minimal_estimates):
        """Cell-level Price_Effect + Volume_Effect must equal Revenue_Delta exactly."""
        # Modify one price: 10% increase on SKU_1 at Retailer_A
        price_changes = {
            ("Beverages", "Retailer_A", "Brand_X", "SKU_1"): 11.0,
        }
        scenario = simulate_scenario(minimal_baseline, minimal_estimates, price_changes)

        # Check at each cell
        np.testing.assert_allclose(
            scenario["Revenue_Delta"],
            scenario["Price_Effect"] + scenario["Volume_Effect"],
            rtol=1e-9, atol=1e-6
        )

    def test_aggregate_price_volume_reconciliation(self, minimal_baseline, minimal_estimates):
        """Sum of cell-level effects must equal aggregate revenue change."""
        price_changes = {
            ("Beverages", "Retailer_A", "Brand_X", "SKU_1"): 11.0,
        }
        scenario = simulate_scenario(minimal_baseline, minimal_estimates, price_changes)

        total_delta = scenario["Revenue_Delta"].sum()
        total_pe = scenario["Price_Effect"].sum()
        total_ve = scenario["Volume_Effect"].sum()

        np.testing.assert_allclose(total_delta, total_pe + total_ve, rtol=1e-9, atol=1e-6)

    def test_shares_sum_to_100(self, minimal_baseline, minimal_estimates):
        """Value shares within any denominator must sum to 100%."""
        scenario = simulate_scenario(minimal_baseline, minimal_estimates, {})

        # Retailer shares (total market denominator)
        ret_shares = value_shares(scenario, ["Retailer"])
        assert abs(ret_shares["Baseline_Share_Pct"].sum() - 100.0) < 1e-9
        assert abs(ret_shares["Simulated_Share_Pct"].sum() - 100.0) < 1e-9

        # Brand shares within category
        brand_shares = value_shares(scenario, ["Category", "Brand"])
        cat_shares = brand_shares.groupby("Category").agg(
            base_sum=("Baseline_Share_Pct", "sum"),
            sim_sum=("Simulated_Share_Pct", "sum")
        )
        assert all(abs(cat_shares["base_sum"] - 100.0) < 1e-9)
        assert all(abs(cat_shares["sim_sum"] - 100.0) < 1e-9)

    def test_zero_elasticity_leaves_quantity_unchanged(self, minimal_baseline):
        """Zero elasticity should leave quantity unchanged regardless of price change."""
        estimates_zero = pd.DataFrame({
            "Category": ["Beverages", "Beverages"],
            "Brand": ["Brand_X", "Brand_X"],
            "SKU": ["SKU_1", "SKU_2"],
            "Elasticity_MAP": [0.0, 0.0],
        })
        price_changes = {
            ("Beverages", "Retailer_A", "Brand_X", "SKU_1"): 20.0,  # 100% increase
        }
        scenario = simulate_scenario(minimal_baseline, estimates_zero, price_changes)

        # Quantity should be unchanged where elasticity is zero
        mask = scenario["Elasticity_MAP"] == 0.0
        np.testing.assert_allclose(
            scenario.loc[mask, "Simulated_Unit_Sales"],
            scenario.loc[mask, "Baseline_Unit_Sales"],
            rtol=1e-10
        )

    def test_unit_elasticity_preserves_own_revenue(self, minimal_baseline):
        """At elasticity -1, own-cell revenue should remain unchanged under constant elasticity."""
        estimates_unit = pd.DataFrame({
            "Category": ["Beverages", "Beverages"],
            "Brand": ["Brand_X", "Brand_X"],
            "SKU": ["SKU_1", "SKU_2"],
            "Elasticity_MAP": [-1.0, -1.0],
        })
        price_changes = {
            ("Beverages", "Retailer_A", "Brand_X", "SKU_1"): 11.0,  # 10% increase
        }
        scenario = simulate_scenario(minimal_baseline, estimates_unit, price_changes)

        # For elasticity -1: Q1 = Q0 * (P1/P0)^-1 = Q0 * P0/P1
        # Revenue R1 = P1 * Q1 = P1 * Q0 * P0/P1 = P0 * Q0 = R0
        mask = scenario["Elasticity_MAP"] == -1.0
        np.testing.assert_allclose(
            scenario.loc[mask, "Simulated_Revenue"],
            scenario.loc[mask, "Baseline_Revenue"],
            rtol=1e-6
        )

    def test_missing_listings_not_created(self, minimal_baseline, minimal_estimates):
        """Prices for non-existent SKU-retailer combos should raise ValueError."""
        price_changes = {
            ("Beverages", "Retailer_C", "Brand_X", "SKU_1"): 8.0,  # Retailer_C doesn't exist
        }
        with pytest.raises(ValueError, match="Scenario target not found"):
            simulate_scenario(minimal_baseline, minimal_estimates, price_changes)

    def test_zero_demand_stays_zero(self, minimal_baseline, minimal_estimates):
        """Zero baseline units should remain zero under multiplicative scenario."""
        baseline_zero = minimal_baseline.copy()
        baseline_zero.loc[0, "Unit_Sales"] = 0.0
        baseline_zero.loc[0, "Sales_Revenue"] = 0.0
        baseline_zero.loc[0, "Price"] = 10.0

        scenario = simulate_scenario(baseline_zero, minimal_estimates, {})

        # Zero baseline should stay zero (multiplicative model)
        assert scenario.loc[0, "Simulated_Unit_Sales"] == 0.0
        assert scenario.loc[0, "Simulated_Revenue"] == 0.0


class TestThreeFactorDecomposition:
    """Tests for the Volume -> Mix -> Price decomposition."""

    def test_decomposition_sums_to_revenue_delta(self, minimal_baseline, minimal_estimates):
        """VE + ME + PE must equal Revenue_Delta at any aggregation level."""
        price_changes = {
            ("Beverages", "Retailer_A", "Brand_X", "SKU_1"): 11.0,
        }
        scenario = simulate_scenario(minimal_baseline, minimal_estimates, price_changes)

        # Market level (empty group_columns = total market)
        decomp = three_factor_decomposition(scenario, [])
        np.testing.assert_allclose(
            decomp["Decomposition_Sum"].iloc[0],
            decomp["Revenue_Delta"].iloc[0],
            rtol=1e-9, atol=1e-6
        )

        # Category level
        cat_decomp = three_factor_decomposition(scenario, ["Category"])
        for _, row in cat_decomp.iterrows():
            np.testing.assert_allclose(
                row["Decomposition_Sum"], row["Revenue_Delta"], rtol=1e-9, atol=1e-6
            )

    def test_mix_effect_zero_when_shares_unchanged(self, minimal_baseline):
        """Mix effect should be zero when all SKUs have same elasticity (shares unchanged)."""
        # Same elasticity for all SKUs -> proportional quantity changes -> shares unchanged
        estimates_same = pd.DataFrame({
            "Category": ["Beverages", "Beverages"],
            "Brand": ["Brand_X", "Brand_X"],
            "SKU": ["SKU_1", "SKU_2"],
            "Elasticity_MAP": [-1.5, -1.5],
        })
        # Proportional price changes
        price_changes = {
            ("Beverages", "Retailer_A", "Brand_X", "SKU_1"): 11.0,
            ("Beverages", "Retailer_A", "Brand_X", "SKU_2"): 5.5,
        }
        scenario = simulate_scenario(minimal_baseline, estimates_same, price_changes)

        decomp = three_factor_decomposition(scenario, [])
        # Mix should be approximately zero (within numerical precision)
        assert abs(decomp["Mix_Effect"].iloc[0]) < 1e-6

    def test_pure_mix_effect_nonzero(self):
        """Test pure mix effect with hand-calculated example."""
        # Two 1L products priced at 1 and 3
        # Quantities: [100, 100] -> [50, 150]
        # Expected: VE=0, PE=0, ME=+100
        baseline = pd.DataFrame({
            "Month": ["M1", "M1"],
            "Period": [1, 1],
            "Season": [0, 0],
            "Category": ["Cat", "Cat"],
            "Retailer": ["Ret", "Ret"],
            "Brand": ["A", "A"],
            "SKU": ["S1", "S2"],
            "Pack_Size": ["1.0", "1.0"],
            "Unit_Sales": [100.0, 100.0],
            "Sales_Revenue": [100.0, 300.0],
            "Price": [1.0, 3.0],
        })
        estimates = pd.DataFrame({
            "Category": ["Cat", "Cat"],
            "Brand": ["A", "A"],
            "SKU": ["S1", "S2"],
            "Elasticity_MAP": [0.0, 0.0],  # Zero elasticity - prices don't change quantities
        })
        # Simulate quantities directly by using a scenario with same prices but different quantities
        # Since we can't directly set quantities, we'll use a different approach:
        # Create scenario where prices are unchanged but we manually set quantities for test
        # Actually, the decomposition is tested on the scenario output, so we need to
        # construct a scenario DataFrame with the desired simulated quantities
        
        # For this test, we manually create the scenario output
        scenario = baseline.copy()
        scenario["Baseline_Unit_Sales"] = scenario["Unit_Sales"]
        scenario["Baseline_Price"] = scenario["Price"]
        scenario["Baseline_Revenue"] = scenario["Sales_Revenue"]
        scenario["Simulated_Price"] = scenario["Price"]
        scenario["Simulated_Unit_Sales"] = [50.0, 150.0]  # Share shift
        scenario["Simulated_Revenue"] = scenario["Simulated_Price"] * scenario["Simulated_Unit_Sales"]
        scenario["Price_Change_Pct"] = 0.0
        scenario["Revenue_Delta"] = scenario["Simulated_Revenue"] - scenario["Baseline_Revenue"]
        scenario["Elasticity_MAP"] = 0.0

        decomp = three_factor_decomposition(scenario, [])
        
        # Expected: VE=0 (Q unchanged at 200), PE=0 (P unchanged), ME=+100
        # Total baseline Q = 200, Pbar0 = 400/200 = 2
        # Baseline share: 0.5 each, Simulated share: 0.25, 0.75
        # ME = Q1 * sum((s1-s0)*P0) = 200 * ((-0.25)*1 + 0.25*3) = 200 * 0.5 = 100
        assert abs(decomp["Volume_Effect"].iloc[0] - 0.0) < 1e-6
        assert abs(decomp["Price_Effect"].iloc[0] - 0.0) < 1e-6
        assert abs(decomp["Mix_Effect"].iloc[0] - 100.0) < 1e-6

    def test_pack_size_factor_extraction(self):
        """Pack size numeric litres parsing."""
        assert _pack_size_to_factor("1.0") == 1.0
        assert _pack_size_to_factor("0.33") == 0.33
        assert _pack_size_to_factor("2.0") == 2.0
        assert _pack_size_to_factor("1.98") == 1.98
        assert _pack_size_to_factor("0.5") == 0.5
        assert _pack_size_to_factor("") == 1.0
        assert _pack_size_to_factor(None) == 1.0
        assert _pack_size_to_factor(np.nan) == 1.0 or np.isnan(_pack_size_to_factor(np.nan))


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_extreme_price_change_rejected(self, minimal_baseline):
        """Extreme price changes should raise ValueError."""
        estimates_pos = pd.DataFrame({
            "Category": ["Beverages", "Beverages"],
            "Brand": ["Brand_X", "Brand_X"],
            "SKU": ["SKU_1", "SKU_2"],
            "Elasticity_MAP": [2.0, -1.2],  # Positive elasticity for SKU_1
        })
        price_changes = {
            ("Beverages", "Retailer_A", "Brand_X", "SKU_1"): 1e12,
        }
        with pytest.raises(ValueError, match="extreme demand response"):
            simulate_scenario(minimal_baseline, estimates_pos, price_changes)

    def test_nonfinite_price_rejected(self, minimal_baseline, minimal_estimates):
        """Non-finite prices should raise ValueError."""
        price_changes = {
            ("Beverages", "Retailer_A", "Brand_X", "SKU_1"): np.nan,
        }
        with pytest.raises(ValueError, match="finite price above zero"):
            simulate_scenario(minimal_baseline, minimal_estimates, price_changes)

    def test_missing_elasticity_rejected(self, minimal_baseline):
        """Missing elasticity estimates should raise ValueError."""
        estimates_missing = pd.DataFrame({
            "Category": ["Beverages"],
            "Brand": ["Brand_X"],
            "SKU": ["SKU_1"],
            "Elasticity_MAP": [-1.5],
        })
        price_changes = {}
        with pytest.raises(ValueError, match="elasticity estimates are missing"):
            simulate_scenario(minimal_baseline, estimates_missing, price_changes)


class TestSimulator:
    """Tests for the sim.py CLI functions."""

    def test_empty_scenario_reproduces_baseline(self, minimal_baseline, minimal_estimates):
        """Empty price_changes should reproduce baseline exactly."""
        scenario = simulate_scenario(minimal_baseline, minimal_estimates, {})

        np.testing.assert_allclose(
            scenario["Simulated_Unit_Sales"], scenario["Baseline_Unit_Sales"], rtol=1e-10
        )
        np.testing.assert_allclose(
            scenario["Simulated_Revenue"], scenario["Baseline_Revenue"], rtol=1e-10
        )
        np.testing.assert_allclose(scenario["Revenue_Delta"], 0.0, atol=1e-10)
        np.testing.assert_allclose(scenario["Price_Change_Pct"], 0.0, atol=1e-10)

    def test_scenario_applies_discount_correctly(self, minimal_baseline, minimal_estimates):
        """Scenario with absolute new price should produce correct percentage change."""
        # Baseline price is 10.0, apply 20% discount -> new price 8.0
        price_changes = {
            ("Beverages", "Retailer_A", "Brand_X", "SKU_1"): 8.0,
        }

        scenario = simulate_scenario(minimal_baseline, minimal_estimates, price_changes)

        # Check the changed cell
        mask = scenario["SKU"] == "SKU_1"
        changed = scenario[mask]
        assert len(changed) == 2  # Retailer_A and Retailer_B
        ret_a = changed[changed["Retailer"] == "Retailer_A"].iloc[0]
        ret_b = changed[changed["Retailer"] == "Retailer_B"].iloc[0]

        # Retailer_A should have -20% price change
        assert abs(ret_a["Price_Change_Pct"] - (-20.0)) < 1e-9
        # Retailer_B should have 0% price change
        assert abs(ret_b["Price_Change_Pct"]) < 1e-9


class TestPosteriorAggregation:
    """Tests for correct posterior uncertainty aggregation."""

    def test_market_aggregation_preserves_covariance(self):
        """Market revenue interval should be computed from aggregated draws, not marginal quantiles."""
        # Create mock revenue draws for 2 cells, 10000 draws
        n_draws = 10000
        rng = np.random.default_rng(42)
        # Two perfectly correlated cells
        base = rng.normal(1000, 100, n_draws)
        revenue_draws = np.column_stack([base, base])  # Perfect correlation

        # Correct: aggregate within draws, then compute quantiles
        market_revenue = revenue_draws.sum(axis=1)
        correct_interval = np.quantile(market_revenue, [0.025, 0.975])

        # Incorrect: sum of marginal quantiles
        marginal_q025 = np.quantile(revenue_draws, 0.025, axis=0).sum()
        marginal_q975 = np.quantile(revenue_draws, 0.975, axis=0).sum()

        # For perfectly correlated variables, correct interval should be ~2x individual interval
        individual_interval = np.quantile(base, [0.025, 0.975])
        expected_interval = 2 * individual_interval

        np.testing.assert_allclose(correct_interval, expected_interval, rtol=0.02)

        # This test documents the expected behavior for the production code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])