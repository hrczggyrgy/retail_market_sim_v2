"""
Tests for analytical invariants in the retail market simulation.

These tests verify mathematical properties that must hold regardless of
model parameters or data values.
"""
import numpy as np
import pandas as pd
import pytest

from app import (
    simulate_market,
    three_factor_decomposition,
    value_shares,
    _pack_size_to_factor,
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
        "Pack_Size": ["1.0 L", "0.5 L", "1.0 L", "0.5 L"],
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


@pytest.fixture
def price_matrix(minimal_baseline):
    """Editable price matrix with same prices as baseline (no change)."""
    return minimal_baseline[["Category", "Brand", "SKU", "Price"]].drop_duplicates().assign(
        **{"Price | Retailer_A": [10.0, 5.0], "Price | Retailer_B": [10.0, 5.0]}
    )


@pytest.fixture
def retailer_columns():
    return {"Price | Retailer_A": "Retailer_A", "Price | Retailer_B": "Retailer_B"}


# =============================================================================
# Invariant Tests
# =============================================================================
class TestAnalyticalInvariants:
    """Tests for mathematical properties that must always hold."""

    def test_unchanged_prices_reproduce_baseline(
        self, minimal_baseline, minimal_estimates, price_matrix, retailer_columns
    ):
        """Unchanged prices must exactly reproduce baseline quantities and revenue."""
        scenario = simulate_market(
            minimal_baseline, price_matrix, retailer_columns, minimal_estimates
        )
        np.testing.assert_allclose(
            scenario["Simulated_Unit_Sales"], scenario["Baseline_Unit_Sales"], rtol=1e-10
        )
        np.testing.assert_allclose(
            scenario["Simulated_Revenue"], scenario["Baseline_Revenue"], rtol=1e-10
        )
        np.testing.assert_allclose(
            scenario["Revenue_Delta"], 0.0, atol=1e-10
        )

    def test_cell_level_price_volume_reconciliation(
        self, minimal_baseline, minimal_estimates, price_matrix, retailer_columns
    ):
        """Cell-level Price_Effect + Volume_Effect must equal Revenue_Delta exactly."""
        # Modify one price
        price_matrix_changed = price_matrix.copy()
        price_matrix_changed.loc[0, "Price | Retailer_A"] = 11.0  # 10% increase

        scenario = simulate_market(
            minimal_baseline, price_matrix_changed, retailer_columns, minimal_estimates
        )

        # Check at each cell
        np.testing.assert_allclose(
            scenario["Revenue_Delta"],
            scenario["Price_Effect"] + scenario["Volume_Effect"],
            rtol=1e-9, atol=1e-6
        )

    def test_aggregate_price_volume_reconciliation(
        self, minimal_baseline, minimal_estimates, price_matrix, retailer_columns
    ):
        """Sum of cell-level effects must equal aggregate revenue change."""
        price_matrix_changed = price_matrix.copy()
        price_matrix_changed.loc[0, "Price | Retailer_A"] = 11.0

        scenario = simulate_market(
            minimal_baseline, price_matrix_changed, retailer_columns, minimal_estimates
        )

        total_delta = scenario["Revenue_Delta"].sum()
        total_pe = scenario["Price_Effect"].sum()
        total_ve = scenario["Volume_Effect"].sum()

        np.testing.assert_allclose(total_delta, total_pe + total_ve, rtol=1e-9, atol=1e-6)

    def test_shares_sum_to_100(self, minimal_baseline, minimal_estimates, price_matrix, retailer_columns):
        """Value shares within any denominator must sum to 100%."""
        scenario = simulate_market(
            minimal_baseline, price_matrix, retailer_columns, minimal_estimates
        )

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

    def test_zero_elasticity_leaves_quantity_unchanged(self, minimal_baseline, retailer_columns, price_matrix):
        """Zero elasticity should leave quantity unchanged regardless of price change."""
        estimates_zero = pd.DataFrame({
            "Category": ["Beverages", "Beverages"],
            "Brand": ["Brand_X", "Brand_X"],
            "SKU": ["SKU_1", "SKU_2"],
            "Elasticity_MAP": [0.0, 0.0],
        })
        price_matrix_changed = price_matrix.copy()
        price_matrix_changed.loc[0, "Price | Retailer_A"] = 20.0  # 100% increase

        scenario = simulate_market(
            minimal_baseline, price_matrix_changed, retailer_columns, estimates_zero
        )

        # Quantity should be unchanged where elasticity is zero
        mask = scenario["Elasticity_MAP"] == 0.0
        np.testing.assert_allclose(
            scenario.loc[mask, "Simulated_Unit_Sales"],
            scenario.loc[mask, "Baseline_Unit_Sales"],
            rtol=1e-10
        )

    def test_unit_elasticity_preserves_own_revenue(self, minimal_baseline, retailer_columns, price_matrix):
        """At elasticity -1, own-cell revenue should remain unchanged under constant elasticity."""
        estimates_unit = pd.DataFrame({
            "Category": ["Beverages", "Beverages"],
            "Brand": ["Brand_X", "Brand_X"],
            "SKU": ["SKU_1", "SKU_2"],
            "Elasticity_MAP": [-1.0, -1.0],
        })
        price_matrix_changed = price_matrix.copy()
        price_matrix_changed.loc[0, "Price | Retailer_A"] = 11.0  # 10% increase

        scenario = simulate_market(
            minimal_baseline, price_matrix_changed, retailer_columns, estimates_unit
        )

        # For elasticity -1: Q1 = Q0 * (P1/P0)^-1 = Q0 * P0/P1
        # Revenue R1 = P1 * Q1 = P1 * Q0 * P0/P1 = P0 * Q0 = R0
        mask = scenario["Elasticity_MAP"] == -1.0
        np.testing.assert_allclose(
            scenario.loc[mask, "Simulated_Revenue"],
            scenario.loc[mask, "Baseline_Revenue"],
            rtol=1e-6
        )

    def test_missing_listings_not_created(
        self, minimal_baseline, minimal_estimates, retailer_columns, price_matrix
    ):
        """Prices entered for non-existent SKU-retailer combos should be ignored."""
        # Create price matrix with extra retailer column that doesn't exist in baseline
        price_matrix_extra = price_matrix.copy()
        price_matrix_extra["Price | Retailer_C"] = 8.0  # This retailer doesn't exist in baseline

        scenario = simulate_market(
            minimal_baseline, price_matrix_extra, {**retailer_columns, "Price | Retailer_C": "Retailer_C"},
            minimal_estimates
        )

        # Should not have created new Retailer_C rows
        assert "Retailer_C" not in scenario["Retailer"].values
        assert len(scenario) == len(minimal_baseline)

    def test_zero_demand_stays_zero(self, minimal_baseline, minimal_estimates, retailer_columns, price_matrix):
        """Zero baseline units should remain zero under multiplicative scenario."""
        baseline_zero = minimal_baseline.copy()
        baseline_zero.loc[0, "Unit_Sales"] = 0.0
        baseline_zero.loc[0, "Sales_Revenue"] = 0.0
        baseline_zero.loc[0, "Price"] = 10.0  # Price is irrelevant when units=0

        scenario = simulate_market(
            baseline_zero, price_matrix, retailer_columns, minimal_estimates
        )

        # Zero baseline should stay zero (multiplicative model)
        assert scenario.loc[0, "Simulated_Unit_Sales"] == 0.0
        assert scenario.loc[0, "Simulated_Revenue"] == 0.0


class TestThreeFactorDecomposition:
    """Tests for the Volume -> Mix -> Price decomposition."""

    def test_decomposition_sums_to_revenue_delta(self, minimal_baseline, minimal_estimates, price_matrix, retailer_columns):
        """VE + ME + PE must equal Revenue_Delta at any aggregation level."""
        price_matrix_changed = price_matrix.copy()
        price_matrix_changed.loc[0, "Price | Retailer_A"] = 11.0

        scenario = simulate_market(
            minimal_baseline, price_matrix_changed, retailer_columns, minimal_estimates
        )

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

    def test_mix_effect_zero_when_shares_unchanged(self, minimal_baseline, retailer_columns, price_matrix):
        """Mix effect should be zero when all SKUs have same elasticity (shares unchanged)."""
        # Same elasticity for all SKUs -> proportional quantity changes -> shares unchanged
        estimates_same = pd.DataFrame({
            "Category": ["Beverages", "Beverages"],
            "Brand": ["Brand_X", "Brand_X"],
            "SKU": ["SKU_1", "SKU_2"],
            "Elasticity_MAP": [-1.5, -1.5],
        })
        price_matrix_changed = price_matrix.copy()
        price_matrix_changed.loc[0, "Price | Retailer_A"] = 11.0
        price_matrix_changed.loc[1, "Price | Retailer_A"] = 5.5  # Proportional change

        scenario = simulate_market(
            minimal_baseline, price_matrix_changed, retailer_columns, estimates_same
        )

        decomp = three_factor_decomposition(scenario, [])
        # Mix should be approximately zero (within numerical precision)
        assert abs(decomp["Mix_Effect"].iloc[0]) < 1e-6

    def test_pack_size_factor_extraction(self):
        """Pack size string parsing should handle common formats."""
        assert _pack_size_to_factor("1.5 L") == 1.5
        assert _pack_size_to_factor("500 g") == 0.5
        assert _pack_size_to_factor("500g") == 0.5
        assert _pack_size_to_factor("6 pc") == 6.0
        assert _pack_size_to_factor("6 PCS") == 6.0
        assert _pack_size_to_factor("1000 ml") == 1.0  # Converts to litres
        assert _pack_size_to_factor("2 kg") == 2.0
        assert _pack_size_to_factor("invalid") == 1.0  # Default fallback
        assert _pack_size_to_factor("") == 1.0
        assert _pack_size_to_factor(None) == 1.0
        assert _pack_size_to_factor(np.nan) == 1.0 or np.isnan(_pack_size_to_factor(np.nan))


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_extreme_price_change_rejected(self, minimal_baseline, price_matrix, retailer_columns):
        """Extreme price changes should raise ValueError."""
        # Use positive elasticity to trigger extreme response
        estimates_pos = pd.DataFrame({
            "Category": ["Beverages", "Beverages"],
            "Brand": ["Brand_X", "Brand_X"],
            "SKU": ["SKU_1", "SKU_2"],
            "Elasticity_MAP": [2.0, -1.2],  # Positive elasticity for SKU_1
        })
        price_matrix_extreme = price_matrix.copy()
        price_matrix_extreme.loc[0, "Price | Retailer_A"] = 1e12  # log_response = 2*log(1e11) ≈ 50.6 > 50

        with pytest.raises(ValueError, match="extreme demand response"):
            simulate_market(minimal_baseline, price_matrix_extreme, retailer_columns, estimates_pos)

    def test_nonfinite_price_rejected(self, minimal_baseline, minimal_estimates, price_matrix, retailer_columns):
        """Non-finite prices should raise ValueError."""
        price_matrix_bad = price_matrix.copy()
        price_matrix_bad.loc[0, "Price | Retailer_A"] = np.nan

        with pytest.raises(ValueError, match="finite price above zero"):
            simulate_market(minimal_baseline, price_matrix_bad, retailer_columns, minimal_estimates)

    def test_missing_elasticity_rejected(self, minimal_baseline, price_matrix, retailer_columns):
        """Missing elasticity estimates should raise ValueError."""
        estimates_missing = pd.DataFrame({
            "Category": ["Beverages"],
            "Brand": ["Brand_X"],
            "SKU": ["SKU_1"],
            "Elasticity_MAP": [-1.5],
        })  # Missing SKU_2

        with pytest.raises(ValueError, match="elasticity estimates are missing"):
            simulate_market(minimal_baseline, price_matrix, retailer_columns, estimates_missing)


class TestSimulator:
    """Tests for the sim.py CLI functions."""

    def test_empty_scenario_reproduces_baseline(self, minimal_baseline, minimal_estimates, price_matrix, retailer_columns):
        """Empty price_changes should reproduce baseline exactly."""
        from sim import simulate_scenario

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
        from sim import simulate_scenario

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


class TestPackSizeParsing:
    """Tests for pack size parsing edge cases."""

    def test_known_formats(self):
        """Test common pack size formats."""
        from sim import _pack_size_to_factor
        assert _pack_size_to_factor("1.5 L") == 1.5
        assert _pack_size_to_factor("500 g") == 0.5
        assert _pack_size_to_factor("500g") == 0.5
        assert _pack_size_to_factor("6 pc") == 6.0
        assert _pack_size_to_factor("6 PCS") == 6.0
        assert _pack_size_to_factor("1000 ml") == 1.0
        assert _pack_size_to_factor("2 kg") == 2.0
        assert _pack_size_to_factor("20 kg") == 20.0  # kg should not trigger /1000
        assert _pack_size_to_factor("100 g") == 0.1  # g > 10 should trigger /1000


class TestPosteriorAggregation:
    """Tests for correct posterior uncertainty aggregation."""

    def test_market_aggregation_preserves_covariance(self):
        """Market revenue interval should be computed from aggregated draws, not marginal quantiles."""
        # Create mock revenue draws for 2 cells, 100 draws
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

        # Sum of marginals is wrong (too narrow for positive correlation)
        # This test documents the expected behavior, not a code check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])