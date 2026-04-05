"""Tests for src.analysis.monte_carlo."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.monte_carlo import (
    compute_percentile_paths,
    estimate_target_date,
    simulate_dca,
    simulate_paths,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def daily_returns() -> np.ndarray:
    """Synthetic daily returns with known seed for reproducibility."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0004, 0.012, size=756)  # ~3 years of daily returns


# ------------------------------------------------------------------
# simulate_paths
# ------------------------------------------------------------------

class TestSimulatePaths:

    def test_paths_shape(self, daily_returns: np.ndarray) -> None:
        paths = simulate_paths(daily_returns, n_simulations=50, n_days=252, random_seed=1)
        assert paths.shape == (252, 50)

    def test_paths_start_at_one(self, daily_returns: np.ndarray) -> None:
        paths = simulate_paths(daily_returns, n_simulations=100, n_days=252, random_seed=1)
        np.testing.assert_allclose(paths[0, :], 1.0, atol=1e-10)

    def test_deterministic_with_seed(self, daily_returns: np.ndarray) -> None:
        paths_a = simulate_paths(daily_returns, n_simulations=50, n_days=100, random_seed=99)
        paths_b = simulate_paths(daily_returns, n_simulations=50, n_days=100, random_seed=99)
        np.testing.assert_array_equal(paths_a, paths_b)

    def test_log_return_compounding(self, daily_returns: np.ndarray) -> None:
        """Arithmetic compound would differ from log compound over 252 days.

        We verify that the simulation uses log returns by checking that
        the median path endpoint is consistent with log-normal growth.
        """
        paths = simulate_paths(daily_returns, n_simulations=5000, n_days=252, random_seed=42)

        # Log return parameters
        clipped = np.clip(daily_returns, -0.9999, None)
        log_ret = np.log(1 + clipped)
        mu = np.mean(log_ret)

        # Expected median of log-normal: exp(mu * n_days)
        expected_median = np.exp(mu * 252)
        actual_median = np.median(paths[-1, :])

        # Should be within 5% -- Monte Carlo is noisy but with 5000 paths
        # this is well within tolerance
        assert abs(actual_median - expected_median) / expected_median < 0.05

    def test_max_simulations_raises_value_error(self, daily_returns: np.ndarray) -> None:
        with pytest.raises(ValueError, match="exceeds maximum"):
            simulate_paths(daily_returns, n_simulations=10001)

    def test_clip_protects_against_log_zero(self) -> None:
        """Returns containing -1.0 (total loss) should not raise."""
        bad_returns = np.array([0.01, -1.0, 0.02, -0.5, 0.03])
        # Should not raise
        paths = simulate_paths(bad_returns, n_simulations=10, n_days=10, random_seed=1)
        assert paths.shape == (10, 10)
        assert np.all(np.isfinite(paths))

    def test_single_simulation_no_shape_error(self, daily_returns: np.ndarray) -> None:
        paths = simulate_paths(daily_returns, n_simulations=1, n_days=50, random_seed=1)
        assert paths.shape == (50, 1)
        assert np.isfinite(paths).all()


# ------------------------------------------------------------------
# compute_percentile_paths
# ------------------------------------------------------------------

class TestPercentilePaths:

    def test_percentile_paths_ordering(self, daily_returns: np.ndarray) -> None:
        paths = simulate_paths(daily_returns, n_simulations=500, n_days=252, random_seed=1)
        pct = compute_percentile_paths(paths, [5, 50, 95])

        assert "p5" in pct
        assert "p50" in pct
        assert "p95" in pct

        # p5 <= p50 <= p95 at every time step
        assert np.all(pct["p5"] <= pct["p50"] + 1e-10)
        assert np.all(pct["p50"] <= pct["p95"] + 1e-10)


# ------------------------------------------------------------------
# simulate_dca
# ------------------------------------------------------------------

class TestSimulateDca:

    def test_dca_shape(self, daily_returns: np.ndarray) -> None:
        result = simulate_dca(
            daily_returns,
            monthly_contribution=500.0,
            n_months=24,
            n_simulations=50,
            random_seed=1,
        )
        assert result.shape == (24, 50)

    def test_dca_adds_contributions_correctly(self) -> None:
        """With zero returns, final value = starting + contribution * n_months."""
        zero_returns = np.zeros(500)
        result = simulate_dca(
            zero_returns,
            monthly_contribution=100.0,
            n_months=12,
            starting_value=1000.0,
            n_simulations=10,
            random_seed=1,
        )
        expected_final = 1000.0 + 100.0 * 12
        # With zero mean and zero std, returns are all zero -> (1+0)=1
        # Each month: (prev + 100) * 1.0 = prev + 100
        np.testing.assert_allclose(result[-1, :], expected_final, atol=1.0)


# ------------------------------------------------------------------
# estimate_target_date
# ------------------------------------------------------------------

class TestEstimateTargetDate:

    def test_target_date_probability_between_0_and_1(
        self, daily_returns: np.ndarray,
    ) -> None:
        portfolio = simulate_dca(
            daily_returns,
            monthly_contribution=500.0,
            n_months=60,
            starting_value=10000.0,
            n_simulations=200,
            random_seed=1,
        )
        result = estimate_target_date(portfolio, target_value=20000.0)
        assert 0.0 <= result["p90_probability"] <= 1.0

    def test_target_date_p50_none_when_target_unreachable(
        self, daily_returns: np.ndarray,
    ) -> None:
        """Target of 1 trillion in 12 months should be unreachable."""
        portfolio = simulate_dca(
            daily_returns,
            monthly_contribution=100.0,
            n_months=12,
            starting_value=0.0,
            n_simulations=100,
            random_seed=1,
        )
        result = estimate_target_date(portfolio, target_value=1e12)
        assert result["p50_months"] is None
        assert result["p90_probability"] == 0.0
