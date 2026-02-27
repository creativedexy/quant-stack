import numpy as np
import pytest

from src.quant_stack.returns import simple_return, log_return, cumulative_returns


class TestSimpleReturn:
    def test_basic(self):
        prices = [100, 110, 105]
        result = simple_return(prices)
        expected = np.array([0.1, -5 / 110])
        np.testing.assert_allclose(result, expected)

    def test_constant_prices(self):
        result = simple_return([50, 50, 50])
        np.testing.assert_allclose(result, [0.0, 0.0])


class TestLogReturn:
    def test_basic(self):
        prices = [100, 110, 105]
        result = log_return(prices)
        expected = np.diff(np.log([100, 110, 105]))
        np.testing.assert_allclose(result, expected)

    def test_consistent_with_simple(self):
        prices = [100, 101]
        sr = simple_return(prices)[0]
        lr = log_return(prices)[0]
        assert pytest.approx(lr) == np.log(1 + sr)


class TestCumulativeReturns:
    def test_basic(self):
        returns = [0.1, -0.05, 0.02]
        result = cumulative_returns(returns)
        expected = np.cumprod([1.1, 0.95, 1.02]) - 1.0
        np.testing.assert_allclose(result, expected)

    def test_zero_returns(self):
        result = cumulative_returns([0.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 0.0])
