import numpy as np
import pytest

from src.quant_stack.risk import volatility, sharpe_ratio, max_drawdown


class TestVolatility:
    def test_known_values(self):
        returns = [0.01, -0.02, 0.015, -0.005, 0.01]
        vol = volatility(returns, annualise=False)
        expected = float(np.std(returns, ddof=1))
        assert pytest.approx(vol) == expected

    def test_annualised(self):
        returns = [0.01, -0.02, 0.015, -0.005, 0.01]
        vol_ann = volatility(returns, annualise=True, periods_per_year=252)
        vol_raw = volatility(returns, annualise=False)
        assert pytest.approx(vol_ann) == vol_raw * np.sqrt(252)


class TestSharpeRatio:
    def test_zero_std(self):
        assert sharpe_ratio([0.0, 0.0, 0.0]) == 0.0

    def test_positive_returns(self):
        returns = [0.01] * 100
        sr = sharpe_ratio(returns, risk_free_rate=0.0)
        assert sr > 0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        prices = [100, 110, 120, 130]
        assert max_drawdown(prices) == 0.0

    def test_known_drawdown(self):
        prices = [100, 120, 90, 110]
        dd = max_drawdown(prices)
        assert pytest.approx(dd) == (90 - 120) / 120

    def test_full_series(self):
        prices = [100, 80, 90, 70, 100]
        dd = max_drawdown(prices)
        assert pytest.approx(dd) == (70 - 100) / 100
