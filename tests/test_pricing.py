import math

import pytest

from src.quant_stack.pricing import black_scholes_call, black_scholes_put


class TestBlackScholesCall:
    def test_atm(self):
        # ATM option should have positive value
        price = black_scholes_call(s=100, k=100, t=1.0, r=0.05, sigma=0.2)
        assert price > 0

    def test_deep_itm(self):
        price = black_scholes_call(s=200, k=100, t=1.0, r=0.05, sigma=0.2)
        intrinsic = 200 - 100 * math.exp(-0.05)
        assert price >= intrinsic

    def test_deep_otm(self):
        price = black_scholes_call(s=50, k=200, t=0.01, r=0.05, sigma=0.2)
        assert price == pytest.approx(0.0, abs=1e-6)


class TestBlackScholesPut:
    def test_atm(self):
        price = black_scholes_put(s=100, k=100, t=1.0, r=0.05, sigma=0.2)
        assert price > 0

    def test_put_call_parity(self):
        s, k, t, r, sigma = 100, 100, 1.0, 0.05, 0.2
        call = black_scholes_call(s, k, t, r, sigma)
        put = black_scholes_put(s, k, t, r, sigma)
        # C - P = S - K * e^{-rT}
        lhs = call - put
        rhs = s - k * math.exp(-r * t)
        assert pytest.approx(lhs, rel=1e-10) == rhs

    def test_deep_itm_put(self):
        price = black_scholes_put(s=50, k=200, t=1.0, r=0.05, sigma=0.2)
        intrinsic = 200 * math.exp(-0.05) - 50
        assert price >= intrinsic
