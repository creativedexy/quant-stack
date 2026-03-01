"""Tests for trading strategies and the StrategyRegistry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.strategy import (
    MACDCrossoverStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    Strategy,
    StrategyRegistry,
    strategy_registry,
)
from src.data.synthetic import generate_synthetic_ohlcv
from src.features.pipeline import FeaturePipeline


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    """Synthetic OHLCV with all indicators, ready for strategy testing."""
    ohlcv = generate_synthetic_ohlcv("TEST", days=504, seed=42)
    pipeline = FeaturePipeline()
    return pipeline.run(ohlcv)


@pytest.fixture
def single_row_features() -> pd.DataFrame:
    """Single-row feature DataFrame for edge-case testing."""
    dates = pd.bdate_range("2024-01-02", periods=1, freq="B")
    return pd.DataFrame(
        {
            "Close": [100.0],
            "rsi_14": [50.0],
            "sma_50": [95.0],
            "macd_histogram": [0.5],
        },
        index=dates,
    )


@pytest.fixture
def all_nan_features() -> pd.DataFrame:
    """Feature DataFrame where all indicator values are NaN."""
    dates = pd.bdate_range("2024-01-02", periods=10, freq="B")
    return pd.DataFrame(
        {
            "Close": [float("nan")] * 10,
            "rsi_14": [float("nan")] * 10,
            "sma_50": [float("nan")] * 10,
            "macd_histogram": [float("nan")] * 10,
        },
        index=dates,
    )


# ===================================================================
# Instantiation
# ===================================================================

class TestStrategyInstantiation:
    """All three strategies can be instantiated."""

    def test_mean_reversion_instantiates(self) -> None:
        strat = MeanReversionStrategy()
        assert isinstance(strat, Strategy)
        assert strat.name == "mean_reversion"

    def test_momentum_instantiates(self) -> None:
        strat = MomentumStrategy()
        assert isinstance(strat, Strategy)
        assert strat.name == "momentum"

    def test_macd_crossover_instantiates(self) -> None:
        strat = MACDCrossoverStrategy()
        assert isinstance(strat, Strategy)
        assert strat.name == "macd_crossover"

    def test_strategy_is_abstract(self) -> None:
        """Cannot instantiate Strategy directly."""
        with pytest.raises(TypeError):
            Strategy("test")  # type: ignore[abstract]

    def test_describe_returns_string(self) -> None:
        for cls in (MeanReversionStrategy, MomentumStrategy, MACDCrossoverStrategy):
            strat = cls()
            desc = strat.describe()
            assert isinstance(desc, str)
            assert len(desc) > 0


# ===================================================================
# Signal shape and values
# ===================================================================

class TestSignalOutput:
    """generate_signals returns a DataFrame with values in {-1, 0, 1}."""

    def test_mean_reversion_returns_dataframe(self, features_df: pd.DataFrame) -> None:
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(features_df)
        assert isinstance(signals, pd.DataFrame)

    def test_momentum_returns_dataframe(self, features_df: pd.DataFrame) -> None:
        strat = MomentumStrategy()
        signals = strat.generate_signals(features_df)
        assert isinstance(signals, pd.DataFrame)

    def test_macd_returns_dataframe(self, features_df: pd.DataFrame) -> None:
        strat = MACDCrossoverStrategy()
        signals = strat.generate_signals(features_df)
        assert isinstance(signals, pd.DataFrame)

    def test_mean_reversion_values_in_range(self, features_df: pd.DataFrame) -> None:
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(features_df)
        unique = set(signals.iloc[:, 0].unique())
        assert unique.issubset({-1, 0, 1})

    def test_momentum_values_in_range(self, features_df: pd.DataFrame) -> None:
        strat = MomentumStrategy()
        signals = strat.generate_signals(features_df)
        unique = set(signals.iloc[:, 0].unique())
        assert unique.issubset({-1, 0, 1})

    def test_macd_values_in_range(self, features_df: pd.DataFrame) -> None:
        strat = MACDCrossoverStrategy()
        signals = strat.generate_signals(features_df)
        unique = set(signals.iloc[:, 0].unique())
        assert unique.issubset({-1, 0, 1})


# ===================================================================
# Mean Reversion specifics
# ===================================================================

class TestMeanReversion:
    """MeanReversion produces buy signals when RSI < 30 and sell when > 70."""

    def test_buy_when_rsi_below_30(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        df = pd.DataFrame(
            {"rsi_14": [15, 20, 25, 28, 29]},
            index=dates,
        )
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(df)
        assert (signals.iloc[:, 0] == 1).all()

    def test_sell_when_rsi_above_70(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        df = pd.DataFrame(
            {"rsi_14": [75, 80, 85, 90, 95]},
            index=dates,
        )
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(df)
        assert (signals.iloc[:, 0] == -1).all()

    def test_flat_when_rsi_in_neutral_zone(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        df = pd.DataFrame(
            {"rsi_14": [40, 45, 50, 55, 60]},
            index=dates,
        )
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(df)
        assert (signals.iloc[:, 0] == 0).all()

    def test_config_overrides_thresholds(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=3, freq="B")
        df = pd.DataFrame({"rsi_14": [35, 50, 65]}, index=dates)
        strat = MeanReversionStrategy(
            config={"oversold_threshold": 40, "overbought_threshold": 60},
        )
        signals = strat.generate_signals(df).iloc[:, 0]
        assert signals.iloc[0] == 1   # 35 < 40
        assert signals.iloc[1] == 0   # 40 <= 50 <= 60
        assert signals.iloc[2] == -1  # 65 > 60

    def test_produces_buy_in_synthetic_data(self, features_df: pd.DataFrame) -> None:
        """On realistic synthetic data, at least some buy signals appear."""
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(features_df).iloc[:, 0]
        assert (signals == 1).any(), "Expected at least one buy signal in synthetic data"

    def test_produces_sell_in_synthetic_data(self, features_df: pd.DataFrame) -> None:
        """On realistic synthetic data, at least some sell signals appear."""
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(features_df).iloc[:, 0]
        assert (signals == -1).any(), "Expected at least one sell signal in synthetic data"


# ===================================================================
# MACD Crossover specifics
# ===================================================================

class TestMACDCrossover:
    """MACD strategy produces signals only at crossover points."""

    def test_signals_only_at_crossover(self) -> None:
        """Signals should only appear where the histogram crosses zero."""
        dates = pd.bdate_range("2024-01-02", periods=8, freq="B")
        df = pd.DataFrame(
            {
                "macd_histogram": [-0.5, -0.3, -0.1, 0.2, 0.5, 0.3, -0.1, -0.4],
            },
            index=dates,
        )
        strat = MACDCrossoverStrategy()
        signals = strat.generate_signals(df).iloc[:, 0]
        # Crossover from negative to positive at index 3 → buy
        assert signals.iloc[3] == 1
        # Crossover from positive to negative at index 6 → sell
        assert signals.iloc[6] == -1
        # Non-crossover bars should be 0
        assert signals.iloc[1] == 0
        assert signals.iloc[4] == 0
        assert signals.iloc[7] == 0

    def test_not_every_bar_gets_signal(self) -> None:
        """Most bars should be 0 (hold), not ±1."""
        dates = pd.bdate_range("2024-01-02", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "macd_histogram": [-1, -0.5, 0.2, 0.5, 0.8, 0.3, -0.1, -0.5, -0.3, 0.1],
            },
            index=dates,
        )
        strat = MACDCrossoverStrategy()
        signals = strat.generate_signals(df).iloc[:, 0]
        n_signals = (signals != 0).sum()
        assert n_signals < len(signals), "MACD should not signal on every bar"

    def test_sustained_positive_histogram_no_repeat_buy(self) -> None:
        """A histogram that stays positive after crossover should not re-signal."""
        dates = pd.bdate_range("2024-01-02", periods=6, freq="B")
        df = pd.DataFrame(
            {"macd_histogram": [-0.5, 0.1, 0.3, 0.5, 0.7, 0.9]},
            index=dates,
        )
        strat = MACDCrossoverStrategy()
        signals = strat.generate_signals(df).iloc[:, 0]
        # Only the first crossover (index 1) should be a buy
        assert signals.iloc[1] == 1
        assert (signals.iloc[2:] == 0).all()


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Strategies handle edge cases: all NaN features, single row."""

    def test_all_nan_mean_reversion(self, all_nan_features: pd.DataFrame) -> None:
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(all_nan_features).iloc[:, 0]
        assert (signals == 0).all()

    def test_all_nan_momentum(self, all_nan_features: pd.DataFrame) -> None:
        strat = MomentumStrategy()
        signals = strat.generate_signals(all_nan_features).iloc[:, 0]
        assert (signals == 0).all()

    def test_all_nan_macd(self, all_nan_features: pd.DataFrame) -> None:
        strat = MACDCrossoverStrategy()
        signals = strat.generate_signals(all_nan_features).iloc[:, 0]
        assert (signals == 0).all()

    def test_single_row_mean_reversion(self, single_row_features: pd.DataFrame) -> None:
        strat = MeanReversionStrategy()
        signals = strat.generate_signals(single_row_features)
        assert len(signals) == 1

    def test_single_row_momentum(self, single_row_features: pd.DataFrame) -> None:
        strat = MomentumStrategy()
        signals = strat.generate_signals(single_row_features)
        assert len(signals) == 1

    def test_single_row_macd(self, single_row_features: pd.DataFrame) -> None:
        strat = MACDCrossoverStrategy()
        signals = strat.generate_signals(single_row_features)
        assert len(signals) == 1


# ===================================================================
# Strategy Registry
# ===================================================================

class TestStrategyRegistry:
    """StrategyRegistry returns all three strategies."""

    def test_list_contains_all_three(self) -> None:
        names = strategy_registry.list_strategies()
        assert "mean_reversion" in names
        assert "momentum" in names
        assert "macd_crossover" in names

    def test_create_mean_reversion(self) -> None:
        strat = strategy_registry.create("mean_reversion")
        assert isinstance(strat, MeanReversionStrategy)

    def test_create_momentum(self) -> None:
        strat = strategy_registry.create("momentum")
        assert isinstance(strat, MomentumStrategy)

    def test_create_macd_crossover(self) -> None:
        strat = strategy_registry.create("macd_crossover")
        assert isinstance(strat, MACDCrossoverStrategy)

    def test_create_with_config(self) -> None:
        strat = strategy_registry.create(
            "mean_reversion",
            config={"oversold_threshold": 25, "overbought_threshold": 75},
        )
        assert strat._oversold == 25
        assert strat._overbought == 75

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown strategy"):
            strategy_registry.create("nonexistent")

    def test_register_custom_strategy(self) -> None:
        """Can register and create a new strategy at runtime."""
        registry = StrategyRegistry()

        class DummyStrategy(Strategy):
            def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(
                    np.zeros((len(features), 1), dtype=int),
                    index=features.index,
                    columns=["signal"],
                )

        registry.register("dummy", DummyStrategy)
        assert "dummy" in registry.list_strategies()
        strat = registry.create("dummy")
        assert isinstance(strat, DummyStrategy)
