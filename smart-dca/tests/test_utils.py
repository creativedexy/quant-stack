"""Tests for utility modules: config_loader and validators."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config_loader import clear_cache, load_assets, load_settings
from src.utils.exceptions import ConfigError
from src.utils.validators import validate_amount_gbp, validate_score


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Clear config cache before and after each test."""
    clear_cache()
    yield
    clear_cache()


class TestLoadSettings:
    """Tests for load_settings()."""

    def test_returns_expected_structure(self) -> None:
        """load_settings() returns dict with all required top-level keys."""
        settings = load_settings()
        assert "binance" in settings
        assert "schedule" in settings
        assert "scoring" in settings
        assert "signals" in settings
        assert "risk" in settings

    def test_scoring_threshold_present(self) -> None:
        """Scoring section contains buy_threshold."""
        settings = load_settings()
        assert "buy_threshold" in settings["scoring"]
        assert isinstance(settings["scoring"]["buy_threshold"], (int, float))

    def test_env_var_override_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SMARTDCA_ env var overrides settings values."""
        monkeypatch.setenv("SMARTDCA_SCORING__BUY_THRESHOLD", "70")
        settings = load_settings()
        assert settings["scoring"]["buy_threshold"] == 70

    def test_missing_file_raises_config_error(self, tmp_path: Path) -> None:
        """Missing settings file raises ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            load_settings(tmp_path / "nonexistent.yaml")


class TestLoadAssets:
    """Tests for load_assets()."""

    def test_returns_assets_dict(self) -> None:
        """load_assets() returns dict with 'assets' key."""
        assets = load_assets()
        assert "assets" in assets
        assert "BTCGBP" in assets["assets"]

    def test_asset_has_required_fields(self) -> None:
        """Each asset has weekly_allocation_gbp, min_order_gbp, enabled."""
        assets = load_assets()
        for symbol, cfg in assets["assets"].items():
            assert "weekly_allocation_gbp" in cfg, f"{symbol} missing weekly_allocation_gbp"
            assert "min_order_gbp" in cfg, f"{symbol} missing min_order_gbp"
            assert "enabled" in cfg, f"{symbol} missing enabled"


class TestValidateAmountGbp:
    """Tests for validate_amount_gbp()."""

    def test_rejects_above_max_single_order(self) -> None:
        """Amount above max_single_order_gbp is rejected."""
        # max_single_order_gbp is 50.0 in config
        assert validate_amount_gbp(100.0, "BTCGBP") is False

    def test_accepts_valid_amount(self) -> None:
        """Amount within limits is accepted."""
        assert validate_amount_gbp(10.0, "BTCGBP") is True

    def test_rejects_below_min_order(self) -> None:
        """Amount below min_order_gbp is rejected."""
        # BTCGBP min_order_gbp is 5.0
        assert validate_amount_gbp(1.0, "BTCGBP") is False

    def test_rejects_unknown_asset(self) -> None:
        """Unknown asset symbol is rejected."""
        assert validate_amount_gbp(10.0, "FAKEGBP") is False


class TestValidateScore:
    """Tests for validate_score()."""

    def test_rejects_below_zero(self) -> None:
        """Score below 0.0 is rejected."""
        assert validate_score(-1.0) is False

    def test_rejects_above_hundred(self) -> None:
        """Score above 100.0 is rejected."""
        assert validate_score(101.0) is False

    def test_accepts_valid_score(self) -> None:
        """Score within 0-100 is accepted."""
        assert validate_score(50.0) is True

    def test_accepts_boundaries(self) -> None:
        """Score at exact boundaries (0.0, 100.0) is accepted."""
        assert validate_score(0.0) is True
        assert validate_score(100.0) is True
