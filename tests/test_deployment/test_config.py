"""Tests for src/deployment/config.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.deployment.config import (
    DeploymentConfigError,
    get_active_tier,
    get_evaluation_weights,
    get_opus_config,
    get_structural_exclusions,
    is_live_execution_enabled,
    load_deployment_config,
)


# ------------------------------------------------------------------
# load_deployment_config
# ------------------------------------------------------------------


class TestLoadDeploymentConfig:
    """Tests for loading the deployment YAML config."""

    def test_loads_from_default_path(self):
        """Config loads successfully from config/capital_deployment.yaml."""
        config = load_deployment_config()
        assert isinstance(config, dict)
        assert "evaluation" in config
        assert "scanner" in config
        assert "sizing" in config

    def test_loads_from_custom_path(self, tmp_path):
        """Config loads from a custom YAML file path."""
        custom = tmp_path / "custom.yaml"
        custom.write_text(
            yaml.dump({
                "evaluation": {
                    "weights": {
                        "growth_score": 0.40,
                        "diversification_score": 0.25,
                        "risk_adjusted_score": 0.20,
                        "income_score": 0.15,
                    },
                    "thresholds": {},
                },
                "scanner": {},
                "sizing": {"scaling": {}, "constraints": {}},
                "execution": {},
                "risk": {},
                "opus": {},
                "memory": {},
                "pipeline": {},
            }),
            encoding="utf-8",
        )
        config = load_deployment_config(custom)
        assert config["evaluation"]["weights"]["growth_score"] == 0.40

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError if config file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_deployment_config(tmp_path / "nonexistent.yaml")

    def test_missing_required_sections_raises(self, tmp_path):
        """DeploymentConfigError if required sections are absent."""
        custom = tmp_path / "bad.yaml"
        custom.write_text(
            yaml.dump({"evaluation": {"weights": {"a": 1.0}}}),
            encoding="utf-8",
        )
        with pytest.raises(DeploymentConfigError, match="missing required"):
            load_deployment_config(custom)

    def test_weights_not_summing_to_one_raises(self, tmp_path):
        """DeploymentConfigError if evaluation weights do not sum to ~1.0."""
        custom = tmp_path / "bad_weights.yaml"
        custom.write_text(
            yaml.dump({
                "evaluation": {
                    "weights": {
                        "growth_score": 0.50,
                        "diversification_score": 0.50,
                        "risk_adjusted_score": 0.50,
                        "income_score": 0.50,
                    },
                },
                "scanner": {},
                "sizing": {},
                "execution": {},
                "risk": {},
                "opus": {},
                "memory": {},
                "pipeline": {},
            }),
            encoding="utf-8",
        )
        with pytest.raises(DeploymentConfigError, match="weights must sum"):
            load_deployment_config(custom)

    def test_non_dict_yaml_raises(self, tmp_path):
        """DeploymentConfigError if YAML root is not a mapping."""
        custom = tmp_path / "list.yaml"
        custom.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(DeploymentConfigError, match="YAML mapping"):
            load_deployment_config(custom)


# ------------------------------------------------------------------
# get_active_tier
# ------------------------------------------------------------------


class TestGetActiveTier:
    """Tests for sizing tier selection."""

    def test_tier_1_for_500_gbp(self, deployment_config):
        """GBP 500 falls into tier 1 (0-1000)."""
        tier = get_active_tier(500.0, deployment_config)
        assert tier["tier_name"] == "tier_1"
        assert tier["max_positions"] == 5
        assert tier["prefer_etfs"] is True

    def test_tier_2_for_5000_gbp(self, deployment_config):
        """GBP 5000 falls into tier 2 (1000-10000)."""
        tier = get_active_tier(5000.0, deployment_config)
        assert tier["tier_name"] == "tier_2"
        assert tier["max_positions"] == 8
        assert tier["prefer_etfs"] is False

    def test_tier_3_for_50000_gbp(self, deployment_config):
        """GBP 50000 falls into tier 3 (10000-100000)."""
        tier = get_active_tier(50000.0, deployment_config)
        assert tier["tier_name"] == "tier_3"
        assert tier["max_positions"] == 15

    def test_tier_4_for_large_capital(self, deployment_config):
        """GBP 200000 falls into tier 4 (100000+)."""
        tier = get_active_tier(200000.0, deployment_config)
        assert tier["tier_name"] == "tier_4"
        assert tier["max_positions"] == 25

    def test_zero_capital_gets_tier_1(self, deployment_config):
        """GBP 0 is still valid and maps to tier 1."""
        tier = get_active_tier(0.0, deployment_config)
        assert tier["tier_name"] == "tier_1"

    def test_boundary_1000_gets_tier_2(self, deployment_config):
        """GBP 1000 exactly maps to tier 2 (min_capital <= capital < max_capital)."""
        tier = get_active_tier(1000.0, deployment_config)
        assert tier["tier_name"] == "tier_2"


# ------------------------------------------------------------------
# get_structural_exclusions
# ------------------------------------------------------------------


class TestGetStructuralExclusions:
    """Tests for structural exclusion list."""

    def test_default_is_empty(self, deployment_config):
        """Default config has no structural exclusions."""
        result = get_structural_exclusions(deployment_config)
        assert result == []

    def test_returns_configured_exclusions(self, deployment_config):
        """Returns tickers from config when set."""
        deployment_config["scanner"]["global_exclusions"]["structural"] = [
            "TSLA",
            "GME",
        ]
        result = get_structural_exclusions(deployment_config)
        assert result == ["TSLA", "GME"]


# ------------------------------------------------------------------
# is_live_execution_enabled
# ------------------------------------------------------------------


class TestIsLiveExecutionEnabled:
    """Tests for the live execution gate."""

    def test_true_when_both_keys_set(self, deployment_config):
        """Returns True when mode=LIVE and confirm_live=true."""
        assert is_live_execution_enabled(deployment_config) is True

    def test_false_when_mode_missing(self, deployment_config):
        """Returns False when execution.mode is missing."""
        del deployment_config["execution"]["mode"]
        assert is_live_execution_enabled(deployment_config) is False

    def test_false_when_mode_is_paper(self, deployment_config):
        """Returns False when mode is not LIVE."""
        deployment_config["execution"]["mode"] = "PAPER"
        assert is_live_execution_enabled(deployment_config) is False

    def test_false_when_confirm_live_is_false(self, deployment_config):
        """Returns False when confirm_live is explicitly False."""
        deployment_config["execution"]["confirm_live"] = False
        assert is_live_execution_enabled(deployment_config) is False

    def test_false_when_confirm_live_missing(self, deployment_config):
        """Returns False when confirm_live key is absent."""
        del deployment_config["execution"]["confirm_live"]
        assert is_live_execution_enabled(deployment_config) is False

    def test_case_insensitive_mode(self, deployment_config):
        """Mode comparison is case-insensitive."""
        deployment_config["execution"]["mode"] = "live"
        assert is_live_execution_enabled(deployment_config) is True


# ------------------------------------------------------------------
# get_evaluation_weights
# ------------------------------------------------------------------


class TestGetEvaluationWeights:
    """Tests for evaluation weight retrieval."""

    def test_returns_four_weights(self, deployment_config):
        """Returns all four evaluation dimension weights."""
        weights = get_evaluation_weights(deployment_config)
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_growth_is_highest(self, deployment_config):
        """Growth has the highest weight (0.40)."""
        weights = get_evaluation_weights(deployment_config)
        assert weights["growth_score"] == 0.40


# ------------------------------------------------------------------
# get_opus_config
# ------------------------------------------------------------------


class TestGetOpusConfig:
    """Tests for Opus configuration retrieval."""

    def test_returns_model_and_params(self, deployment_config):
        """Returns model string and operational parameters."""
        opus = get_opus_config(deployment_config)
        assert "model" in opus
        assert opus["max_tokens"] == 4096
        assert opus["timeout"] == 120

    def test_includes_api_key_availability(self, deployment_config, monkeypatch):
        """Includes a flag indicating whether ANTHROPIC_API_KEY is set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        opus = get_opus_config(deployment_config)
        assert opus["api_key_available"] is True

    def test_warns_when_no_api_key(self, deployment_config, monkeypatch):
        """Flags api_key_available=False when env var is missing."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        opus = get_opus_config(deployment_config)
        assert opus["api_key_available"] is False
