"""Tests for src/deployment/analyst.py.

All tests use mocked API calls. NEVER make real Opus calls in tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.deployment import (
    EnrichedCandidate,
    PipelineContext,
    ScoredCandidate,
)
from src.deployment.analyst import (
    ANALYST_SYSTEM_PROMPT,
    KeywordAnalyst,
    OpusAnalyst,
)


@pytest.fixture
def opus_analyst(deployment_config) -> OpusAnalyst:
    """OpusAnalyst with mocked API key."""
    return OpusAnalyst(config=deployment_config, memory=None)


@pytest.fixture
def keyword_analyst(deployment_config) -> KeywordAnalyst:
    """KeywordAnalyst instance."""
    return KeywordAnalyst(config=deployment_config)


@pytest.fixture
def pipeline_context(deployment_config, synthetic_portfolio) -> PipelineContext:
    """Pipeline context with synthetic portfolio positions."""
    return PipelineContext(
        cycle_id="cycle_001",
        memory=None,
        config=deployment_config,
        portfolio_positions=synthetic_portfolio,
    )


# ------------------------------------------------------------------
# KeywordAnalyst (always available, no API)
# ------------------------------------------------------------------


class TestKeywordAnalyst:
    """Tests for the keyword fallback analyst."""

    def test_produces_valid_scores(
        self, keyword_analyst, synthetic_enriched, pipeline_context,
    ):
        """Keyword analyst returns ScoredCandidate with valid scores."""
        scored = keyword_analyst.analyse(synthetic_enriched, pipeline_context)
        assert len(scored) == len(synthetic_enriched)
        for sc in scored:
            assert isinstance(sc, ScoredCandidate)
            for key in ["growth", "diversification", "risk_adjusted", "income", "composite"]:
                assert key in sc.scores
                assert 0.0 <= sc.scores[key] <= 1.0

    def test_composite_is_weighted_sum(
        self, keyword_analyst, synthetic_enriched, pipeline_context,
    ):
        """Composite score equals the weighted sum of dimensions."""
        scored = keyword_analyst.analyse(synthetic_enriched, pipeline_context)
        weights = pipeline_context.config["evaluation"]["weights"]

        for sc in scored:
            expected = (
                sc.scores["growth"] * weights["growth_score"]
                + sc.scores["diversification"] * weights["diversification_score"]
                + sc.scores["risk_adjusted"] * weights["risk_adjusted_score"]
                + sc.scores["income"] * weights["income_score"]
            )
            assert abs(sc.scores["composite"] - expected) < 0.01, (
                f"{sc.ticker}: composite={sc.scores['composite']}, "
                f"expected={expected}"
            )

    def test_reasoning_non_empty(
        self, keyword_analyst, synthetic_enriched, pipeline_context,
    ):
        """All reasoning fields are non-empty strings."""
        scored = keyword_analyst.analyse(synthetic_enriched, pipeline_context)
        for sc in scored:
            for key in ["growth", "diversification", "risk_adjusted", "income"]:
                assert key in sc.reasoning
                assert len(sc.reasoning[key]) > 0

    def test_conviction_levels(
        self, keyword_analyst, synthetic_enriched, pipeline_context,
    ):
        """Conviction is one of HIGH, MEDIUM, LOW."""
        scored = keyword_analyst.analyse(synthetic_enriched, pipeline_context)
        for sc in scored:
            assert sc.conviction in ("HIGH", "MEDIUM", "LOW")

    def test_sector_in_portfolio_scores_low_diversification(
        self, keyword_analyst, pipeline_context,
    ):
        """Candidate in same sector as existing holdings scores low on diversification."""
        # Portfolio has Technology sector.
        tech_candidate = EnrichedCandidate(
            ticker="AAPL",
            name="Apple Inc",
            exchange="NASDAQ",
            sector="Technology",
            market_cap_usd=3e12,
            source_universe="us_growth",
            fundamentals={"earnings_growth_yoy": 0.10, "revenue_growth_yoy": 0.05, "dividend_yield": 0.005},
            technicals={"rsi_14": 55, "volatility_21d": 0.20},
            gbp_price=140.0,
        )
        scored = keyword_analyst.analyse([tech_candidate], pipeline_context)
        assert scored[0].scores["diversification"] < 0.50

    def test_absent_sector_scores_high_diversification(
        self, keyword_analyst, pipeline_context,
    ):
        """Candidate in absent sector scores high on diversification."""
        # Portfolio does NOT have Energy.
        energy_candidate = EnrichedCandidate(
            ticker="XOM",
            name="Exxon Mobil",
            exchange="NYSE",
            sector="Energy",
            market_cap_usd=400e9,
            source_universe="contrarian_value",
            fundamentals={"earnings_growth_yoy": 0.05, "revenue_growth_yoy": 0.02, "dividend_yield": 0.035},
            technicals={"rsi_14": 50, "volatility_21d": 0.18},
            gbp_price=85.0,
        )
        scored = keyword_analyst.analyse([energy_candidate], pipeline_context)
        assert scored[0].scores["diversification"] >= 0.70

    def test_risks_include_currency(
        self, keyword_analyst, pipeline_context,
    ):
        """Non-GBP candidates get a currency risk flag."""
        usd_candidate = EnrichedCandidate(
            ticker="NVDA",
            name="NVIDIA",
            exchange="NASDAQ",
            sector="Technology",
            market_cap_usd=2.8e12,
            source_universe="us_growth",
            currency="USD",
            fundamentals={"earnings_growth_yoy": 0.35},
            technicals={"rsi_14": 60, "volatility_21d": 0.30},
            gbp_price=105.0,
        )
        scored = keyword_analyst.analyse([usd_candidate], pipeline_context)
        assert any("currency" in r.lower() for r in scored[0].risks)

    def test_high_pe_flagged_as_risk(
        self, keyword_analyst, pipeline_context,
    ):
        """High P/E ratio is flagged as a risk."""
        expensive = EnrichedCandidate(
            ticker="EXPEN",
            name="Expensive Corp",
            exchange="NASDAQ",
            sector="Technology",
            market_cap_usd=100e9,
            source_universe="us_growth",
            fundamentals={"pe_ratio": 80, "earnings_growth_yoy": 0.20},
            technicals={"rsi_14": 55, "volatility_21d": 0.22},
            gbp_price=200.0,
        )
        scored = keyword_analyst.analyse([expensive], pipeline_context)
        assert any("valuation" in r.lower() for r in scored[0].risks)


# ------------------------------------------------------------------
# OpusAnalyst (mocked API)
# ------------------------------------------------------------------


class TestOpusAnalyst:
    """Tests for the Opus analyst with mocked API calls."""

    def test_falls_back_when_no_api_key(
        self, deployment_config, synthetic_enriched, pipeline_context,
    ):
        """Uses keyword fallback when ANTHROPIC_API_KEY is not set."""
        analyst = OpusAnalyst(config=deployment_config)
        analyst._api_key = ""  # Simulate missing key.

        scored = analyst.analyse(synthetic_enriched, pipeline_context)
        assert len(scored) == len(synthetic_enriched)
        # Should still produce valid scores via fallback.
        for sc in scored:
            assert 0.0 <= sc.composite_score <= 1.0

    def test_score_validation_catches_out_of_range(self, opus_analyst):
        """Scores outside [0, 1] are rejected."""
        bad_scores = {
            "growth": 1.5,
            "diversification": 0.5,
            "risk_adjusted": 0.5,
            "income": 0.5,
            "composite": 0.7,
        }
        assert opus_analyst._validate_scores(bad_scores) is False

    def test_score_validation_catches_missing_key(self, opus_analyst):
        """Missing required score key is rejected."""
        incomplete = {
            "growth": 0.8,
            "diversification": 0.6,
            # Missing risk_adjusted, income, composite.
        }
        assert opus_analyst._validate_scores(incomplete) is False

    def test_score_validation_auto_corrects_composite(self, opus_analyst):
        """Composite mismatch is auto-corrected rather than rejected."""
        scores = {
            "growth": 0.80,
            "diversification": 0.60,
            "risk_adjusted": 0.50,
            "income": 0.30,
            "composite": 0.99,  # Wrong.
        }
        result = opus_analyst._validate_scores(scores)
        assert result is True
        # Composite should be corrected.
        expected = 0.80 * 0.40 + 0.60 * 0.25 + 0.50 * 0.20 + 0.30 * 0.15
        assert abs(scores["composite"] - expected) < 0.01

    def test_opus_prompt_includes_portfolio(
        self, opus_analyst, synthetic_enriched, pipeline_context,
    ):
        """Opus prompt includes the portfolio positions context."""
        prompt = opus_analyst._build_opus_prompt(
            synthetic_enriched, pipeline_context
        )
        assert "Existing Portfolio" in prompt
        assert "VUSA.L" in prompt  # From synthetic_portfolio fixture.

    def test_opus_prompt_includes_candidate_data(
        self, opus_analyst, synthetic_enriched, pipeline_context,
    ):
        """Opus prompt includes enriched candidate data."""
        prompt = opus_analyst._build_opus_prompt(
            synthetic_enriched, pipeline_context
        )
        for ec in synthetic_enriched:
            assert ec.ticker in prompt

    def test_opus_prompt_includes_weights(
        self, opus_analyst, synthetic_enriched, pipeline_context,
    ):
        """Opus prompt includes evaluation weights."""
        prompt = opus_analyst._build_opus_prompt(
            synthetic_enriched, pipeline_context
        )
        assert "growth_score" in prompt
        assert "0.4" in prompt

    def test_parse_opus_response_valid(
        self, opus_analyst, synthetic_enriched, mock_opus_response,
    ):
        """Valid Opus response is parsed into ScoredCandidate."""
        # Only NVDA is in both the response and the enriched list.
        scored = opus_analyst._parse_opus_response(
            mock_opus_response, synthetic_enriched
        )
        assert len(scored) >= 1
        nvda = next((s for s in scored if s.ticker == "NVDA"), None)
        assert nvda is not None
        assert nvda.scores["growth"] == 0.88

    def test_parse_skips_unknown_ticker(
        self, opus_analyst, synthetic_enriched,
    ):
        """Tickers in response but not in enriched set are skipped."""
        response = {
            "candidates": [
                {
                    "ticker": "UNKNOWN",
                    "scores": {
                        "growth": 0.5, "diversification": 0.5,
                        "risk_adjusted": 0.5, "income": 0.5,
                        "composite": 0.5,
                    },
                    "reasoning": {
                        "growth": "x", "diversification": "x",
                        "risk_adjusted": "x", "income": "x",
                    },
                    "conviction": "LOW",
                    "risks": [],
                    "position_suggestion": "",
                }
            ]
        }
        scored = opus_analyst._parse_opus_response(
            response, synthetic_enriched
        )
        assert len(scored) == 0

    def test_token_usage_logged(
        self, deployment_config, synthetic_enriched, pipeline_context,
    ):
        """Opus call logs token usage to memory."""
        mock_memory = MagicMock()
        mock_memory.get_learnings.return_value = {}
        mock_memory.load.return_value = {"last_cycle_id": "cycle_001"}

        analyst = OpusAnalyst(
            config=deployment_config, memory=mock_memory,
        )
        analyst._api_key = "sk-test"

        # Mock the Anthropic client.
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "candidates": [],
            "portfolio_level_commentary": "",
            "regime_assessment": "",
        }))]
        mock_response.usage.input_tokens = 1500
        mock_response.usage.output_tokens = 800

        # Mock the anthropic module at import time.
        mock_anthropic_mod = MagicMock()
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        import sys
        sys.modules["anthropic"] = mock_anthropic_mod
        try:
            analyst.analyse(synthetic_enriched, pipeline_context)
        finally:
            sys.modules.pop("anthropic", None)

        mock_memory.log_opus_usage.assert_called_once()

    def test_system_prompt_exists(self):
        """Analyst system prompt is a non-empty string."""
        assert len(ANALYST_SYSTEM_PROMPT) > 100
        assert "portfolio analyst" in ANALYST_SYSTEM_PROMPT.lower()
