"""Opus analyst -- scores candidates using Claude Opus or keyword fallback.

The Opus analyst is the core value of the deployment engine.  It receives
enriched candidates plus the live portfolio context, and returns structured
scores with reasoning for each evaluation dimension.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.deployment import (
    AbstractAnalyst,
    EnrichedCandidate,
    PipelineContext,
    ScoredCandidate,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# System prompt (code, not config)
# ------------------------------------------------------------------

ANALYST_SYSTEM_PROMPT = """\
You are a senior portfolio analyst at a multi-strategy fund. You are reviewing
candidate investments for a new capital allocation that MUST complement -- not
duplicate -- the existing portfolio.

Your mandate:
- Growth-first, but never reckless. You seek asymmetric risk/reward.
- Diversification is structural, not cosmetic. If the existing book is
  already heavy in a sector, geography, or theme, buying more of the same
  is not diversification -- even if the specific ticker is different.
- You think in terms of portfolio construction, not individual stock picks.
  Every candidate is evaluated in the context of what is already held.
  The existing portfolio is provided in context -- study it carefully before
  scoring. Identify the concentration risks, missing exposures, and sector
  gaps yourself. Do not rely on pre-computed summaries.
- You are honest about uncertainty. A score of 0.50 with high confidence
  is more useful than a score of 0.85 with no supporting evidence.
- You explain your reasoning in plain English. No jargon. The end user
  has low financial literacy and needs to understand WHY, not just WHAT.

Constraints:
- The existing portfolio (provided in context) is FIXED. You cannot suggest
  changes to it. You are only scoring NEW candidates for fresh capital.
- Maximum correlation to existing portfolio: 0.60. If a candidate moves
  in lockstep with what is already held, it fails diversification regardless
  of other scores.
- You MUST flag any candidate where your growth thesis depends on a single
  binary event (e.g., FDA approval, election outcome). These are bets, not
  investments.
- Currency risk is real. Non-GBP positions carry FX exposure. Factor this
  into the risk-adjusted score.
- If the existing portfolio already holds ETFs or funds covering a theme
  (e.g., S&P 500 tracker, semiconductor fund), a candidate in the same
  space must offer something materially different to score well on
  diversification. Owning two S&P 500 trackers is not diversification.
- All scores must be between 0.0 and 1.0. Do not inflate. A composite of
  0.60 is a decent opportunity. 0.80+ is exceptional and rare.

Output format: Return ONLY valid JSON matching the schema provided.
No markdown, no preamble, no commentary outside the JSON structure.
"""


class OpusAnalyst(AbstractAnalyst):
    """Scores candidates using Claude Opus with structured JSON output.

    Falls back to KeywordAnalyst when ANTHROPIC_API_KEY is not set.

    Args:
        config: Deployment config dict.
        memory: MemoryManager for cost tracking and learnings.
    """

    def __init__(
        self,
        config: dict[str, Any],
        memory: Any | None = None,
    ) -> None:
        self._config = config
        self._memory = memory
        self._opus_config = config.get("opus", {})
        self._eval_weights = config.get("evaluation", {}).get("weights", {})
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = self._opus_config.get("model", "claude-opus-4-20250514")
        self._max_retries = self._opus_config.get("max_retries", 3)

    def analyse(
        self,
        enriched: list[EnrichedCandidate],
        context: PipelineContext,
    ) -> list[ScoredCandidate]:
        """Score enriched candidates. Uses Opus if API key is set, else fallback.

        Args:
            enriched: Enriched candidates from the enricher.
            context: Pipeline context with portfolio and config.

        Returns:
            List of ScoredCandidate objects with scores and reasoning.
        """
        if not self._api_key:
            logger.warning(
                "No ANTHROPIC_API_KEY -- using keyword fallback analyst"
            )
            fallback = KeywordAnalyst(self._config)
            return fallback.analyse(enriched, context)

        return self._analyse_with_opus(enriched, context)

    def _analyse_with_opus(
        self,
        enriched: list[EnrichedCandidate],
        context: PipelineContext,
    ) -> list[ScoredCandidate]:
        """Send enriched candidates to Opus and parse the response."""
        prompt = self._build_opus_prompt(enriched, context)

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._call_opus(prompt)
                scored = self._parse_opus_response(response, enriched)
                return scored
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(
                    "Opus response parse failed (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    e,
                )
                if attempt < self._max_retries:
                    # Retry with stricter instruction.
                    prompt += (
                        "\n\nIMPORTANT: Your previous response was not valid "
                        "JSON. Return ONLY a valid JSON object matching the "
                        "schema. No markdown fences, no commentary."
                    )
                else:
                    logger.error(
                        "Opus analysis failed after %d attempts -- "
                        "falling back to keyword analyst",
                        self._max_retries,
                    )
                    fallback = KeywordAnalyst(self._config)
                    return fallback.analyse(enriched, context)

        return []

    def _build_opus_prompt(
        self,
        enriched: list[EnrichedCandidate],
        context: PipelineContext,
    ) -> str:
        """Build the user message for Opus with full context."""
        # Portfolio context.
        portfolio_section = json.dumps(
            context.portfolio_positions, indent=2, default=str
        )

        # Evaluation weights.
        weights_section = json.dumps(self._eval_weights, indent=2)

        # Learnings from memory.
        learnings = {}
        if self._memory is not None:
            try:
                learnings = self._memory.get_learnings()
            except Exception:
                pass
        learnings_section = json.dumps(learnings, indent=2, default=str)

        # Candidate data.
        candidates_data = []
        for ec in enriched:
            candidates_data.append({
                "ticker": ec.ticker,
                "name": ec.name,
                "exchange": ec.exchange,
                "sector": ec.sector,
                "asset_type": ec.asset_type,
                "currency": ec.currency,
                "gbp_price": ec.gbp_price,
                "fundamentals": ec.fundamentals,
                "technicals": ec.technicals,
                "news_sentiment": ec.news_sentiment,
            })
        candidates_section = json.dumps(candidates_data, indent=2, default=str)

        # Output schema.
        schema = {
            "candidates": [
                {
                    "ticker": "STRING",
                    "scores": {
                        "growth": "FLOAT 0.0-1.0",
                        "diversification": "FLOAT 0.0-1.0",
                        "risk_adjusted": "FLOAT 0.0-1.0",
                        "income": "FLOAT 0.0-1.0",
                        "composite": "FLOAT = weighted sum",
                    },
                    "reasoning": {
                        "growth": "STRING",
                        "diversification": "STRING",
                        "risk_adjusted": "STRING",
                        "income": "STRING",
                    },
                    "conviction": "HIGH | MEDIUM | LOW",
                    "risks": ["STRING"],
                    "position_suggestion": "STRING",
                }
            ],
            "portfolio_level_commentary": "STRING",
            "regime_assessment": "STRING",
        }

        return f"""\
## Existing Portfolio (LIVE -- read at runtime, not hardcoded)
{portfolio_section}

## Evaluation Weights
{weights_section}

Composite = (growth * {self._eval_weights.get('growth_score', 0.4)}) + \
(diversification * {self._eval_weights.get('diversification_score', 0.25)}) + \
(risk_adjusted * {self._eval_weights.get('risk_adjusted_score', 0.2)}) + \
(income * {self._eval_weights.get('income_score', 0.15)})

## Learnings from Previous Cycles
{learnings_section}

## Candidate Data
{candidates_section}

## Required Output Schema
{json.dumps(schema, indent=2)}

Score each candidate. Return ONLY valid JSON matching the schema above.
No markdown, no preamble, no commentary outside the JSON structure.
"""

    def _call_opus(self, user_message: str) -> dict[str, Any]:
        """Call the Claude Opus API and return the parsed response."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        response = client.messages.create(
            model=self._model,
            max_tokens=self._opus_config.get("max_tokens", 4096),
            system=ANALYST_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        # Track token usage.
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens

        if self._memory is not None:
            try:
                cycle_id = ""
                if hasattr(self._memory, "_load_unlocked"):
                    state = self._memory.load()
                    cycle_id = state.get("last_cycle_id", "")
                self._memory.log_opus_usage(
                    cycle_id, tokens_in, tokens_out, self._model
                )
            except Exception as e:
                logger.warning("Failed to log Opus usage: %s", e)

        logger.info(
            "Opus response: %d input tokens, %d output tokens",
            tokens_in,
            tokens_out,
        )

        # Extract text content.
        text = response.content[0].text.strip()

        # Strip markdown fences if present.
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences).
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        return json.loads(text)

    def _parse_opus_response(
        self,
        response: dict[str, Any],
        enriched: list[EnrichedCandidate],
    ) -> list[ScoredCandidate]:
        """Parse Opus JSON response into ScoredCandidate objects."""
        enriched_map = {ec.ticker: ec for ec in enriched}
        scored: list[ScoredCandidate] = []

        for candidate_data in response.get("candidates", []):
            ticker = candidate_data.get("ticker", "")
            ec = enriched_map.get(ticker)

            if ec is None:
                logger.warning(
                    "Opus returned ticker %s not in enriched set -- skipping",
                    ticker,
                )
                continue

            scores = candidate_data.get("scores", {})
            reasoning = candidate_data.get("reasoning", {})

            # Validate scores.
            if not self._validate_scores(scores):
                logger.warning(
                    "Invalid scores for %s: %s -- skipping", ticker, scores
                )
                continue

            scored.append(ScoredCandidate.from_enriched(
                ec,
                scores=scores,
                reasoning=reasoning,
                conviction=candidate_data.get("conviction", "MEDIUM"),
                risks=candidate_data.get("risks", []),
                position_suggestion=candidate_data.get(
                    "position_suggestion", ""
                ),
            ))

        return scored

    def _validate_scores(self, scores: dict[str, float]) -> bool:
        """Validate that all scores are in [0.0, 1.0] and composite is correct."""
        required = ["growth", "diversification", "risk_adjusted", "income", "composite"]

        for key in required:
            if key not in scores:
                return False
            val = scores[key]
            if not isinstance(val, (int, float)):
                return False
            if val < 0.0 or val > 1.0:
                return False

        # Verify composite is the weighted sum (within tolerance).
        expected_composite = (
            scores["growth"] * self._eval_weights.get("growth_score", 0.4)
            + scores["diversification"] * self._eval_weights.get("diversification_score", 0.25)
            + scores["risk_adjusted"] * self._eval_weights.get("risk_adjusted_score", 0.2)
            + scores["income"] * self._eval_weights.get("income_score", 0.15)
        )

        if abs(scores["composite"] - expected_composite) > 0.02:
            logger.warning(
                "Composite mismatch: reported=%.4f, expected=%.4f",
                scores["composite"],
                expected_composite,
            )
            # Auto-correct the composite rather than rejecting.
            scores["composite"] = round(expected_composite, 4)

        return True


# ------------------------------------------------------------------
# Keyword fallback analyst (works offline)
# ------------------------------------------------------------------


class KeywordAnalyst(AbstractAnalyst):
    """Heuristic-based scoring when Opus is unavailable.

    Uses simple rules based on fundamentals and technicals to produce
    scores. Less nuanced than Opus but works fully offline.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._eval_weights = config.get("evaluation", {}).get("weights", {})

    def analyse(
        self,
        enriched: list[EnrichedCandidate],
        context: PipelineContext,
    ) -> list[ScoredCandidate]:
        """Score candidates using keyword heuristics."""
        portfolio_sectors = {
            p.get("sector", "").lower()
            for p in context.portfolio_positions
        }

        scored: list[ScoredCandidate] = []

        for ec in enriched:
            growth = self._score_growth(ec)
            diversification = self._score_diversification(
                ec, portfolio_sectors
            )
            risk_adjusted = self._score_risk_adjusted(ec)
            income = self._score_income(ec)

            composite = (
                growth * self._eval_weights.get("growth_score", 0.4)
                + diversification * self._eval_weights.get("diversification_score", 0.25)
                + risk_adjusted * self._eval_weights.get("risk_adjusted_score", 0.2)
                + income * self._eval_weights.get("income_score", 0.15)
            )

            scores = {
                "growth": round(growth, 4),
                "diversification": round(diversification, 4),
                "risk_adjusted": round(risk_adjusted, 4),
                "income": round(income, 4),
                "composite": round(composite, 4),
            }

            reasoning = {
                "growth": self._growth_reasoning(ec, growth),
                "diversification": self._diversification_reasoning(
                    ec, diversification, portfolio_sectors
                ),
                "risk_adjusted": self._risk_reasoning(ec, risk_adjusted),
                "income": self._income_reasoning(ec, income),
            }

            conviction = (
                "HIGH" if composite >= 0.70
                else "MEDIUM" if composite >= 0.55
                else "LOW"
            )

            scored.append(ScoredCandidate.from_enriched(
                ec,
                scores=scores,
                reasoning=reasoning,
                conviction=conviction,
                risks=self._identify_risks(ec),
                position_suggestion=self._suggest_position(ec, composite),
            ))

        return scored

    def _score_growth(self, ec: EnrichedCandidate) -> float:
        """Score growth potential from fundamentals."""
        earnings_growth = ec.fundamentals.get("earnings_growth_yoy", 0) or 0
        revenue_growth = ec.fundamentals.get("revenue_growth_yoy", 0) or 0

        if earnings_growth > 0.25:
            score = 0.85
        elif earnings_growth > 0.15:
            score = 0.70
        elif earnings_growth > 0.05:
            score = 0.50
        else:
            score = 0.30

        # Boost for strong revenue growth.
        if revenue_growth > 0.20:
            score = min(1.0, score + 0.10)

        # RSI momentum bonus.
        rsi = ec.technicals.get("rsi_14", 50)
        if 40 < rsi < 70:
            score = min(1.0, score + 0.05)

        return min(1.0, max(0.0, score))

    def _score_diversification(
        self,
        ec: EnrichedCandidate,
        portfolio_sectors: set[str],
    ) -> float:
        """Score diversification benefit relative to existing portfolio."""
        sector = ec.sector.lower()

        if sector not in portfolio_sectors:
            score = 0.85
        else:
            score = 0.25

        # ETFs that track different themes get a bonus.
        if ec.asset_type == "etf" and sector not in portfolio_sectors:
            score = min(1.0, score + 0.10)

        return min(1.0, max(0.0, score))

    def _score_risk_adjusted(self, ec: EnrichedCandidate) -> float:
        """Score risk-adjusted return potential."""
        vol = ec.technicals.get("volatility_21d", 0.25)
        rsi = ec.technicals.get("rsi_14", 50)
        atr = ec.technicals.get("atr_14", 5)

        # Lower volatility is better.
        if vol < 0.15:
            score = 0.80
        elif vol < 0.25:
            score = 0.60
        elif vol < 0.40:
            score = 0.40
        else:
            score = 0.25

        # RSI not extreme is better.
        if 30 < rsi < 70:
            score = min(1.0, score + 0.10)

        return min(1.0, max(0.0, score))

    def _score_income(self, ec: EnrichedCandidate) -> float:
        """Score income potential from dividend data."""
        div_yield = ec.fundamentals.get("dividend_yield", 0) or 0

        if div_yield > 0.04:
            return 0.80
        elif div_yield > 0.02:
            return 0.50
        elif div_yield > 0.01:
            return 0.30
        else:
            return 0.10

    def _growth_reasoning(
        self, ec: EnrichedCandidate, score: float,
    ) -> str:
        """Generate plain English growth reasoning."""
        eg = ec.fundamentals.get("earnings_growth_yoy", 0) or 0
        rg = ec.fundamentals.get("revenue_growth_yoy", 0) or 0
        return (
            f"Earnings growth {eg:.0%}, revenue growth {rg:.0%}. "
            f"Score: {score:.2f}."
        )

    def _diversification_reasoning(
        self,
        ec: EnrichedCandidate,
        score: float,
        portfolio_sectors: set[str],
    ) -> str:
        """Generate plain English diversification reasoning."""
        sector = ec.sector.lower()
        if sector not in portfolio_sectors:
            return (
                f"{ec.sector} is not represented in the current portfolio. "
                f"This adds genuine diversification. Score: {score:.2f}."
            )
        return (
            f"{ec.sector} is already held in the portfolio. "
            f"Limited diversification benefit. Score: {score:.2f}."
        )

    def _risk_reasoning(
        self, ec: EnrichedCandidate, score: float,
    ) -> str:
        """Generate plain English risk reasoning."""
        vol = ec.technicals.get("volatility_21d", 0)
        return (
            f"21-day volatility: {vol:.1%}. "
            f"Risk-adjusted score: {score:.2f}."
        )

    def _income_reasoning(
        self, ec: EnrichedCandidate, score: float,
    ) -> str:
        """Generate plain English income reasoning."""
        dy = ec.fundamentals.get("dividend_yield", 0) or 0
        return f"Dividend yield: {dy:.1%}. Income score: {score:.2f}."

    def _identify_risks(self, ec: EnrichedCandidate) -> list[str]:
        """Identify key risks for the candidate."""
        risks: list[str] = []
        pe = ec.fundamentals.get("pe_ratio")
        if pe is not None and pe > 40:
            risks.append(f"High valuation at {pe:.0f}x earnings")
        rsi = ec.technicals.get("rsi_14", 50)
        if rsi > 70:
            risks.append("RSI above 70 -- potentially overbought")
        if rsi < 30:
            risks.append("RSI below 30 -- may be falling for a reason")
        if ec.currency != "GBP":
            risks.append(f"Currency risk: {ec.currency}/GBP exposure")
        return risks

    def _suggest_position(
        self, ec: EnrichedCandidate, composite: float,
    ) -> str:
        """Suggest a position size in plain English."""
        if composite >= 0.70:
            return "3-5% of capital as a conviction position"
        elif composite >= 0.55:
            return "1-3% of capital as a starter position"
        else:
            return "Watch only -- score below threshold"
