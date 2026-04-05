"""Capital Deployment Engine -- autonomous scanning, analysis, and execution.

Pipeline: SCAN -> ENRICH -> ANALYSE (Opus) -> CHECKPOINT -> SIZE -> QUEUE -> PERSIST -> LEARN

This module provides:
- Dataclasses for each pipeline stage
- Abstract interfaces for pluggable implementations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


# ------------------------------------------------------------------
# Dataclasses -- one per pipeline stage
# ------------------------------------------------------------------

@dataclass
class Candidate:
    """Raw candidate from the scanner."""

    ticker: str
    name: str
    exchange: str
    sector: str
    market_cap_usd: float
    source_universe: str
    avg_daily_volume: float = 0.0
    currency: str = "USD"
    asset_type: str = "stock"  # "stock" or "etf"


@dataclass
class EnrichedCandidate:
    """Candidate enriched with fundamentals, technicals, and sentiment."""

    # Identity (carried from Candidate)
    ticker: str
    name: str
    exchange: str
    sector: str
    market_cap_usd: float
    source_universe: str
    avg_daily_volume: float = 0.0
    currency: str = "USD"
    asset_type: str = "stock"

    # Enrichment data
    fundamentals: dict[str, Any] = field(default_factory=dict)
    technicals: dict[str, Any] = field(default_factory=dict)
    news_sentiment: dict[str, Any] = field(default_factory=dict)
    gbp_price: float = 0.0
    raw_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_candidate(
        cls,
        candidate: Candidate,
        fundamentals: dict[str, Any] | None = None,
        technicals: dict[str, Any] | None = None,
        news_sentiment: dict[str, Any] | None = None,
        gbp_price: float = 0.0,
        raw_data: dict[str, Any] | None = None,
    ) -> EnrichedCandidate:
        """Create an EnrichedCandidate from a Candidate plus enrichment data."""
        return cls(
            ticker=candidate.ticker,
            name=candidate.name,
            exchange=candidate.exchange,
            sector=candidate.sector,
            market_cap_usd=candidate.market_cap_usd,
            source_universe=candidate.source_universe,
            avg_daily_volume=candidate.avg_daily_volume,
            currency=candidate.currency,
            asset_type=candidate.asset_type,
            fundamentals=fundamentals or {},
            technicals=technicals or {},
            news_sentiment=news_sentiment or {},
            gbp_price=gbp_price,
            raw_data=raw_data or {},
        )


@dataclass
class ScoredCandidate:
    """Candidate scored by the analyst (Opus or keyword fallback)."""

    # Identity (carried forward)
    ticker: str
    name: str
    exchange: str
    sector: str
    market_cap_usd: float
    source_universe: str
    avg_daily_volume: float = 0.0
    currency: str = "USD"
    asset_type: str = "stock"

    # Enrichment (carried forward)
    fundamentals: dict[str, Any] = field(default_factory=dict)
    technicals: dict[str, Any] = field(default_factory=dict)
    news_sentiment: dict[str, Any] = field(default_factory=dict)
    gbp_price: float = 0.0

    # Scoring
    scores: dict[str, float] = field(default_factory=dict)
    reasoning: dict[str, str] = field(default_factory=dict)
    conviction: str = "MEDIUM"  # HIGH / MEDIUM / LOW
    risks: list[str] = field(default_factory=list)
    position_suggestion: str = ""

    @property
    def composite_score(self) -> float:
        """Return the composite score, or 0.0 if not computed."""
        return self.scores.get("composite", 0.0)

    @classmethod
    def from_enriched(
        cls,
        enriched: EnrichedCandidate,
        scores: dict[str, float] | None = None,
        reasoning: dict[str, str] | None = None,
        conviction: str = "MEDIUM",
        risks: list[str] | None = None,
        position_suggestion: str = "",
    ) -> ScoredCandidate:
        """Create a ScoredCandidate from an EnrichedCandidate plus scoring data."""
        return cls(
            ticker=enriched.ticker,
            name=enriched.name,
            exchange=enriched.exchange,
            sector=enriched.sector,
            market_cap_usd=enriched.market_cap_usd,
            source_universe=enriched.source_universe,
            avg_daily_volume=enriched.avg_daily_volume,
            currency=enriched.currency,
            asset_type=enriched.asset_type,
            fundamentals=enriched.fundamentals,
            technicals=enriched.technicals,
            news_sentiment=enriched.news_sentiment,
            gbp_price=enriched.gbp_price,
            scores=scores or {},
            reasoning=reasoning or {},
            conviction=conviction,
            risks=risks or [],
            position_suggestion=position_suggestion,
        )


@dataclass
class SizedPosition:
    """A scored candidate with a concrete position size."""

    ticker: str
    name: str
    exchange: str
    sector: str
    asset_type: str = "stock"
    gbp_price: float = 0.0

    # Sizing
    shares: int = 0
    gbp_amount: float = 0.0
    pct_of_capital: float = 0.0
    order_type: str = "LIMIT"  # LIMIT or MARKET

    # Carried scoring
    composite_score: float = 0.0
    conviction: str = "MEDIUM"
    reasoning_summary: str = ""


@dataclass
class TradeOrder:
    """An executable trade order ready for IB submission."""

    order_id: str
    ticker: str
    direction: str = "BUY"
    quantity: int = 0
    order_type: str = "LIMIT"
    limit_price: float | None = None
    reasoning_summary: str = ""

    # State
    status: str = "PENDING_APPROVAL"  # PENDING_APPROVAL / APPROVED / REJECTED / SUBMITTED / FILLED / FAILED
    rejection_reason: str = ""

    # Execution results (populated after fill)
    fill_price: float | None = None
    fill_quantity: int | None = None
    commission: float | None = None
    fill_timestamp: str | None = None
    slippage: float | None = None  # fill_price - limit_price


@dataclass
class PipelineContext:
    """Mutable context passed through the pipeline."""

    cycle_id: str
    memory: Any  # MemoryManager (avoid circular import)
    config: dict[str, Any]
    portfolio_positions: list[dict[str, Any]] = field(default_factory=list)

    # Pipeline state
    current_step: str = "IDLE"
    candidates: list[Candidate] = field(default_factory=list)
    enriched: list[EnrichedCandidate] = field(default_factory=list)
    scored: list[ScoredCandidate] = field(default_factory=list)
    sized: list[SizedPosition] = field(default_factory=list)
    orders: list[TradeOrder] = field(default_factory=list)

    # Metadata
    started_at: str = ""
    capital_available_gbp: float = 0.0
    errors: list[dict[str, Any]] = field(default_factory=list)
    opus_tokens_used: int = 0


@dataclass
class StepResult:
    """Result from a single pipeline step execution."""

    status: Literal["OK", "RETRY", "ESCALATE"]
    data: Any = None
    error: str | None = None
    attempts: int = 1
    duration_seconds: float = 0.0


@dataclass
class CycleResult:
    """Summary of a completed deployment cycle."""

    cycle_id: str
    status: Literal["COMPLETE", "AWAITING_APPROVAL", "FAILED", "ESCALATED", "HALTED"]
    candidates_scanned: int = 0
    candidates_enriched: int = 0
    candidates_scored: int = 0
    candidates_above_threshold: int = 0
    trades_queued: int = 0
    total_gbp_deployed: float = 0.0
    errors: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    opus_tokens_used: int = 0


# ------------------------------------------------------------------
# Abstract interfaces -- contracts for each pipeline stage
# ------------------------------------------------------------------

class AbstractScanner(ABC):
    """Contract for universe scanning."""

    @abstractmethod
    def scan(self, exclusions: list[str] | None = None) -> list[Candidate]:
        """Scan configured universes and return filtered candidates.

        Args:
            exclusions: Tickers to exclude (portfolio + cooldowns + structural).

        Returns:
            Deduplicated list of candidates passing all filters.
        """
        ...


class AbstractEnricher(ABC):
    """Contract for candidate enrichment."""

    @abstractmethod
    def enrich(self, candidates: list[Candidate]) -> list[EnrichedCandidate]:
        """Enrich candidates with fundamentals, technicals, and sentiment.

        Args:
            candidates: Raw candidates from the scanner.

        Returns:
            Enriched candidates (some may be dropped if data fetch fails).
        """
        ...


class AbstractAnalyst(ABC):
    """Contract for candidate analysis and scoring."""

    @abstractmethod
    def analyse(
        self,
        enriched: list[EnrichedCandidate],
        context: PipelineContext,
    ) -> list[ScoredCandidate]:
        """Score enriched candidates against evaluation criteria.

        Args:
            enriched: Enriched candidates from the enricher.
            context: Pipeline context including portfolio and config.

        Returns:
            Scored candidates with reasoning.
        """
        ...


class AbstractSizer(ABC):
    """Contract for position sizing."""

    @abstractmethod
    def size(
        self,
        scored: list[ScoredCandidate],
        capital_gbp: float,
        constraints: dict[str, Any] | None = None,
    ) -> list[SizedPosition]:
        """Size positions for approved candidates.

        Args:
            scored: Scored candidates above threshold.
            capital_gbp: Available capital in GBP (queried from IB).
            constraints: Tier-specific constraints.

        Returns:
            Sized positions respecting all constraints.
        """
        ...


class AbstractQueueManager(ABC):
    """Contract for trade queue management."""

    @abstractmethod
    def generate_orders(
        self, sized: list[SizedPosition],
    ) -> list[TradeOrder]:
        """Generate executable trade orders from sized positions.

        Args:
            sized: Sized positions from the sizer.

        Returns:
            Trade orders ready for approval.
        """
        ...
