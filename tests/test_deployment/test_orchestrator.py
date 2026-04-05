"""Tests for src/deployment/orchestrator.py."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from src.deployment import (
    Candidate,
    CycleResult,
    EnrichedCandidate,
    ScoredCandidate,
    SizedPosition,
    TradeOrder,
)
from src.deployment.orchestrator import DeploymentPipeline


@pytest.fixture
def mock_scanner():
    """Scanner that returns synthetic candidates."""
    scanner = MagicMock()
    scanner.scan.return_value = [
        Candidate("NVDA", "NVIDIA", "NASDAQ", "Technology", 2.8e12, "us_growth", 50e6),
        Candidate("XOM", "Exxon", "NYSE", "Energy", 400e9, "contrarian_value", 20e6),
    ]
    return scanner


@pytest.fixture
def mock_enricher():
    """Enricher that returns enriched candidates."""
    enricher = MagicMock()
    enricher.enrich.return_value = [
        EnrichedCandidate(
            ticker="NVDA", name="NVIDIA", exchange="NASDAQ",
            sector="Technology", market_cap_usd=2.8e12,
            source_universe="us_growth", gbp_price=100.0,
        ),
        EnrichedCandidate(
            ticker="XOM", name="Exxon", exchange="NYSE",
            sector="Energy", market_cap_usd=400e9,
            source_universe="contrarian_value", gbp_price=85.0,
        ),
    ]
    return enricher


@pytest.fixture
def mock_analyst():
    """Analyst that returns scored candidates."""
    analyst = MagicMock()
    analyst.analyse.return_value = [
        ScoredCandidate(
            ticker="NVDA", name="NVIDIA", exchange="NASDAQ",
            sector="Technology", market_cap_usd=2.8e12,
            source_universe="us_growth", gbp_price=100.0,
            scores={"growth": 0.85, "diversification": 0.70, "risk_adjusted": 0.65, "income": 0.30, "composite": 0.72},
            conviction="HIGH",
        ),
        ScoredCandidate(
            ticker="XOM", name="Exxon", exchange="NYSE",
            sector="Energy", market_cap_usd=400e9,
            source_universe="contrarian_value", gbp_price=85.0,
            scores={"growth": 0.60, "diversification": 0.80, "risk_adjusted": 0.70, "income": 0.55, "composite": 0.66},
            conviction="MEDIUM",
        ),
    ]
    return analyst


@pytest.fixture
def mock_sizer():
    """Sizer that returns sized positions."""
    sizer = MagicMock()
    sizer.size.return_value = [
        SizedPosition(
            ticker="NVDA", name="NVIDIA", exchange="NASDAQ",
            sector="Technology", gbp_price=100.0, shares=3,
            gbp_amount=300.0, pct_of_capital=0.06, order_type="LIMIT",
            composite_score=0.72, conviction="HIGH",
        ),
    ]
    return sizer


@pytest.fixture
def mock_queue():
    """Queue manager that returns trade orders."""
    queue = MagicMock()
    queue.generate_orders.return_value = [
        TradeOrder(
            order_id="DEP-TEST0001", ticker="NVDA",
            direction="BUY", quantity=3,
            order_type="LIMIT", limit_price=100.50,
        ),
    ]
    queue.cancel_all.return_value = 1
    return queue


@pytest.fixture
def mock_memory():
    """Memory manager mock."""
    memory = MagicMock()
    memory.load.return_value = {}
    memory.get_learnings.return_value = {}
    memory.get_rejected_tickers.return_value = []
    return memory


@pytest.fixture
def pipeline(
    deployment_config,
    mock_scanner, mock_enricher, mock_analyst,
    mock_sizer, mock_queue, mock_memory,
) -> DeploymentPipeline:
    """Pipeline with all mocked components."""
    return DeploymentPipeline(
        config=deployment_config,
        memory=mock_memory,
        scanner=mock_scanner,
        enricher=mock_enricher,
        analyst=mock_analyst,
        sizer=mock_sizer,
        queue_manager=mock_queue,
    )


def _auto_approve(pipe: DeploymentPipeline, delay: float = 0.1) -> None:
    """Background thread that auto-approves checkpoint."""
    def approve():
        for _ in range(100):
            if pipe.state == "AWAITING_APPROVAL":
                pipe.approve_checkpoint()
                return
            time.sleep(delay)
    t = threading.Thread(target=approve, daemon=True)
    t.start()


# ------------------------------------------------------------------
# Basic pipeline execution
# ------------------------------------------------------------------


class TestPipelineExecution:
    """Tests for full pipeline execution."""

    def test_full_cycle_completes(self, pipeline):
        """Full cycle runs to COMPLETE with auto-approve."""
        _auto_approve(pipeline)
        result = pipeline.run_cycle(
            cycle_id="test_001", capital_gbp=5000.0,
        )
        assert result is not None
        assert result.status == "COMPLETE"
        assert result.cycle_id == "test_001"

    def test_candidates_scanned_counted(self, pipeline):
        """Scanned candidate count is reported."""
        _auto_approve(pipeline)
        result = pipeline.run_cycle(
            cycle_id="test_002", capital_gbp=5000.0,
        )
        assert result.candidates_scanned == 2

    def test_candidates_enriched_counted(self, pipeline):
        """Enriched candidate count is reported."""
        _auto_approve(pipeline)
        result = pipeline.run_cycle(
            cycle_id="test_003", capital_gbp=5000.0,
        )
        assert result.candidates_enriched == 2

    def test_candidates_scored_counted(self, pipeline):
        """Scored candidate count is reported."""
        _auto_approve(pipeline)
        result = pipeline.run_cycle(
            cycle_id="test_004", capital_gbp=5000.0,
        )
        assert result.candidates_scored == 2

    def test_trades_queued_counted(self, pipeline):
        """Queued trade count is reported."""
        _auto_approve(pipeline)
        result = pipeline.run_cycle(
            cycle_id="test_005", capital_gbp=5000.0,
        )
        assert result.trades_queued == 1

    def test_duration_positive(self, pipeline):
        """Duration is positive."""
        _auto_approve(pipeline)
        result = pipeline.run_cycle(
            cycle_id="test_006", capital_gbp=5000.0,
        )
        assert result.duration_seconds > 0

    def test_memory_start_end_called(self, pipeline, mock_memory):
        """Memory start_cycle and end_cycle are called."""
        _auto_approve(pipeline)
        pipeline.run_cycle(cycle_id="test_007", capital_gbp=5000.0)
        mock_memory.start_cycle.assert_called_once_with("test_007")
        mock_memory.end_cycle.assert_called_once()

    def test_memory_learn_called(self, pipeline, mock_memory):
        """Memory learn_from_cycle is called."""
        _auto_approve(pipeline)
        pipeline.run_cycle(cycle_id="test_008", capital_gbp=5000.0)
        mock_memory.learn_from_cycle.assert_called_once()


# ------------------------------------------------------------------
# State management
# ------------------------------------------------------------------


class TestStateManagement:
    """Tests for pipeline state transitions."""

    def test_initial_state_is_idle(self, pipeline):
        """Pipeline starts in IDLE state."""
        assert pipeline.state == "IDLE"

    def test_state_returns_to_complete(self, pipeline):
        """Pipeline ends in COMPLETE state after successful cycle."""
        _auto_approve(pipeline)
        pipeline.run_cycle(cycle_id="test_009", capital_gbp=5000.0)
        assert pipeline.state == "COMPLETE"

    def test_cannot_start_while_running(self, pipeline):
        """Cannot start a new cycle while one is running."""
        # Manually set state to SCANNING.
        pipeline._set_state("SCANNING")
        result = pipeline.run_cycle(
            cycle_id="test_010", capital_gbp=5000.0,
        )
        assert result is None


# ------------------------------------------------------------------
# Checkpoint (approval gate)
# ------------------------------------------------------------------


class TestCheckpoint:
    """Tests for the AWAITING_APPROVAL checkpoint."""

    def test_pipeline_waits_at_checkpoint(self, pipeline):
        """Pipeline blocks at AWAITING_APPROVAL until approved."""
        # Run in background.
        result_holder = [None]

        def run():
            result_holder[0] = pipeline.run_cycle(
                cycle_id="test_011", capital_gbp=5000.0,
            )

        t = threading.Thread(target=run, daemon=True)
        t.start()

        # Wait for checkpoint.
        for _ in range(50):
            if pipeline.state == "AWAITING_APPROVAL":
                break
            time.sleep(0.05)

        assert pipeline.state == "AWAITING_APPROVAL"

        # Approve and wait for completion.
        pipeline.approve_checkpoint()
        t.join(timeout=5.0)
        assert result_holder[0] is not None
        assert result_holder[0].status == "COMPLETE"


# ------------------------------------------------------------------
# Emergency halt
# ------------------------------------------------------------------


class TestEmergencyHalt:
    """Tests for emergency halt."""

    def test_halt_cancels_orders(self, pipeline, mock_queue):
        """Halt cancels pending orders via queue."""
        cancelled = pipeline.halt()
        assert cancelled == 1
        mock_queue.cancel_all.assert_called_once()

    def test_halt_sets_state(self, pipeline):
        """Halt sets pipeline state to HALTED."""
        pipeline.halt()
        assert pipeline.state == "HALTED"

    def test_halt_during_approval_unblocks(self, pipeline):
        """Halt unblocks the approval wait."""
        result_holder = [None]

        def run():
            result_holder[0] = pipeline.run_cycle(
                cycle_id="test_halt", capital_gbp=5000.0,
            )

        t = threading.Thread(target=run, daemon=True)
        t.start()

        # Wait for checkpoint.
        for _ in range(50):
            if pipeline.state == "AWAITING_APPROVAL":
                break
            time.sleep(0.05)

        pipeline.halt()
        t.join(timeout=5.0)
        assert result_holder[0] is not None
        assert result_holder[0].status == "HALTED"


# ------------------------------------------------------------------
# Retry behaviour
# ------------------------------------------------------------------


class TestRetryBehaviour:
    """Tests for step retry logic."""

    def test_retries_on_failure(self, pipeline, mock_scanner):
        """Scanner failure triggers retry."""
        call_count = 0

        def flaky_scan(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Transient error")
            return [
                Candidate("NVDA", "NVIDIA", "NASDAQ", "Tech", 2e12, "us_growth", 50e6),
            ]

        mock_scanner.scan.side_effect = flaky_scan
        _auto_approve(pipeline)
        result = pipeline.run_cycle(
            cycle_id="test_retry", capital_gbp=5000.0,
        )
        assert result.status == "COMPLETE"
        assert call_count == 2

    def test_all_retries_exhausted_escalates(
        self, pipeline, mock_scanner,
    ):
        """Exhausting all retries escalates to FAILED."""
        mock_scanner.scan.side_effect = RuntimeError("Permanent failure")
        _auto_approve(pipeline)
        result = pipeline.run_cycle(
            cycle_id="test_fail", capital_gbp=5000.0,
        )
        # Pipeline should handle the escalation gracefully.
        assert result is not None
        # The scan step returns empty data on escalation.
        assert result.candidates_scanned == 0


# ------------------------------------------------------------------
# Capital resolution
# ------------------------------------------------------------------


class TestCapitalResolution:
    """Tests for capital resolution."""

    def test_uses_provided_capital(self, pipeline):
        """Provided capital is used directly."""
        _auto_approve(pipeline)
        result = pipeline.run_cycle(
            cycle_id="test_cap", capital_gbp=1234.0,
        )
        assert pipeline.context.capital_available_gbp == 1234.0

    def test_falls_back_to_config_default(self, pipeline):
        """Without broker or explicit capital, uses config default."""
        _auto_approve(pipeline)
        result = pipeline.run_cycle(cycle_id="test_default")
        # Config default is 500.0.
        assert pipeline.context.capital_available_gbp == 500.0
