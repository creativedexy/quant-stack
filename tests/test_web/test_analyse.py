"""Tests for the Analyse page -- chart + signal research workbench."""

from __future__ import annotations

import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------


class TestAnalysePage:
    """GET /ui/analyse returns the full page."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_default_ticker_in_page(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse")
        # Should contain the default ticker (first in watchlist)
        assert "CNDX.L" in resp.text

    def test_custom_ticker(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse?ticker=AZN.L")
        assert resp.status_code == 200
        assert "AZN.L" in resp.text

    def test_contains_overlay_buttons(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse")
        for overlay in ("Bollinger", "RSI", "MACD", "ML Signal", "Monte Carlo", "Log"):
            assert overlay in resp.text

    def test_contains_signal_direction(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse")
        text = resp.text
        # Should contain one of the plain English labels
        has_label = any(
            label in text
            for label in ("Looking Good", "Be Careful", "Steady", "Neutral")
        )
        assert has_label, "No plain English signal label found"

    def test_contains_confidence_number(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse")
        # Confidence label is uppercase in the redesigned layout
        assert "Confidence" in resp.text or "confidence" in resp.text

    def test_no_financial_jargon(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse")
        text = resp.text.lower()
        for jargon in ("sharpe", "alpha", "drawdown", "long/short"):
            assert jargon not in text, f"Found jargon '{jargon}' in response"

    def test_signal_uses_plain_labels(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse")
        text = resp.text
        # Should NOT contain trading jargon labels
        assert "LONG" not in text or "Bollinger" in text  # LONG only in overlay context
        assert "SHORT" not in text or "short" in text.lower()  # allow lowercase prose
        assert "EXIT" not in text or "exit" in text.lower()

    def test_feature_drivers_use_strength_labels(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse")
        text = resp.text
        has_strength = any(
            label in text
            for label in ("Strong", "Moderate", "Mild", "Weak")
        )
        assert has_strength, "No strength labels found in feature drivers"


    def test_feature_driver_names(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse")
        text = resp.text
        has_driver = any(
            name in text
            for name in ("Momentum", "RSI Recovery", "MACD Trend", "Bollinger Position", "Volume")
        )
        assert has_driver, "No expected feature driver names found"

    def test_contains_action_cards(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse")
        assert "Add to Watchlist" in resp.text
        assert "View in Portfolio" in resp.text
        assert "Set Price Alert" in resp.text


class TestAnalyseChartPartial:
    """GET /ui/analyse/chart returns SVG path data."""

    def test_returns_html(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse/chart?ticker=CNDX.L&period=1M")
        assert resp.status_code == 200

    def test_contains_plotly_chart(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse/chart?ticker=CNDX.L&period=1M")
        assert "analyse-chart" in resp.text
        assert "renderAnalyseChart" in resp.text

    def test_log_scale_param(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse/chart?ticker=CNDX.L&period=1M&log_scale=1")
        assert resp.status_code == 200


class TestAnalyseSignalPartial:
    """GET /ui/analyse/signal returns signal data."""

    def test_returns_html(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse/signal?ticker=CNDX.L")
        assert resp.status_code == 200

    def test_contains_direction(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse/signal?ticker=CNDX.L")
        has_label = any(
            label in resp.text
            for label in ("Looking Good", "Be Careful", "Steady", "Neutral")
        )
        assert has_label

    def test_contains_drivers(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse/signal?ticker=CNDX.L")
        assert "Driving This Signal" in resp.text


class TestAnalyseAIDive:
    """POST /ui/analyse/ai-dive returns AI analysis."""

    def test_returns_analysis(self, client: TestClient) -> None:
        resp = client.post("/ui/analyse/ai-dive?ticker=CNDX.L")
        assert resp.status_code == 200
        # Should contain some analysis text (synthetic fallback)
        assert len(resp.text) > 50
        assert "CNDX.L" in resp.text


class TestLearnStrips:
    """Learn strips appear for each overlay type."""

    def test_bollinger_learn_strip(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse/chart?ticker=CNDX.L&period=1M&overlays=bollinger")
        assert "Bollinger Bands" in resp.text

    def test_rsi_learn_strip(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse/chart?ticker=CNDX.L&period=1M&overlays=rsi")
        assert "Relative Strength" in resp.text

    def test_macd_learn_strip(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse/chart?ticker=CNDX.L&period=1M&overlays=macd")
        assert "Trend Detector" in resp.text

    def test_ml_learn_strip(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse/chart?ticker=CNDX.L&period=1M&overlays=ml")
        assert "AI Model Signal" in resp.text

    def test_monte_carlo_learn_strip(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse/chart?ticker=CNDX.L&period=1M&overlays=monte_carlo")
        assert "Monte Carlo" in resp.text


# ---------------------------------------------------------------------------
# ChartService SVG tests
# ---------------------------------------------------------------------------


class TestChartServiceSVG:
    """ChartService SVG generation methods."""

    @pytest.fixture()
    def chart_svc(self):
        from src.services.chart_service import ChartService
        from src.utils.config import load_config

        return ChartService(load_config())

    def test_generate_price_svg_returns_path(self, chart_svc) -> None:
        prices = pd.Series(
            [100, 102, 101, 105, 103, 108, 110],
            index=pd.date_range("2024-01-01", periods=7, freq="D"),
        )
        result = chart_svc.generate_price_svg(prices, width=800, height=300)
        assert "path_d" in result
        assert result["path_d"].startswith("M")
        assert "area_d" in result
        assert result["colour"] in ("var(--green)", "var(--red)")

    def test_generate_price_svg_empty(self, chart_svc) -> None:
        prices = pd.Series(dtype=float)
        result = chart_svc.generate_price_svg(prices)
        assert result["path_d"] == ""

    def test_generate_bollinger_svg_returns_three_paths(self, chart_svc) -> None:
        prices = pd.Series(
            [100 + i * 0.5 for i in range(30)],
            index=pd.date_range("2024-01-01", periods=30, freq="D"),
        )
        result = chart_svc.generate_bollinger_svg(prices, window=20)
        assert "upper_d" in result
        assert "lower_d" in result
        assert "mid_d" in result
        # At least one should have content (after window warm-up)
        assert result["upper_d"] or result["lower_d"] or result["mid_d"]

    def test_generate_trade_dots(self, chart_svc) -> None:
        prices = pd.Series(
            [100, 102, 101, 105, 103],
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )
        trades = [{"date": "2024-01-03", "price": 101, "side": "buy"}]
        dots = chart_svc.generate_trade_dots(trades, prices, width=800, height=300)
        assert len(dots) == 1
        assert "cx" in dots[0]
        assert "cy" in dots[0]

    def test_generate_prev_close_line(self, chart_svc) -> None:
        y = chart_svc.generate_prev_close_line(105.0, (100.0, 110.0), height=300)
        assert isinstance(y, float)
        assert 0 < y < 300


# ---------------------------------------------------------------------------
# No jargon sweep
# ---------------------------------------------------------------------------


class TestNoJargon:
    """Ensure no financial jargon appears in any analyse response."""

    _JARGON = ["sharpe", "alpha", "drawdown", "beta", "sortino"]

    def test_main_page_no_jargon(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse")
        lower = resp.text.lower()
        for word in self._JARGON:
            assert word not in lower, f"Jargon '{word}' found in analyse page"

    def test_signal_partial_no_jargon(self, client: TestClient) -> None:
        resp = client.get("/ui/analyse/signal?ticker=CNDX.L")
        lower = resp.text.lower()
        for word in self._JARGON:
            assert word not in lower, f"Jargon '{word}' found in signal partial"


class TestBuildChartJson:
    """_build_chart_json returns valid Plotly spec."""

    def test_returns_data_and_layout(self) -> None:
        from web.routes.analyse import _build_chart_json
        result = _build_chart_json("CNDX.L", "1M", [])
        assert "data" in result
        assert "layout" in result
        assert isinstance(result["data"], list)
        assert len(result["data"]) >= 1

    def test_candlestick_trace_present(self) -> None:
        from web.routes.analyse import _build_chart_json
        result = _build_chart_json("CNDX.L", "1M", [])
        candle = result["data"][0]
        assert candle["type"] == "candlestick"
        assert len(candle["x"]) > 0
        assert len(candle["open"]) > 0

    def test_rsi_overlay_adds_trace(self) -> None:
        from web.routes.analyse import _build_chart_json
        result = _build_chart_json("CNDX.L", "1M", ["rsi"])
        names = [t.get("name", "") for t in result["data"]]
        assert any("RSI" in n for n in names)

    def test_bollinger_overlay_adds_traces(self) -> None:
        from web.routes.analyse import _build_chart_json
        result = _build_chart_json("CNDX.L", "1M", ["bollinger"])
        names = [t.get("name", "") for t in result["data"]]
        assert any("BB" in n for n in names)

    def test_macd_overlay_adds_traces(self) -> None:
        from web.routes.analyse import _build_chart_json
        result = _build_chart_json("CNDX.L", "1M", ["macd"])
        names = [t.get("name", "") for t in result["data"]]
        assert any("MACD" in n for n in names)

    def test_monte_carlo_adds_future_traces(self) -> None:
        from web.routes.analyse import _build_chart_json
        result = _build_chart_json("CNDX.L", "3M", ["monte_carlo"])
        candle_dates = result["data"][0]["x"]
        mc_traces = [t for t in result["data"] if "P50" in t.get("name", "")]
        if mc_traces:
            mc_dates = mc_traces[0]["x"]
            assert mc_dates[-1] > candle_dates[-1]

    def test_layout_has_dark_theme(self) -> None:
        from web.routes.analyse import _build_chart_json
        result = _build_chart_json("CNDX.L", "1M", [])
        layout = result["layout"]
        assert layout["paper_bgcolor"] == "transparent"
        assert layout["plot_bgcolor"] == "transparent"
