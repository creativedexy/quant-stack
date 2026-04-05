"""Tests for the Session 14 design system: theme tokens, base nav,
partials, Jinja2 filters, and AIService synthetic fallbacks.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Theme CSS tokens
# ---------------------------------------------------------------------------

_THEME_CSS = (
    Path(__file__).resolve().parents[2] / "web" / "static" / "css" / "theme.css"
)


class TestThemeCSS:
    """Verify theme.css contains all required design tokens."""

    @pytest.fixture(autouse=True)
    def _load_css(self) -> None:
        self.css = _THEME_CSS.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "var",
        [
            "--bg",
            "--card",
            "--card-hi",
            "--border",
            "--border-hi",
            "--t1",
            "--t2",
            "--t3",
            "--green",
            "--green-dim",
            "--red",
            "--red-dim",
            "--amber",
            "--amber-dim",
            "--cyan",
            "--cyan-dim",
            "--purple",
            "--purple-dim",
            "--mono",
            "--sans",
        ],
    )
    def test_required_css_variable(self, var: str) -> None:
        assert var in self.css, f"Missing CSS variable {var}"

    @pytest.mark.parametrize(
        "cls",
        [
            ".text-green",
            ".text-red",
            ".text-amber",
            ".text-cyan",
            ".text-purple",
            ".text-muted",
            ".font-mono",
            ".card",
            ".dot-live",
            ".dot-action",
            ".dot-idle",
            ".chip-green",
            ".chip-red",
            ".chip-amber",
            ".ai-btn",
            ".ai-card",
            ".ai-card-green",
            ".ai-card-red",
            ".ai-card-amber",
            ".ai-card-purple",
            ".ai-card-cyan",
        ],
    )
    def test_required_css_class(self, cls: str) -> None:
        assert cls in self.css, f"Missing CSS class {cls}"


# ---------------------------------------------------------------------------
# Base template navigation
# ---------------------------------------------------------------------------


class TestBaseHTML:
    """Verify base.html renders the 5-tab nav correctly."""

    @pytest.fixture(autouse=True)
    def _setup_env(self) -> None:
        from jinja2 import Environment, FileSystemLoader

        templates_dir = (
            Path(__file__).resolve().parents[2] / "web" / "templates"
        )
        self.env = Environment(loader=FileSystemLoader(str(templates_dir)))
        # Register filters used by partials
        self.env.filters["gbp"] = lambda v: f"GBP{float(v):,.2f}" if v else "--"
        self.env.filters["pct"] = lambda v: f"+{float(v)*100:.2f}%" if v else "--"
        self.env.globals["lse_market"] = lambda: {
            "lse_open": True,
            "lse_time": "14:30",
        }

    def test_five_nav_tabs(self) -> None:
        tpl = self.env.get_template("base.html")
        html = tpl.render(
            request=_FakeRequest("/ui/overview"),
            active_tab="dashboard",
            tab_statuses={},
            css_v="0",
        )
        for tab in ["Dashboard", "Watchlist", "Portfolio", "Analyse", "Execute"]:
            assert tab in html, f"Nav tab '{tab}' not found"

    def test_active_tab_highlight(self) -> None:
        tpl = self.env.get_template("base.html")
        html = tpl.render(
            request=_FakeRequest("/ui/watchlist"),
            active_tab="watchlist",
            tab_statuses={},
            css_v="0",
        )
        # The active link should have the 'active' class
        assert 'class="nav-link active"' in html or "nav-link active" in html

    def test_lse_status_renders(self) -> None:
        tpl = self.env.get_template("base.html")
        html = tpl.render(
            request=_FakeRequest("/ui/overview"),
            active_tab="dashboard",
            tab_statuses={},
            css_v="0",
        )
        assert "LSE" in html

    def test_ticker_bar_block_exists(self) -> None:
        tpl = self.env.get_template("base.html")
        # The block should render (even if empty)
        html = tpl.render(
            request=_FakeRequest("/ui/overview"),
            active_tab="dashboard",
            tab_statuses={},
            css_v="0",
        )
        # Just confirm template renders without error
        assert "<nav" in html


# ---------------------------------------------------------------------------
# Ticker bar partial
# ---------------------------------------------------------------------------


class TestTickerBarPartial:
    """Verify ticker_bar.html renders with correct data."""

    @pytest.fixture(autouse=True)
    def _setup_env(self) -> None:
        from jinja2 import Environment, FileSystemLoader

        templates_dir = (
            Path(__file__).resolve().parents[2] / "web" / "templates"
        )
        self.env = Environment(loader=FileSystemLoader(str(templates_dir)))
        self.env.filters["gbp"] = lambda v: f"GBP{float(v):,.2f}" if v else "--"
        self.env.filters["pct"] = lambda v: f"+{float(v)*100:.2f}%" if v else "--"

    def test_renders_ticker_data(self) -> None:
        tpl = self.env.get_template("partials/ticker_bar.html")
        html = tpl.render(
            ticker="AZN.L",
            company_name="AstraZeneca",
            price=112.50,
            change_pct=0.023,
            logo_colour="#7B2D8E",
        )
        assert "AstraZeneca" in html
        assert "AZN.L" in html
        assert "GBP112.50" in html


# ---------------------------------------------------------------------------
# AI card partial
# ---------------------------------------------------------------------------


class TestAICardPartial:
    """Verify ai_card.html renders all colour variants."""

    @pytest.fixture(autouse=True)
    def _setup_env(self) -> None:
        from jinja2 import Environment, FileSystemLoader

        templates_dir = (
            Path(__file__).resolve().parents[2] / "web" / "templates"
        )
        self.env = Environment(loader=FileSystemLoader(str(templates_dir)))

    @pytest.mark.parametrize("colour", ["green", "red", "amber", "purple", "cyan"])
    def test_colour_variant(self, colour: str) -> None:
        tpl = self.env.get_template("partials/ai_card.html")
        html = tpl.render(
            colour=colour,
            icon_svg="<svg></svg>",
            title="Test Insight",
            text="Some explanation text.",
            chips=[{"label": "AAPL", "colour": "cyan"}],
        )
        assert f"ai-card-{colour}" in html
        assert "Test Insight" in html
        assert "AAPL" in html


# ---------------------------------------------------------------------------
# Jinja2 filters
# ---------------------------------------------------------------------------


class TestJinja2Filters:
    """Verify the custom template filters produce correct output."""

    def test_gbp_filter(self) -> None:
        from web.routes.ui import _gbp

        assert _gbp(1234.5) == "GBP1,234.50"
        assert _gbp(0) == "GBP0.00"
        assert _gbp(-50) == "-GBP50.00"

    def test_pct_filter(self) -> None:
        from web.routes.ui import _pct

        result = _pct(0.0312)
        assert "+3.12%" in result
        assert "text-green" in result

    def test_pct_negative(self) -> None:
        from web.routes.ui import _pct

        result = _pct(-0.05)
        assert "-5.00%" in result
        assert "text-red" in result

    def test_pct_plain(self) -> None:
        from web.routes.ui import _pct_plain

        assert _pct_plain(0.0312) == "+3.12%"
        assert _pct_plain(-0.05) == "-5.00%"

    def test_signal_label_looking_good(self) -> None:
        from web.routes.ui import _signal_label

        assert _signal_label(0.78) == "Looking Good"

    def test_signal_label_be_careful(self) -> None:
        from web.routes.ui import _signal_label

        assert _signal_label(0.28) == "Be Careful"

    def test_signal_label_steady(self) -> None:
        from web.routes.ui import _signal_label

        assert _signal_label(0.50) == "Steady"

    def test_signal_label_neutral(self) -> None:
        from web.routes.ui import _signal_label

        assert _signal_label(0.40) == "Neutral"

    def test_ago_filter(self) -> None:
        from datetime import datetime, timedelta, timezone

        from web.routes.ui import _ago

        now = datetime.now(timezone.utc)
        assert _ago(now - timedelta(seconds=10)) == "Just now"
        assert "hour" in _ago(now - timedelta(hours=2))
        assert "day" in _ago(now - timedelta(days=3))


# ---------------------------------------------------------------------------
# AIService
# ---------------------------------------------------------------------------


class TestAIService:
    """Verify AIService synthetic fallbacks return correct structures."""

    def test_analyse_ticker_returns_string(self) -> None:
        from src.services.ai_service import AIService

        svc = AIService()
        result = svc.analyse_ticker("AAPL", {"direction": "up", "confidence": 0.7}, {})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_review_portfolio_returns_list_of_dicts(self) -> None:
        from src.services.ai_service import AIService

        svc = AIService()
        result = svc.review_portfolio([], {}, [])
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert isinstance(item, dict)
            assert "colour" in item
            assert "title" in item
            assert "text" in item

    def test_review_execution_returns_list_of_dicts(self) -> None:
        from src.services.ai_service import AIService

        svc = AIService()
        result = svc.review_execution({}, [], [])
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert isinstance(item, dict)
            assert "colour" in item
            assert "title" in item
            assert "text" in item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in for a Starlette Request object."""

    def __init__(self, path: str = "/") -> None:
        self.url = _FakeURL(path)


class _FakeURL:
    def __init__(self, path: str) -> None:
        self.path = path
        self.query = ""
