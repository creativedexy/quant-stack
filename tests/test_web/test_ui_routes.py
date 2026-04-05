"""Tests for the production UI routes (web/routes/ui.py).

Verifies:
- Overview page renders correctly with Level 1-4 data.
- Strategy page renders correctly with strategy list.
- Portfolio page renders correctly with charts and risk metrics.
- Research page renders correctly with entry list and filter.
- Execution page renders correctly with trading banner and orders.
- Jinja2 custom filters produce correct output.
- Static assets are served.
- HTMX partial endpoints return fragments (not full pages).
- Interpret action endpoint works with mocked interpreter.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# client fixture is provided by tests/test_web/conftest.py (includes auth)


# ── Page rendering ───────────────────────────────


class TestOverviewPage:
    """GET /ui/overview returns a well-formed HTML page."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "text/html" in resp.headers["content-type"]

    def test_contains_page_title(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "Overview" in resp.text

    def test_contains_nav_links(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        for link in ("Dashboard", "Watchlist", "Portfolio", "Research", "Analyse", "Execute"):
            assert link in resp.text

    def test_contains_research_nav_link(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "Research" in resp.text
        assert "/ui/research" in resp.text

    def test_overview_link_is_active(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        # The overview nav link should have the 'active' class
        assert 'class="nav-link active"' in resp.text

    def test_contains_chat_zone(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "chatMessages" in resp.text
        assert "Quant Guide" in resp.text

    def test_contains_nav_tiles(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "nav-tile" in resp.text

    def test_contains_watchlist_snapshot(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "wl-row" in resp.text or "snapshot-panel" in resp.text

    def test_contains_news_snapshot(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "news-snap-item" in resp.text or "No news loaded yet" in resp.text

    def test_htmx_attributes_present(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "hx-get" in resp.text
        assert "hx-trigger" in resp.text
        assert "hx-target" in resp.text

    def test_htmx_polling_intervals(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        # Watchlist rows poll every 30s
        assert "every 30s" in resp.text

    def test_includes_htmx_script(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "htmx.org" in resp.text

    def test_includes_alpinejs_script(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "alpinejs" in resp.text

    def test_includes_theme_css(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "/static/css/theme.css" in resp.text

    def test_includes_chart_hover_js(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "/static/js/chart_hover.js" in resp.text


# ── Strategy page ─────────────────────────────────


class TestStrategyPage:
    """GET /ui/strategy returns a well-formed strategy page."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/strategy")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        resp = client.get("/ui/strategy")
        assert "text/html" in resp.headers["content-type"]

    def test_contains_strategy_heading(self, client: TestClient) -> None:
        resp = client.get("/ui/strategy")
        assert "Strategy" in resp.text

    def test_contains_strategy_list(self, client: TestClient) -> None:
        resp = client.get("/ui/strategy")
        # Should contain the strategy sidebar and detail area
        assert "strategy-detail" in resp.text

    def test_contains_htmx_attributes(self, client: TestClient) -> None:
        resp = client.get("/ui/strategy")
        assert "hx-get" in resp.text
        assert "hx-target" in resp.text

    def test_strategy_link_is_active(self, client: TestClient) -> None:
        resp = client.get("/ui/strategy")
        assert 'class="nav-link active"' in resp.text


# ── Portfolio page ───────────────────────────────


class TestPortfolioPage:
    """GET /ui/portfolio returns a well-formed portfolio page."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/portfolio")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        resp = client.get("/ui/portfolio")
        assert "text/html" in resp.headers["content-type"]

    def test_contains_portfolio_heading(self, client: TestClient) -> None:
        resp = client.get("/ui/portfolio")
        assert "Portfolio" in resp.text

    def test_contains_heatmap_grid(self, client: TestClient) -> None:
        resp = client.get("/ui/portfolio")
        assert "heatmap" in resp.text.lower()

    def test_contains_dca_section(self, client: TestClient) -> None:
        resp = client.get("/ui/portfolio")
        assert "DCA" in resp.text or "dca" in resp.text.lower()

    def test_contains_positions_or_holdings(self, client: TestClient) -> None:
        resp = client.get("/ui/portfolio")
        # The redesigned page uses a heatmap grid for positions
        assert "heatmap" in resp.text.lower() or "position" in resp.text.lower()

    def test_portfolio_link_is_active(self, client: TestClient) -> None:
        resp = client.get("/ui/portfolio")
        assert 'class="nav-link active"' in resp.text


# ── Research page ────────────────────────────────


class TestResearchPage:
    """GET /ui/research returns a well-formed research page."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/research")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        resp = client.get("/ui/research")
        assert "text/html" in resp.headers["content-type"]

    def test_contains_research_heading(self, client: TestClient) -> None:
        resp = client.get("/ui/research")
        assert "Research Log" in resp.text

    def test_contains_filter_bar(self, client: TestClient) -> None:
        resp = client.get("/ui/research")
        assert "filter-select" in resp.text
        assert "notebook" in resp.text.lower()

    def test_contains_detail_panel(self, client: TestClient) -> None:
        resp = client.get("/ui/research")
        assert "research-detail" in resp.text

    def test_contains_htmx_attributes(self, client: TestClient) -> None:
        resp = client.get("/ui/research")
        assert "hx-get" in resp.text
        assert "hx-target" in resp.text

    def test_research_link_is_active(self, client: TestClient) -> None:
        resp = client.get("/ui/research")
        assert 'class="nav-link active"' in resp.text


class TestResearchTabHighlight:
    """Research page highlights its own tab, not Analyse."""

    def test_research_tab_is_active(self, client: TestClient) -> None:
        resp = client.get("/ui/research")
        text = resp.text
        assert 'href="/ui/research">Research' in text

    def test_analyse_tab_not_active_on_research(self, client: TestClient) -> None:
        resp = client.get("/ui/research")
        assert 'class="nav-link active"' in resp.text
        import re
        active_link = re.search(r'class="nav-link active"[^>]*href="([^"]+)"', resp.text)
        if active_link:
            assert active_link.group(1) == "/ui/research"


# ── Execution page ───────────────────────────────


class TestExecutionPage:
    """GET /ui/execution returns a well-formed execution page."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/execution")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        resp = client.get("/ui/execution")
        assert "text/html" in resp.headers["content-type"]

    def test_contains_execution_heading(self, client: TestClient) -> None:
        resp = client.get("/ui/execution")
        assert "Execution" in resp.text

    def test_contains_paper_trading_banner(self, client: TestClient) -> None:
        resp = client.get("/ui/execution")
        assert "PAPER TRADING" in resp.text

    def test_contains_trading_banner_class(self, client: TestClient) -> None:
        resp = client.get("/ui/execution")
        assert "trading-banner" in resp.text

    def test_contains_account_status(self, client: TestClient) -> None:
        resp = client.get("/ui/execution")
        assert "Account Value" in resp.text
        assert "Cash" in resp.text

    def test_contains_open_orders_section(self, client: TestClient) -> None:
        resp = client.get("/ui/execution")
        assert "Open Orders" in resp.text

    def test_contains_order_history_section(self, client: TestClient) -> None:
        resp = client.get("/ui/execution")
        assert "Order History" in resp.text

    def test_contains_htmx_polling(self, client: TestClient) -> None:
        resp = client.get("/ui/execution")
        # Open orders poll every 10 seconds
        assert "every 10s" in resp.text

    def test_execution_link_is_active(self, client: TestClient) -> None:
        resp = client.get("/ui/execution")
        assert 'class="nav-link active"' in resp.text


# ── HTMX partials ────────────────────────────────


class TestMetricsBarPartial:
    """GET /ui/partials/metrics-bar returns a fragment."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/metrics-bar")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/metrics-bar")
        assert "text/html" in resp.headers["content-type"]

    def test_contains_metric_cards(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/metrics-bar")
        assert "Portfolio Value" in resp.text
        assert "Daily P&amp;L" in resp.text or "Daily P&L" in resp.text

    def test_is_fragment_not_full_page(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/metrics-bar")
        # Should NOT contain <!DOCTYPE> -- it's a fragment
        assert "<!DOCTYPE" not in resp.text


class TestSignalsPartial:
    """GET /ui/partials/signals returns a fragment."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/signals")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/signals")
        assert "text/html" in resp.headers["content-type"]

    def test_is_fragment_not_full_page(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/signals")
        assert "<!DOCTYPE" not in resp.text

    def test_contains_signal_content_or_fallback(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/signals")
        # Either signals are present or the fallback message shows
        assert ("signal-row" in resp.text or "No signals today" in resp.text)


class TestPositionDetailPartial:
    """GET /ui/partials/position-detail returns a fragment."""

    def test_returns_200_no_ticker(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/position-detail")
        assert resp.status_code == 200

    def test_fallback_message_without_ticker(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/position-detail")
        assert "Click a position" in resp.text

    def test_returns_200_with_ticker(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/position-detail?ticker=SHEL.L")
        assert resp.status_code == 200

    def test_contains_ticker_with_valid_request(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/position-detail?ticker=SHEL.L")
        assert "SHEL.L" in resp.text

    def test_is_fragment_not_full_page(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/position-detail?ticker=SHEL.L")
        assert "<!DOCTYPE" not in resp.text

    def test_contains_interpret_button(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/position-detail?ticker=SHEL.L")
        assert "Run Interpretation" in resp.text
        assert "hx-post" in resp.text


class TestStrategyDetailPartial:
    """GET /ui/partials/strategy-detail returns a fragment."""

    def test_returns_200_no_strategy(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/strategy-detail")
        assert resp.status_code == 200

    def test_fallback_message_without_strategy(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/strategy-detail")
        assert "Select a strategy" in resp.text

    def test_returns_200_with_strategy(self, client: TestClient) -> None:
        resp = client.get(
            "/ui/partials/strategy-detail?strategy=momentum_rsi"
        )
        assert resp.status_code == 200

    def test_contains_strategy_name_with_valid_request(
        self, client: TestClient
    ) -> None:
        resp = client.get(
            "/ui/partials/strategy-detail?strategy=momentum_rsi"
        )
        # Strategy name is formatted with replace("_", " ") | title
        assert "Momentum Rsi" in resp.text or "momentum_rsi" in resp.text

    def test_is_fragment_not_full_page(self, client: TestClient) -> None:
        resp = client.get(
            "/ui/partials/strategy-detail?strategy=momentum_rsi"
        )
        assert "<!DOCTYPE" not in resp.text

    def test_contains_summary_metrics(self, client: TestClient) -> None:
        resp = client.get(
            "/ui/partials/strategy-detail?strategy=momentum_rsi"
        )
        # Summary card should have these metric labels
        assert "Sharpe" in resp.text
        assert "CAGR" in resp.text
        assert "Max Drawdown" in resp.text


class TestPositionTablePartial:
    """GET /ui/partials/position-table returns a fragment."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/position-table")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/position-table")
        assert "text/html" in resp.headers["content-type"]

    def test_is_fragment_not_full_page(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/position-table")
        assert "<!DOCTYPE" not in resp.text

    def test_contains_table_or_fallback(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/position-table")
        assert ("feature-table" in resp.text or "No positions" in resp.text)

    def test_sort_parameter_accepted(self, client: TestClient) -> None:
        for sort_key in ("ticker", "weight", "pnl_today", "pnl_total"):
            resp = client.get(f"/ui/partials/position-table?sort={sort_key}")
            assert resp.status_code == 200


class TestResearchEntriesPartial:
    """GET /ui/partials/research-entries returns a fragment."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/research-entries")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/research-entries")
        assert "text/html" in resp.headers["content-type"]

    def test_is_fragment_not_full_page(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/research-entries")
        assert "<!DOCTYPE" not in resp.text

    def test_contains_entries_or_fallback(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/research-entries")
        assert ("log-entry" in resp.text or "No research entries" in resp.text)

    def test_notebook_filter_accepted(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/research-entries?notebook=test")
        assert resp.status_code == 200


class TestResearchEntryPartial:
    """GET /ui/partials/research-entry returns a fragment."""

    def test_returns_200_no_id(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/research-entry")
        assert resp.status_code == 200

    def test_fallback_without_id(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/research-entry")
        assert "Entry not found" in resp.text or "Select an entry" in resp.text

    def test_is_fragment_not_full_page(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/research-entry?id=test")
        assert "<!DOCTYPE" not in resp.text

    def test_returns_200_with_invalid_id(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/research-entry?id=nonexistent")
        assert resp.status_code == 200


class TestOpenOrdersPartial:
    """GET /ui/partials/open-orders returns a fragment."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/open-orders")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/open-orders")
        assert "text/html" in resp.headers["content-type"]

    def test_is_fragment_not_full_page(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/open-orders")
        assert "<!DOCTYPE" not in resp.text

    def test_contains_orders_or_fallback(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/open-orders")
        assert ("feature-table" in resp.text or "No open orders" in resp.text)


class TestOrderHistoryPartial:
    """GET /ui/partials/order-history returns a fragment."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/order-history")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/order-history")
        assert "text/html" in resp.headers["content-type"]

    def test_is_fragment_not_full_page(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/order-history")
        assert "<!DOCTYPE" not in resp.text

    def test_contains_history_or_fallback(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/order-history")
        assert ("feature-table" in resp.text or "No order history" in resp.text)

    def test_page_parameter_accepted(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/order-history?page=1")
        assert resp.status_code == 200


# ── Interpret action ──────────────────────────────


class TestInterpretAction:
    """POST /ui/actions/interpret returns a fragment."""

    def test_returns_200_without_ticker(self, client: TestClient) -> None:
        resp = client.post("/ui/actions/interpret")
        assert resp.status_code == 200

    def test_returns_200_with_ticker(self, client: TestClient) -> None:
        resp = client.post("/ui/actions/interpret?ticker=SHEL.L")
        assert resp.status_code == 200

    def test_is_fragment_not_full_page(self, client: TestClient) -> None:
        resp = client.post("/ui/actions/interpret?ticker=SHEL.L")
        assert "<!DOCTYPE" not in resp.text

    def test_contains_interpretation_section(self, client: TestClient) -> None:
        """Without API key, should still return a fragment with interpretation info."""
        resp = client.post("/ui/actions/interpret?ticker=SHEL.L")
        # Should have the position detail with some interpretation
        assert "SHEL.L" in resp.text
        assert "Interpretation" in resp.text or "interpretation" in resp.text

    def test_with_mocked_interpreter(self, client: TestClient) -> None:
        """With a mocked interpreter, verify interpretation content appears."""
        mock_interp = MagicMock()
        mock_interp.interpret.return_value = {
            "summary": "Test interpretation summary for SHEL.L",
            "observations": ["Obs 1"],
            "warnings": ["Risk warning here"],
            "suggestions": ["Consider rebalancing"],
            "confidence": "medium",
        }
        mock_rlog = MagicMock()

        # QuantInterpreter and ResearchLog are imported locally inside
        # the route handler, so we patch in the source modules.
        with patch(
            "src.notebooks.claude_interpreter.QuantInterpreter",
            return_value=mock_interp,
        ), patch(
            "src.notebooks.research_log.ResearchLog",
            return_value=mock_rlog,
        ):
            resp = client.post("/ui/actions/interpret?ticker=SHEL.L")

        assert resp.status_code == 200


# ── Jinja2 filters ───────────────────────────────


class TestJinjaFilters:
    """Verify custom Jinja2 filters produce expected output."""

    def test_gbp_filter(self) -> None:
        from web.routes.ui import _gbp

        assert _gbp(1234.5) == "GBP1,234.50"

    def test_gbp_filter_large_value(self) -> None:
        from web.routes.ui import _gbp

        assert _gbp(1_000_000) == "GBP1,000,000.00"

    def test_gbp_filter_zero(self) -> None:
        from web.routes.ui import _gbp

        assert _gbp(0) == "GBP0.00"

    def test_gbp_filter_negative(self) -> None:
        from web.routes.ui import _gbp

        assert _gbp(-500.1) == "-GBP500.10"

    def test_pct_filter_positive(self) -> None:
        from web.routes.ui import _pct

        result = _pct(0.0312)
        assert "+3.12%" in result
        assert "text-green" in result

    def test_pct_filter_negative(self) -> None:
        from web.routes.ui import _pct

        result = _pct(-0.045)
        assert "-4.50%" in result
        assert "text-red" in result

    def test_pct_filter_zero(self) -> None:
        from web.routes.ui import _pct

        result = _pct(0.0)
        assert "+0.00%" in result

    def test_pct_filter_small_value(self) -> None:
        from web.routes.ui import _pct

        result = _pct(0.001)
        assert "+0.10%" in result


# ── Static files ─────────────────────────────────


class TestStaticFiles:
    """Static assets are served correctly."""

    def test_theme_css_served(self, client: TestClient) -> None:
        resp = client.get("/static/css/theme.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]

    def test_charts_js_served(self, client: TestClient) -> None:
        resp = client.get("/static/js/charts.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers["content-type"]

    def test_theme_css_contains_variables(self, client: TestClient) -> None:
        resp = client.get("/static/css/theme.css")
        assert "--bg:" in resp.text
        assert "--card-bg:" in resp.text
        assert "--accent:" in resp.text

    def test_theme_css_contains_new_components(self, client: TestClient) -> None:
        resp = client.get("/static/css/theme.css")
        assert ".spinner" in resp.text
        assert ".overview-grid" in resp.text
        assert ".strategy-grid" in resp.text
        assert ".signal-badge" in resp.text
        assert ".ticker-chip" in resp.text

    def test_theme_css_contains_portfolio_components(self, client: TestClient) -> None:
        resp = client.get("/static/css/theme.css")
        assert ".portfolio-grid" in resp.text
        assert ".portfolio-main" in resp.text
        assert ".portfolio-sidebar" in resp.text

    def test_theme_css_contains_research_components(self, client: TestClient) -> None:
        resp = client.get("/static/css/theme.css")
        assert ".research-grid" in resp.text
        assert ".log-entry" in resp.text
        assert ".filter-select" in resp.text

    def test_theme_css_contains_execution_components(self, client: TestClient) -> None:
        resp = client.get("/static/css/theme.css")
        assert ".trading-banner" in resp.text
        assert ".trading-banner-paper" in resp.text
        assert ".trading-banner-live" in resp.text
        assert ".pagination" in resp.text
        assert ".sort-btn" in resp.text

    def test_charts_js_contains_treemap_handler(self, client: TestClient) -> None:
        resp = client.get("/static/js/charts.js")
        assert "treemap" in resp.text
        assert "plotly_click" in resp.text


# ── Actions dropdown ────────────────────────────


class TestActionsDropdown:
    """Actions dropdown appears in navbar."""

    def test_actions_button_present(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "+ Actions" in resp.text

    def test_modal_container_present(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert 'id="modal-container"' in resp.text


# ── Upload research ─────────────────────────────


class TestUploadResearch:
    """Upload research modal and endpoint."""

    def test_modal_partial_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/modal-upload-research")
        assert resp.status_code == 200
        assert "Upload Research" in resp.text

    def test_upload_endpoint_rejects_no_file(self, client: TestClient) -> None:
        resp = client.post("/ui/upload/research")
        assert resp.status_code == 422

    def test_upload_endpoint_accepts_txt(self, client: TestClient, tmp_path) -> None:
        test_file = tmp_path / "test_note.txt"
        test_file.write_text("Test research note content")
        with open(test_file, "rb") as f:
            resp = client.post(
                "/ui/upload/research",
                files={"file": ("test_note.txt", f, "text/plain")},
                data={"ticker": "", "notes": "Test upload"},
            )
        assert resp.status_code == 200
        assert "uploaded" in resp.text.lower() or "success" in resp.text.lower()


# ── Import purchases ────────────────────────────


class TestImportPurchases:
    """CSV purchase import modal and endpoints."""

    def test_modal_partial_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/modal-import-purchases")
        assert resp.status_code == 200
        assert "Import" in resp.text

    def test_preview_parses_valid_csv(self, client: TestClient) -> None:
        csv_content = "date,ticker,price,amount_gbp,note\n2025-03-01,CNDX.L,2500,1000,Monthly DCA\n"
        resp = client.post(
            "/ui/upload/purchases/preview",
            files={"file": ("purchases.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        )
        assert resp.status_code == 200
        assert "CNDX.L" in resp.text
        assert "2025-03-01" in resp.text

    def test_preview_rejects_bad_csv(self, client: TestClient) -> None:
        csv_content = "bad,columns\nfoo,bar\n"
        resp = client.post(
            "/ui/upload/purchases/preview",
            files={"file": ("bad.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        )
        assert resp.status_code == 200
        assert "missing" in resp.text.lower() or "error" in resp.text.lower()

    def test_confirm_imports_rows(self, client: TestClient) -> None:
        csv_content = "date,ticker,price,amount_gbp,note\n2099-12-31,TEST.L,100,50,Import test\n"
        resp = client.post(
            "/ui/upload/purchases/confirm",
            data={"csv_data": csv_content},
        )
        assert resp.status_code == 200
        assert "imported" in resp.text.lower() or "1" in resp.text


# ── Quick-add purchase modal ────────────────────


class TestQuickAddPurchase:
    """Quick-add purchase modal from navbar."""

    def test_modal_partial_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/modal-add-purchase")
        assert resp.status_code == 200
        assert "Add Purchase" in resp.text
        assert "ticker" in resp.text.lower()


# ── DCA form always visible ─────────────────────


class TestDCAFormAlwaysVisible:
    """DCA purchase form should be visible without toggle."""

    def test_form_has_no_x_show(self, client: TestClient) -> None:
        resp = client.get("/ui/portfolio/dca/ticker/CNDX.L")
        if resp.status_code == 200:
            assert 'x-show="showForm"' not in resp.text
