# Interactive Plotly Chart — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the analyse page's server-rendered SVG chart with an interactive Plotly candlestick chart featuring synchronised sub-panels (RSI, MACD, ML), a date timeline, and future space for Monte Carlo projections.

**Architecture:** Server builds a Plotly JSON dict in `_build_chart_json()`. The template injects it into a `<div>` and calls `renderAnalyseChart()` from `charts.js`. Overlay toggles return updated HTML+JSON via HTMX. Plotly handles zoom, pan, crosshair, tooltips natively. Sub-panels share the X-axis for synchronised interaction.

**Tech Stack:** Plotly.js (CDN), existing FastAPI + HTMX + Jinja2 stack, existing `chart_service.py` data methods.

---

### Task 1: Add Plotly.js CDN to base template

**Files:**
- Modify: `web/templates/base.html:10-11` (after HTMX script)

**Step 1: Add Plotly CDN script tag**

In `web/templates/base.html`, after the HTMX script line, add:

```html
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
```

**Step 2: Verify page still loads**

Run: Load `http://localhost:8000/ui/analyse` in preview
Expected: Page loads normally, no console errors

**Step 3: Verify Plotly is available**

Run in browser console: `typeof Plotly`
Expected: `"object"`

---

### Task 2: Build `_build_chart_json()` server function

**Files:**
- Modify: `web/routes/analyse.py` (add new function after `_build_chart_context`)
- Test: `tests/test_web/test_analyse.py`

**Step 1: Write the test**

Add to `tests/test_web/test_analyse.py`:

```python
class TestBuildChartJson:
    """_build_chart_json returns valid Plotly spec."""

    def test_returns_data_and_layout(self) -> None:
        from web.routes.analyse import _build_chart_json
        result = _build_chart_json("CNDX.L", "1M", [])
        assert "data" in result
        assert "layout" in result
        assert isinstance(result["data"], list)
        assert len(result["data"]) >= 1  # At least the candlestick trace

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
        types = [t.get("name", "") for t in result["data"]]
        assert any("RSI" in n for n in types)

    def test_bollinger_overlay_adds_traces(self) -> None:
        from web.routes.analyse import _build_chart_json
        result = _build_chart_json("CNDX.L", "1M", ["bollinger"])
        names = [t.get("name", "") for t in result["data"]]
        assert any("Upper" in n or "BB" in n for n in names)

    def test_macd_overlay_adds_traces(self) -> None:
        from web.routes.analyse import _build_chart_json
        result = _build_chart_json("CNDX.L", "1M", ["macd"])
        names = [t.get("name", "") for t in result["data"]]
        assert any("MACD" in n for n in names)

    def test_monte_carlo_adds_future_traces(self) -> None:
        from web.routes.analyse import _build_chart_json
        result = _build_chart_json("CNDX.L", "3M", ["monte_carlo"])
        # MC traces should have dates beyond the last price date
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
```

**Step 2: Run tests — should fail**

Run: `py -3 -m pytest tests/test_web/test_analyse.py::TestBuildChartJson -v`
Expected: FAIL (function doesn't exist yet)

**Step 3: Implement `_build_chart_json()`**

Add to `web/routes/analyse.py` after `_build_chart_context()`:

```python
def _build_chart_json(
    ticker: str,
    period: str,
    overlays: list[str],
    log_scale: bool = False,
) -> dict[str, Any]:
    """Build a Plotly JSON spec for the interactive analyse chart.

    Returns a dict with ``data`` (list of traces) and ``layout``
    ready for ``Plotly.react()`` on the client.
    """
    svcs = _get_services()
    chart_svc = svcs["chart"]

    # -- Fetch data ------------------------------------------------
    price_data = chart_svc.get_price_history(ticker, period=_map_period(period))
    dates = price_data.get("dates", [])
    opens = price_data.get("open", [])
    highs = price_data.get("high", [])
    lows = price_data.get("low", [])
    closes = price_data.get("close", [])

    if not dates or len(dates) < 2:
        return {"data": [], "layout": _chart_layout(1)}

    # -- Determine subplot layout ----------------------------------
    has_rsi = "rsi" in overlays
    has_macd = "macd" in overlays
    has_ml = "ml" in overlays

    n_sub = 1 + int(has_rsi or has_ml) + int(has_macd)
    if n_sub == 1:
        row_heights = [1.0]
    elif n_sub == 2:
        row_heights = [0.75, 0.25]
    else:
        row_heights = [0.60, 0.20, 0.20]

    traces: list[dict[str, Any]] = []

    # -- Row 1: Candlestick ---------------------------------------
    traces.append({
        "type": "candlestick",
        "x": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "name": ticker,
        "increasing": {"line": {"color": "#23c55e"}},
        "decreasing": {"line": {"color": "#ef4444"}},
        "xaxis": "x",
        "yaxis": "y",
    })

    # Bollinger bands
    if "bollinger" in overlays:
        bb_upper = price_data.get("bb_upper", [])
        bb_middle = price_data.get("bb_middle", [])
        bb_lower = price_data.get("bb_lower", [])
        if bb_upper:
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": dates, "y": bb_upper, "name": "BB Upper",
                "line": {"color": "#f59e0b", "width": 1, "dash": "dot"},
                "opacity": 0.6, "xaxis": "x", "yaxis": "y",
                "showlegend": False,
            })
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": dates, "y": bb_lower, "name": "BB Lower",
                "line": {"color": "#f59e0b", "width": 1, "dash": "dot"},
                "opacity": 0.6, "fill": "tonexty",
                "fillcolor": "rgba(245,158,11,0.06)",
                "xaxis": "x", "yaxis": "y",
                "showlegend": False,
            })
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": dates, "y": bb_middle, "name": "BB Mid",
                "line": {"color": "#f59e0b", "width": 1, "dash": "dash"},
                "opacity": 0.35, "xaxis": "x", "yaxis": "y",
                "showlegend": False,
            })

    # Previous close line (as shape in layout, not here)

    # Monte Carlo projection
    if "monte_carlo" in overlays:
        mc_svc = svcs["monte_carlo"]
        mc_data = mc_svc.run_portfolio_projection(ticker, period_years=1)
        if mc_data.get("dates_forward"):
            mc_dates = mc_data["dates_forward"]
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": mc_dates, "y": mc_data["p95"], "name": "P95",
                "line": {"color": "#7c3aed", "width": 1, "dash": "dash"},
                "opacity": 0.5, "xaxis": "x", "yaxis": "y",
                "showlegend": False,
            })
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": mc_dates, "y": mc_data["p5"], "name": "P05",
                "line": {"color": "#7c3aed", "width": 0.5, "dash": "dash"},
                "opacity": 0.4, "fill": "tonexty",
                "fillcolor": "rgba(124,58,237,0.08)",
                "xaxis": "x", "yaxis": "y",
                "showlegend": False,
            })
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": mc_dates, "y": mc_data["p50"], "name": "P50 (median)",
                "line": {"color": "#7c3aed", "width": 2},
                "opacity": 0.8, "xaxis": "x", "yaxis": "y",
                "showlegend": False,
            })

    # -- Row 2: RSI or ML -----------------------------------------
    sub_row = 2
    if has_rsi:
        rsi_data = chart_svc.get_rsi_series(ticker, period=_map_period(period))
        if rsi_data.get("rsi"):
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": rsi_data["dates"],
                "y": [v for v in rsi_data["rsi"]],
                "name": f"RSI ({rsi_data['window']})",
                "line": {"color": "#06b6d4", "width": 1.5},
                "xaxis": "x", "yaxis": f"y{sub_row}",
            })
    elif has_ml:
        ml_data = chart_svc.get_ml_confidence_series(
            ticker, period=_map_period(period),
        )
        if ml_data.get("confidence"):
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": ml_data["dates"],
                "y": ml_data["confidence"],
                "name": "ML Confidence",
                "line": {"color": "#23c55e", "width": 1.5},
                "xaxis": "x", "yaxis": f"y{sub_row}",
            })

    # -- Row 3: MACD -----------------------------------------------
    if has_macd:
        macd_row = sub_row + (1 if (has_rsi or has_ml) else 0)
        if macd_row == sub_row:
            macd_row = 2  # Only MACD, no RSI/ML
        macd_data = chart_svc.get_macd_series(
            ticker, period=_map_period(period),
        )
        if macd_data.get("macd_histogram"):
            hist_colors = [
                "#23c55e" if (v or 0) >= 0 else "#ef4444"
                for v in macd_data["macd_histogram"]
            ]
            traces.append({
                "type": "bar",
                "x": macd_data["dates"],
                "y": macd_data["macd_histogram"],
                "name": "Histogram",
                "marker": {"color": hist_colors, "opacity": 0.5},
                "xaxis": "x", "yaxis": f"y{macd_row}",
                "showlegend": False,
            })
        if macd_data.get("macd_line"):
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": macd_data["dates"],
                "y": macd_data["macd_line"],
                "name": "MACD",
                "line": {"color": "#06b6d4", "width": 1.2},
                "xaxis": "x", "yaxis": f"y{macd_row}",
            })
        if macd_data.get("macd_signal"):
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": macd_data["dates"],
                "y": macd_data["macd_signal"],
                "name": "Signal",
                "line": {"color": "#f59e0b", "width": 1, "dash": "dash"},
                "xaxis": "x", "yaxis": f"y{macd_row}",
            })

    # -- Layout ----------------------------------------------------
    layout = _chart_layout(n_sub, row_heights, log_scale)

    # Prev close horizontal line
    if len(closes) >= 2:
        layout.setdefault("shapes", []).append({
            "type": "line", "xref": "paper", "yref": "y",
            "x0": 0, "x1": 1,
            "y0": closes[-2], "y1": closes[-2],
            "line": {"color": "#f59e0b", "width": 1, "dash": "dash"},
            "opacity": 0.5,
        })

    # RSI threshold lines
    if has_rsi and n_sub >= 2:
        for level, color in [(70, "#ef4444"), (30, "#23c55e"), (50, "#666")]:
            layout["shapes"].append({
                "type": "line", "xref": "paper", "yref": "y2",
                "x0": 0, "x1": 1, "y0": level, "y1": level,
                "line": {"color": color, "width": 0.5, "dash": "dot"},
                "opacity": 0.4,
            })

    # MACD zero line
    if has_macd:
        macd_yref = f"y{n_sub}"
        layout["shapes"].append({
            "type": "line", "xref": "paper", "yref": macd_yref,
            "x0": 0, "x1": 1, "y0": 0, "y1": 0,
            "line": {"color": "#666", "width": 0.5, "dash": "dot"},
            "opacity": 0.4,
        })

    return {"data": traces, "layout": layout}


def _chart_layout(
    n_rows: int,
    row_heights: list[float] | None = None,
    log_scale: bool = False,
) -> dict[str, Any]:
    """Build the Plotly layout dict for the analyse chart."""
    if row_heights is None:
        row_heights = [1.0]

    layout: dict[str, Any] = {
        "paper_bgcolor": "transparent",
        "plot_bgcolor": "transparent",
        "font": {"color": "#f9fafb", "size": 11},
        "margin": {"l": 50, "r": 16, "t": 8, "b": 40},
        "showlegend": False,
        "hovermode": "x unified",
        "dragmode": "zoom",
        "shapes": [],
        "xaxis": {
            "gridcolor": "#2d2d2d",
            "linecolor": "#2d2d2d",
            "rangeslider": {"visible": False},
            "type": "date",
        },
        "yaxis": {
            "gridcolor": "#2d2d2d",
            "linecolor": "#2d2d2d",
            "side": "right",
            "type": "log" if log_scale else "linear",
        },
    }

    if n_rows == 1:
        layout["xaxis"]["domain"] = [0, 1]
        layout["yaxis"]["domain"] = [0, 1]
    elif n_rows == 2:
        layout["yaxis"]["domain"] = [0.28, 1.0]
        layout["yaxis2"] = {
            "gridcolor": "#2d2d2d", "linecolor": "#2d2d2d",
            "side": "right", "domain": [0, 0.24],
            "anchor": "x",
        }
    else:
        layout["yaxis"]["domain"] = [0.44, 1.0]
        layout["yaxis2"] = {
            "gridcolor": "#2d2d2d", "linecolor": "#2d2d2d",
            "side": "right", "domain": [0.22, 0.40],
            "anchor": "x",
        }
        layout["yaxis3"] = {
            "gridcolor": "#2d2d2d", "linecolor": "#2d2d2d",
            "side": "right", "domain": [0, 0.18],
            "anchor": "x",
        }

    return layout
```

**Step 4: Run tests**

Run: `py -3 -m pytest tests/test_web/test_analyse.py::TestBuildChartJson -v`
Expected: All PASS

---

### Task 3: Add `renderAnalyseChart()` to client JS

**Files:**
- Modify: `web/static/js/charts.js` (append function)

**Step 1: Add the render function**

Append to `web/static/js/charts.js`:

```javascript
/**
 * Render the analyse page interactive chart.
 *
 * @param {string} elementId - DOM id of the chart container.
 * @param {object} spec - Plotly spec with {data, layout}.
 */
function renderAnalyseChart(elementId, spec) {
  var el = document.getElementById(elementId);
  if (!el || !spec) return;

  var config = {
    responsive: true,
    displayModeBar: false,
    scrollZoom: true,
  };

  Plotly.react(el, spec.data || [], spec.layout || {}, config);
}
```

**Step 2: Verify file is syntactically valid**

Run: Load any page in preview, check no JS errors in console.

---

### Task 4: Wire the template to use Plotly

**Files:**
- Modify: `web/templates/partials/analyse_chart.html` (replace SVG with Plotly div)
- Modify: `web/routes/analyse.py` (add `chart_json` to both route contexts)

**Step 1: Update `analyse_chart_partial` route**

In `web/routes/analyse.py`, update `analyse_chart_partial()` to build and return `chart_json`:

```python
@router.get("/analyse/chart", response_class=HTMLResponse)
async def analyse_chart_partial(
    request: Request,
    ticker: str = Query(default="CNDX.L"),
    period: str = Query(default="1M"),
    overlays: str = Query(default=""),
    log_scale: int = Query(default=0),
) -> HTMLResponse:
    """HTMX partial -- returns chart fragment only."""
    overlay_list = [o.strip() for o in overlays.split(",") if o.strip()]

    chart_json = _build_chart_json(
        ticker, period, overlay_list, log_scale=bool(log_scale),
    )
    learn_strips = _build_learn_strips(
        overlay_list, None, None, None, None, ticker,
    )

    return templates.TemplateResponse(
        "partials/analyse_chart.html",
        {
            "request": request,
            "ticker": ticker,
            "period": period,
            "overlays": overlay_list,
            "log_scale": bool(log_scale),
            "chart_json": chart_json,
            "learn_strips": learn_strips,
        },
    )
```

**Step 2: Update `analyse_page` route**

Add `chart_json` to the `analyse_page()` context alongside existing fields:

```python
chart_json = _build_chart_json(ticker, period, overlays)
```

Pass `"chart_json": chart_json` in the template context dict.

**Step 3: Rewrite `analyse_chart.html`**

Replace the entire file with:

```html
{# Chart fragment -- toolbar + Plotly chart, swapped via HTMX #}

{# Overlay toolbar #}
<div class="card" style="display:flex;align-items:center;gap:0.35rem;padding:0.4rem 0.75rem;">
  {% for ov, label, dot_colour in [
    ('bollinger', 'Bollinger', 'var(--amber)'),
    ('rsi', 'RSI', 'var(--cyan)'),
    ('macd', 'MACD', 'var(--amber)'),
    ('ml', 'ML Signal', 'var(--green)'),
    ('monte_carlo', 'Monte Carlo', 'var(--purple)'),
  ] %}
  <button class="btn btn-sm overlay-btn{% if ov in overlays %} active{% endif %}"
          hx-get="/ui/analyse/chart?ticker={{ ticker }}&period={{ period }}&overlays={% if ov in overlays %}{{ overlays | reject('equalto', ov) | join(',') }}{% else %}{{ (overlays + [ov]) | join(',') }}{% endif %}&log_scale={{ 1 if log_scale else 0 }}"
          hx-target="#chart-zone"
          hx-swap="innerHTML">
    <span class="ov-dot" style="background:{{ dot_colour }};"></span>
    {{ label }}
  </button>
  {% endfor %}

  <button class="btn btn-sm overlay-btn{% if log_scale %} active{% endif %}"
          hx-get="/ui/analyse/chart?ticker={{ ticker }}&period={{ period }}&overlays={{ overlays | join(',') }}&log_scale={% if log_scale %}0{% else %}1{% endif %}"
          hx-target="#chart-zone"
          hx-swap="innerHTML"
          style="margin-left:auto;">
    Log
  </button>
</div>

{# Plotly chart #}
<div class="card" style="padding:0;">
  <div id="analyse-chart" style="width:100%;height:600px;"></div>
</div>

<script>
  renderAnalyseChart('analyse-chart', {{ chart_json | tojson }});
</script>

{# Learn strips #}
{% if learn_strips %}
<div class="card" style="padding:0.5rem 0.75rem;">
  {% for strip in learn_strips %}
  <div style="display:flex;align-items:baseline;gap:0.5rem;padding:0.35rem 0;">
    <span style="width:8px;height:8px;border-radius:50%;background:{{ strip.dot_colour | default('var(--amber)') }};flex-shrink:0;position:relative;top:1px;"></span>
    <span style="font-size:0.8rem;font-weight:600;white-space:nowrap;">{{ strip.name }}</span>
    <span class="text-muted" style="font-size:0.75rem;">{{ strip.explanation }}</span>
  </div>
  {% endfor %}
</div>
{% endif %}

<div id="forensic-panel"></div>
```

**Step 4: Load the page and verify**

Run: Load `http://localhost:8000/ui/analyse` in preview
Expected: Interactive candlestick chart with date timeline, zoom/pan works, dark theme

---

### Task 5: Verify all overlays work

**Step 1: Test each overlay**

Click each overlay button and verify:
- **Bollinger**: Amber dotted bands around candles
- **RSI**: Sub-panel appears below with RSI line, threshold lines at 30/50/70
- **MACD**: Sub-panel with histogram bars + MACD line + signal line
- **ML Signal**: Sub-panel with confidence line
- **Monte Carlo**: Purple fan chart extending into future whitespace
- **Log**: Y-axis switches to logarithmic scale
- **Multiple overlays**: RSI + MACD shows two sub-panels, zoom syncs

**Step 2: Test period buttons**

Click 1D, 1W, 1M, 3M, 6M, 1Y — chart updates with correct date range.

**Step 3: Test ticker selector**

Click dropdown, select a different ticker — full page reloads with new data.

---

### Task 6: Run full test suite

**Step 1: Run web tests**

Run: `py -3 -m pytest tests/test_web/ -v`
Expected: All pass

**Step 2: Run full suite**

Run: `py -3 -m pytest tests/ -v --ignore=tests/test_models/test_automl.py`
Expected: 1169+ passed, 0 failures
