# Interactive Plotly Chart — Design Document

**Date**: 2026-03-22
**Goal**: Replace the analyse page's server-rendered SVG chart with interactive Plotly subplots featuring a date timeline, synchronised panels, candlestick price chart, and future space for Monte Carlo projections.

## Architecture

Server builds a Plotly-compatible JSON dict. Client renders it with `Plotly.react()`. On overlay toggle or period change, server returns updated JSON via HTMX, client re-renders in-place.

```
Server (analyse.py)                    Browser
─────────────────                      ───────
_build_chart_json() ──→ JSON string ──→ renderAnalyseChart()
  - OHLC candlestick data                - Main panel (row 1, ~65% height)
  - Bollinger band traces                 - RSI panel (row 2, ~17%)
  - RSI series                            - MACD panel (row 3, ~18%)
  - MACD line + signal + histogram        - Shared X-axis with date timeline
  - ML confidence line                    - Future whitespace for Monte Carlo
  - Monte Carlo P5/P50/P95 fan           - Crosshair synced across all panels
  - dates (past + 60 trading days future) - Zoom, pan, hover natively
```

## Key Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Main chart | `go.Candlestick` (OHLC) | User requested candlestick for buy decisions |
| Sub-panels | Plotly `make_subplots(shared_xaxes=True)` | One render, automatic timeline sync |
| Future space | Extend date axis 60 trading days beyond last price | Monte Carlo projection room |
| Timeline | X-axis with Plotly auto date formatting | Handles 1D to 1Y ranges gracefully |
| Overlays | Toggle via HTMX, server rebuilds full JSON | Same UX pattern as current |
| Prev close | `add_hline` shape annotation, dashed amber | Matches current design |
| Monte Carlo | `go.Scatter` fill='tonexty' for P5-P95 band | Standard fan chart pattern |
| Bollinger | `go.Scatter` traces on main panel | Amber, semi-transparent |
| Theme | Dark background matching `--card` (#1a1a1a) | Consistent with existing UI |
| Rangeslider | Disabled | Clean look, use period buttons instead |

## Layout

Plotly subplots with `rows=3, row_heights=[0.65, 0.17, 0.18]`:

- **Row 1**: Candlestick + Bollinger + Monte Carlo + prev close line
- **Row 2**: RSI line with overbought (70) / oversold (30) horizontal lines
- **Row 3**: MACD histogram + MACD line + signal line + zero line

Only active overlays get traces. Empty rows are hidden (row height collapses).

## Data Flow

1. `_build_chart_json(ticker, period, overlays, log_scale)` returns a dict:
   ```python
   {
       "data": [...],       # Plotly trace dicts
       "layout": {...},     # Axes, shapes, annotations, theme
   }
   ```

2. Template injects JSON into the page:
   ```html
   <div id="analyse-chart" style="width:100%;height:600px;"></div>
   <script>
     renderAnalyseChart({{ chart_json | tojson | safe }});
   </script>
   ```

3. HTMX overlay toggle returns new HTML fragment with updated JSON.
   `Plotly.react()` updates the existing chart smoothly (no destroy/recreate).

## Subplot Visibility Logic

| Overlay active? | Subplots shown |
|----------------|----------------|
| None | Row 1 only (100% height) |
| RSI | Row 1 (75%) + Row 2 (25%) |
| MACD | Row 1 (75%) + Row 3 (25%) |
| RSI + MACD | Row 1 (60%) + Row 2 (20%) + Row 3 (20%) |
| Any + ML | ML shares Row 2 with RSI (or gets its own row) |

## Files Changed

| File | Change |
|------|--------|
| `web/routes/analyse.py` | New `_build_chart_json()` function; route returns `chart_json` in context |
| `web/templates/partials/analyse_chart.html` | Replace SVG with Plotly `<div>` + render script |
| `web/static/js/charts.js` | Add `renderAnalyseChart(json)` for subplot rendering |
| `partials/chart_rsi.html` | **Removed** — RSI is now a Plotly subplot row |
| `partials/chart_macd.html` | **Removed** — MACD is now a Plotly subplot row |
| `partials/chart_ml.html` | **Removed** — ML is now a Plotly subplot row |
| `partials/chart_monte_carlo.html` | **Removed** — MC is now Plotly traces on main panel |
| `tests/test_web/test_analyse.py` | Update to check for Plotly div instead of SVG |

## What Stays the Same

- Overlay toolbar (toggle buttons in `analyse_chart.html`)
- Signal panel on the right
- Learn strips below chart
- Watchlist dropdown in ticker bar
- Period selector buttons (1D–1Y)
- HTMX swap pattern (`#chart-zone`)
- All service layer code (`chart_service.py` methods still used for data)

## Theme Colours

```
Background:   transparent (card handles it)
Candle up:    #23c55e (--positive / green)
Candle down:  #ef4444 (--negative / red)
Bollinger:    #f59e0b (--amber)
RSI line:     #06b6d4 (--cyan)
MACD line:    #06b6d4 (--cyan)
Signal line:  #f59e0b (--amber)
Histogram +: #23c55e
Histogram -: #ef4444
ML line:      #23c55e (--green)
Monte Carlo:  #7c3aed (--purple)
Prev close:   #f59e0b dashed
Grid:         #2d2d2d
Text:         #f9fafb
```
