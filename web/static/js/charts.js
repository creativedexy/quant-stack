/**
 * Quant Stack — Thin Plotly chart wrapper.
 *
 * Receives Plotly JSON from the Jinja2 template and renders it
 * with consistent theme defaults. No business logic lives here.
 */

/* global Plotly */

/**
 * Render a Plotly chart into the given element.
 *
 * @param {string} elementId  - DOM id of the target container.
 * @param {string} plotlyJsonString - JSON string with {data, layout, config}.
 */
function renderChart(elementId, plotlyJsonString) {
  var el = document.getElementById(elementId);
  if (!el) {
    console.warn("renderChart: element #" + elementId + " not found");
    return;
  }

  var parsed;
  try {
    parsed = JSON.parse(plotlyJsonString);
  } catch (e) {
    console.error("renderChart: invalid JSON for #" + elementId, e);
    return;
  }

  var data = parsed.data || [];
  var layout = Object.assign(
    {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "#f9fafb" },
      xaxis: { gridcolor: "#2d2d2d" },
      yaxis: { gridcolor: "#2d2d2d" },
      margin: { l: 48, r: 16, t: 32, b: 40 },
    },
    parsed.layout || {}
  );
  var config = Object.assign(
    { responsive: true, displayModeBar: false },
    parsed.config || {}
  );

  Plotly.react(el, data, layout, config);

  // For treemap/heatmap charts, forward Plotly click events
  // as a standard CustomEvent so inline scripts and HTMX can listen.
  if (data.length > 0 && data[0].type === "treemap") {
    el.on("plotly_click", function (eventData) {
      el.dispatchEvent(
        new CustomEvent("plotly_click", { detail: eventData, bubbles: true })
      );
    });
  }
}

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
