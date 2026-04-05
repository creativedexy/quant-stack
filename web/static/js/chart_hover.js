/**
 * Quant Stack -- Intraday SVG chart with crosshair hover.
 *
 * Draws filled line charts from data-* attributes on [data-prices]
 * elements and adds interactive crosshair + tooltip on hover.
 *
 * Re-runs after HTMX content swaps via htmx:afterSwap listener.
 */

function initCharts() {
  document.querySelectorAll('[data-prices]:not([data-chart-init])').forEach(area => {
    area.setAttribute('data-chart-init', '1');
    drawChart(area);
    attachHover(area);
  });
}

function formatPriceLabel(price, ticker) {
  if (!ticker) return price.toFixed(2);
  // LSE tickers end in .L -- show in pence
  if (ticker.endsWith('.L')) {
    return price > 100
      ? price.toFixed(0) + 'p'
      : price.toFixed(1) + 'p';
  }
  return '$' + price.toFixed(2);
}

function drawChart(area) {
  const svg = area.querySelector('.chart-svg');
  if (!svg) return;

  const rawPrices = area.dataset.prices;
  if (!rawPrices) return;
  const prices = rawPrices.split(',').map(Number);
  if (prices.length < 2) return;

  const prev   = parseFloat(area.dataset.prev) || prices[0];
  const dir    = area.dataset.direction;
  const color  = dir === 'up' ? '#23c55e' : '#ef4444';
  const ticker = area.dataset.ticker || '';

  const W = 400, H = 120, PAD = 6;

  // Scale with 15% headroom so line never touches edges
  const priceRange = Math.max(...prices) - Math.min(...prices) || 1;
  const mn = Math.min(...prices, prev) - priceRange * 0.15;
  const mx = Math.max(...prices, prev) + priceRange * 0.15;
  const range = mx - mn;

  const toX = i => PAD + (i / (prices.length - 1)) * (W - PAD * 2);
  const toY = v => H - PAD - ((v - mn) / range) * (H - PAD * 2);

  const pts      = prices.map((v, i) => `${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ');
  const lastX    = toX(prices.length - 1).toFixed(1);
  const polyPts  = `${pts} ${lastX},${H} ${toX(0).toFixed(1)},${H}`;
  const pcY      = toY(prev).toFixed(1);

  // Build SVG via innerHTML (trusted server data, not user input)
  svg.innerHTML = `
    <defs>
      <linearGradient id="g-${ticker}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%"   stop-color="${color}" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
      </linearGradient>
    </defs>
    <line x1="${PAD}" y1="${pcY}" x2="${W-PAD}" y2="${pcY}"
          stroke="rgba(255,255,255,0.15)" stroke-width="1" stroke-dasharray="3,4"/>
    <polygon points="${polyPts}" fill="url(#g-${ticker})"/>
    <polyline points="${pts}" fill="none" stroke="${color}"
              stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${lastX}" cy="${toY(prices[prices.length-1]).toFixed(1)}"
            r="3.5" fill="${color}" stroke="#1a1a1a" stroke-width="2"/>
  `;

  // Prev close label
  const existing = area.querySelector('.prev-close-label');
  if (existing) existing.remove();

  const lbl = document.createElement('div');
  lbl.className = 'prev-close-label';
  lbl.style.cssText = `
    position:absolute; right:6px; pointer-events:none;
    font-family:'IBM Plex Mono',monospace; font-size:9px; color:#4a4a4a;
    top:${((parseFloat(pcY) / H) * 100 - 1).toFixed(1)}%;
  `;
  lbl.textContent = `Prev close: ${formatPriceLabel(prev, ticker)}`;
  area.appendChild(lbl);
}

function attachHover(area) {
  // Clone to remove previous listeners -- prevents duplicate stacking on HTMX refresh
  const fresh = area.cloneNode(true);
  area.replaceWith(fresh);
  area = fresh;

  // Re-draw after clone (cloneNode copies data attrs but not event state)
  drawChart(area);

  const rawPrices = area.dataset.prices;
  if (!rawPrices) return;
  const prices = rawPrices.split(',').map(Number);
  if (prices.length < 2) return;

  const prev   = parseFloat(area.dataset.prev) || prices[0];
  const dir    = area.dataset.direction;
  const ticker = area.dataset.ticker || '';
  const color  = dir === 'up' ? '#23c55e' : '#ef4444';

  const W = 400, H = 120, PAD = 6;
  const priceRange = Math.max(...prices) - Math.min(...prices) || 1;
  const mn = Math.min(...prices, prev) - priceRange * 0.15;
  const mx = Math.max(...prices, prev) + priceRange * 0.15;
  const range = mx - mn;

  const toY = v => H - PAD - ((v - mn) / range) * (H - PAD * 2);

  const times = [
    '4:00 AM','4:30 AM','5:00 AM','5:30 AM','6:00 AM','6:30 AM',
    '7:00 AM','7:30 AM','8:00 AM','8:30 AM','9:00 AM','9:30 AM',
    '10:00 AM','10:30 AM','11:00 AM','11:30 AM','12:01 PM','12:30 PM',
    '1:00 PM','1:30 PM','2:00 PM'
  ];

  const cv  = area.querySelector('.crosshair-v');
  const ch  = area.querySelector('.crosshair-h');
  const tip = area.querySelector('.hover-tip');
  const tipPrice = area.querySelector('.hover-price');
  const tipDate  = area.querySelector('.hover-date');

  if (!cv || !ch || !tip) return;

  area.addEventListener('mousemove', e => {
    const rect = area.getBoundingClientRect();
    const relX = e.clientX - rect.left;
    const relY = e.clientY - rect.top;
    const scaleFactor = rect.width / W;
    const pct  = Math.max(0, Math.min(1,
      (relX - PAD * scaleFactor) / (rect.width - PAD * 2 * scaleFactor)
    ));
    const idx  = Math.min(prices.length - 1, Math.round(pct * (prices.length - 1)));
    const price = prices[idx];
    const priceYpct = toY(price) / H;
    const priceYpx  = priceYpct * rect.height;

    // Show crosshairs
    cv.style.cssText = `display:block; position:absolute; top:0; bottom:0; width:1px;
      background:rgba(255,255,255,0.2); pointer-events:none; left:${relX}px;`;
    ch.style.cssText = `display:block; position:absolute; left:0; right:0; height:1px;
      background:rgba(255,255,255,0.1); pointer-events:none; top:${priceYpx}px;`;

    // Build tooltip using textContent only -- no innerHTML for user data
    tipPrice.textContent = formatPriceLabel(price, ticker);
    tipDate.textContent  = `Mar 3 \xb7 ${times[idx] || times[times.length - 1]} EST`;
    tipPrice.style.color = color;
    tip.style.color      = color;

    // Position tooltip -- flip left if near right edge
    const TW = 150;
    let tx = relX + 12;
    if (tx + TW > rect.width) tx = relX - TW - 12;
    let ty = priceYpx - 44;
    if (ty < 4) ty = priceYpx + 8;

    tip.style.cssText = `
      display:block; position:absolute; left:${tx}px; top:${ty}px;
      background:#0f0f0f; border:1px solid #363636; border-radius:6px;
      padding:6px 10px; pointer-events:none; z-index:20; white-space:nowrap;
      box-shadow:0 4px 16px rgba(0,0,0,0.5);
    `;
  });

  area.addEventListener('mouseleave', () => {
    cv.style.display  = 'none';
    ch.style.display  = 'none';
    tip.style.display = 'none';
  });
}

// Initial render
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initCharts);
} else {
  initCharts();
}

// Re-render after HTMX swaps (afterSwap for innerHTML, afterSettle for outerHTML)
document.body.addEventListener('htmx:afterSwap', initCharts);
document.body.addEventListener('htmx:afterSettle', initCharts);

// Catch async HTMX load-triggered cards that may settle before listeners attach.
// Debounced via requestAnimationFrame to avoid infinite loops (initCharts mutates DOM).
let _chartRAF = 0;
new MutationObserver(function() {
  cancelAnimationFrame(_chartRAF);
  _chartRAF = requestAnimationFrame(initCharts);
}).observe(document.body, { childList: true, subtree: true });
