# UX Improvements Design — 2026-03-22

## Problem

Three UX issues in the FastAPI + HTMX dashboard:

1. **Research page is hidden** — no navbar tab, highlights "Analyse" instead, only reachable via overview tile
2. **No upload capability** — research tile promises "Upload papers" but no upload exists anywhere
3. **Adding purchases is clunky** — form hidden behind toggle, only accessible from DCA detail sidebar, no bulk import

## Solution: Navbar Actions Hub

### 1. Research Navigation

Add **"Research"** as 6th navbar tab between Portfolio and Analyse.

**Files to modify:**
- `web/templates/base.html` — add tab to `tabs` list, update `_page_map` and `_path_map`

```
Before: Dashboard | Watchlist | Portfolio | Analyse | Execute
After:  Dashboard | Watchlist | Portfolio | Research | Analyse | Execute | [+ Actions]
```

### 2. Actions Dropdown (Navbar)

Add a **"+ Actions"** dropdown button to the right side of the navbar (before LSE status).
Alpine.js `x-data="{ actionsOpen: false }"` toggles a dropdown menu with three items:

- **Upload Research** — opens upload modal
- **Import Purchases (CSV)** — opens CSV import modal
- **Add Purchase** — opens quick-add purchase modal

**Files to modify:**
- `web/templates/base.html` — add dropdown markup after nav-links div
- `web/static/css/theme.css` — dropdown and modal styles

### 3. Upload Research Modal

HTMX-driven modal for uploading research documents.

**UI:**
- File picker (`.pdf`, `.txt`, `.md`)
- Optional ticker association (dropdown)
- Optional notes field
- Submit button

**Backend:**
- `POST /ui/upload/research` — saves file to `data/research/`, appends entry to `data/processed/research_log.jsonl`
- Returns success HTML fragment that closes modal

**Files to create:**
- `web/templates/partials/modal_upload_research.html`
- `data/research/` directory

**Files to modify:**
- `web/routes/ui.py` — add upload endpoint

### 4. CSV Purchase Import Modal

HTMX-driven modal for bulk-importing purchase history.

**UI:**
- File picker (`.csv`)
- Preview table of parsed rows (rendered server-side after upload)
- "Import All" button
- Success summary (X imported, Y skipped as duplicates)

**Expected CSV format:**
```
date,ticker,price,amount_gbp,note
2025-03-01,CNDX.L,2500,1000,Monthly DCA
```

**Backend:**
- `POST /ui/upload/purchases/preview` — parses CSV, returns preview table HTML
- `POST /ui/upload/purchases/confirm` — bulk-inserts via DCAStorage, returns summary

**Files to create:**
- `web/templates/partials/modal_import_purchases.html`
- `web/templates/partials/import_preview_table.html`

**Files to modify:**
- `web/routes/ui.py` — add preview + confirm endpoints

### 5. Quick-Add Purchase Modal

Same fields as existing DCA sidebar form, but accessible from any page via navbar.

**UI:**
- Ticker selector (dropdown of all DCA tickers)
- Date, Price (pence), Amount (GBP), Note
- Submit button

**Backend:**
- Reuses existing `POST /ui/portfolio/dca/purchase` endpoint
- Modal closes on success, shows brief confirmation

**Files to create:**
- `web/templates/partials/modal_add_purchase.html`

### 6. Always-Visible DCA Sidebar Form

Remove the Alpine.js `x-show` toggle from the existing DCA purchase form in
`portfolio_dca.html`. The form is always visible — no "Add Purchase" button needed.

**Files to modify:**
- `web/templates/partials/portfolio_dca.html` — remove `x-data`, `x-show`, `x-cloak`, toggle button

## Modal Pattern

All three modals share a common pattern:
- Rendered as HTMX fragments loaded into a shared `#modal-container` div in `base.html`
- Backdrop click or X button closes modal (`hx-on:click="this.innerHTML=''"` on container)
- Forms POST via HTMX, responses replace modal content with success/error message

## Testing

- Test upload endpoint with synthetic PDF
- Test CSV import with valid/invalid/duplicate rows
- Test quick-add purchase from navbar
- Verify Research tab highlights correctly on `/ui/research`
- Verify all 6 navbar tabs render and link correctly
