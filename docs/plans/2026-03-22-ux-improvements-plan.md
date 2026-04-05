# UX Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three UX issues — add Research navbar tab, add Actions dropdown with upload/import/quick-add modals, make DCA purchase form always visible.

**Architecture:** Extends the existing FastAPI + HTMX + Alpine.js patterns. Modals are HTMX fragments loaded into a shared container in base.html. Uploads use FastAPI's `UploadFile`. All new endpoints live in `web/routes/ui.py` alongside existing routes.

**Tech Stack:** FastAPI, Jinja2, HTMX, Alpine.js, existing DCAService/DCAStorage/ResearchLog

---

### Task 1: Add Research Tab to Navbar

**Files:**
- Modify: `web/templates/base.html:38-60`
- Test: `tests/test_web/test_ui_routes.py`

**Step 1: Write the failing test**

Add to `tests/test_web/test_ui_routes.py` in `TestOverviewPage`:

```python
def test_contains_research_nav_link(self, client: TestClient) -> None:
    resp = client.get("/ui/overview")
    assert "Research" in resp.text
    assert "/ui/research" in resp.text
```

Add a new test class:

```python
class TestResearchTabHighlight:
    """Research page highlights its own tab, not Analyse."""

    def test_research_tab_is_active(self, client: TestClient) -> None:
        resp = client.get("/ui/research")
        # Research should be the active link, not Analyse
        text = resp.text
        # Find the Research nav-link and check it has 'active'
        assert 'href="/ui/research">Research' in text

    def test_analyse_tab_not_active_on_research(self, client: TestClient) -> None:
        resp = client.get("/ui/research")
        # Analyse should NOT be active when viewing research
        assert 'class="nav-link active"' in resp.text
        # The active link should contain Research
        import re
        active_link = re.search(r'class="nav-link active"[^>]*href="([^"]+)"', resp.text)
        if active_link:
            assert active_link.group(1) == "/ui/research"
```

**Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_web/test_ui_routes.py::TestResearchTabHighlight -v`
Expected: FAIL (Research tab doesn't exist yet, `/ui/research` highlights Analyse)

**Step 3: Update base.html navbar**

In `web/templates/base.html`, modify the `tabs` list (line 38) to add Research between Portfolio and Analyse:

```jinja2
{%- set tabs = [
  ('dashboard', 'Dashboard', '/ui/overview'),
  ('watchlist', 'Watchlist', '/ui/watchlist'),
  ('portfolio', 'Portfolio', '/ui/portfolio'),
  ('research',  'Research',  '/ui/research'),
  ('analyse',   'Analyse',   '/ui/analyse'),
  ('execute',   'Execute',   '/ui/execution'),
] -%}
```

Update `_path_map` (line 46) — change `/ui/research` from `analyse` to `research`:

```jinja2
{%- set _path_map = {
  '/ui/overview': 'dashboard', '/ui/strategy': 'analyse',
  '/ui/watchlist': 'watchlist', '/ui/chart': 'analyse',
  '/ui/portfolio': 'portfolio', '/ui/research': 'research',
  '/ui/analyse': 'analyse', '/ui/execution': 'execute',
  '/ui/news': 'dashboard',
} -%}
```

Update `_page_map` (line 53) — change `research` from `analyse` to `research`:

```jinja2
{%- set _page_map = {
  'overview': 'dashboard', 'strategy': 'analyse',
  'watchlist': 'watchlist', 'chart': 'analyse',
  'portfolio': 'portfolio', 'research': 'research',
  'analyse': 'analyse', 'execution': 'execute',
  'news': 'dashboard', 'execute': 'execute',
  'dashboard': 'dashboard',
} -%}
```

**Step 4: Update the existing nav links test**

In `tests/test_web/test_ui_routes.py`, `TestOverviewPage.test_contains_nav_links` (line 44), add "Research" to the checked links:

```python
def test_contains_nav_links(self, client: TestClient) -> None:
    resp = client.get("/ui/overview")
    for link in ("Dashboard", "Watchlist", "Portfolio", "Research", "Analyse", "Execute"):
        assert link in resp.text
```

**Step 5: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_web/test_ui_routes.py -v -k "nav_link or Research"`
Expected: PASS

**Step 6: Commit**

```bash
git add web/templates/base.html tests/test_web/test_ui_routes.py
git commit -m "feat(ui): add Research as own navbar tab"
```

---

### Task 2: Add Modal Container + CSS to base.html

**Files:**
- Modify: `web/templates/base.html:88-98`
- Modify: `web/static/css/theme.css` (append)

**Step 1: Add modal container div to base.html**

After the `</nav>` tag (line 88) and before `{% block ticker_bar %}` (line 90), add:

```html
  {# Modal container -- HTMX loads modal partials here #}
  <div id="modal-container" onclick="if(event.target===this)this.innerHTML=''"></div>
```

**Step 2: Add Actions dropdown to navbar**

In `base.html`, after the closing `</div>` of `.nav-links` (line 83) and before the LSE status div (line 84), add the Actions dropdown:

```html
      <div class="nav-actions" x-data="{ actionsOpen: false }">
        <button class="btn-actions" x-on:click="actionsOpen = !actionsOpen"
                x-on:click.outside="actionsOpen = false">
          + Actions
        </button>
        <div class="actions-dropdown" x-show="actionsOpen" x-cloak x-transition>
          <button class="actions-item"
                  hx-get="/ui/partials/modal-upload-research"
                  hx-target="#modal-container"
                  x-on:click="actionsOpen = false">
            Upload Research
          </button>
          <button class="actions-item"
                  hx-get="/ui/partials/modal-import-purchases"
                  hx-target="#modal-container"
                  x-on:click="actionsOpen = false">
            Import Purchases (CSV)
          </button>
          <button class="actions-item"
                  hx-get="/ui/partials/modal-add-purchase"
                  hx-target="#modal-container"
                  x-on:click="actionsOpen = false">
            Add Purchase
          </button>
        </div>
      </div>
```

**Step 3: Add CSS for modal and dropdown**

Append to `web/static/css/theme.css`:

```css
/* ── Actions dropdown ────────────────────────── */

.nav-actions {
  position: relative;
  margin-left: 0.75rem;
}

.btn-actions {
  background: var(--accent);
  color: #000;
  border: none;
  padding: 0.3rem 0.75rem;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 600;
  font-family: var(--sans);
  cursor: pointer;
}

.btn-actions:hover {
  filter: brightness(1.1);
}

.actions-dropdown {
  position: absolute;
  right: 0;
  top: 100%;
  margin-top: 0.25rem;
  background: var(--card);
  border: 1px solid var(--border-hi);
  border-radius: 6px;
  min-width: 200px;
  z-index: 100;
  overflow: hidden;
}

.actions-item {
  display: block;
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  color: var(--t1);
  padding: 0.6rem 1rem;
  font-size: 0.8rem;
  font-family: var(--sans);
  cursor: pointer;
}

.actions-item:hover {
  background: var(--card-hi);
}

/* ── Modal overlay ───────────────────────────── */

#modal-container:empty {
  display: none;
}

#modal-container {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.6);
  z-index: 200;
  display: flex;
  align-items: center;
  justify-content: center;
}

.modal-card {
  background: var(--card);
  border: 1px solid var(--border-hi);
  border-radius: 8px;
  padding: 1.5rem;
  width: min(480px, 90vw);
  max-height: 80vh;
  overflow-y: auto;
}

.modal-card h3 {
  margin: 0 0 1rem;
  font-size: 1rem;
  color: var(--t1);
}

.modal-card label {
  font-size: 0.75rem;
  display: block;
  margin-bottom: 0.25rem;
  color: var(--t2);
}

.modal-card .btn-accent {
  margin-top: 0.75rem;
}

.modal-success {
  color: var(--positive);
  font-size: 0.85rem;
  padding: 0.5rem 0;
}

.modal-error {
  color: var(--negative);
  font-size: 0.85rem;
  padding: 0.5rem 0;
}
```

**Step 4: Write test for Actions dropdown presence**

Add to `tests/test_web/test_ui_routes.py`:

```python
class TestActionsDropdown:
    """Actions dropdown appears in navbar."""

    def test_actions_button_present(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "+ Actions" in resp.text

    def test_modal_container_present(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert 'id="modal-container"' in resp.text
```

**Step 5: Run tests**

Run: `py -3 -m pytest tests/test_web/test_ui_routes.py::TestActionsDropdown -v`
Expected: PASS

**Step 6: Commit**

```bash
git add web/templates/base.html web/static/css/theme.css tests/test_web/test_ui_routes.py
git commit -m "feat(ui): add Actions dropdown and modal container to navbar"
```

---

### Task 3: Upload Research Modal + Endpoint

**Files:**
- Create: `web/templates/partials/modal_upload_research.html`
- Modify: `web/routes/ui.py` (add 2 endpoints + `UploadFile` import)
- Test: `tests/test_web/test_ui_routes.py`

**Step 1: Write the failing tests**

Add to `tests/test_web/test_ui_routes.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_web/test_ui_routes.py::TestUploadResearch -v`
Expected: FAIL (endpoints don't exist)

**Step 3: Create the modal template**

Create `web/templates/partials/modal_upload_research.html`:

```html
<div class="modal-card">
  <h3>Upload Research</h3>
  <form hx-post="/ui/upload/research"
        hx-target="#modal-container"
        hx-encoding="multipart/form-data">

    <label>File (.pdf, .txt, .md)</label>
    <input type="file" name="file" accept=".pdf,.txt,.md" required
           class="chart-select" style="width: 100%; margin-bottom: 0.5rem;">

    <label>Ticker (optional)</label>
    <input type="text" name="ticker" placeholder="e.g. CNDX.L"
           class="chart-select" style="width: 100%; margin-bottom: 0.5rem;">

    <label>Notes (optional)</label>
    <input type="text" name="notes" placeholder="What is this document about?"
           class="chart-select" style="width: 100%; margin-bottom: 0.5rem;">

    <button type="submit" class="btn btn-sm btn-accent" style="width: 100%;">
      Upload
    </button>
  </form>
</div>
```

**Step 4: Add route for modal partial + upload endpoint**

Add `UploadFile` to the imports in `web/routes/ui.py` (line 22):

```python
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, UploadFile, status
```

Add these routes after the research partials section (after line 1482):

```python
@router.get("/partials/modal-upload-research", response_class=HTMLResponse)
async def modal_upload_research(request: Request) -> HTMLResponse:
    """HTMX partial -- upload research modal."""
    return templates.TemplateResponse(
        "partials/modal_upload_research.html",
        {"request": request},
    )


@router.post("/upload/research", response_class=HTMLResponse)
async def upload_research(
    request: Request,
    file: UploadFile,
    ticker: str = Form(""),
    notes: str = Form(""),
) -> HTMLResponse:
    """Handle research document upload.

    Saves file to data/research/ and appends an entry to the research log.
    """
    # Validate extension
    allowed = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        return HTMLResponse(
            '<div class="modal-card"><p class="modal-error">'
            f"Unsupported file type: {suffix}. Use .pdf, .txt, or .md</p></div>",
            status_code=422,
        )

    # Save file
    research_dir = Path("data/research")
    research_dir.mkdir(parents=True, exist_ok=True)
    dest = research_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)

    # Append to research log
    svcs = _get_services()
    research_log = svcs.get("research_log")
    if research_log is not None:
        data_summary: dict[str, Any] = {
            "file": file.filename,
            "size_bytes": len(content),
        }
        if ticker:
            data_summary["ticker"] = ticker
        research_log.log_entry(
            notebook="web_upload",
            task="document_upload",
            data_summary=data_summary,
            interpretation={"summary": notes or f"Uploaded {file.filename}"},
            notes=notes,
        )

    return HTMLResponse(
        '<div class="modal-card"><p class="modal-success">'
        f"Successfully uploaded {file.filename}</p>"
        '<button class="btn btn-sm" onclick="document.getElementById(\'modal-container\').innerHTML=\'\'">'
        "Close</button></div>"
    )
```

**Step 5: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_web/test_ui_routes.py::TestUploadResearch -v`
Expected: PASS

**Step 6: Commit**

```bash
git add web/templates/partials/modal_upload_research.html web/routes/ui.py tests/test_web/test_ui_routes.py
git commit -m "feat(ui): add research document upload modal and endpoint"
```

---

### Task 4: CSV Purchase Import Modal + Endpoints

**Files:**
- Create: `web/templates/partials/modal_import_purchases.html`
- Create: `web/templates/partials/import_preview_table.html`
- Modify: `web/routes/ui.py` (add 3 endpoints)
- Test: `tests/test_web/test_ui_routes.py`

**Step 1: Write the failing tests**

Add to `tests/test_web/test_ui_routes.py`:

```python
import io


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
        # Should show error message
        assert "missing" in resp.text.lower() or "error" in resp.text.lower()

    def test_confirm_imports_rows(self, client: TestClient) -> None:
        # Use a unique date to avoid conflicts with existing data
        csv_content = "date,ticker,price,amount_gbp,note\n2099-12-31,TEST.L,100,50,Import test\n"
        resp = client.post(
            "/ui/upload/purchases/confirm",
            data={"csv_data": csv_content},
        )
        assert resp.status_code == 200
        assert "imported" in resp.text.lower() or "1" in resp.text
```

**Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_web/test_ui_routes.py::TestImportPurchases -v`
Expected: FAIL

**Step 3: Create import modal template**

Create `web/templates/partials/modal_import_purchases.html`:

```html
<div class="modal-card">
  <h3>Import Purchases (CSV)</h3>
  <p class="muted" style="font-size: 0.75rem; margin-bottom: 0.75rem;">
    Expected columns: <code>date,ticker,price,amount_gbp,note</code><br>
    Price in pence. Amount in GBP. Note is optional.
  </p>
  <form hx-post="/ui/upload/purchases/preview"
        hx-target="#import-preview"
        hx-encoding="multipart/form-data">

    <label>CSV File</label>
    <input type="file" name="file" accept=".csv" required
           class="chart-select" style="width: 100%; margin-bottom: 0.5rem;">

    <button type="submit" class="btn btn-sm btn-accent" style="width: 100%;">
      Preview
    </button>
  </form>
  <div id="import-preview" style="margin-top: 0.75rem;"></div>
</div>
```

**Step 4: Create preview table template**

Create `web/templates/partials/import_preview_table.html`:

```html
{% if error %}
<p class="modal-error">{{ error }}</p>
{% else %}
<table class="feature-table" style="width: 100%; font-size: 0.8rem; margin-bottom: 0.75rem;">
  <thead>
    <tr>
      <th>Date</th>
      <th>Ticker</th>
      <th class="text-right">Price (p)</th>
      <th class="text-right">Amount</th>
      <th>Note</th>
    </tr>
  </thead>
  <tbody>
    {% for row in rows %}
    <tr>
      <td class="font-mono">{{ row.date }}</td>
      <td>{{ row.ticker }}</td>
      <td class="text-right font-mono">{{ row.price }}</td>
      <td class="text-right font-mono">{{ row.amount_gbp }}</td>
      <td class="muted" style="font-size: 0.7rem;">{{ row.note }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
<p class="muted" style="font-size: 0.75rem;">{{ rows | length }} row(s) ready to import. Duplicates will be skipped.</p>
<form hx-post="/ui/upload/purchases/confirm"
      hx-target="#modal-container">
  <input type="hidden" name="csv_data" value="{{ csv_raw }}">
  <button type="submit" class="btn btn-sm btn-accent" style="width: 100%;">
    Import All
  </button>
</form>
{% endif %}
```

**Step 5: Add the 3 route handlers**

Add to `web/routes/ui.py` after the research upload endpoint:

```python
@router.get("/partials/modal-import-purchases", response_class=HTMLResponse)
async def modal_import_purchases(request: Request) -> HTMLResponse:
    """HTMX partial -- CSV import modal."""
    return templates.TemplateResponse(
        "partials/modal_import_purchases.html",
        {"request": request},
    )


@router.post("/upload/purchases/preview", response_class=HTMLResponse)
async def upload_purchases_preview(
    request: Request,
    file: UploadFile,
) -> HTMLResponse:
    """Parse uploaded CSV and return a preview table."""
    import csv as csv_mod
    import io as io_mod

    content = (await file.read()).decode("utf-8-sig")
    reader = csv_mod.DictReader(io_mod.StringIO(content))

    required = {"date", "ticker", "price", "amount_gbp"}
    if not required.issubset(set(reader.fieldnames or [])):
        missing = required - set(reader.fieldnames or [])
        return templates.TemplateResponse(
            "partials/import_preview_table.html",
            {"request": request, "error": f"Missing columns: {', '.join(sorted(missing))}", "rows": []},
        )

    rows = []
    for row in reader:
        try:
            rows.append({
                "date": row["date"].strip(),
                "ticker": row["ticker"].strip(),
                "price": float(row["price"]),
                "amount_gbp": float(row["amount_gbp"]),
                "note": row.get("note", "").strip(),
            })
        except (ValueError, KeyError):
            continue

    if not rows:
        return templates.TemplateResponse(
            "partials/import_preview_table.html",
            {"request": request, "error": "No valid rows found in CSV.", "rows": []},
        )

    return templates.TemplateResponse(
        "partials/import_preview_table.html",
        {"request": request, "rows": rows, "csv_raw": content, "error": None},
    )


@router.post("/upload/purchases/confirm", response_class=HTMLResponse)
async def upload_purchases_confirm(
    request: Request,
    csv_data: str = Form(...),
) -> HTMLResponse:
    """Bulk-import purchases from pre-validated CSV data."""
    import csv as csv_mod
    import io as io_mod

    svcs = _get_services()
    dca_svc = svcs["dca"]

    reader = csv_mod.DictReader(io_mod.StringIO(csv_data))
    imported = 0
    skipped = 0

    for row in reader:
        try:
            dca_svc.add_purchase(
                ticker=row["ticker"].strip(),
                date=row["date"].strip(),
                price=float(row["price"]),
                amount_gbp=float(row["amount_gbp"]),
                note=row.get("note", "").strip(),
            )
            imported += 1
        except ValueError:
            skipped += 1

    return HTMLResponse(
        '<div class="modal-card">'
        f'<p class="modal-success">{imported} purchase(s) imported.</p>'
        + (f'<p class="muted" style="font-size:0.75rem;">{skipped} duplicate(s) skipped.</p>' if skipped else "")
        + '<button class="btn btn-sm" onclick="document.getElementById(\'modal-container\').innerHTML=\'\'">'
        "Close</button></div>"
    )
```

**Step 6: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_web/test_ui_routes.py::TestImportPurchases -v`
Expected: PASS

**Step 7: Commit**

```bash
git add web/templates/partials/modal_import_purchases.html web/templates/partials/import_preview_table.html web/routes/ui.py tests/test_web/test_ui_routes.py
git commit -m "feat(ui): add CSV purchase import modal with preview and bulk insert"
```

---

### Task 5: Quick-Add Purchase Modal

**Files:**
- Create: `web/templates/partials/modal_add_purchase.html`
- Modify: `web/routes/ui.py` (add 1 endpoint)
- Test: `tests/test_web/test_ui_routes.py`

**Step 1: Write the failing test**

Add to `tests/test_web/test_ui_routes.py`:

```python
class TestQuickAddPurchase:
    """Quick-add purchase modal from navbar."""

    def test_modal_partial_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/partials/modal-add-purchase")
        assert resp.status_code == 200
        assert "Add Purchase" in resp.text
        assert "ticker" in resp.text.lower()

    def test_quick_add_submits_purchase(self, client: TestClient) -> None:
        resp = client.post(
            "/ui/portfolio/dca/purchase",
            data={
                "ticker": "TEST.L",
                "date": "2099-06-15",
                "price": "500",
                "amount_gbp": "200",
                "note": "Quick add test",
            },
        )
        # Existing endpoint returns 200 with row partial on success
        assert resp.status_code == 200
```

**Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_web/test_ui_routes.py::TestQuickAddPurchase::test_modal_partial_returns_200 -v`
Expected: FAIL (endpoint doesn't exist)

**Step 3: Create the modal template**

Create `web/templates/partials/modal_add_purchase.html`:

```html
<div class="modal-card">
  <h3>Add Purchase</h3>
  <form hx-post="/ui/portfolio/dca/purchase"
        hx-target="#modal-container"
        hx-on::after-request="if(event.detail.successful){document.getElementById('modal-container').innerHTML='<div class=\'modal-card\'><p class=\'modal-success\'>Purchase saved.</p><button class=\'btn btn-sm\' onclick=\'document.getElementById(\\\'modal-container\\\').innerHTML=\\\'\\\'\'>Close</button></div>'}">

    <label>Ticker</label>
    <input type="text" name="ticker" required placeholder="e.g. CNDX.L"
           class="chart-select" style="width: 100%; margin-bottom: 0.5rem;">

    <label>Date</label>
    <input type="date" name="date" required
           class="chart-select" style="width: 100%; margin-bottom: 0.5rem;">

    <label>Price (pence)</label>
    <input type="number" name="price" step="0.01" min="0" required
           class="chart-select" style="width: 100%; margin-bottom: 0.5rem;">

    <label>Amount (GBP)</label>
    <input type="number" name="amount_gbp" step="0.01" min="0" required
           class="chart-select" style="width: 100%; margin-bottom: 0.5rem;">

    <label>Note</label>
    <input type="text" name="note" placeholder="Optional"
           class="chart-select" style="width: 100%; margin-bottom: 0.5rem;">

    <button type="submit" class="btn btn-sm btn-accent" style="width: 100%;">
      Save
    </button>
  </form>
</div>
```

**Step 4: Add route for modal partial**

Add to `web/routes/ui.py` after the import purchases endpoints:

```python
@router.get("/partials/modal-add-purchase", response_class=HTMLResponse)
async def modal_add_purchase(request: Request) -> HTMLResponse:
    """HTMX partial -- quick-add purchase modal."""
    return templates.TemplateResponse(
        "partials/modal_add_purchase.html",
        {"request": request},
    )
```

**Step 5: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_web/test_ui_routes.py::TestQuickAddPurchase -v`
Expected: PASS

**Step 6: Commit**

```bash
git add web/templates/partials/modal_add_purchase.html web/routes/ui.py tests/test_web/test_ui_routes.py
git commit -m "feat(ui): add quick-add purchase modal accessible from navbar"
```

---

### Task 6: Make DCA Sidebar Form Always Visible

**Files:**
- Modify: `web/templates/partials/portfolio_dca.html:77-110`
- Test: `tests/test_web/test_ui_routes.py`

**Step 1: Write the failing test**

Add to `tests/test_web/test_ui_routes.py`:

```python
class TestDCAFormAlwaysVisible:
    """DCA purchase form should be visible without toggle."""

    def test_form_has_no_x_show(self, client: TestClient) -> None:
        # Load a DCA detail page
        resp = client.get("/ui/portfolio/dca/ticker/CNDX.L")
        if resp.status_code == 200:
            # The form should NOT have x-show (always visible)
            assert 'x-show="showForm"' not in resp.text
```

**Step 2: Run to verify it fails**

Run: `py -3 -m pytest tests/test_web/test_ui_routes.py::TestDCAFormAlwaysVisible -v`
Expected: FAIL (form currently uses `x-show`)

**Step 3: Modify portfolio_dca.html**

In `web/templates/partials/portfolio_dca.html`, replace lines 77-110 (the form card):

Replace this:
```html
    {# Add purchase form #}
    <div class="card" x-data="{ showForm: false }">
      <button class="btn btn-sm btn-accent" style="width: 100%;"
              x-on:click="showForm = !showForm"
              x-text="showForm ? 'Cancel' : 'Add Purchase'"></button>

      <form x-show="showForm" x-cloak
            style="margin-top: 0.75rem;"
```

With this:
```html
    {# Add purchase form — always visible #}
    <div class="card">
      <h4 style="margin-bottom: 0.5rem;">Add Purchase</h4>

      <form style="margin-top: 0;"
```

Remove the `x-on::after-request` reset (keep `hx-on::after-request`) — the existing HTMX reset is fine.

**Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_web/test_ui_routes.py::TestDCAFormAlwaysVisible -v`
Expected: PASS

**Step 5: Commit**

```bash
git add web/templates/partials/portfolio_dca.html tests/test_web/test_ui_routes.py
git commit -m "feat(ui): make DCA purchase form always visible (remove toggle)"
```

---

### Task 7: Run Full Test Suite + Visual Verification

**Step 1: Run the full test suite**

Run: `py -3 -m pytest tests/ -v --ignore=tests/test_models/test_automl.py`
Expected: 0 failures

**Step 2: Visual verification via preview**

- Navigate to `/ui/overview` — verify 6 navbar tabs (Dashboard, Watchlist, Portfolio, Research, Analyse, Execute)
- Verify "+ Actions" dropdown appears and opens
- Click "Upload Research" — verify modal opens
- Click "Import Purchases" — verify modal opens
- Click "Add Purchase" — verify modal opens
- Navigate to `/ui/research` — verify "Research" tab is highlighted (not Analyse)
- Navigate to `/ui/portfolio`, open DCA for a ticker — verify purchase form is always visible

**Step 3: Final commit**

If any fixes were needed, commit them. Otherwise, tag the feature as complete.
