"""Deployment routes -- Capital Deployment Engine UI.

Routes live under /ui/execute/deployment/... as a subsection of Execute.
All computation goes through DeploymentService -- never import from
src/deployment/ directly.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse

from src.utils.logging import get_logger
from web.routes.ui import check_auth, templates

logger = get_logger(__name__)

router = APIRouter(
    prefix="/ui/execute/deployment",
    dependencies=[Depends(check_auth)],
)

# -- Service layer bootstrap -----------------------------------

_service = None


def _get_service():
    """Lazily initialise and cache the deployment service."""
    global _service
    if _service is not None:
        return _service

    from src.services.deployment_service import DeploymentService

    try:
        from src.deployment.config import load_deployment_config
        config = load_deployment_config()
    except Exception as exc:
        logger.warning("Failed to load deployment config: %s", exc)
        config = {}

    _service = DeploymentService(config=config)
    return _service


# -- Dashboard -------------------------------------------------

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def deployment_dashboard(request: Request):
    """Main deployment dashboard."""
    svc = _get_service()
    status = svc.get_cycle_status()
    memory = svc.get_memory_summary()
    buying_power = svc.get_ib_buying_power()
    if buying_power is None:
        # Fall back to configured capital when IB not connected.
        cfg = svc._ensure_config()
        buying_power = cfg.get("capital", {}).get("total_gbp", 500.0)

    return templates.TemplateResponse("deployment/dashboard.html", {
        "request": request,
        "active_page": "execute",
        "cycle_status": status,
        "capital_available": buying_power,
        "capital_deployed": 0.0,
        "memory_summary": memory,
        "last_cycle": None,
    })


# -- Run cycle -------------------------------------------------

@router.post("/run", response_class=HTMLResponse)
async def run_cycle(
    request: Request,
    dry_run: str = Form("true"),
    use_opus: str = Form("true"),
):
    """Start a new deployment cycle."""
    svc = _get_service()

    # For now, run synchronously with synthetic data.
    # Full background execution will use the orchestrator.
    try:
        from src.deployment.config import load_deployment_config
        from src.deployment.scanner import SyntheticScanner
        from src.deployment.enricher import SyntheticEnricher
        from src.deployment.analyst import KeywordAnalyst, OpusAnalyst
        from src.deployment.sizer import KellySizer
        from src.deployment.memory import MemoryManager
        from src.deployment.orchestrator import DeploymentPipeline

        config = svc._ensure_config()
        memory = MemoryManager.from_config(config)
        scanner = SyntheticScanner()
        enricher = SyntheticEnricher()

        if use_opus == "true":
            analyst = OpusAnalyst(config=config, memory=memory)
        else:
            analyst = KeywordAnalyst(config=config)

        sizer = KellySizer(config=config)
        # Use the service's queue so the queue page sees the same orders.
        queue = svc._ensure_queue()

        pipeline = DeploymentPipeline(
            config=config,
            memory=memory,
            scanner=scanner,
            enricher=enricher,
            analyst=analyst,
            sizer=sizer,
            queue_manager=queue,
            service=svc,
        )

        # Auto-approve for now (checkpoint handled via UI in full version).
        import threading
        import time

        def auto_approve():
            for _ in range(200):
                if pipeline.state == "AWAITING_APPROVAL":
                    pipeline.approve_checkpoint()
                    return
                time.sleep(0.1)

        threading.Thread(target=auto_approve, daemon=True).start()

        result = pipeline.run_cycle(
            dry_run=(dry_run == "true"),
            use_opus=(use_opus == "true"),
        )

        status = svc.get_cycle_status()
        memory_summary = svc.get_memory_summary()

        # Return the inner content partial — HTMX swaps into #deployment-content.
        return templates.TemplateResponse("deployment/_dashboard_content.html", {
            "request": request,
            "active_page": "execute",
            "cycle_status": status,
            "capital_available": 500.0,
            "capital_deployed": result.total_gbp_deployed if result else 0.0,
            "memory_summary": memory_summary,
            "last_cycle": {
                "cycle_id": result.cycle_id if result else "unknown",
                "candidates_scanned": result.candidates_scanned if result else 0,
                "candidates_enriched": result.candidates_enriched if result else 0,
                "candidates_scored": result.candidates_scored if result else 0,
                "candidates_above_threshold": result.candidates_above_threshold if result else 0,
                "trades_queued": result.trades_queued if result else 0,
                "duration_seconds": result.duration_seconds if result else 0,
            },
        })
    except Exception as exc:
        logger.error("Failed to run cycle: %s", exc)
        return HTMLResponse(
            f'<div style="color:var(--red);padding:1rem;">Error: {exc}</div>',
            status_code=500,
        )


# -- Status (HTMX polling) ------------------------------------

@router.get("/status", response_class=HTMLResponse)
async def pipeline_status(request: Request):
    """Return pipeline status fragment for HTMX polling."""
    svc = _get_service()
    status = svc.get_cycle_status()

    state = status.get("state", "IDLE")
    extra = ""
    if state == "AWAITING_APPROVAL":
        extra = ' -- <a href="/ui/execute/deployment/checkpoint" style="color:var(--cyan);">Review Candidates</a>'

    return HTMLResponse(
        f'<div class="deploy-section" '
        f'hx-get="/ui/execute/deployment/status" '
        f'hx-trigger="every 5s" hx-swap="outerHTML">'
        f'<h3>Pipeline Status</h3>'
        f'<p style="font-family:\'IBM Plex Mono\',monospace;color:var(--amber);">'
        f'{state}{extra}</p></div>'
    )


# -- Checkpoint ------------------------------------------------

@router.get("/checkpoint", response_class=HTMLResponse)
async def checkpoint_page(request: Request):
    """Checkpoint review page with scored candidates."""
    svc = _get_service()
    candidates = svc.get_scored_candidates()

    return templates.TemplateResponse("deployment/checkpoint.html", {
        "request": request,
        "active_page": "execute",
        "candidates": candidates,
    })


@router.post("/checkpoint/approve", response_class=HTMLResponse)
async def approve_candidate(request: Request, ticker: str = Form(...)):
    """Approve a single candidate (visual feedback)."""
    return HTMLResponse(
        f'<div class="candidate-card" id="candidate-{ticker}" '
        f'style="border-color:var(--green);opacity:0.7;">'
        f'<p style="color:var(--green);font-weight:600;margin:0;">'
        f'{ticker} -- APPROVED</p></div>'
    )


@router.post("/checkpoint/reject", response_class=HTMLResponse)
async def reject_candidate(request: Request, ticker: str = Form(...)):
    """Reject a single candidate (visual feedback)."""
    return HTMLResponse(
        f'<div class="candidate-card" id="candidate-{ticker}" '
        f'style="border-color:var(--red);opacity:0.5;">'
        f'<p style="color:var(--red);font-weight:600;margin:0;">'
        f'{ticker} -- REJECTED</p></div>'
    )


@router.post("/checkpoint/approve-all", response_class=HTMLResponse)
async def approve_all_candidates(request: Request):
    """Approve all candidates and redirect to queue."""
    svc = _get_service()
    svc.approve_all_orders()
    return HTMLResponse(
        '<p style="color:var(--green);font-weight:600;">All candidates approved. '
        '<a href="/ui/execute/deployment/queue" style="color:var(--cyan);">'
        'View Trade Queue</a></p>'
    )


@router.post("/checkpoint/proceed", response_class=HTMLResponse)
async def proceed_to_sizing(request: Request):
    """Proceed past checkpoint to sizing."""
    return HTMLResponse(
        '<p style="color:var(--cyan);">Proceeding to sizing... '
        '<a href="/ui/execute/deployment/queue" style="color:var(--cyan);">'
        'View Trade Queue</a></p>'
    )


# -- Trade queue -----------------------------------------------

@router.get("/queue", response_class=HTMLResponse)
async def queue_page(request: Request):
    """Trade queue review page."""
    svc = _get_service()
    orders = svc.get_trade_queue()
    summary = svc.get_queue_summary()
    buying_power = svc.get_ib_buying_power()

    return templates.TemplateResponse("deployment/queue.html", {
        "request": request,
        "active_page": "execute",
        "orders": orders,
        "summary": summary,
        "buying_power": buying_power,
    })


@router.post("/queue/approve", response_class=HTMLResponse)
async def approve_order(request: Request, order_id: str = Form(...)):
    """Approve a single order."""
    svc = _get_service()
    svc.approve_order(order_id)
    orders = svc.get_trade_queue()
    # Find and return the updated row.
    for o in orders:
        if o.get("order_id") == order_id:
            return HTMLResponse(
                f'<tr id="order-{order_id}">'
                f'<td style="font-family:\'IBM Plex Mono\',monospace;font-weight:600;">{o["ticker"]}</td>'
                f'<td>{o["direction"]}</td>'
                f'<td style="font-family:\'IBM Plex Mono\',monospace;">{o["quantity"]}</td>'
                f'<td>{o["order_type"]}</td>'
                f'<td style="font-family:\'IBM Plex Mono\',monospace;">'
                f'{"&pound;%.4f" % o["limit_price"] if o.get("limit_price") else "MARKET"}</td>'
                f'<td style="font-family:\'IBM Plex Mono\',monospace;">'
                f'{"&pound;%.2f" % (o["limit_price"] * o["quantity"]) if o.get("limit_price") else "--"}</td>'
                f'<td><span class="order-status order-APPROVED">APPROVED</span></td>'
                f'<td></td></tr>'
            )
    return HTMLResponse("")


@router.post("/queue/reject", response_class=HTMLResponse)
async def reject_order(request: Request, order_id: str = Form(...)):
    """Reject a single order."""
    svc = _get_service()
    svc.reject_order(order_id, "Rejected by user")
    return HTMLResponse(
        f'<tr id="order-{order_id}" style="opacity:0.5;">'
        f'<td colspan="8"><span class="order-status order-REJECTED">REJECTED</span> {order_id}</td>'
        f'</tr>'
    )


@router.post("/queue/approve-all", response_class=HTMLResponse)
async def approve_all_orders(request: Request):
    """Approve all pending orders and refresh queue page."""
    svc = _get_service()
    svc.approve_all_orders()
    # HTMX redirect to reload the full queue page cleanly.
    return HTMLResponse(
        "",
        headers={"HX-Redirect": "/ui/execute/deployment/queue"},
    )


@router.post("/queue/submit", response_class=HTMLResponse)
async def submit_orders(request: Request, dry_run: str = Form("true")):
    """Submit approved orders to IB."""
    svc = _get_service()
    results = svc.confirm_and_submit(dry_run=(dry_run == "true"))

    mode = "DRY RUN" if dry_run == "true" else "LIVE"
    submitted = sum(1 for r in results if r.get("status") == "SUBMITTED")
    failed = sum(1 for r in results if r.get("status") == "FAILED")

    return HTMLResponse(
        f'<div style="padding:1rem;">'
        f'<h3 style="color:var(--green);">Submission Complete ({mode})</h3>'
        f'<p>{submitted} orders submitted, {failed} failed.</p>'
        f'<a href="/ui/execute/deployment" style="color:var(--cyan);">Back to Dashboard</a>'
        f'</div>'
    )


# -- Emergency halt --------------------------------------------

@router.post("/halt", response_class=HTMLResponse)
async def emergency_halt(request: Request):
    """Emergency halt -- cancel all pending orders."""
    svc = _get_service()
    result = svc.emergency_stop()
    cancelled = result.get("orders_cancelled", 0)

    return HTMLResponse(
        f'<div style="padding:1rem;">'
        f'<h3 style="color:var(--red);">EMERGENCY HALT</h3>'
        f'<p>{cancelled} orders cancelled.</p>'
        f'<a href="/ui/execute/deployment" style="color:var(--cyan);">Back to Dashboard</a>'
        f'</div>'
    )


# -- Memory page -----------------------------------------------

@router.get("/memory", response_class=HTMLResponse)
async def memory_page(request: Request):
    """Memory inspection page."""
    svc = _get_service()
    summary = svc.get_memory_summary()
    cost = summary.get("cost_summary", {})
    watchlist = summary.get("watchlist", [])
    excluded = summary.get("excluded_tickers", [])

    # Get sector trends from memory if available.
    sector_trends = {}
    try:
        memory = svc._ensure_memory()
        learnings = memory.get_learnings()
        sector_trends = learnings.get("sector_trends", {})
    except Exception:
        pass

    return templates.TemplateResponse("deployment/memory.html", {
        "request": request,
        "active_page": "execute",
        "summary": summary,
        "cost": cost,
        "watchlist": watchlist,
        "excluded": excluded,
        "sector_trends": sector_trends,
    })
