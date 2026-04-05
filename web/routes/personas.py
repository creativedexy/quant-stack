"""FastAPI router for the Personas page (/ui/personas paths).

Renders persona cards, detail views, and handles activation/deactivation
via HTMX. All computation is done in PersonaService -- templates only render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.services.persona_service import PersonaService
from src.utils.config import load_config
from src.utils.logging import get_logger
from web.routes.ui import check_auth

logger = get_logger(__name__)

_WEB_DIR = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _WEB_DIR / "templates"

router = APIRouter(prefix="/ui/personas", dependencies=[Depends(check_auth)])
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Lazy-initialised service
_svc: PersonaService | None = None


def _get_service() -> PersonaService:
    """Lazy-initialise the persona service."""
    global _svc
    if _svc is None:
        _svc = PersonaService(load_config())
    return _svc


# -- State badge colour mapping -----------------------------------

_STATE_CLASSES = {
    "active": "badge-green",
    "pending": "badge-amber",
    "dormant": "badge-grey",
    "paused": "badge-blue",
}


# ------------------------------------------------------------------
# Full page
# ------------------------------------------------------------------

@router.get("", response_class=HTMLResponse)
async def personas_page(request: Request) -> HTMLResponse:
    """Render the personas card grid page."""
    svc = _get_service()
    personas = svc.get_all_personas()
    capital = svc.get_capital_summary()

    # Annotate each persona with a CSS badge class
    for p in personas:
        p["state_class"] = _STATE_CLASSES.get(p.get("state", ""), "badge-grey")

    return templates.TemplateResponse(
        "personas.html",
        {
            "request": request,
            "active_page": "research",
            "personas": personas,
            "capital": capital,
        },
    )


# ------------------------------------------------------------------
# Detail page
# ------------------------------------------------------------------

@router.get("/{persona_id}", response_class=HTMLResponse)
async def persona_detail(request: Request, persona_id: str) -> HTMLResponse:
    """Render the detail page for a single persona."""
    svc = _get_service()
    detail = svc.get_persona_detail(persona_id)

    if detail is None:
        return templates.TemplateResponse(
            "personas.html",
            {
                "request": request,
                "active_page": "research",
                "personas": svc.get_all_personas(),
                "capital": svc.get_capital_summary(),
                "error": f"Persona '{persona_id}' not found.",
            },
        )

    detail["state_class"] = _STATE_CLASSES.get(
        detail.get("state", ""), "badge-grey"
    )

    return templates.TemplateResponse(
        "persona_detail.html",
        {
            "request": request,
            "active_page": "research",
            "persona": detail,
        },
    )


# ------------------------------------------------------------------
# HTMX actions (return partial card)
# ------------------------------------------------------------------

@router.post("/{persona_id}/activate", response_class=HTMLResponse)
async def activate_persona(
    request: Request, persona_id: str
) -> HTMLResponse:
    """Activate a persona and return the updated card partial."""
    svc = _get_service()
    result = svc.activate_persona(persona_id)
    personas = svc.get_all_personas()

    # Find the updated persona
    persona = next(
        (p for p in personas if p["persona_id"] == persona_id), None
    )
    if persona:
        persona["state_class"] = _STATE_CLASSES.get(
            persona.get("state", ""), "badge-grey"
        )

    return templates.TemplateResponse(
        "partials/persona_card.html",
        {
            "request": request,
            "persona": persona or {"persona_id": persona_id, "name": persona_id},
            "message": result.get("message", ""),
        },
    )


@router.post("/{persona_id}/deactivate", response_class=HTMLResponse)
async def deactivate_persona(
    request: Request, persona_id: str
) -> HTMLResponse:
    """Deactivate a persona and return the updated card partial."""
    svc = _get_service()
    result = svc.deactivate_persona(persona_id)
    personas = svc.get_all_personas()

    persona = next(
        (p for p in personas if p["persona_id"] == persona_id), None
    )
    if persona:
        persona["state_class"] = _STATE_CLASSES.get(
            persona.get("state", ""), "badge-grey"
        )

    return templates.TemplateResponse(
        "partials/persona_card.html",
        {
            "request": request,
            "persona": persona or {"persona_id": persona_id, "name": persona_id},
            "message": result.get("message", ""),
        },
    )


@router.post("/{persona_id}/pause", response_class=HTMLResponse)
async def pause_persona(
    request: Request, persona_id: str
) -> HTMLResponse:
    """Pause a persona and return the updated card partial."""
    svc = _get_service()
    result = svc.pause_persona(persona_id)
    personas = svc.get_all_personas()

    persona = next(
        (p for p in personas if p["persona_id"] == persona_id), None
    )
    if persona:
        persona["state_class"] = _STATE_CLASSES.get(
            persona.get("state", ""), "badge-grey"
        )

    return templates.TemplateResponse(
        "partials/persona_card.html",
        {
            "request": request,
            "persona": persona or {"persona_id": persona_id, "name": persona_id},
            "message": result.get("message", ""),
        },
    )


@router.post("/{persona_id}/brief", response_class=HTMLResponse)
async def generate_brief(
    request: Request, persona_id: str
) -> HTMLResponse:
    """Generate a brief now and return the brief panel."""
    svc = _get_service()
    brief = svc.generate_brief_now(persona_id)

    return templates.TemplateResponse(
        "partials/persona_brief.html",
        {
            "request": request,
            "brief": brief,
            "persona_id": persona_id,
        },
    )
