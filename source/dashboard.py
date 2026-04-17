"""Serves the single-page admin dashboard for KnowledgeDB."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])

_TEMPLATE_PATH = Path(__file__).parent / "dashboard.html"


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return HTMLResponse(html)
