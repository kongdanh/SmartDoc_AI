"""
SmartDoc AI — Dashboard Router.

Serves a simple HTML dashboard for server health & domain overview.
Kept minimal — the main UI is the Streamlit app.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Lightweight HTML dashboard (redirect users to Streamlit for full UI)."""
    html = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>SmartDoc AI — Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Inter', -apple-system, sans-serif;
                background: #0f1117; color: #e4e6f0;
                display: flex; align-items: center; justify-content: center;
                min-height: 100vh;
            }
            .card {
                background: linear-gradient(145deg, #1a1d27, #242836);
                border: 1px solid #2e3348; border-radius: 16px;
                padding: 48px; text-align: center; max-width: 520px;
                box-shadow: 0 8px 40px rgba(0,0,0,0.3);
            }
            h1 {
                font-size: 28px; font-weight: 800;
                background: linear-gradient(135deg, #6c8aff, #4caf84);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                margin-bottom: 8px;
            }
            .subtitle { color: #9498b0; font-size: 14px; margin-bottom: 24px; }
            .status {
                display: inline-flex; align-items: center; gap: 8px;
                background: rgba(76, 175, 132, 0.15); color: #4caf84;
                padding: 8px 20px; border-radius: 20px; font-weight: 600;
                font-size: 14px; margin-bottom: 24px;
            }
            .dot {
                width: 8px; height: 8px; border-radius: 50%;
                background: #4caf84; box-shadow: 0 0 8px rgba(76,175,132,0.6);
            }
            .link {
                display: inline-block; margin-top: 16px; padding: 12px 32px;
                background: linear-gradient(135deg, #6c8aff, #3d5afe);
                color: #fff; text-decoration: none; border-radius: 10px;
                font-weight: 600; font-size: 14px;
                box-shadow: 0 4px 15px rgba(61, 90, 254, 0.4);
                transition: transform 0.2s;
            }
            .link:hover { transform: translateY(-2px); }
            p { color: #9498b0; font-size: 13px; line-height: 1.6; margin-top: 16px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🧠 SmartDoc AI</h1>
            <div class="subtitle">GraphRAG Knowledge Management System</div>
            <div class="status"><div class="dot"></div> Server Online</div>
            <br>
            <a class="link" href="http://localhost:8501" target="_blank">
                🚀 Open Full Dashboard (Streamlit)
            </a>
            <p>
                API Docs: <a href="/docs" style="color:#6c8aff">/docs</a> |
                Health: <a href="/health" style="color:#6c8aff">/health</a>
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
