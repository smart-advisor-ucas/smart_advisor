"""
FastAPI application entrypoint.

Run with:
    uvicorn src.ui.main:app --reload --port 8000

The frontend should call POST /chat with {"session_id": "...", "message": "..."}.
In production the built React bundle (src/ui/frontend/dist) is served from "/";
without a build, only the API routes are available.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.ui.routes import chat, voice

app = FastAPI(title="UCAS Smart Advisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(voice.router)

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Built React frontend (registered last so API routes always win) ──────
_FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"

if _FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def serve_index() -> FileResponse:
        return FileResponse(_FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> FileResponse:
        # Real files in dist/ (favicon, vite.svg, …) are served as-is; any
        # other path falls back to index.html so client-side routing works.
        candidate = (_FRONTEND_DIST / full_path).resolve()
        if candidate.is_file() and candidate.is_relative_to(_FRONTEND_DIST):
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")
