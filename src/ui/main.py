"""
FastAPI application entrypoint.

Run with:
    uvicorn src.main:app --reload --port 8000

The frontend should call POST /chat with {"session_id": "...", "message": "..."}.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.ui.routes import chat

app = FastAPI(title="UCAS Smart Advisor API")

# Allow the frontend (running on a different origin during development) to call this API.
# TODO: restrict allow_origins to your actual frontend domain before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/health")
def health():
    return {"status": "ok"}
