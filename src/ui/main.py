"""
FastAPI application entrypoint.

Run with:
    uvicorn src.main:app --reload --port 8000

The frontend should call POST /chat with {"session_id": "...", "message": "..."}.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
