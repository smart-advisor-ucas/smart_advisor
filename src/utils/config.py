"""
Central configuration: environment variables, shared API clients, and constants.
Import from here instead of calling os.getenv() everywhere.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

# Load .env from project root (one level above src/)
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


# ── API Keys ─────────────────────────────────────────────────────────────
HF_KEY           = os.getenv("HF_TOKEN")
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID  = _env("TELEGRAM_CHAT_ID")
# Cloudflare Worker relay; trailing slash must be stripped so we don't build
# https://worker.example//bot<token>/sendMessage
TELEGRAM_API_BASE = (_env("TELEGRAM_API_BASE") or "https://api.telegram.org").rstrip("/")

# ── Advisor fallback notifications ───────────────────────────────────────
# NOTIFY_CHANNEL: "email_api" | "email" | "telegram" | "both" — validated and
# defaulted (to "email_api") in fallback_service.resolve_notify_channel,
# mirroring how the voice layer resolves VOICE_TTS. All are optional: the app
# must boot without them, exactly as the Telegram pair was optional.
NOTIFY_CHANNEL    = _env("NOTIFY_CHANNEL")
SMTP_EMAIL        = _env("SMTP_EMAIL")
SMTP_APP_PASSWORD = _env("SMTP_APP_PASSWORD")
ADVISOR_EMAIL     = _env("ADVISOR_EMAIL")
# Resend HTTPS email API ("email_api" channel) — SMTP is blocked on HF Spaces,
# HTTPS on 443 is not. RESEND_FROM is the verified sender address.
RESEND_API_KEY    = _env("RESEND_API_KEY")
RESEND_FROM       = _env("RESEND_FROM")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY") 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing. Add it to .env and re-run.")

if not GROQ_API_KEY:                                     # NEW
    raise ValueError("GROQ_API_KEY is missing. Add it to .env and re-run.")

# ── Shared LLM client (GitHub Models API) ──────────────────────────────────
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

groq_client   = Groq(api_key=GROQ_API_KEY)

# ── Vector DB ────────────────────────────────────────────────────────────
DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db")
COLLECTION_NAME = "ucas_knowledge_base"

# ── Retrieval tuning ─────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.4
TOP_K = 3
MEMORY_WINDOW_K = 3  # conversation turns kept in memory
MAX_CHUNKS = 6              # NEW — cap on unique chunks kept after ranking
MAX_CONTEXT_TOKENS = 3000   # NEW — token budget for merged context
QUERY_VARIANTS_N = 2        # NEW — generate_queries default changed from 3 → 2
