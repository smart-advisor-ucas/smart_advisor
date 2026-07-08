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

# ── API Keys ─────────────────────────────────────────────────────────────
HF_KEY           = os.getenv("HF_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY") 

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN is missing. Add it to .env and re-run.")

if not GROQ_API_KEY:                                     # NEW
    raise ValueError("GROQ_API_KEY is missing. Add it to .env and re-run.")

# ── Shared LLM client (GitHub Models API) ──────────────────────────────────
github_client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=GITHUB_TOKEN,
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
