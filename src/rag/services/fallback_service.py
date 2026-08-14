"""
Fallback mechanism: notify the human academic advisor via Telegram when the
system can't answer a question, and the LLM tool schema used to trigger this.
"""
import json
import threading
import time
from datetime import datetime

import requests
from requests.exceptions import ReadTimeout, ConnectionError

from src.utils.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_API_BASE


def send_fallback_telegram(student: dict, question: str) -> bool:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [                                        # CHANGED — reverted escaping (legacy Markdown doesn't support it)
        "*[Smart Advisor] Unanswered Question*",
        "",
        "*Student Details*",
        f"Name  : {student.get('name',  'Not provided')}",
        f"Email : {student.get('email', 'Not provided')}",
        f"Phone : {student.get('phone', 'Not provided')}",
        f"Time  : {timestamp}",
        "",
        "*Unanswered Question*",
        f"{question}",
        "",
        "_Sent automatically by the UCAS Smart Advisor system._",
    ]
    message = "\n".join(lines)
    # Goes through the Cloudflare Worker relay (TELEGRAM_API_BASE) instead of
    # api.telegram.org directly, to avoid Telegram blocking the datacenter egress IP
    # that Hugging Face Spaces assigns on restart.
    relay = (TELEGRAM_API_BASE or "https://api.telegram.org").rstrip("/")
    url = f"{relay}/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "Markdown",
    }

    print(f"[Telegram] posting via {relay}", flush=True)

    max_retries = 3                                  # NEW — retry with backoff
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=60)   # CHANGED — was 15, now 60 for weak connections
            result = resp.json()
            if result.get("ok"):
                print("[Telegram] sendMessage ok", flush=True)
                return True
            print(f"[Telegram] API returned error: {result}", flush=True)
            return False  # API error — no point retrying

        except (ReadTimeout, ConnectionError) as e:
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"[Telegram] Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                print("[Telegram] All retries exhausted — question will be logged locally only.")
                return False

        except Exception as e:
            print(f"[Telegram] Unexpected error: {e}")
            return False

    return False


def _send_fallback_background(student: dict, question: str) -> None:
    """
    Runs in a background thread: does the actual Telegram send (with its
    internal retries/backoff, now up to 60s per attempt) and, on failure,
    writes to the local backup log. Never touches the request/response path.
    """
    success = send_fallback_telegram(student, question)
    if not success:                                  # local backup log
        with open("failed_questions.log", "a", encoding="utf-8") as f:
            f.write(
                f"\n---\nTime: {datetime.now()}\n"
                f"Name: {student.get('name')}\n"
                f"Email: {student.get('email')}\n"
                f"Phone: {student.get('phone')}\n"
                f"Question: {question}\n"
            )


def record_unknown_question(question: str, name: str, email: str = None, phone: str = None) -> dict:
    """
    Fires the Telegram notification in the background and returns immediately
    — so a slow/blocked Telegram call never delays the reply to the student
    or the processing of their next question. Actual delivery success/failure
    is only known inside the background thread (logged to failed_questions.log
    on failure), not reflected in this return value.
    """
    student = {"name": name, "email": email, "phone": phone}
    threading.Thread(
        target=_send_fallback_background,
        args=(student, question),
        daemon=True,
    ).start()

    return {"recorded": "pending"}


# ── LLM tool schema ──────────────────────────────────────────────────────
record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": (
        "Always use this tool to record any question that couldn't be answered. "
        "Also records the student's details so the advisor can follow up."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
            "name": {"type": "string", "description": "The student's full name"},
            "email": {"type": "string", "description": "The student's email address"},
            "phone": {"type": "string", "description": "The student's phone number"},
        },
        "required": ["question", "name"],
        "additionalProperties": False,
    },
}

tools = [{"type": "function", "function": record_unknown_question_json}]


def handle_tool_calls(tool_calls) -> list[dict]:
    results = []
    for tc in tool_calls:
        args = json.loads(tc.function.arguments)
        print(f"Tool called: {tc.function.name}", flush=True)
        result = record_unknown_question(**args) if tc.function.name == "record_unknown_question" else {}
        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tc.id})
    return results