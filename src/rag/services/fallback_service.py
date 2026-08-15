"""
Fallback mechanism: notify the human academic advisor when the system can't
answer a question, and the LLM tool schema used to trigger this.

Two delivery channels, selected by NOTIFY_CHANNEL:

  - "email" (default): Gmail SMTP. Works from Hugging Face Spaces — Spaces
    shares outbound IPs across many users, and Telegram aggressively blocks
    datacenter IP ranges that have seen abuse, so Telegram delivery from a
    Space is unreliable regardless of the bot token/chat ID being correct.
    SMTP does not have this problem.
  - "telegram": the original bot notification (optionally through the
    Cloudflare Worker relay, TELEGRAM_API_BASE). Reliable locally.
  - "both": send on both channels.

Whichever channel is selected, the send runs in a daemon thread so delivery
latency (up to ~60s per attempt) never blocks the /chat request. If the
selected channel fails — in "both" mode, only if BOTH fail — the question is
appended to failed_questions.log instead.
"""
import json
import smtplib
import threading
import time
from datetime import datetime
from email.header import Header
from email.mime.text import MIMEText

import requests
from requests.exceptions import ReadTimeout, ConnectionError

from src.utils.config import (
    ADVISOR_EMAIL,
    NOTIFY_CHANNEL,
    SMTP_APP_PASSWORD,
    SMTP_EMAIL,
    TELEGRAM_API_BASE,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# ── Channel selection (same pattern as VOICE_TTS in src/voice/config.py) ──
DEFAULT_NOTIFY_CHANNEL = "email"
VALID_NOTIFY_CHANNELS = {"email", "telegram", "both"}


def resolve_notify_channel(explicit: str | None = None) -> str:
    ch = (explicit or NOTIFY_CHANNEL or DEFAULT_NOTIFY_CHANNEL).lower()
    if ch not in VALID_NOTIFY_CHANNELS:
        raise ValueError(f"notify channel must be one of {VALID_NOTIFY_CHANNELS}, got {ch!r}")
    return ch


def _send_via_email(student: dict, question: str) -> bool:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    body = (
        f"[Smart Advisor] Unanswered Question\n\n"
        f"Student Details\n"
        f"Name  : {student.get('name',  'Not provided')}\n"
        f"Email : {student.get('email', 'Not provided')}\n"
        f"Phone : {student.get('phone', 'Not provided')}\n"
        f"Time  : {timestamp}\n\n"
        f"Unanswered Question\n{question}\n\n"
        f"Sent automatically by the UCAS Smart Advisor system."
    )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header("[Smart Advisor] سؤال بحاجة متابعة", "utf-8")
    msg["From"] = SMTP_EMAIL
    msg["To"] = ADVISOR_EMAIL

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as server:   # 60s — same reasoning as the old Telegram timeout: tolerate a slow/weak connection
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
            server.sendmail(SMTP_EMAIL, [ADVISOR_EMAIL], msg.as_string())
        return True

    except smtplib.SMTPAuthenticationError as e:
        print(f"[Email] Authentication failed — check SMTP_EMAIL / SMTP_APP_PASSWORD: {e}")
        return False

    except (smtplib.SMTPException, OSError) as e:
        print(f"[Email] Send failed: {e}")
        return False

    except Exception as e:
        print(f"[Email] Unexpected error: {e}")
        return False


def _send_via_telegram(student: dict, question: str) -> bool:
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
    Runs in a background thread: does the actual send on the configured
    channel(s) and, on failure, writes to the local backup log. Never touches
    the request/response path, so a slow channel never delays the reply to
    the student or blocks processing of their next question.
    """
    try:
        channel = resolve_notify_channel()
    except ValueError as e:
        print(f"[Notify] {e} — using default {DEFAULT_NOTIFY_CHANNEL!r}", flush=True)
        channel = DEFAULT_NOTIFY_CHANNEL

    success = False
    if channel in ("email", "both"):
        success = _send_via_email(student, question) or success
    if channel in ("telegram", "both"):
        success = _send_via_telegram(student, question) or success

    if not success:                                  # local backup log ("both": only when both channels failed)
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
    Fires the advisor notification in a background thread and returns
    immediately — so a slow/blocked channel never delays the reply to the
    student or the processing of their next question. Actual delivery
    success/failure is only known inside the background thread (logged to
    failed_questions.log on failure), not reflected in this return value.
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
