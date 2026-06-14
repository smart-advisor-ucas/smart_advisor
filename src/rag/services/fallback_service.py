"""
Fallback mechanism: notify the human academic advisor via Telegram when the
system can't answer a question, and the LLM tool schema used to trigger this.
"""
import json
from datetime import datetime

import requests

from src.utils.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def send_fallback_telegram(student: dict, question: str) -> bool:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    message = (
        f"*[Smart Advisor] Unanswered Question*\n\n"
        f"*Student Details*\n"
        f"Name  : {student.get('name', 'Not provided')}\n"
        f"Email : {student.get('email', 'Not provided')}\n"
        f"Phone : {student.get('phone', 'Not provided')}\n"
        f"Time  : {timestamp}\n\n"
        f"*Unanswered Question*\n{question}\n\n"
        f"_Sent automatically by the UCAS Smart Advisor system._"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
        timeout=10,
    )
    return resp.json().get("ok", False)


def record_unknown_question(question: str, name: str, email: str = None, phone: str = None) -> dict:
    student = {"name": name, "email": email, "phone": phone}
    success = send_fallback_telegram(student, question)
    print(f"Recording unanswered question from {name}: {question}")
    return {"recorded": "ok" if success else "failed"}


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
