"""
Classifies whether a student's message is conversational (greeting, thanks,
small talk) vs. a real academic question needing retrieval.
"""
import json
import re

from src.utils.config import github_client, groq_client

_CONVERSATIONAL_PATTERNS = [
    "شكرا", "شكراً", "تمام", "ماشي", "مرحبا", "أهلا", "اهلا", "يسلمو",
    "حسنا", "حسناً", "ممتاز", "السلام عليكم", "وعليكم السلام",
    "كيف حالك", "شو اخبارك", "مع السلامة", "باي", "طيب",
]

def _quick_conversational_check(user_message: str) -> bool | None:
    """Returns True/False if confidently classifiable without an LLM call,
    None if genuinely ambiguous and needs the LLM fallback."""
    stripped = user_message.strip()
    if len(stripped) <= 3:
        return True  # extremely short messages are almost always filler
    if any(p in stripped for p in _CONVERSATIONAL_PATTERNS) and len(stripped) < 25:
        return True
    if "؟" in stripped or "?" in stripped or len(stripped) > 25:
        return False  # a question mark or longer message is very likely a real question
    return None  # ambiguous — defer to LLM

_CONVERSATIONAL_SYSTEM = """You are a classifier for an academic advisor chatbot.
Decide if the student's message is a real academic question that needs retrieval, or just conversational
(greeting, thanks, acknowledgement, filler, or off-topic small talk).

Return ONLY JSON in this exact format:
{
  "is_conversational": true or false,
  "response": "a short polite Arabic reply if is_conversational is true, otherwise null"
}

Examples of conversational (is_conversational: true):
- "شكراً", "تمام", "ماشي", "مرحبا", "أهلاً", "يسلمو", "حسناً", "ممتاز", "السلام عليكم"
- "كيف حالك", "من أنت", "ما اسمك"
- Any greeting, farewell, or expression of thanks

Examples of real questions (is_conversational: false):
- "ما هي المنح الدراسية", "ما خطة الدراسة", "ما شرط القبول"
- "هل أنا مقبول بمعدل 75%", "ما المساقات في السنة الأولى"
- Any question about the program, courses, admission, or career

Rules:
- Return only JSON, no markdown, no explanation.
- The response field must always be in Arabic.
- Keep the response short (1-2 sentences), warm, and relevant to what the student said."""


def is_conversational(user_message: str) -> tuple[bool, str | None]:
    """
    Returns (True, polite_response) if message is a greeting/thanks/non-question.
    Returns (False, None) if message is a real academic question.
    """
    quick = _quick_conversational_check(user_message)
    if quick is False:
        return False, None
    if quick is True:
        # Still return a plausible generic Arabic reply — cheap, no LLM call
        return True, "أهلاً بك! كيف يمكنني مساعدتك؟"

    messages = [
        {"role": "system", "content": """You are a classifier for an academic advisor chatbot.
Decide if the student's message is a real academic question that needs retrieval, or just conversational
(greeting, thanks, acknowledgement, filler, or off-topic small talk).

Return ONLY JSON in this exact format:
{
  "is_conversational": true or false,
  "response": "a short polite Arabic reply if is_conversational is true, otherwise null"
}

Examples of conversational (is_conversational: true):
- "شكراً", "تمام", "ماشي", "مرحبا", "أهلاً", "يسلمو", "حسناً", "ممتاز", "السلام عليكم"
- "كيف حالك", "من أنت", "ما اسمك"
- Any greeting, farewell, or expression of thanks

Examples of real questions (is_conversational: false):
- "ما هي المنح الدراسية", "ما خطة الدراسة", "ما شرط القبول"
- "هل أنا مقبول بمعدل 75%", "ما المساقات في السنة الأولى"
- Any question about the program, courses, admission, or career

Rules:
- Return only JSON, no markdown, no explanation.
- The response field must always be in Arabic.
- Keep the response short (1-2 sentences), warm, and relevant to what the student said."""},
        {"role": "user", "content": user_message}
    ]

    def _try(client, model):
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=100
        )
        raw = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
        return json.loads(raw)

    try:
        result = _try(groq_client, "llama-3.1-8b-instant")
    except Exception as e:
        print(f"[is_conversational/Groq error] {e} — falling back to gpt-4o-mini")
        try:
            result = _try(github_client, "gpt-4o-mini")
        except Exception as e2:
            print(f"[is_conversational/fallback error] {e2} — treating as real question")
            return False, None   # ← this is the line that actually prevents the crash

    if result.get("is_conversational"):
        return True, result.get("response")
    return False, None