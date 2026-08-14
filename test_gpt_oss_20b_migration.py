"""
Validation script for migrating conversational.py and profile.py
from llama-3.1-8b-instant to openai/gpt-oss-20b on Groq.

Run this BEFORE swapping the model string in the production files.
Requires GROQ_API_KEY in your .env (same as the rest of the project).

Usage:
    python test_gpt_oss_20b_migration.py
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

dotenv_path = Path.cwd() / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
NEW_MODEL = "openai/gpt-oss-20b"

# ── System prompts copied verbatim from conversational.py / profile.py ─────
CONVERSATIONAL_SYSTEM = """You are a classifier for an academic advisor chatbot.
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

EXTRACTION_SYSTEM = """
You are an assistant who extracts student information from a conversation message.
Return ONLY JSON object with these keys (if the information is not exist use null):
{
  "gpa": floating number from 0 to 100 or null,
  "academic_track": one of [علمي، صناعي، تكنولوجيا معلومات، أدبي، تجاري، أخرى] or null,
  "likes_math": null or true/false,
  "interest_areas": a JSON array containing one or more of [علم البيانات، الذكاء الاصطناعي، أمن المعلومات، شبكات الحاسوب، هندسة الحاسوب، غير محدد], or an empty array [] if none mentioned,
  "degree_preference": one of [بكالوريوس، دبلوم، غير محدد] or null
}
Rules:
- Only extract a field if the student's message is actually answering the question about THAT field,
  or explicitly volunteers that information. A short answer like "نعم"/"لا" only applies to the field
  whose question was just asked — do not let it populate any other field.
- The student's message may be a short answer (e.g. "نعم", "لا", "أيوة", "yes", "no") to a question
  the assistant just asked. Use the conversation context to determine which field this answer applies to,
  and map Arabic/English affirmatives (نعم، أجل، أيوة، صح، yes) to true and negatives (لا، مش، no) to false.
- For degree_preference: if the question asked about degree/duration and the student answers with a
  duration, map it accordingly — "سنتان"، "سنتين"، "2"، "two years" → دبلوم;
  "اربع سنوات"، "4"، "أربعة"، "four years" → بكالوريوس.
- Do not invent information that is not in the message or implied by the immediate question context.
- Do not modify fields in the current file unless the student explicitly corrects information.
- Return only JSON, without explanation or markdown.
"""

EMPTY_PROFILE_JSON = json.dumps({
    "gpa": None, "academic_track": None, "likes_math": None,
    "interest_areas": [], "degree_preference": None,
}, ensure_ascii=False)


def call_json(system: str, user: str, max_tokens: int = 500):
    resp = groq_client.chat.completions.create(
        model=NEW_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=max_tokens,
        reasoning_effort="low",    
        include_reasoning=False,
    )
    raw = resp.choices[0].message.content
    cleaned = re.sub(r"```json|```", "", raw or "").strip()
    try:
        return json.loads(cleaned), raw
    except (json.JSONDecodeError, TypeError):
        return None, raw


# ── TEST SET 1: is_conversational ───────────────────────────────────────────
# Only messages that reach the LLM in production (no "؟", length 4-25,
# not in the hardcoded _CONVERSATIONAL_PATTERNS list) are useful here —
# everything else is decided by _quick_conversational_check() without a model call.
conversational_cases = [
    ("من أنت", True),             # identity question -> listed as conversational in the prompt
    ("ما اسمك", True),
    ("وين الدكتور", False),        # real question, no "؟"
    ("بدي اعرف المنح", False),     # real request, no "؟"
    ("تصبح على خير", True),        # farewell, not in the quick-pattern list
]

print("=" * 70)
print(f"TEST 1: is_conversational classifier  (model={NEW_MODEL})")
print("=" * 70)
for msg, expected in conversational_cases:
    parsed, raw = call_json(CONVERSATIONAL_SYSTEM, msg, max_tokens=100)
    ok = parsed is not None and parsed.get("is_conversational") == expected
    print(f"{'✅' if ok else '❌'} msg={msg!r:20} expected={expected} got={parsed}")
    if parsed is None:
        print(f"    ⚠️ JSON parse failed. Raw: {raw!r}")

# ── TEST SET 2: extract_profile ─────────────────────────────────────────────
profile_cases = [
    ("معدلي 88 وفرعي علمي", {"gpa": 88.0, "academic_track": "علمي"}),
    ("أربع سنوات", {"degree_preference": "بكالوريوس"}),
    ("سنتين", {"degree_preference": "دبلوم"}),
    ("أنا مهتم بالذكاء الاصطناعي وحابب الرياضيات كتير",
     {"interest_areas": ["الذكاء الاصطناعي"], "likes_math": True}),
    # ⚠️ No last-question context is passed to the LLM in extract_profile(),
    # so this case is genuinely ambiguous regardless of model — see note above.
    ("نعم", None),
]

print()
print("=" * 70)
print(f"TEST 2: extract_profile  (model={NEW_MODEL})")
print("=" * 70)
for msg, expected_subset in profile_cases:
    user_content = f"الملف الحالي: {EMPTY_PROFILE_JSON}\n\nرسالة الطالب: {msg}"
    parsed, raw = call_json(EXTRACTION_SYSTEM, user_content, max_tokens=400)
    print(f"msg={msg!r}")
    print(f"    got={parsed}")
    if expected_subset:
        mismatches = {
            k: (v, parsed.get(k) if parsed else None)
            for k, v in expected_subset.items()
            if not parsed or parsed.get(k) != v
        }
        print(f"    {'✅ matches expected' if not mismatches else f'❌ mismatch: {mismatches}'}")
    if parsed is None:
        print(f"    ⚠️ JSON parse failed. Raw: {raw!r}")
    print()