"""
Student profile extraction (LLM-based) and the onboarding question sequencer.
"""
import json
import re

from src.utils.config import groq_client, openrouter_client     
from src.utils.schemas import StudentProfile

_EXTRACTION_SYSTEM = """
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

# Short answers that should map to a boolean field (likes_math) without an LLM call.
_YES_ANSWERS = {
    "نعم", "أجل", "اجل", "أيوة", "ايوة", "أيوه", "ايوه", "أيوا", "ايوا",
    "آه", "اه", "ايه", "صح", "أكيد", "اكيد", "طبعا", "طبعاً",
    "yes", "y", "yeah", "yep", "true",
}
_NO_ANSWERS = {
    "لا", "لأ", "لاء", "كلا", "أبدا", "ابدا", "مش", "مو",
    "no", "n", "nope", "false",
}
_BOOLEAN_FIELDS = {"likes_math"}


def _parse_yes_no(user_message: str) -> bool | None:
    """Return True/False if the whole message is a yes/no, else None."""
    text = re.sub(r"^[\s\"'«»]+|[\s.!?؟،,;؛»«\"']+$", "", user_message.strip())
    if not text:
        return None
    lowered = text.lower()
    if text in _YES_ANSWERS or lowered in _YES_ANSWERS:
        return True
    if text in _NO_ANSWERS or lowered in _NO_ANSWERS:
        return False
    return None


def _coerce_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        yn = _parse_yes_no(value)
        if yn is not None:
            return yn
        low = value.strip().lower()
        if low in ("true", "1"):
            return True
        if low in ("false", "0"):
            return False
    return None


def extract_profile(
    user_message: str,
    current: StudentProfile,
    pending_field: str | None = None,
) -> StudentProfile:
    """Call LLM to extract profile fields from student message and merge with current profile.

    pending_field is the onboarding field whose question was just asked. Short
    yes/no answers like "نعم" are otherwise ambiguous and would leave the field
    empty, causing the same question to be repeated.
    """
    yn = _parse_yes_no(user_message)
    if (
        yn is not None
        and pending_field in _BOOLEAN_FIELDS
        and getattr(current, pending_field, None) is None
    ):
        return current.model_copy(update={pending_field: yn})

    pending_ctx = ""
    if pending_field:
        asked = _FIELD_QUESTIONS.get(pending_field, pending_field)
        pending_ctx = (
            f"السؤال الذي طرحه المستشار للتو (الحقل: {pending_field}): {asked}\n"
            f"إذا كانت رسالة الطالب إجابة قصيرة (نعم/لا أو ما يعادلها)، "
            f"املأ الحقل {pending_field} فقط.\n\n"
        )

    try:
        resp = groq_client.chat.completions.create(          
            model="openai/gpt-oss-20b", 
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user", "content": (
                    f"{pending_ctx}"
                    f"الملف الحالي: {current.model_dump_json()}\n\n"
                    f"رسالة الطالب: {user_message}"
                )},
            ],
            temperature=0.0,
            max_tokens=500,
            reasoning_effort="low",    
            include_reasoning=False,
        )
        raw = re.sub(r"```json|```", "", resp.choices[0].message.content or "").strip()
        try:
            extracted = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return current

        # Strip whitespace from all extracted string/list values
        cleaned = {}
        for key, value in extracted.items():
            if isinstance(value, str):
                cleaned[key] = value.strip()
            elif isinstance(value, list):
                cleaned[key] = [v.strip() if isinstance(v, str) else v for v in value]
            else:
                cleaned[key] = value

        # Merge: only fill None / empty-list fields from extraction
        current_data = current.model_dump()
        for key, value in cleaned.items():
            if value is None:
                continue
            if key == "likes_math":
                value = _coerce_bool(value)
                if value is None:
                    continue
            if key == "interest_areas" and isinstance(value, str):
                value = [value]
            existing = current_data.get(key)
            if existing is None or (isinstance(existing, list) and len(existing) == 0):
                current_data[key] = value
        return StudentProfile(**current_data)
    except Exception as e:
        print(f"[Profile extraction error] {e}")
        return current  # return unchanged on any failure


# Maps missing profile fields -> natural Arabic questions.
# No LLM needed — pure lookup table for speed and reliability.
_FIELD_QUESTIONS: dict[str, str] = {
    "gpa": (
        "للبدء، ما معدلك في الثانوية العامة (التوجيهي)؟ "
        "هذا يساعدني في معرفة البرامج التي تؤهل للقبول."
    ),
    "academic_track": (
        "ما فرعك الدراسي في الثانوية؟ "
        "(علمي / صناعي / تكنولوجيا معلومات / أدبي / تجاري / أخرى)"
    ),
    "likes_math": (
        "هل تستمتع بالرياضيات والإحصاء؟ "
        "أسألك لأن بعض التخصصات كعلم البيانات تعتمد عليهما بشكل كبير."
    ),
    "interest_areas": (
        "ما الذي يثير اهتمامك أكثر؟ "
        "(تحليل البيانات / الذكاء الاصطناعي / أمن المعلومات / الشبكات / هندسة الحاسوب)"
    ),
    "degree_preference": (
        "هل تفضل الحصول على درجة البكالوريوس (4 سنوات) أم الدبلوم (سنتان)؟"
    ),
}


def next_onboarding_question(profile: StudentProfile) -> str | None:
    """Return the next onboarding question to ask, or None if profile is complete."""
    for field in profile.missing_fields():
        q = _FIELD_QUESTIONS.get(field)
        if q:
            return q
    return None
