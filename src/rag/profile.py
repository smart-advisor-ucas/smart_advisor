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
  "interest_areas": one of [علم البيانات، الذكاء الاصطناعي، أمن المعلومات، شبكات الحاسوب، هندسة الحاسوب، غير محدد] or null,
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


def extract_profile(user_message: str, current: StudentProfile) -> StudentProfile:
    """Call LLM to extract profile fields from student message and merge with current profile."""
    try:
        resp = groq_client.chat.completions.create(          
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user", "content": (
                    f"الملف الحالي: {current.model_dump_json()}\n\n"
                    f"رسالة الطالب: {user_message}"
                )},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
        extracted = json.loads(raw)

        # NEW — strip whitespace from all extracted string/list values
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
