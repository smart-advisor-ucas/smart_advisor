"""
LLM-based detection of program/course/study-plan filters from a student's query.
Returns a ChromaDB 'where' clause to restrict retrieval.
"""
import json
import re

from src.utils.config import github_client

# Known program names in the DB (must match metadata exactly)
KNOWN_PROGRAMS = [
    "علم البيانات والذكاء الاصطناعي",
    "هندسة أمن المعلومات السيبراني",
    "شبكات الحاسوب والإنترنت",
    "صيانة الأجهزة الذكية",
    "أمن المعلومات",
    "هندسة الحاسوب",
    "هندسة الذكاء الاصطناعي",
]

KNOWN_COURSES = [
    "قرآن كريم",
    "اللغة الإنجليزية",
    "أساسيات علم البيانات والذكاء الاصطناعي",
    "لغة برمجة (عملي)",
    "مقدمة في الحوسبة (عملي)",
    "لغة برمجة",
    "مقدمة في الحوسبة",
    "تفاضل وتكامل 1",
    "تفاضل وتكامل 2",
    "دراسات في السيرة",
    "لغة إنجليزية تخصصية",
    "تراكيب بيانات",
    "لغات برمجة علم البيانات (عملي)",
    "الرياضيات المنفصلة",
    "لغات برمجة علم البيانات",
    "مبادئ الإحصاء والاحتمالات",
    "الجبر الخطي",
    "مقدمة في قواعد البيانات (عملي)",
    "برمجة الذكاء الاصطناعي (عملي)",
    "تحليل وتمثيل البيانات",
    "مقدمة في قواعد البيانات",
    "برمجة للذكاء الاصطناعي",
    "تحليل وتمثيل البيانات (عملي)",
    "التنقيب عن البيانات (عملي)",
    "معمارية الحاسوب",
    "مبادئ الشبكات",
    "برمجة تعلم الآلة",
    "تصميم وتحليل الخوارزميات",
    "التنقيب عن البيانات",
    "برمجة تعلم الآلة (عملي)",
    "اللغة العربية",
    "النظم المغموسة (عملي)",
    "نظم التشغيل",
    "إنترنت الأشياء",
    "الحوسبة السحابية",
    "تمييز الأنماط",
    "النظم المغموسة",
    "العمل الحر",
    "التعلم العميق (عملي)",
    "مناهج البحث العلمي والكتابة العلمية",
    "مخازن البيانات",
    "معالجة الصور الرقمية",
    "التعلم العميق",
    "الأنظمة الخبيرة",
    "ريادة الأعمال",
    "معالجة اللغات الطبيعية (عملي)",
    "أخلاقيات الذكاء الاصطناعي",
    "برمجة الروبوت (التعليم المعزز)",
    "برمجة الروبوت (التعليم المعزز) عملي",
    "عمليات تعلم الآلة (MLOps)",
    "معالجة اللغات الطبيعية",
    "مشروع تخرج (2)",
    "متطلب تخصص اختياري",
    "دراسات في العقيدة",
    "هندسة برمجيات",
    "التدريب الميداني",
    "البيانات الكبيرة",
    "استرجاع المعلومات",
    "علم الإدراك والمعرفة",
    "برمجة الروبوت",
]

_FILTER_SYSTEM = f"""
You are an assistant who determines whether a student's question refers to a specific academic program or course.

List of available programs:
{chr(10).join(f'- {p}' for p in KNOWN_PROGRAMS)}

Your task:
1. Extract ALL programs mentioned or clearly referred to. Return their exact names from the list {KNOWN_PROGRAMS}.
2. Extract ALL course codes mentioned (e.g. DSAI1301). Return them as a list.
3. Extract ALL course names mentioned or clearly referred to. Return their exact names from the list {KNOWN_COURSES}.
4. If no specific program is mentioned, return "علم البيانات والذكاء الاصطناعي".
5. If no specific course is mentioned, return null.
6. Determine if the student is asking about the full study plan / curriculum / semester structure of a program. If so, set "is_study_plan" to true.

Return ONLY JSON:
{{"programs": [...] or ["علم البيانات والذكاء الاصطناعي"], "course_codes": [...] or null, "course_names": [...] or null, "is_study_plan": true or false}}
"""


def detect_metadata_filter(query: str) -> tuple[dict | None, bool]:
    """
    Ask LLM to detect program/course/study-plan intent in a query.
    Returns (where_clause_or_None, is_study_plan).
    """
    try:
        resp = github_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _FILTER_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        raw = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
        detected = json.loads(raw)

        raw_programs = detected.get("programs") or []
        if isinstance(raw_programs, str):
            raw_programs = [raw_programs]
        programs = [p for p in raw_programs if p in KNOWN_PROGRAMS]

        course_codes = detected.get("course_codes") or []

        raw_course_names = detected.get("course_names") or []
        if isinstance(raw_course_names, str):
            raw_course_names = [raw_course_names]
        course_names = [p for p in raw_course_names if p in KNOWN_COURSES]

        is_study_plan = detected.get("is_study_plan", False)

        # Study plan query — filter by category AND program
        if is_study_plan and programs:
            if len(programs) > 1:
                f = {"$and": [
                    {"category": "study_plan"},
                    {"program": {"$in": programs}},
                ]}
            else:
                f = {"$and": [
                    {"category": "study_plan"},
                    {"program": programs[0]},
                ]}
            return f, True

        # Course is more specific than program — takes priority
        if course_codes:
            f = {"course_code": {"$in": course_codes}} if len(course_codes) > 1 else {"course_code": course_codes[0]}
            if programs:
                prog_f = {"program": {"$in": programs}} if len(programs) > 1 else {"program": programs[0]}
                f = {"$and": [f, prog_f]}
            return f, False

        if course_names:
            f = {"course_name": {"$in": course_names}} if len(course_names) > 1 else {"course_name": course_names[0]}
            if programs:
                prog_f = {"program": {"$in": programs}} if len(programs) > 1 else {"program": programs[0]}
                f = {"$and": [f, prog_f]}
            return f, False

        if programs:
            f = {"program": {"$in": programs}} if len(programs) > 1 else {"program": programs[0]}
            return f, False

        return None, False

    except Exception as e:
        print(f"[Filter detection error] {e}")
        return None, False
