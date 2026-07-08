"""
LLM-based detection of program/course/study-plan filters from a student's query.
Returns a ChromaDB 'where' clause to restrict retrieval.
"""
import json
import re

from langchain_classic.memory import ConversationBufferWindowMemory

from src.utils.config import github_client

# Known program names in the DB (must match metadata exactly)
KNOWN_PROGRAMS = [
    "علم البيانات والذكاء الاصطناعي",
    "هندسة أمن المعلومات السيبراني",
    "شبكات الحاسوب والإنترنت",
    "صيانة الأجهزة الذكية",
    "أمن المعلومات"
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
    "برمجة الروبوت"
]

# Programs that share a generic field name but differ in degree level/duration.
# If the student uses the generic name without disambiguating, search both.
AMBIGUOUS_PROGRAM_GROUPS: list[set[str]] = [
    {"هندسة أمن المعلومات السيبراني", "أمن المعلومات"},
]

_DEGREE_DISAMBIGUATION_KEYWORDS = [
    "بكالوريوس", "دبلوم", "سنتين", "سنتان", "أربع سنوات", "اربع سنوات",
    "هندسة أمن", "سيبراني",
]

def _expand_ambiguous_programs(programs: list[str], query: str) -> list[str]:
    """Safety net on top of the LLM extraction: if an ambiguous field name
    was matched and the student didn't specify degree/duration, expand to
    the full ambiguous group so both programs get searched."""
    if any(k in query for k in _DEGREE_DISAMBIGUATION_KEYWORDS):
        return programs
    result = set(programs)
    for group in AMBIGUOUS_PROGRAM_GROUPS:
        if result & group:
            result |= group
    return list(result)




_FILTER_SYSTEM = f"""
You are an assistant who determines whether a student's question refers to a specific academic program or course, and what category of information they want.

List of available programs:
{chr(10).join(f'- {p}' for p in KNOWN_PROGRAMS)}

Your task:
1. Extract ALL programs mentioned or clearly referred to. Return their exact names from the list {KNOWN_PROGRAMS}.
2. Extract ALL course codes mentioned (e.g. DSAI1301). Return them as a list.
3. Extract ALL course names mentioned or clearly referred to. Return their exact names from the list {KNOWN_COURSES}.
4. If no specific program is mentioned AND the question is not a general/comparison question about "all majors" (see rule 9), return "علم البيانات والذكاء الاصطناعي".
5. If no specific course is mentioned, return null.
6. Set "is_study_plan" to true if the student is asking about the full study plan,
   curriculum, semester structure, OR the set of courses/subjects a program offers
   — including comparisons like "ما الفرق بين مساقات X ومساقات Y؟" or
   "ما هي المساقات في تخصص كذا؟". This is broader than just literal "خطة دراسية" wording.
7. Determine if the student is asking about general program facts — admission requirement/GPA cutoff, degree type, duration, credit hours, college/department, or a general overview of the program (but NOT the full semester-by-semester curriculum) or something like that. If so, set "is_program_info" to true.
8. Determine if the student is asking about scholarships, financial aid, grants, or fee discounts something like: منحة، منح، مساعدة مالية، إعفاء. or any other similar words. If so, set "is_scholarship" to true.
9. Determine if the student is asking a broad question about ALL specializations at UCAS, OR comparing a program to "other majors" / "بقية التخصصات" / "باقي البرامج" without naming those other majors specifically. If so, set "all_programs" to true and include ALL programs from the list in "programs".

CRITICAL RULE — Ambiguous Program Names:
=========================================
"أمن المعلومات" / "الأمن السيبراني" / "أمن سيبراني" is ambiguous — it can mean
EITHER of TWO different programs:
  - "هندسة أمن المعلومات السيبراني" (بكالوريوس - أربع سنوات)
  - "أمن المعلومات" (دبلوم - سنتان)
If the student's message does not clearly specify which one (no mention of
"بكالوريوس", "دبلوم", "أربع سنوات", "سنتين", or "هندسة"), extract BOTH program
names into "programs". Only extract one if the student disambiguates
(e.g. "دبلوم" → diploma only, "بكالوريوس"/"هندسة" → the engineering bachelor's only).

CRITICAL RULES for course name matching:
==========================================
- Only extract a course name if the student is EXPLICITLY asking about that 
  specific course by name or code.
- Do NOT match a course just because the query contains a word that appears 
  in a course name.

Examples of when NOT to match a course:
- "لماذا الرياضيات مهمة؟" → do NOT match "الرياضيات المنفصلة" 
  (student is asking about math in general, not that specific course)
- "هل البرمجة صعبة؟" → do NOT match "لغة برمجة" 
  (student is asking generally, not about that course)
- "ما أهمية الذكاء الاصطناعي؟" → do NOT match "أساسيات علم البيانات والذكاء الاصطناعي"

Examples of when TO match a course:
- "ما محتوى مساق الرياضيات المنفصلة؟" → match "الرياضيات المنفصلة" ✓
- "من يدرّس DSAI1308؟" → match by course code ✓
- "ما متطلبات مساق تراكيب البيانات؟" → match "تراكيب بيانات" ✓

The student must be asking ABOUT the course itself, not just mentioning 
a topic that relates to it.

Examples for the category flags:
- "كم عدد الساعات المعتمدة لتخصص علم البيانات؟" → is_program_info=true, programs=["علم البيانات والذكاء الاصطناعي"]
- "ما هو معدل القبول لتخصص أمن المعلومات؟" → is_program_info=true, programs=["هندسة أمن المعلومات السيبراني", "أمن المعلومات"]
- "ما هي خطة دراسة تخصص أمن المعلومات؟" → is_study_plan=true, programs=["هندسة أمن المعلومات السيبراني", "أمن المعلومات"]
- "هل يوجد منح دراسية؟" → is_scholarship=true
- "ما شروط منحة ويبقى الأمل؟" → is_scholarship=true
- "ما الفرق بين علم البيانات وباقي التخصصات في الجامعة؟" → all_programs=true, programs=[all programs]
- "ما هي التخصصات المتاحة في UCAS؟" → all_programs=true, programs=[all programs]
- "ما الفرق بين علم البيانات والأمن السيبراني؟" → all_programs=false, programs=["علم البيانات والذكاء الاصطناعي", "هندسة أمن المعلومات السيبراني"]

Return ONLY JSON:
{{"programs": [...] or ["علم البيانات والذكاء الاصطناعي"], "course_codes": [...] or null, "course_names": [...] or null, "is_study_plan": true or false, "is_program_info": true or false, "is_scholarship": true or false, "all_programs": true or false}}
"""

def _category_and_programs_filter(category: str, programs: list[str]) -> dict:
    prog_f = {"program": {"$in": programs}} if len(programs) > 1 else {"program": programs[0]}
    return {"$and": [{"category": category}, prog_f]}

def detect_metadata_filter(query: str, memory: "ConversationBufferWindowMemory | None" = None) -> dict | None:
    """
    Ask LLM to detect program/course/category mentioned in query.
    Returns (chromadb_where_clause_or_None, mode) where mode is:
      "exact"      -> caller should bypass embedding search (collection.get)
      "similarity" -> caller should run normal embedding similarity search
    """
    history_msgs = []
    if memory is not None:
        chat_history = memory.load_memory_variables({})["history"]
        # keep it short — this call doesn't need full context, just enough
        # to resolve "هذا التخصص" / "نفس البرنامج" type references
        history_msgs = [
            {"role": ("user" if m.type == "human" else "assistant"), "content": m.content}
            for m in chat_history[-4:]
        ]

    try:
        resp = github_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _FILTER_SYSTEM},
                {"role": "user",   "content": query}
            ],
            temperature=0.0,
            max_tokens=200
        )
        raw = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
        print(f"[detect_metadata_filter raw] {raw}")
        detected = json.loads(raw)

        raw_programs = detected.get('programs') or []
        if isinstance(raw_programs, str):
            raw_programs = [raw_programs]  # wrap stray string in a list
        programs = [p for p in raw_programs if p in KNOWN_PROGRAMS]

        if detected.get("all_programs", False):
            programs = list(KNOWN_PROGRAMS)

        # Safety net for the أمن المعلومات ambiguity, independent of the LLM call
        programs = _expand_ambiguous_programs(programs, query)
        print(f"[Filter] programs={programs}")
        
        course_codes = detected.get("course_codes") or []
        print(f"[Filter] course_codes={course_codes}")

        raw_course_names = detected.get("course_names") or []
        if isinstance(raw_course_names, str):
            raw_course_names = [raw_course_names]  # wrap stray string in a list
        course_names     = [p for p in raw_course_names if p in KNOWN_COURSES]
        print(f"[Filter] course_names={course_names}")

        is_study_plan = detected.get("is_study_plan", False)
        is_study_plan   = detected.get("is_study_plan", False)
        is_program_info = detected.get("is_program_info", False)
        is_scholarship  = detected.get("is_scholarship", False)
        print(f"is_study_plan={is_study_plan} is_program_info={is_program_info} is_scholarship={is_scholarship}")

        # ── Scholarship: exact category fetch, not tied to a program ───────
        if is_scholarship:
            f = {"category": "scholarship"}
            print("[Filter] scholarship — exact category fetch")
            return f, "exact"


        if is_study_plan and programs:
            f = _category_and_programs_filter("study_plan", programs)
            print(f"[Filter] study_plan, programs={programs}")
            return f, "exact"

        # ── NEW: broad "all programs" comparison — exact fetch, not similarity ──
        # Bypasses embedding search entirely so retrieval isn't biased by how the
        # question happens to be phrased, and isn't dominated by whichever program
        # has the most chunks in the DB (DS&AI has a full syllabus; diploma
        # programs only have 1-2 chunks each). Guarantees every program gets
        # equal representation: one program_info + one career_opportunities
        # chunk per program.
        if detected.get("all_programs", False) and not is_study_plan:
            f = {"$and": [
                {"category": {"$in": ["program_info", "career_opportunities"]}},
                {"program": {"$in": programs}}
            ]}
            print(f"[Filter] all_programs comparison — exact fetch, programs={programs}")
            return f, "exact"

        # ── Course code / name — similarity search, unchanged from before ──
        if course_codes:
            f = {"course_code": {"$in": course_codes}} if len(course_codes) > 1 else {"course_code": course_codes[0]}
            if programs:
                prog_f = {"program": {"$in": programs}} if len(programs) > 1 else {"program": programs[0]}
                f = {"$and": [f, prog_f]}
            print(f"[Filter] course_codes={course_codes}, programs={programs}")
            return f, "similarity"

        if course_names:
            f = {"course_name": {"$in": course_names}} if len(course_names) > 1 else {"course_name": course_names[0]}
            if programs:
                prog_f = {"program": {"$in": programs}} if len(programs) > 1 else {"program": programs[0]}
                f = {"$and": [f, prog_f]}
            print(f"[Filter] course_names={course_names}, programs={programs}")
            return f, "similarity"
            
        # ── Program info: exact category + program fetch ────────────────────
        if is_program_info:
            if not programs:
                programs = ["علم البيانات والذكاء الاصطناعي"]
            f = _category_and_programs_filter("program_info", programs)
            print(f"[Filter] program_info, programs={programs}")
            return f, "exact"

        if programs:
            f = {"program": {"$in": programs}} if len(programs) > 1 else {"program": programs[0]}
            print(f"[Filter] programs={programs}")
            return f, "similarity"
        return None, "similarity"

    except Exception as e:
        print(f"[Filter detection error] {e}")
        return None, "similarity"
