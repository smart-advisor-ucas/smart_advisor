"""
Core chat logic, converted from the Gradio prototype's gradio_chat() function.

A ChatSession holds all per-conversation state (profile, contact info, memory,
onboarding/awaiting-info flags). The SessionStore keeps sessions in memory,
keyed by session_id, so the frontend just needs to send a session_id with
each request.

NOTE: in-memory storage is wiped on server restart and won't work across
multiple server processes/workers. For production, swap SessionStore's
internals for Redis or a database — the ChatSession dataclass and
process_message() function don't need to change.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import threading

from groq import APIStatusError
from src.utils.config import openrouter_client, groq_client
from src.rag.memory import (
    get_empty_memory, save_to_memory, history_as_messages,
    strip_personalization_tail, _save_to_memory_background,
)
from src.rag.profile import extract_profile, next_onboarding_question
from src.rag.retrieval import multi_query_search, condense_followup_question
from src.rag.system_prompt import SYSTEM_PROMPT
from src.rag.conversational import is_conversational
from src.utils.schemas import StudentProfile, StudentContact
from src.rag.services.fallback_service import record_unknown_question, handle_tool_calls, tools
from src.utils.context_builder import build_context

FALLBACK_MARKER = "لا تتوفر"  # used to locate the original question after a fallback


@dataclass
class ChatSession:
    student: StudentContact = field(default_factory=StudentContact)
    awaiting_info: bool = False
    profile: StudentProfile = field(default_factory=StudentProfile)
    onboarding: bool = True
    memory: object = field(default_factory=get_empty_memory)
    history: list[dict] = field(default_factory=list)  # [{"role": ..., "content": ...}, ...]
    pending_question: str | None = None
    _memory_lock: threading.Lock = field(default_factory=threading.Lock)

class SessionStore:
    """In-memory session store. Replace with Redis/DB for multi-worker deployments."""

    def __init__(self):
        self._sessions: dict[str, ChatSession] = {}

    def get(self, session_id: str) -> ChatSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession()
        return self._sessions[session_id]

    def reset(self, session_id: str) -> ChatSession:
        self._sessions[session_id] = ChatSession()
        return self._sessions[session_id]


session_store = SessionStore()


def _extract_student_info(user_message: str, student: StudentContact) -> StudentContact:
    """Try to extract name/email/phone from the student's message."""
    import json as _json
    try:
        resp = openrouter_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Extract student contact details from the message if present."
                    "Respond ONLY in JSON: {'name': '...', 'email': '...', 'phone': '...'}"
                    "Use null for missing fields."
                )},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
        extracted = _json.loads(resp.choices[0].message.content)
        for key in ("name", "email", "phone"):
            if extracted.get(key):
                setattr(student, key, extracted[key])
    except Exception:
        pass
    return student

def _save_to_memory_background_locked(session: ChatSession, user_msg: str, assistant_msg: str) -> None:
    with session._memory_lock:
        _save_to_memory_background(session.memory, user_msg, assistant_msg)

def process_message(session: ChatSession, user_message: str) -> str:
    """
    Process one user message against the given session, mutating session state
    in place, and return the assistant's reply text.
    """

    with session._memory_lock:
        pass  # wait for any in-flight background save to finish before reading memory

    if not user_message.strip():
        return ""

    session.history.append({"role": "user", "content": user_message})

    # ── Phase 1: Onboarding — collect student profile ──────────────────
    if session.onboarding:
        session.profile = extract_profile(user_message, session.profile)
        next_q = next_onboarding_question(session.profile)

        if next_q:
            response = next_q
        else:
            session.onboarding = False
            response = (
                "شكراً! الآن لديّ صورة واضحة عن وضعك الأكاديمي وأهدافك. "
                "يمكنني مساعدتك بشكل أفضل. ما سؤالك؟"
            )

        session.history.append({"role": "assistant", "content": response})
        return response

    # ── Phase 2: Info-collection mode (post-fallback) ───────────────────
    if session.awaiting_info:
        session.student = _extract_student_info(user_message, session.student)
        still_missing = session.student.missing()

        if still_missing == "name":
            response = "شكراً! ما اسمك الكريم؟"
        elif still_missing == "contact":
            response = "شكراً! هل يمكنك تزويدي ببريدك الإلكتروني أو رقم هاتفك حتى يتمكن المرشد من التواصل معك؟"
        else:
            record_unknown_question(                      
                question=session.pending_question,
                name=session.student.name,
                email=session.student.email,
                phone=session.student.phone,
            )
            response = "شكراً! تم إحالة سؤالك إلى المرشد الأكاديمي وسيتواصل معك قريباً. هل لديك أي سؤال آخر؟"
            session.awaiting_info = False
            session.pending_question = None

        session.history.append({"role": "assistant", "content": response})
        return response

    # ── Conversational check — before any retrieval ──────────────────────  # NEW BLOCK
    conversational, polite_response = is_conversational(user_message)
    if conversational:
        session.history.append({"role": "assistant", "content": polite_response or ""})
        return polite_response

    # ── Phase 3: Normal RAG flow ─────────────────────────────────────────
    resolved_message = condense_followup_question(user_message, session.memory)
    search_result = multi_query_search(resolved_message, session.memory, session.profile)

    # Layer 1: below similarity threshold — no relevant chunks at all
    if not search_result["has_answer"]:
        still_missing = session.student.missing()
        if still_missing:
            session.awaiting_info = True
            pending_question = resolved_message
            response = (
                "لا تتوفر لديّ معلومات كافية للإجابة على هذا السؤال.\n"
                + ("سأحيل سؤالك إلى المرشد الأكاديمي. ما اسمك الكريم؟"
                   if still_missing == "name"
                   else "سأحيل سؤالك إلى المرشد الأكاديمي. هل يمكنك تزويدي ببريدك الإلكتروني أو رقم هاتفك؟")
            )
        else:
            record_unknown_question(
                question=resolved_message,
                name=session.student.name,
                email=session.student.email,
                phone=session.student.phone,
            )
            response = "لا تتوفر لديّ معلومات كافية للإجابة على هذا السؤال. تم إحالة سؤالك إلى المرشد الأكاديمي وسيتواصل معك قريباً."

        session.history.append({"role": "assistant", "content": response})
        return response

    # Layer 2: pass chunks + profile to LLM, allow tool call as second-level fallback
    context = build_context(search_result["documents"])
    history_msgs = history_as_messages(session.memory)

    # ── NEW: force coverage of every program actually retrieved ─────────
    programs_in_context = sorted({
        m.get("program") for m in search_result["metadatas"] if m.get("program")
    })
    coverage_instruction = ""
    if len(programs_in_context) > 2:
        coverage_instruction = (
            f"\n\n Note: The above context covers this programs {len(programs_in_context)}:"
            f"{'، '.join(programs_in_context)}.\n"
            f"Your answer should address each major on this list individually (even if the previous conversation mentioned fewer majors)\n"
            "don't limit yourself to majors already discussed or those that seem most relevant to the student's interests.\n"
            "Place any personal recommendations based on the student's profile in a separate, clear paragraph at the end of your answer, only after you have covered all the majors."
        )

    profile_ctx = session.profile.to_context_string()

    prompts = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history_msgs,
        {"role": "user", "content": (
            f"معلومات الطالب: {profile_ctx}\n"
            f"السياق:\n{context}\n"
            f"السؤال: {resolved_message}\n"
            f"Use student information only if the question explicitly concerns its relevance to them."
            f"(e.g., Is it suitable for me? Can I enroll? Is my GPA sufficient?) or other relative questions"
            f"Otherwise, answer the question as is, without analyzing eligibility or suitability.\n"
            f"{coverage_instruction}\n"
        )},
    ]

    response = ""
    done = False
    while not done:
        try:
            resp_obj = groq_client.chat.completions.create(
                model="qwen/qwen3-32b", messages=prompts, tools=tools, temperature=0.1
            )
        except APIStatusError as e:
            resp_obj = openrouter_client.chat.completions.create(
                model="openai/gpt-4o-mini", messages=prompts, tools=tools, temperature=0.1
            )

        finish_reason = resp_obj.choices[0].finish_reason
        message = resp_obj.choices[0].message

        if finish_reason == "tool_calls":
            still_missing = session.student.missing()
            if still_missing:
                session.awaiting_info = True
                pending_question = resolved_message
                response = (
                    "لا تتوفر لديّ معلومات كافية للإجابة على هذا السؤال.\n"
                    + ("سأحيل سؤالك إلى المرشد الأكاديمي. ما اسمك الكريم؟"
                       if still_missing == "name"
                       else "سأحيل سؤالك إلى المرشد الأكاديمي. هل يمكنك تزويدي ببريدك الإلكتروني أو رقم هاتفك؟")
                )
            else:
                record_unknown_question(
                    question=resolved_message,
                    name=session.student.name,
                    email=session.student.email,
                    phone=session.student.phone,
                )
                response = "لا تتوفر لديّ معلومات كافية للإجابة على هذا السؤال. تم إحالة سؤالك إلى المرشد الأكاديمي وسيتواصل معك قريباً."
            done = True
        else:
            response = message.content
            done = True

    if not session.awaiting_info:
        threading.Thread(
            target=_save_to_memory_background_locked,
            args=(session, user_message, response),
            daemon=True,
        ).start()

    session.history.append({"role": "assistant", "content": response})
    return response
