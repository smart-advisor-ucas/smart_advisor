"""
Pydantic models:
- StudentProfile: structured student data collected during onboarding
- API request/response schemas used by the FastAPI routes
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    """Stores structured information about the student, collected during onboarding.
    Used to personalise query rewriting and LLM answers."""

    # Admission eligibility
    gpa: float | None = Field(None, description="Tawjihi GPA as a percentage (0-100)", ge=0, le=100)
    academic_track: Literal["علمي", "صناعي", "تكنولوجيا معلومات", "أدبي", "تجاري", "أخرى"] | None = Field(
        None, description="High school academic track"
    )
    # enrollment_status: Literal["طالب حالي", "طالب جديد", "مهتم بالالتحاق"] | None = Field(
    #     None, description="Whether the student is currently enrolled or prospective"
    # )

    # Interests
    likes_math: bool | None = Field(None, description="Enjoys mathematics and statistics")
    has_programming_background: bool | None = Field(None, description="Has prior programming experience")
    interest_areas: list[Literal[
        "علم البيانات", "الذكاء الاصطناعي", "أمن المعلومات",
        "شبكات الحاسوب", "هندسة الحاسوب", "غير محدد"
    ]] = Field(default_factory=list, description="Areas of academic or professional interest")

    # Academic goals
    degree_preference: Literal["بكالوريوس", "دبلوم", "غير محدد"] | None = Field(
        None, description="Preferred degree level"
    )
    # needs_financial_aid: bool | None = Field(None, description="Interested in scholarships")

    # Priority fields that drive onboarding — ordered by importance
    _PRIORITY = ["gpa", "academic_track", "likes_math", "interest_areas", "degree_preference"]

    def missing_fields(self) -> list[str]:
        """Return priority fields still None (or empty list for interest_areas)."""
        result = []
        for f in self._PRIORITY:
            v = getattr(self, f)
            if v is None or (isinstance(v, list) and len(v) == 0):
                result.append(f)
        return result

    def is_complete(self) -> bool:
        return len(self.missing_fields()) == 0

    def to_context_string(self) -> str:
        """Arabic summary injected into prompts."""
        parts = []
        if self.gpa is not None:
            parts.append(f"المعدل في الثانوية: {self.gpa}%")
        if self.academic_track:
            parts.append(f"الفرع الدراسي: {self.academic_track}")
        # if self.enrollment_status:
        #     parts.append(f"الحالة: {self.enrollment_status}")
        if self.likes_math is not None:
            parts.append("يستمتع بالرياضيات" if self.likes_math else "لا يفضل الرياضيات")
        # if self.has_programming_background is not None:
        #     parts.append("لديه خلفية برمجية" if self.has_programming_background else "لا توجد خلفية برمجية")
        if self.interest_areas:
            parts.append(f"اهتماماته: {', '.join(self.interest_areas)}")
        if self.degree_preference:
            parts.append(f"يريد: {self.degree_preference}")
        # if self.needs_financial_aid:
        #     parts.append("مهتم بالمنح الدراسية")
        return " | ".join(parts) if parts else "لا توجد معلومات بعد"


class StudentContact(BaseModel):
    """Contact info collected only when a fallback to a human advisor is needed."""
    name: str | None = None
    email: str | None = None
    phone: str | None = None

    def missing(self) -> str | None:
        if not self.name:
            return "name"
        if not self.email and not self.phone:
            return "contact"
        return None


# ── API I/O schemas ─────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    onboarding: bool
    awaiting_info: bool
