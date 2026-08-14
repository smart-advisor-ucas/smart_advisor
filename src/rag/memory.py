"""
Conversation memory helpers (LangChain ConversationBufferWindowMemory with
a manually enforced window size) and post-response memory hygiene.
"""
from langchain_classic.memory import ConversationBufferWindowMemory

from src.utils.config import MEMORY_WINDOW_K, groq_client

# NOTE: matches notebook's current (possibly unintentional) mismatch — the
# memory object itself is built with MEMORY_WINDOW_K (5), but this local
# trim constant is 3, making 3 the effective operative window. See the
# config.py note above — flag for confirmation before treating as final.
_SAVE_TRIM_K = 3


def get_empty_memory() -> ConversationBufferWindowMemory:
    return ConversationBufferWindowMemory(k=MEMORY_WINDOW_K, return_messages=True, memory_key="history")


def save_to_memory(memory: ConversationBufferWindowMemory, user_msg: str, assistant_msg: str) -> None:
    """Save a turn and manually enforce the (tighter) trim window."""
    memory.save_context({"input": user_msg}, {"output": assistant_msg})
    limit = _SAVE_TRIM_K * 2
    if len(memory.chat_memory.messages) > limit:
        memory.chat_memory.messages = memory.chat_memory.messages[-limit:]


def history_as_messages(memory: ConversationBufferWindowMemory) -> list[dict]:
    """Convert stored memory into OpenAI-style chat messages."""
    chat_history = memory.load_memory_variables({})["history"]
    return [
        {"role": ("user" if m.type == "human" else "assistant"), "content": m.content}
        for m in chat_history
    ]


# ── NEW: memory hygiene — strip personalization before saving ──────────────
_MEMORY_STRIP_SYSTEM = """You are given an academic advisor's answer to a student.
Your task: return ONLY the factual/informational part of the answer — remove
any section that personalizes the answer to the specific student (GPA checks,
"يناسبك"/"يتوافق مع اهتماماتك" framing, eligibility verdicts, program
recommendations, "الاختيار الأمثل" or similar closing recommendations).

Rules:
- If the answer has NO personalization content at all, return it UNCHANGED,
  word-for-word, in full. This is the common case — most answers (e.g. course
  lists, faculty information, study plans, admission requirements) contain no
  personalization at all and must come back exactly as given.
- Removing personalization means deleting a specific sentence or clause that
  ties information to the individual student. It does NOT mean summarizing,
  shortening, or omitting factual content that just happens to be near a
  personalized sentence — keep everything else word-for-word.
- Example: a question about faculty members with their names, ranks, and
  specializations has NO personalization to remove — return it unchanged,
  even if it explains what each specialization covers.
- Only remove personalization/recommendation content, nothing else.
- Return ONLY the trimmed answer text, no explanation, no markdown fences.
"""

# Arabic runs ~2.2 chars/token (see project token-budget heuristic). The
# expected output is very often the SAME length as the input (most answers
# have nothing to strip), so the budget must cover a full-length response
# plus room for reasoning_effort="low" overhead — not half the input.
_CHARS_PER_TOKEN = 2.2
_MIN_KEEP_RATIO = 0.5  # if the model returns less than this fraction of the
                       # original length, treat it as over-trimming and keep
                       # the original instead of trusting the output.


def strip_personalization_tail(response: str) -> str:
    """
    Before saving an assistant turn to memory, remove personalization/
    recommendation content so a fit-related answer doesn't bias the tone
    of unrelated future turns still inside the k-window. The full
    personalized answer is still shown to the student this turn — only
    what gets remembered is trimmed.
    """
    if not response or len(response) < 50:
        return response
    try:
        estimated_tokens = int(len(response) / _CHARS_PER_TOKEN)
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": _MEMORY_STRIP_SYSTEM},
                {"role": "user", "content": response},
            ],
            temperature=0.0,
            max_tokens=estimated_tokens + 300,  # full-length budget + reasoning/margin
            reasoning_effort="low",
            include_reasoning=False,
        )
        trimmed = (resp.choices[0].message.content or "").strip()

        if not trimmed:
            print("[Memory strip] empty result — keeping original")
            return response

        # Safety net: a plausible strip removes a sentence or two, not most
        # of the answer. If most of the content disappeared, something went
        # wrong (truncation or over-aggressive stripping) — don't trust it.
        ratio = len(trimmed) / len(response)
        if ratio < _MIN_KEEP_RATIO:
            print(
                f"[Memory strip] suspicious over-trim ({ratio:.0%} of original "
                f"kept) — keeping original instead. Trimmed output was:\n{trimmed}"
            )
            return response

        print(f"[Memory] stripped {len(response) - len(trimmed)} chars of personalization")
        print(trimmed)
        return trimmed
    except Exception as e:
        print(f"[Memory strip error] {e}")
        return response

def _save_to_memory_background(memory: ConversationBufferWindowMemory, user_msg: str, assistant_msg: str) -> None:
    """
    Runs on a background thread, after the response has already been returned
    to the caller — memory hygiene (personalization stripping + save) doesn't
    need to block the response, since it only affects what future turns see,
    not the current one.
    """
    try:
        cleaned = strip_personalization_tail(assistant_msg)
        save_to_memory(memory, user_msg, cleaned)
    except Exception as e:
        print(f"[Background memory save error] {e}")
        save_to_memory(memory, user_msg, assistant_msg)  # fail safe — save unstripped rather than lose the turn