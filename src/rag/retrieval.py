"""
Multi-query retrieval: generate diverse query variants, search ChromaDB
with the detected metadata filter, and merge/deduplicate results.
Also handles query condensation — resolving elliptical follow-ups into
standalone questions before retrieval ever sees them.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_classic.memory import ConversationBufferWindowMemory

from src.utils.config import (
    openrouter_client, SIMILARITY_THRESHOLD, TOP_K,
    MAX_CHUNKS, MAX_CONTEXT_TOKENS, QUERY_VARIANTS_N,
)
from src.rag.metadata_filter import detect_metadata_filter
from src.knowledge_base.vector_store import search
from src.utils.schemas import StudentProfile


def generate_queries(
    user_message: str,
    memory: ConversationBufferWindowMemory,
    profile: StudentProfile,
    n: int = QUERY_VARIANTS_N,
) -> list[str]:
    """Generate n diverse search query variants, resolving pronouns via chat history."""
    chat_history = memory.load_memory_variables({})["history"]
    history_msgs = [
        {"role": ("user" if m.type == "human" else "assistant"), "content": m.content}
        for m in chat_history
    ]
    profile_ctx = profile.to_context_string()

    prompt = [
        {"role": "system", "content": f"""You are a search optimisation assistant for an academic knowledge base.
Student profile: {profile_ctx}
Given the student's question, generate {n} search queries approaching the topic from completely different angles.
Each query must:
- Use different keywords than the others
- Target a different aspect of the topic
- IMPORTANT: If the question contains pronouns or references (e.g. 'خطته', 'هذا البرنامج', 'نفس التخصص'),
  resolve them using the conversation history before generating queries.
  Never generate queries with unresolved pronouns.
- Use the student profile to be specific where helpful
- Avoid mere synonyms — think about what section headings or labels appear in the document

You MUST respond with ONLY a raw JSON array of {n} strings, no markdown, no backticks, no explanation.
Correct format: ["query 1", "query 2", "query 3"]"""},
        *history_msgs,
        {"role": "user", "content": user_message},
    ]

    try:
        resp = openrouter_client.chat.completions.create(
            model="openai/gpt-4o-mini", messages=prompt, temperature=0.7, max_tokens=300
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        queries = json.loads(raw)
        return [user_message] + queries[:n]
    except Exception as e:
        print(f"[generate_queries error] {e}")
        return [user_message]


def estimate_tokens(text: str) -> int:
    """Rough token estimate without a tokenizer dependency (~2.2 chars/token)."""
    return max(1, int(len(text) / 2.2))

def _related_faq_filter(category: str) -> dict:
    """
    FAQ paragraphs are sometimes hand-tagged as relevant to a specific
    category even though their own category is "faq" (e.g. a paragraph
    discussing course load lives under category=faq but carries
    "study_plan_related": true). Whenever a category filter fires, we
    ALSO pull in any faq chunk carrying "<category>_related": true.
 
    If that flag doesn't exist yet for a given category (e.g. nobody has
    tagged any FAQ paragraph as "career_opportunities_related" yet), this
    where-clause simply matches zero chunks — safe no-op, no crash.
    """
    return {"$and": [{"category": "faq"}, {f"{category}_related": True}]}

def _round_robin_trim(buckets: "dict[str, list[tuple[str, dict]]]", max_context_tokens: int):
    """
    Given {category: [(doc, meta), ...]}, returns (documents, metadatas,
    running_tokens) built by taking one chunk from each non-empty category
    in turn, stopping once the token budget is hit — but always giving
    every category at least ONE chunk first, even if that alone exceeds
    budget. This is what keeps a compound query (e.g. scholarship +
    career_opportunities, ~3,600 tokens combined if dumped in full) from
    silently starving one category or blowing past a small model's total
    request limit (e.g. GitHub Models' 8,000-token cap on gpt-4o-mini).
    """
    buckets = {c: list(chunks) for c, chunks in buckets.items() if chunks}
    merged_docs: list[str] = []
    merged_meta: list[dict] = []
    running_tokens = 0
    guaranteed = set()  # categories that already got their first chunk
 
    categories = list(buckets.keys())
    round_idx = 0
    while categories:
        cat = categories[round_idx % len(categories)]
        doc, meta = buckets[cat].pop(0)
        doc_tokens = estimate_tokens(doc)
 
        must_keep = cat not in guaranteed  # first chunk per category is free
        if not must_keep and running_tokens + doc_tokens > max_context_tokens and merged_docs:
            if not buckets[cat]:
                categories.remove(cat)
                continue
            round_idx = (round_idx + 1) % len(categories)
            continue
 
        merged_docs.append(doc)
        merged_meta.append(meta)
        running_tokens += doc_tokens
        guaranteed.add(cat)
 
        if not buckets[cat]:
            categories.remove(cat)
        if categories:
            round_idx = (round_idx + 1) % len(categories)
 
        if running_tokens >= max_context_tokens and len(guaranteed) == len(buckets):
            break
 
    return merged_docs, merged_meta, running_tokens
    

def multi_query_search(
    user_message: str,
    memory: ConversationBufferWindowMemory,
    profile: StudentProfile,
    top_k: int = TOP_K,
    max_chunks: int = MAX_CHUNKS,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> dict:
    """
    1. Detect ALL active category filters from the query (not just one).
    2. Exact-fetch every active filter, PLUS its related FAQ paragraphs
       (via "<category>_related": true), and merge everything together.
    3. Only if NONE of the exact filters found anything, fall back to the
       existing multi-query similarity search (unchanged from before).
    """
    filters, programs, course_codes, course_names = detect_metadata_filter(user_message, memory)
 
    if filters:
        # Per-category buckets (not one flat list) so we can round-robin
        # across categories below — a compound query must not let one big
        # category (e.g. career_opportunities) starve a smaller one
        # (e.g. scholarship) out of the token budget entirely.
        seen_keys: set[str] = set()
        buckets: dict[str, list[tuple[str, dict]]] = {}
 
        for f in filters:
            bucket = buckets.setdefault(f["category"], [])
 
            result = search(user_message, where=f["where"], is_exact_fetch=True)
            if result["has_answer"]:
                for doc, meta in zip(result["documents"], result["metadatas"]):
                    key = re.sub(r"\s+", "", doc)[:200]
                    if key not in seen_keys:
                        seen_keys.add(key)
                        bucket.append((doc, meta))
 
            related = search(user_message, where=_related_faq_filter(f["category"]), is_exact_fetch=True)
            if related["has_answer"]:
                for doc, meta in zip(related["documents"], related["metadatas"]):
                    key = re.sub(r"\s+", "", doc)[:200]
                    if key not in seen_keys:
                        seen_keys.add(key)
                        bucket.append((doc, meta))
 
        if any(buckets.values()):
            merged_docs, merged_meta, running_tokens = _round_robin_trim(buckets, max_context_tokens)
            print(f"[Multi-Query] {len(filters)} active filter(s) -> {len(merged_docs)} chunks kept "
                  f"across {len(buckets)} categories (~{running_tokens} est. tokens, budget={max_context_tokens})")
            return {"has_answer": True, "documents": merged_docs, "metadatas": merged_meta, "best_score": 1.0}
 
        print("  [Multi-Query] none of the exact-category filters matched anything — falling back to similarity search")
 
    # ── Similarity-search fallback: identical to the previous implementation ──
    metadata_filter = None
    if course_codes:
        metadata_filter = {"course_code": {"$in": course_codes}} if len(course_codes) > 1 else {"course_code": course_codes[0]}
        if programs:
            prog_f = {"program": {"$in": programs}} if len(programs) > 1 else {"program": programs[0]}
            metadata_filter = {"$and": [metadata_filter, prog_f]}
    elif course_names:
        metadata_filter = {"course_name": {"$in": course_names}} if len(course_names) > 1 else {"course_name": course_names[0]}
        if programs:
            prog_f = {"program": {"$in": programs}} if len(programs) > 1 else {"program": programs[0]}
            metadata_filter = {"$and": [metadata_filter, prog_f]}
    elif programs:
        metadata_filter = {"program": {"$in": programs}} if len(programs) > 1 else {"program": programs[0]}
 
    queries = generate_queries(user_message, memory, profile)
    print(f"[Multi-Query] {len(queries)} queries: {queries}")
 
    chunk_map: dict[str, tuple[float, str, dict]] = {}
    best_score = 0.0
 
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        futures = {executor.submit(search, q, top_k, metadata_filter): q for q in queries}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"[Multi-Query] search failed for a query variant: {e}")
                continue
            if result["best_score"] > best_score:
                best_score = result["best_score"]
            for doc, meta, score in zip(result["documents"], result["metadatas"], result["scores"]):
                key = re.sub(r"\s+", "", doc)[:200]
                if key not in chunk_map or score > chunk_map[key][0]:
                    chunk_map[key] = (score, doc, meta)
 
    if not chunk_map or best_score < SIMILARITY_THRESHOLD:
        return {"has_answer": False, "documents": [], "metadatas": [], "best_score": best_score}
 
    ranked = sorted(chunk_map.values(), key=lambda x: x[0], reverse=True)[:max_chunks]
 
    merged_docs, merged_meta, running_tokens = [], [], 0
    for score, doc, meta in ranked:
        doc_tokens = estimate_tokens(doc)
        if running_tokens + doc_tokens > max_context_tokens and merged_docs:
            break
        merged_docs.append(doc)
        merged_meta.append(meta)
        running_tokens += doc_tokens
 
    print(f"[Multi-Query] {len(chunk_map)} unique chunks found, kept {len(merged_docs)} "
          f"(~{running_tokens} estimated tokens, budget={max_context_tokens})")
    return {"has_answer": True, "documents": merged_docs, "metadatas": merged_meta, "best_score": best_score}

# ── NEW: query condensation — resolve elliptical follow-ups upstream ───────
_CONDENSE_SYSTEM = """You are a query-rewriting component in an academic advising
system. Your ONLY job is to determine whether the student's latest message is a
complete, self-contained question, and if not, rewrite it into one using the
conversation history. You do NOT answer the question. You do NOT add facts that
aren't already present in the history or the message itself.

═══════════════════════════════════════════════════════════════════
STEP 1 — DECIDE: does this message depend on prior context to make sense?
═══════════════════════════════════════════════════════════════════
A message is SELF-CONTAINED (leave UNCHANGED) if a person with NO access to the
conversation history could understand exactly what is being asked — the subject
(program/course/policy), and any condition (a GPA, a track, a comparison target),
are all named explicitly in the message itself.

A message is DEPENDENT (must be rewritten) if understanding it requires knowing
something from an earlier turn — a pronoun, an omitted subject, an omitted
condition, or an implied continuation of a prior comparison/scenario.

═══════════════════════════════════════════════════════════════════
STEP 2 — IF DEPENDENT: identify which category of dependency this is, then repair it
═══════════════════════════════════════════════════════════════════

CATEGORY A — Pronoun / demonstrative reference
  Markers: "هذا التخصص", "هذا البرنامج", "هذا المساق", "هذه المنحة", "نفسه"
  Fix: replace the pronoun with the specific name it refers to, found by
  scanning backward through history for the most recently discussed program/
  course/policy. If more than one candidate could match, pick the one most
  recently and most centrally discussed (not just mentioned in passing).

CATEGORY B — Elliptical continuation ("و..." / "وماذا عن" / "ماذا عن")
  Markers: message starts with "و" attached to a bare noun phrase, or "ماذا عن"،
  with no verb or condition of its own — it's a sentence fragment, not a
  complete question.
  Fix: take the FULL question structure (verb, condition, question type) from
  the most recent prior question, and substitute only the new subject the
  fragment introduces. Everything else about the prior question's structure
  is preserved as-is.

CATEGORY C — Carried-over hypothetical condition
  Markers: the prior turn described a hypothetical (a GPA number, a track, a
  scenario) that does NOT belong to the actual student, and the new message
  continues discussing "that same scenario" without restating the condition.
  Fix: explicitly restate the hypothetical condition from the prior turn
  in the rewritten question, word-for-word as it was originally stated.
  CRITICAL: never substitute the actual student's own profile data (their
  real GPA, track, interests) for a hypothetical condition someone raised
  about a DIFFERENT, unnamed student. A hypothetical "طالب معدله 60" is not
  the current student — keep it hypothetical in the rewrite.

CATEGORY D — Comparative extension ("وبالمقارنة مع" / implied "what about the other one")
  Markers: the new message adds one more item to a comparison already
  established in a prior turn, without restating what's being compared.
  Fix: name both the new item and the original comparison subject(s)
  explicitly, and preserve what ASPECT is being compared (admission, courses,
  career paths, etc.) from the earlier turn.

CATEGORY E — Topic shift disguised as continuation
  Markers: the new message uses a pronoun or short form, but is actually
  about something with NO clear referent in recent history (the conversation
  moved on, or the reference is genuinely new).
  Fix: do NOT force a resolution. If no confident referent exists in the
  last few turns, return the message UNCHANGED rather than guessing — a
  wrong guess is worse than leaving it ambiguous for the next stage to handle.

═══════════════════════════════════════════════════════════════════
HARD CONSTRAINTS — apply to every rewrite
═══════════════════════════════════════════════════════════════════
1. Output ONLY the rewritten question. No explanation, no labels, no markdown,
   no quotes around it.
2. Keep the same language as the input (Arabic stays Arabic).
3. Do NOT answer the question — you are rewriting it, not responding to it.
4. Do NOT introduce any fact, number, program name, or condition that isn't
   explicitly present somewhere in the provided history or the message itself.
5. Do NOT merge unrelated subjects from different, distant turns just because
   they both appeared somewhere in history — only resolve against the most
   recent relevant turn(s).
6. If genuinely unsure whether a rewrite is needed or what it should be,
   prefer returning the message UNCHANGED over guessing.

═══════════════════════════════════════════════════════════════════
CRITICAL — THE EXAMPLES BELOW ARE ILLUSTRATIVE ONLY
═══════════════════════════════════════════════════════════════════
The examples use placeholder names like "تخصص أ" and "مساق ب" — these are
NOT real programs and NEVER appear in the actual conversation you're given.
Do NOT copy, adapt, or reuse any wording from these examples in your output,
even if the real conversation happens to resemble one of them closely.
Your output must be derived ONLY from the actual history and message you are
given in this specific call — never from this instructions text.

═══════════════════════════════════════════════════════════════════
EXAMPLES (placeholder names only — never mirror real program names)
═══════════════════════════════════════════════════════════════════

[Category A — pronoun reference]
History: user asked about "مساق أ" course content.
New: "من يدرّس هذا المساق؟"
Rewritten: "من يدرّس مساق أ؟"

[Category B — elliptical continuation]
History: user: "ما هي مساقات تخصص أ في السنة الأولى؟"
New: "وفي السنة الثانية؟"
Rewritten: "ما هي مساقات تخصص أ في السنة الثانية؟"

[Category C — hypothetical condition carryover]
History: user: "هل يمكن لطالب معدله أقل من 60 دخول تخصص أ؟"
         assistant: (eligibility answer)
New: "وتخصص ب؟"
Rewritten: "هل يمكن لطالب معدله أقل من 60 دخول تخصص ب؟"
(NOT the current student's own GPA — the hypothetical "أقل من 60" must be preserved.)

[Category D — comparative extension]
History: user: "ما الفرق من ناحية شروط القبول بين تخصص أ وتخصص ب؟"
New: "وتخصص ج؟"
Rewritten: "ما هو شرط القبول لتخصص ج، مقارنة بتخصص أ وتخصص ب؟"

[Category E — no confident referent, leave unchanged]
History: last 3 turns were all about scholarships, no program discussed recently.
New: "هل هذا يتطلب مقابلة شخصية؟"
Rewritten: "هل هذا يتطلب مقابلة شخصية؟"  (unchanged — "هذا" has no clear referent; do not guess)

[Already self-contained — no rewrite needed]
New: "من يدرّس مساق تراكيب بيانات؟"
Rewritten: "من يدرّس مساق تراكيب بيانات؟"  (unchanged — already names the subject explicitly)

CRITICAL FAILURE MODE TO AVOID:
Your output must NEVER still contain an unresolved reference word ("وماذا عن",
"هذا", "هذه", "نفسه") — if your output still has one of these, you have
failed to do your job. A correct rewrite REPLACES these words with the
actual subject; it never just returns the fragment as-is or restates a
DIFFERENT unresolved fragment.

CRITICAL — YOU ARE NOT A CONVERSATIONAL ASSISTANT:
=====================================================
You NEVER ask the student a clarifying question yourself. You NEVER request
additional information from them. You NEVER produce sub-questions about their
interests, GPA, or preferences unless the ORIGINAL message already asked
about exactly that. Your only two valid outputs are: (1) the original
message unchanged, or (2) the same question with its ambiguous reference
replaced by an explicit subject. Nothing else is a valid output, ever.
"""


def _low_lexical_overlap(original: str, rewritten: str, threshold: float = 0.2) -> bool:
    """
    A correct condensation PRESERVES most of the original question's wording,
    substituting only the unresolved pronoun/ellipsis. If the rewritten text
    shares almost no content words with the original, the model likely
    drifted into generating a DIFFERENT question entirely (e.g. asking the
    student a clarifying question, or answering) rather than resolving the
    one it was given.
    """
    def content_words(s: str) -> set:
        return set(w for w in re.findall(r"[\w\u0600-\u06FF]+", s) if len(w) > 2)

    orig = content_words(original)
    new = content_words(rewritten)
    if not orig:
        return False
    overlap = orig & new
    return (len(overlap) / len(orig)) < threshold


def condense_followup_question(user_message: str, memory: ConversationBufferWindowMemory) -> str:
    """
    Resolve elliptical follow-ups into standalone questions BEFORE retrieval
    or generation ever see them.
    """
    chat_history = memory.load_memory_variables({})["history"]
    if not chat_history:
        return user_message

    history_msgs = [
        {"role": ("user" if m.type == "human" else "assistant"), "content": m.content}
        for m in chat_history[-4:]
    ]
    try:
        resp = openrouter_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": _CONDENSE_SYSTEM},
                *history_msgs,
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        rewritten = resp.choices[0].message.content.strip()
        if rewritten and not _low_lexical_overlap(user_message, rewritten):
            print(f"[Condense] '{user_message}' → '{rewritten}'")
            return rewritten

        print(f"[Condense] rejected (failed validation) — original: '{user_message}' | got: '{rewritten}'")
        return user_message
    except Exception as e:
        print(f"[Condense error] {e}")
        return user_message