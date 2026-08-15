# RAG Pipeline — Smart Advisor

Technical documentation of the Retrieval-Augmented Generation pipeline that powers the Smart Advisor chatbot. This covers how a student's message travels from input to a grounded Arabic answer, including retrieval, filtering, and fallback logic.

---

## 1. High-Level Flow

```
Student message
   │
   ▼
Onboarding check ──(incomplete profile)──► extract_profile() → ask next question
   │ (profile complete)
   ▼
Awaiting-info check ──(true)──► collect name/email/phone → record_unknown_question()
   │ (false)
   ▼
detect_metadata_filter()  →  program / course / study-plan filter
   │
   ├── is_study_plan == True  ──► collection.get(where=filter)  (exact fetch, no embeddings)
   │
   └── is_study_plan == False ──► multi_query_search()
                                       │
                                       ├── generate_queries()  → N reformulated queries
                                       ├── search() per query  → ChromaDB similarity search
                                       └── merge + dedupe + best_score
   │
   ▼
Layer 1: best_score < SIMILARITY_THRESHOLD ?
   │ yes → no relevant chunks → fallback (Telegram escalation)
   │ no
   ▼
build_context()  →  numbered chunk list
   │
   ▼
Layer 2: LLM (GPT-4o-mini via OpenRouter) generates answer
   │
   ├── finish_reason == "tool_calls" → record_unknown_question() → fallback message
   └── finish_reason == "stop"       → answer returned to student
   │
   ▼
save_to_memory()  →  ConversationBufferWindowMemory (k=5)
```

---

## 2. Components

### 2.1 Embeddings — `embed_texts()`

- Model: `BAAI/bge-m3` (1024-dim), served via HuggingFace Inference API (`hf-inference` provider).
- Vectors are L2-normalized (`vector / np.linalg.norm(vector)`) before storage/query, so ChromaDB's cosine distance behaves consistently.
- A `time.sleep(0.1)` per call avoids HF free-tier rate limits.
- Production codebase parallelizes multi-query embedding calls with `ThreadPoolExecutor` — validated safe against the HF free-tier rate limit.

### 2.2 Vector Store — ChromaDB

- Collection: `ucas_knowledge_base`, persisted at `./chroma_db`.
- Two retrieval modes:
  - `collection.query()` — embedding-based similarity search (top-k), used for open-ended questions.
  - `collection.get(where=...)` — exact metadata fetch, used for study-plan/curriculum questions where round-robin similarity search would arbitrarily drop semesters.
- Metadata schema includes: `program`, `category` (`program_info`, `study_plan`, `career_opportunities`, `scholarship`, `faculty`, `course_basic`, `syllabus_main`, `syllabus_topics`), `course_code`, `course_name`, `year`, `semester`, `college`, `instructor`.

### 2.3 Metadata Filter — `detect_metadata_filter()`

An LLM call (GPT-4o-mini, `temperature=0.0`) classifies the query against two closed vocabularies (`KNOWN_PROGRAMS`, `KNOWN_COURSES`) so the returned filter values are guaranteed to exactly match ChromaDB metadata strings.
Detection order (most specific wins):
1. `is_study_plan` + program(s) → `{"$and": [{"category": "study_plan"}, {"program": ...}]}`
2. `course_codes` (+ program if present)
3. `course_names` (+ program if present)
4. `programs` only
5. No filter (broad search)

If no program is mentioned at all, the filter defaults to `"علم البيانات والذكاء الاصطناعي"` (the program this advisor represents) rather than leaving retrieval unscoped.

**Known failure mode:** if the filter matches zero chunks (e.g. a course exists in `KNOWN_COURSES` but wasn't actually chunked/ingested for that program), the query silently returns nothing — the pipeline logs this (`[search] No chunks matched the filter`) rather than crashing, but it is a silent-failure risk documented below.

### 2.4 Multi-Query Search — `multi_query_search()`

1. Calls `detect_metadata_filter()` once.
2. If it's a study-plan query, bypasses embeddings entirely and does an exact `collection.get()`.
3. Otherwise, calls `generate_queries()` — an LLM call that reformulates the student's question into `n` (default 3) queries from different angles, resolving pronouns/ellipsis against `ConversationBufferWindowMemory` history first (e.g. "خطته" → "خطة تخصص علم البيانات والذكاء الاصطناعي").
4. Runs `search()` for the original question **plus** each generated variant, all scoped to the same metadata filter.
5. Merges results, deduplicating by a whitespace-normalized 200-char fingerprint of each chunk.
6. Tracks the single best similarity score across all sub-queries; this score — not any one sub-query's score — is what Layer 1 checks against the threshold.

### 2.5 Two-Layer Fallback

| Layer | Trigger | Cost | Purpose |
|---|---|---|---|
| 1 — Similarity threshold | `best_score < SIMILARITY_THRESHOLD` | No LLM call | Catches questions with no related chunks at all |
| 2 — LLM judgment | Chunks exist but don't answer the question (e.g. comparison with another university) | One LLM call with `tools` | Only the LLM can judge whether retrieved chunks are actually sufficient |

`SIMILARITY_THRESHOLD` is currently `0.4` in the notebook (tuned down from an earlier `0.75` during evaluation — cosine similarity on BGE-M3 tends to run lower than dot-product embeddings for this domain). Tune further during evaluation weeks against a labeled query set.

Both layers converge on the same tool: `record_unknown_question(question, name, email, phone)`, which fires the advisor notification in a background thread and returns `{"recorded": "pending"}` immediately. The delivery channel is selected by `NOTIFY_CHANNEL` — `email` (Gmail SMTP, default; works from Hugging Face Spaces), `telegram` (works locally; Spaces egress IPs are blocked by Telegram), or `both`. On failure the question is appended to `failed_questions.log`.

### 2.6 System Prompt

Key design choices:
- Answers are grounded strictly in retrieved context — no outside knowledge, no fabrication.
- Explicit **comparison protocol**: when two programs/courses are compared, cover every shared dimension found in context and never invent differences; ask one clarifying question first if the student is asking "which is better *for me*."
- Explicit rule against recommending other universities' programs, even if more suitable — this is a college-policy constraint, not a factual one.
- The tool-call rule was tightened after early testing: only escalate when the topic has **zero overlap** with retrieved chunks, not merely an imperfect match — this reduced over-triggering of the fallback on general/adjacent questions (e.g. scholarship questions that apply college-wide, not DS&AI-specific).
- Always responds in Arabic regardless of the question's language.

### 2.7 Student Profile & Onboarding

- `StudentProfile` (Pydantic model): GPA, academic track, enrollment status, math/programming interest, interest areas, degree preference, financial aid need.
- `extract_profile()` — LLM call at `temperature=0.0` that merges extracted fields into the existing profile, **never overwriting** already-filled fields unless the student explicitly corrects them.
- `next_onboarding_question()` — deterministic lookup table (`_FIELD_QUESTIONS`), no LLM needed, driven by `StudentProfile._PRIORITY` field order.
- Once `profile.is_complete()`, the profile context string (`to_context_string()`) is injected into every Layer 2 prompt so answers can be personalized (e.g. "does this student's GPA meet the admission requirement?").
- `strip_personalization_tail` (production codebase) prevents this personalized/eligibility content from leaking into `ConversationBufferWindowMemory` and skewing unrelated future turns.

### 2.8 Memory

- `ConversationBufferWindowMemory(k=5, return_messages=True)` — keeps the last 5 turns.
- `save_to_memory()` manually trims `memory.chat_memory.messages` to `K*2` after each save as a belt-and-suspenders guard against the window not being enforced correctly.
- History is converted to OpenAI-style role dicts (`human` → `user`, else → `assistant`) before being spliced into both `generate_queries()` and the Layer 2 answer prompt.

---

## 3. LLM Providers

| Purpose | Model | Provider | Notes |
|---|---|---|---|
| Answer generation (Layer 2) | `gpt-4o-mini` | OpenRouter (`openai/gpt-4o-mini`) | Migrated from GitHub Models after its July 2026 retirement; prepaid, auto top-up disabled |
| Profile extraction | `gpt-4o-mini` | OpenRouter | `temperature=0.0` for deterministic parsing |
| Metadata filter detection | `gpt-4o-mini` | OpenRouter | `temperature=0.0`, `max_tokens=80` |
| Query reformulation | `gpt-4o-mini` | OpenRouter | `temperature=0.7` for diverse angles |
| Utility calls (e.g. `is_conversational` fallback) | `llama-3.1-8b-instant` | Groq | Fast/cheap; used after a `RateLimitError` crash surfaced the need for a fallback path |
| Preferred generation (quota permitting) | `qwen/qwen3-32b` | Groq | Higher quality, but quota-limited — `generate_answer_with_fallback()` handles routing around 413/429 |

---

## 5. Tuning Notes

- `SIMILARITY_THRESHOLD` — tune during evaluation weeks (7–8) against a labeled query set; current working value is `0.4` (down from an initial `0.75` guess, which was too aggressive for BGE-M3 cosine scores on this corpus).
- `top_k` per sub-query — currently 3; increasing this widens recall per query variant but increases token budget for `build_context()`.
- Number of reformulated queries (`n` in `generate_queries()`) — currently 3, plus the original question, for 4 total searches per turn. This is the main cost driver of Layer 2 latency; reducing `n` trades recall for speed.
- Token budget: system prompt (~3100 tokens) + memory history + merged chunk context must fit within the OpenRouter/provider TPM limit — use `estimate_tokens()` (2.2 chars/token, conservative for Arabic) when adjusting `top_k` or `n`.
