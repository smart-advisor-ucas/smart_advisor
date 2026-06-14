"""
Multi-query retrieval: generate diverse query variants, search ChromaDB
with the detected metadata filter, and merge/deduplicate results.
"""
import json
import re

from langchain_classic.memory import ConversationBufferWindowMemory

from src.utils.config import github_client, SIMILARITY_THRESHOLD, TOP_K
from metadata_filter import detect_metadata_filter
from src.knowledge_base.vector_store import search
from src.utils.schemas import StudentProfile


def generate_queries(
    user_message: str,
    memory: ConversationBufferWindowMemory,
    profile: StudentProfile,
    n: int = 3,
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
        resp = github_client.chat.completions.create(
            model="gpt-4o-mini", messages=prompt, temperature=0.7, max_tokens=300
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        queries = json.loads(raw)
        return [user_message] + queries[:n]
    except Exception as e:
        print(f"[generate_queries error] {e}")
        return [user_message]


def multi_query_search(
    user_message: str,
    memory: ConversationBufferWindowMemory,
    profile: StudentProfile,
    top_k: int = TOP_K,
) -> dict:
    """
    1. Detect metadata filter from query (program/course/study_plan).
    2. If study plan request -> fetch all matching chunks directly, skip multi-query.
    3. Otherwise generate diverse query variants (profile-steered) and search with filter.
    4. Merge results, deduplicating by normalised text fingerprint.
    """
    metadata_filter, is_study_plan = detect_metadata_filter(user_message)

    # Study plan: bypass embedding/ranking entirely, return full matching set
    if is_study_plan and metadata_filter:
        return search(user_message, where=metadata_filter, is_study_plan=True)

    queries = generate_queries(user_message, memory, profile)

    seen_keys: set[str] = set()
    merged_docs: list[str] = []
    merged_meta: list[dict] = []
    best_score = 0.0

    for query in queries:
        result = search(query, top_k=top_k, metadata_filter=metadata_filter)
        if result["best_score"] > best_score:
            best_score = result["best_score"]

        for doc, meta in zip(result["documents"], result["metadatas"]):
            key = re.sub(r"\s+", "", doc)[:200]
            if key not in seen_keys:
                seen_keys.add(key)
                merged_docs.append(doc)
                merged_meta.append(meta)

    if not merged_docs or best_score < SIMILARITY_THRESHOLD:
        return {"has_answer": False, "documents": [], "metadatas": [], "best_score": best_score}

    return {"has_answer": True, "documents": merged_docs, "metadatas": merged_meta, "best_score": best_score}
