"""
ChromaDB collection access and similarity/metadata search.
"""
import chromadb

from src.utils.config import DB_PATH, COLLECTION_NAME, SIMILARITY_THRESHOLD
from src.knowledge_base.embeddings import embed_texts_with_retry

_client = chromadb.PersistentClient(path=DB_PATH)
collection = _client.get_collection(COLLECTION_NAME)


def search(
    query: str,
    top_k: int = 3,
    metadata_filter: dict | None = None,
    where: dict | None = None,
    is_exact_fetch: bool = False,
) -> dict:
    """
    Embed query and search ChromaDB.

    metadata_filter: optional 'where' clause for similarity search.
    where:           'where' clause for a direct metadata lookup (used when is_exact_fetch=True).
    is_exact_fetch:  skip embedding similarity entirely and fetch ALL chunks matching `where`
                      via collection.get(). Used for structured category lookups
                      (study_plan / program_info / scholarship / all_programs comparison)
                      where the whole matching set is wanted, not a top-k similarity guess.

    NOTE: matches current notebook behavior — has_answer is returned True even when
    documents is empty on the exact-fetch path (only a warning is printed). This means
    the fallback-to-similarity-search in multi_query_search never actually triggers on
    a truly empty exact-fetch result. Recommend restoring an `if documents:` guard here
    if you want the fallback to work as originally intended — flagging for your decision.
    """
    # ── Study plan: direct metadata lookup, no ranking ─────────────────
    if is_exact_fetch and where:
        results = collection.get(where=where)
        documents = results["documents"]
        metadatas = results["metadatas"]

        if not documents:
            print("  [search] No exact-category chunks matched — falling back to similarity search")

        scores = [1.0] * len(documents)
        return {"has_answer": True, "documents": documents, "metadatas": metadatas,
                "scores": scores, "best_score": 1.0}

    # ── Embedding-based similarity search ──────────────────────────────
    query_embedding = embed_texts_with_retry([query])[0]

    query_params = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if metadata_filter:
        query_params["where"] = metadata_filter

    results = collection.query(**query_params)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Fallback: if filter matched nothing, retry without the filter
    if not documents and metadata_filter:
        query_params.pop("where", None)
        results = collection.query(**query_params)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

    if not documents:
        return {"has_answer": False, "documents": [], "metadatas": [], "scores": [], "best_score": 0.0}

    scores     = [1 - d for d in distances]
    best_score = max(scores)

    if best_score < SIMILARITY_THRESHOLD:
        return {"has_answer": False, "documents": [], "metadatas": [], "scores": [], "best_score": best_score}

    return {"has_answer": True, "documents": documents, "metadatas": metadatas, "scores": scores, "best_score": best_score}
