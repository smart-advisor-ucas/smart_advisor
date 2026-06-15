"""
ChromaDB collection access and similarity/metadata search.
"""
import chromadb

from src.utils.config import DB_PATH, COLLECTION_NAME, SIMILARITY_THRESHOLD
from src.knowledge_base.embeddings import embed_texts

_client = chromadb.PersistentClient(path=DB_PATH)
collection = _client.get_collection(COLLECTION_NAME)


def search(
    query: str,
    top_k: int = 3,
    metadata_filter: dict | None = None,
    where: dict | None = None,
    is_study_plan: bool = False,
) -> dict:
    """
    Embed query and search ChromaDB.

    metadata_filter: optional ChromaDB 'where' clause for similarity search,
                      e.g. {"program": "علم البيانات والذكاء الاصطناعي"}
    where:            ChromaDB 'where' clause used for direct metadata lookup
                      (only used when is_study_plan=True)
    is_study_plan:    if True, attempt a direct metadata lookup (collection.get)
                      before falling back to similarity search
    """
    # ── Study plan: direct metadata lookup, no ranking ─────────────────
    if is_study_plan and where:
        results = collection.get(where=where)
        documents = results["documents"]
        metadatas = results["metadatas"]

        if documents:
            return {"has_answer": True, "documents": documents, "metadatas": metadatas, "best_score": 1.0}
        # else: fall through to similarity search below

    # ── Embedding-based similarity search ──────────────────────────────
    query_embedding = embed_texts([query])[0]

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
        return {"has_answer": False, "documents": [], "metadatas": [], "best_score": 0.0}

    similarities = [1 - d for d in distances]
    best_score = max(similarities)

    if best_score < SIMILARITY_THRESHOLD:
        return {"has_answer": False, "documents": [], "metadatas": [], "best_score": best_score}

    return {"has_answer": True, "documents": documents, "metadatas": metadatas, "best_score": best_score}
