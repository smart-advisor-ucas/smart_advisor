"""
Build the numbered context string passed to the LLM from retrieved chunks.
"""


def build_context(documents: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(documents))
