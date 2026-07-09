"""
BGE-M3 embeddings via HuggingFace Inference API, with retry-on-failure.
"""
import time
import random
import numpy as np
from huggingface_hub import InferenceClient

from src.utils.config import HF_KEY

_hf_client = InferenceClient(provider="hf-inference", api_key=HF_KEY)


def embed_texts_with_retry(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """Embed a list of texts using BGE-M3, with exponential backoff on failure."""
    embeddings = []
    for text in texts:
        for attempt in range(max_retries):
            try:
                vector = np.array(_hf_client.feature_extraction(text, model="BAAI/bge-m3"))
                if vector.ndim > 1:
                    vector = vector.squeeze()
                vector = vector / np.linalg.norm(vector)
                embeddings.append(vector.tolist())
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                backoff = (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(backoff)
    return embeddings