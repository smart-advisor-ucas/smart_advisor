"""
BGE-M3 embeddings via HuggingFace Inference API.
"""
import time
import numpy as np
from huggingface_hub import InferenceClient

from src.utils.config import HF_KEY

_hf_client = InferenceClient(provider="hf-inference", api_key=HF_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using BGE-M3, with L2 normalisation."""
    embeddings = []
    for text in texts:
        vector = np.array(_hf_client.feature_extraction(text, model="BAAI/bge-m3"))
        if vector.ndim > 1:
            vector = vector.squeeze()
        vector = vector / np.linalg.norm(vector)
        embeddings.append(vector.tolist())
        time.sleep(0.1)  # avoid hitting rate limits on free tier
    return embeddings
