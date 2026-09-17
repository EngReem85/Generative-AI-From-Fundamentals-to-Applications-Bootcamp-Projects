from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts: list[str]):
        # E5 models are trained with explicit passage/query prefixes.
        inputs = [f"passage: {t}" for t in texts]
        return self.model.encode(
            inputs,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_query(self, query: str):
        return self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
