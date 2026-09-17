from __future__ import annotations

import numpy as np
import faiss


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks: list[dict] = []

    def build(self, vectors: np.ndarray, chunks: list[dict]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        vectors = np.asarray(vectors, dtype="float32")
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.chunks = chunks

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Index has not been built")
        k = min(top_k, len(self.chunks))
        q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, ids = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            item = dict(self.chunks[int(idx)])
            item["score"] = float(score)
            results.append(item)
        return results

    @property
    def size(self) -> int:
        return 0 if self.index is None else self.index.ntotal
