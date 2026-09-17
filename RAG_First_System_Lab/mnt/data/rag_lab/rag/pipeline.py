from __future__ import annotations

from pathlib import Path

from .chunking import chunk_documents
from .embeddings import Embedder
from .generator import Generator
from .loaders import load_documents
from .prompts import build_prompt
from .vector_store import VectorStore


class RAGPipeline:
    def __init__(
        self,
        embedding_model: str = "intfloat/multilingual-e5-small",
        generation_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        chunk_size: int = 180,
        chunk_overlap: int = 40,
    ):
        self.embedding_model_name = embedding_model
        self.generation_model_name = generation_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = Embedder(embedding_model)
        self.generator = Generator(generation_model)
        self.store = VectorStore()
        self.documents: list[dict] = []
        self.chunks: list[dict] = []

    def build_from_directory(self, directory: str | Path) -> dict:
        self.documents = load_documents(directory)
        self.chunks = chunk_documents(
            self.documents,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        if not self.chunks:
            raise ValueError("No readable document content found")
        vectors = self.embedder.embed_passages([c["text"] for c in self.chunks])
        self.store.build(vectors, self.chunks)
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "vector_dimension": int(vectors.shape[1]),
        }

    def answer(self, question: str, top_k: int = 4) -> dict:
        q_vector = self.embedder.embed_query(question)
        retrieved = self.store.search(q_vector, top_k=top_k)
        prompt = build_prompt(question, retrieved)
        answer = self.generator.generate(prompt)
        # Labels refer to retrieved rank, not source IDs stored elsewhere.
        citations = sorted({int(n) for n in __import__("re").findall(r"\[S(\d+)\]", answer)})
        return {
            "question": question,
            "answer": answer,
            "retrieved": retrieved,
            "prompt": prompt,
            "citations": citations,
        }
