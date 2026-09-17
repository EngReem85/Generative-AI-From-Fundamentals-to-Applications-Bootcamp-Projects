from __future__ import annotations

import re


def chunk_document(document: dict, chunk_size: int = 180, overlap: int = 40) -> list[dict]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = re.findall(r"\S+", document["text"])
    chunks = []
    start = 0
    chunk_no = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        text = " ".join(words[start:end]).strip()
        if text:
            chunks.append({
                "chunk_id": f"{document['doc_id']}::chunk-{chunk_no}",
                "doc_id": document["doc_id"],
                "source": document["source"],
                "chunk_index": chunk_no,
                "text": text,
            })
        start += step
        chunk_no += 1

    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 180, overlap: int = 40) -> list[dict]:
    chunks: list[dict] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, chunk_size, overlap))
    return chunks
