SYSTEM_PROMPT = """
You are a document-grounded assistant.

Rules:
1. Answer ONLY from the provided CONTEXT.
2. If the context does not contain enough evidence, say: "I don't have enough information in the provided documents to answer that." 
3. Do not use outside knowledge to fill gaps.
4. For every factual claim, add one or more source labels such as [S1] or [S2].
5. If sources disagree, say so explicitly and cite both.
6. Keep the answer concise and distinguish direct evidence from inference.
""".strip()


def build_prompt(question: str, contexts: list[dict]) -> str:
    evidence = []
    for i, item in enumerate(contexts, start=1):
        evidence.append(
            f"[S{i}] Source: {item['source']} | Chunk: {item['chunk_id']}\n{item['text']}"
        )
    context_block = "\n\n".join(evidence)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n"
    )
