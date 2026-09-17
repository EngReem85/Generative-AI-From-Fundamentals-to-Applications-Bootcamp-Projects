from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from rag.evaluation import evaluate_case, load_eval_set, summarize
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="RAG Lab", page_icon="🔎", layout="wide")

st.title("🔎 Lab — Build Your First RAG System")
st.caption("Documents → Knowledge Base → Retrieval → Relevant Context → LLM → Grounded Answer → Citations")

DEFAULT_DOCS = Path(__file__).resolve().parents[1] / "documents"

with st.sidebar:
    st.header("1. Configure")
    embedding_model = st.text_input(
        "Embedding model",
        "intfloat/multilingual-e5-small",
        help="Must produce sentence/document embeddings.",
    )
    generation_model = st.text_input(
        "Generation model",
        "Qwen/Qwen2.5-0.5B-Instruct",
        help="A small instruct model is used by default for an inexpensive lab.",
    )
    chunk_size = st.slider("Chunk size (words)", 80, 350, 180, 10)
    chunk_overlap = st.slider("Chunk overlap (words)", 0, 100, 40, 10)
    top_k = st.slider("Top-K retrieval", 1, 8, 4)

    uploads = st.file_uploader(
        "Optional: upload your own documents",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )

    build = st.button("Build / rebuild knowledge base", type="primary", use_container_width=True)


def save_uploaded_files(uploaded_files):
    if not uploaded_files:
        return DEFAULT_DOCS
    folder = Path(tempfile.mkdtemp(prefix="rag_docs_"))
    for uploaded in uploaded_files:
        safe_name = Path(uploaded.name).name
        (folder / safe_name).write_bytes(uploaded.getvalue())
    return folder


if "rag" not in st.session_state:
    st.session_state.rag = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

if build:
    try:
        docs_dir = save_uploaded_files(uploads)
        with st.spinner("Building knowledge base…"):
            rag = RAGPipeline(
                embedding_model=embedding_model,
                generation_model=generation_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            stats = rag.build_from_directory(docs_dir)
        st.session_state.rag = rag
        st.session_state.stats = stats
        st.success("Knowledge base built successfully.")
    except Exception as exc:
        st.error(f"Could not build the knowledge base: {exc}")

if st.session_state.stats:
    a, b, c = st.columns(3)
    a.metric("Documents", st.session_state.stats["documents"])
    b.metric("Chunks", st.session_state.stats["chunks"])
    c.metric("Vector dimension", st.session_state.stats["vector_dimension"])

st.divider()

ask_tab, eval_tab, inspect_tab = st.tabs(["💬 Ask", "🧪 Evaluation", "🔬 Inspect pipeline"])

with ask_tab:
    st.subheader("Ask the knowledge base")
    question = st.text_input("Question", placeholder="e.g. What is the current GPU allowance?")
    if st.button("Ask", use_container_width=True):
        if st.session_state.rag is None:
            st.warning("Build the knowledge base first.")
        elif not question.strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Retrieving evidence and generating a grounded answer…"):
                st.session_state.last_result = st.session_state.rag.answer(question, top_k=top_k)

    result = st.session_state.last_result
    if result:
        st.markdown("### Grounded answer")
        st.write(result["answer"])

        st.markdown("### Retrieved context")
        for i, item in enumerate(result["retrieved"], start=1):
            with st.expander(f"[S{i}] {item['source']} — score {item['score']:.4f}"):
                st.write(item["text"])
                st.caption(f"Chunk: {item['chunk_id']}")

with eval_tab:
    st.subheader("Evaluate retrieval + grounding")
    st.write("The test set intentionally includes seven question types so failure modes become visible.")
    eval_path = Path(__file__).resolve().parents[1] / "data" / "evaluation.json"
    cases = load_eval_set(eval_path)
    df = pd.DataFrame(cases)
    st.dataframe(df[["id", "type", "question"]], use_container_width=True, hide_index=True)

    run_eval = st.button("Run full evaluation", use_container_width=True)
    if run_eval:
        if st.session_state.rag is None:
            st.warning("Build the knowledge base first.")
        else:
            rows = []
            progress = st.progress(0)
            for idx, case in enumerate(cases, start=1):
                result = st.session_state.rag.answer(case["question"], top_k=top_k)
                rows.append(evaluate_case(case, result))
                progress.progress(idx / len(cases))
            summary = summarize(rows)
            st.session_state.eval_rows = rows
            st.session_state.eval_summary = summary

    if "eval_summary" in st.session_state:
        s = st.session_state.eval_summary
        cols = st.columns(4)
        cols[0].metric("Retrieval hit rate", f"{s['retrieval_hit_rate']:.0%}")
        cols[1].metric("Citation precision", f"{s['citation_precision']:.0%}")
        cols[2].metric("Citation recall", f"{s['citation_recall']:.0%}")
        cols[3].metric("Abstention accuracy", f"{s['abstention_accuracy']:.0%}")
        st.dataframe(pd.DataFrame(st.session_state.eval_rows), use_container_width=True, hide_index=True)

with inspect_tab:
    st.subheader("See the RAG pipeline")
    st.code(
        """DOCUMENTS\n   ↓\nLOAD → PARSE → CLEAN\n   ↓\nCHUNKS\n   ↓\nEMBEDDING MODEL\n   ↓\nVECTORS\n   ↓\nFAISS INDEX\n   ↓\nQUESTION → QUERY EMBEDDING → TOP-K\n   ↓\nRETRIEVED CONTEXT\n   ↓\nGROUNDED PROMPT\n   ↓\nLLM\n   ↓\nANSWER + [S1] [S2] CITATIONS""",
        language="text",
    )
    st.markdown("#### What to inspect")
    st.write(
        "1. Changing chunk size changes what the retriever can match. "
        "2. Changing Top-K changes the evidence available to the LLM. "
        "3. The LLM does not search the documents directly: retrieval happens first. "
        "4. Citations point back to retrieved chunks, not to model memory."
    )

st.divider()
st.caption("Educational lab. The evaluation metrics are retrieval/grounding proxies; human review is still needed for nuanced answer correctness.")
