# Lab — Build Your First RAG System

A from-scratch educational RAG lab that exposes the full pipeline:

`Documents → Parse/Clean → Chunk → Embeddings → FAISS Index → Retrieval → Prompt → LLM → Grounded Answer + Citations → Evaluation`

## What learners build

A document-grounded assistant that:
- answers from a small knowledge base;
- retrieves the most relevant chunks;
- shows exactly which chunks were used;
- cites sources as `[S1]`, `[S2]`, …;
- refuses to invent an answer when the knowledge base does not support it;
- exposes retrieval scores and evaluation results.

## Default models

- Embeddings: `intfloat/multilingual-e5-small`
- Generation: `Qwen/Qwen2.5-0.5B-Instruct`

Both are pulled from Hugging Face on first use. The generator is intentionally small for a teaching lab; a larger instruct model can be configured later.

## Run

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The first launch downloads the models and may take a few minutes.

## Files

```text
rag_lab/
├── app/streamlit_app.py       # Streamlit UI
├── rag/
│   ├── pipeline.py             # End-to-end RAG pipeline
│   ├── loaders.py              # TXT / MD / PDF / DOCX parsing
│   ├── chunking.py             # Recursive word-window chunking
│   ├── embeddings.py            # SentenceTransformer wrapper
│   ├── vector_store.py          # FAISS index + metadata
│   ├── generator.py             # Hugging Face LLM wrapper
│   └── prompts.py               # Grounding prompt
├── documents/                  # Starter knowledge base
├── data/evaluation.json        # Evaluation set
├── tests/test_pipeline.py      # Lightweight unit tests
├── requirements.txt
└── README.md
```

## Lab stages

### 1. Prepare data
Load and parse documents, then normalize whitespace and preserve source metadata.

### 2. Chunking
Split documents into overlapping chunks. In this lab, chunk size and overlap are explicit controls so learners can experiment.

### 3. Embeddings
Convert each chunk into a dense vector using a multilingual sentence embedding model.

### 4. Index
Store normalized vectors in a FAISS inner-product index. With normalized vectors, inner product corresponds to cosine similarity.

### 5. Retrieval
Embed the question, search the index, inspect Top-K chunks and scores, and decide whether the evidence is sufficient.

### 6. Generation
Send only the retrieved evidence to the LLM. The prompt requires source labels and an explicit “not enough information” response when the answer is unsupported.

### 7. Evaluation
The included test set contains:
- Easy
- Normal
- Ambiguous
- Multi-part
- Missing information
- Outdated information
- Irrelevant

The evaluator checks retrieval recall, whether the answer cites retrieved sources, citation precision/recall against the expected sources, and abstention behavior.

## Teaching challenge

Run the same evaluation after changing:
1. chunk size / overlap;
2. Top-K;
3. embedding model;
4. generation model;
5. grounding prompt.

Ask learners: **Which change improved retrieval, which improved answer quality, and which only changed wording?**

## Sources

Official documentation used for the implementation pattern:
- Hugging Face Transformers pipeline: https://huggingface.co/docs/transformers/main/pipeline_tutorial
- Sentence Transformers semantic search: https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html
- FAISS getting started: https://github.com/facebookresearch/faiss/wiki/Getting-started
- Streamlit file uploader: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
