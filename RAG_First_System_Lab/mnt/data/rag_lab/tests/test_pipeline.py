from rag.chunking import chunk_document
from rag.evaluation import citation_labels
from rag.loaders import clean_text


def test_clean_text():
    assert clean_text("A\r\n\r\n\r\n B") == "A\n\nB"


def test_chunk_overlap():
    doc = {"doc_id": "x", "source": "x.txt", "text": "one two three four five six"}
    chunks = chunk_document(doc, chunk_size=4, overlap=1)
    assert len(chunks) == 2
    assert "four" in chunks[1]["text"]


def test_citation_parser():
    assert citation_labels("Answer [S1] and [S2], not [S1].") == [1, 2]
