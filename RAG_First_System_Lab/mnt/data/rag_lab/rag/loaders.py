from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_txt_or_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def load_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def load_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix in {".txt", ".md"}:
        raw = load_txt_or_md(path)
    elif suffix == ".pdf":
        raw = load_pdf(path)
    else:
        raw = load_docx(path)
    return clean_text(raw)


def load_documents(directory: str | Path) -> list[dict]:
    directory = Path(directory)
    documents: list[dict] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = load_path(path)
            if text:
                documents.append({
                    "doc_id": path.stem,
                    "source": path.name,
                    "path": str(path),
                    "text": text,
                })
    return documents
