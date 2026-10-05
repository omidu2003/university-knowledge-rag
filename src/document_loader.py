"""
Document loading for the University Knowledge RAG Assistant.

Responsibilities (Stage 2):
- find supported files (.txt, .pdf) in a directory
- extract text (page by page for PDFs)
- apply light text cleaning
- attach metadata used later for source attribution
- validate input and skip unreadable files without crashing
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def clean_text(text: str) -> str:
    """Light cleaning: normalise line endings and whitespace.

    Deliberately conservative - aggressive cleaning can destroy meaning.
    """
    text = text.replace("\x00", "")           # null characters
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)       # repeated spaces/tabs
    text = re.sub(r" *\n *", "\n", text)      # spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)    # many blank lines -> one
    return text.strip()


def _make_record(
    text: str, file_path: Path, file_type: str, page: Optional[int]
) -> Dict:
    return {
        "text": text,
        "source": file_path.name,   # kept for compatibility with later stages
        "file_type": file_type,
        "page": page,               # None for TXT, 1-based for PDF
    }


def load_text_file(file_path: Path) -> List[Dict]:
    """Load a TXT file as one record. Falls back to cp1252 if not UTF-8."""
    try:
        raw = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("%s is not UTF-8; retrying as cp1252.", file_path.name)
        raw = file_path.read_text(encoding="cp1252", errors="replace")

    text = clean_text(raw)
    if not text:
        logger.warning("Skipping %s: file is empty.", file_path.name)
        return []
    return [_make_record(text, file_path, "txt", None)]


def load_pdf_file(file_path: Path) -> List[Dict]:
    """Load a PDF as one record per page that contains extractable text."""
    try:
        reader = PdfReader(str(file_path))
        if reader.is_encrypted:
            # Many PDFs are 'encrypted' with an empty password.
            if reader.decrypt("") == 0:
                logger.warning("Skipping %s: password protected.", file_path.name)
                return []
    except Exception as error:
        logger.warning("Skipping %s: cannot open PDF (%s).", file_path.name, error)
        return []

    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception as error:
            logger.warning(
                "%s page %d: extraction failed (%s).",
                file_path.name, page_number, error,
            )
            continue
        if text:
            records.append(_make_record(text, file_path, "pdf", page_number))

    if not records:
        logger.warning(
            "Skipping %s: no extractable text (scanned PDF? OCR needed).",
            file_path.name,
        )
    return records


def load_documents(data_directory: Path) -> List[Dict]:
    """Load all supported documents from a directory.

    Returns a list of records with keys:
    text, source, file_type, page.
    """
    data_directory = Path(data_directory)

    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory not found: {data_directory}")
    if not data_directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_directory}")

    documents: List[Dict] = []

    for file_path in sorted(data_directory.iterdir()):   # sorted = reproducible
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            if not file_path.name.startswith("."):       # ignore .gitkeep etc.
                logger.info("Skipping unsupported file: %s", file_path.name)
            continue

        if suffix == ".txt":
            documents.extend(load_text_file(file_path))
        else:
            documents.extend(load_pdf_file(file_path))

    if not documents:
        logger.warning("No documents were loaded from %s.", data_directory)

    return documents