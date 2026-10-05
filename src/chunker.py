from typing import List, Dict


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[str]:
    """
    Split text into overlapping character-based chunks.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap

    return chunks


def create_chunks(documents: List[Dict]) -> List[Dict]:
    """
    Convert documents into chunks while preserving source information.
    """

    chunked_documents = []

    for document in documents:

        chunks = chunk_text(
            document["text"]
        )

        for index, chunk in enumerate(chunks):

            chunked_documents.append(
                {
                    "text": chunk,
                    "source": document["source"],
                    "chunk_id": index
                }
            )

    return chunked_documents