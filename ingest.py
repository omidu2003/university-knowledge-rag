from src.config import (
    RAW_DATA_DIR,
    EMBEDDING_MODEL_NAME
)

from src.document_loader import load_documents
from src.chunker import create_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


def main():

    print("Loading documents...")

    documents = load_documents(
        RAW_DATA_DIR
    )

    print(f"Loaded {len(documents)} documents.")

    print("Creating chunks...")

    chunks = create_chunks(
        documents
    )

    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")

    embedding_model = EmbeddingModel(
        EMBEDDING_MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Creating embeddings...")

    embeddings = embedding_model.encode_documents(
        texts
    )

    print("Storing documents in Chroma...")

    vector_store = VectorStore()

    vector_store.add_documents(
        chunks,
        embeddings
    )

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()