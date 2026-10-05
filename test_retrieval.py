from src.config import (
    EMBEDDING_MODEL_NAME
)

from src.retriever import Retriever


def main():

    retriever = Retriever(
        EMBEDDING_MODEL_NAME
    )

    question = (
        "What evidence should students collect "
        "for their portfolio?"
    )

    results = retriever.retrieve(
        question,
        top_k=3
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    print("\nQUERY:")
    print(question)

    print("\nRETRIEVED RESULTS:")

    for index, (
        document,
        metadata,
        distance
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances
        ),
        start=1
    ):

        print(
            f"\n--- Result {index} ---"
        )

        print(
            f"Source: {metadata['source']}"
        )

        print(
            f"Chunk: {metadata['chunk_id']}"
        )

        print(
            f"Distance: {distance:.4f}"
        )

        print(
            f"Text: {document}"
        )


if __name__ == "__main__":
    main()