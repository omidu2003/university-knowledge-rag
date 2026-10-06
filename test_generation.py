from src.config import EMBEDDING_MODEL_NAME
from src.retriever import Retriever
from src.generator import Generator


def main():
    question = "What evidence should students collect for their portfolio?"

    print("Initialising retriever...")
    retriever = Retriever(EMBEDDING_MODEL_NAME)

    print("Retrieving relevant documents...")
    results = retriever.retrieve(question, top_k=3)

    print("Initialising generator...")
    generator = Generator()

    print("Generating answer...")
    answer = generator.generate(question, results)

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    seen_sources = set()

    for metadata in results["metadatas"][0]:
        source = metadata.get("source", "Unknown source")
        chunk_id = metadata.get("chunk_id", "Unknown")

        if source not in seen_sources:
            print(f"- {source}")
            seen_sources.add(source)

        print(f"  Chunk: {chunk_id}")


if __name__ == "__main__":
    main()