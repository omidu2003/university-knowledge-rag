import json
from pathlib import Path

from src.config import EMBEDDING_MODEL_NAME
from src.retriever import Retriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    questions = load_questions()

    retriever = Retriever(EMBEDDING_MODEL_NAME)

    total = len(questions)
    correct = 0

    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    for item in questions:
        question = item["question"]
        expected_sources = set(item["expected_sources"])

        results = retriever.retrieve(question, top_k=3)

        retrieved_sources = {
            metadata.get("source")
            for metadata in results["metadatas"][0]
        }

        if not expected_sources:
            passed = True
        else:
            passed = bool(expected_sources.intersection(retrieved_sources))

        if passed:
            correct += 1

        print(f"\nID: {item['id']}")
        print(f"Question: {question}")
        print(f"Expected: {sorted(expected_sources)}")
        print(f"Retrieved: {sorted(retrieved_sources)}")
        print(f"Result: {'PASS' if passed else 'FAIL'}")

    accuracy = correct / total if total else 0

    print("\n" + "=" * 70)
    print(f"RESULT: {correct}/{total} questions passed")
    print(f"Retrieval accuracy: {accuracy:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()