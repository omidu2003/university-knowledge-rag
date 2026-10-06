import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class Generator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY was not found. "
                "Add it to your .env file."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        self.model = model

    def generate(self, question: str, retrieved_results: dict) -> str:
        documents = retrieved_results["documents"][0]
        metadatas = retrieved_results["metadatas"][0]

        context_parts = []

        for index, (document, metadata) in enumerate(
            zip(documents, metadatas),
            start=1
        ):
            source = metadata.get("source", "Unknown source")
            chunk_id = metadata.get("chunk_id", "Unknown")

            context_parts.append(
                f"[Source {index}: {source}, chunk {chunk_id}]\n"
                f"{document}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a university knowledge assistant.

Your task is to answer the user's question using ONLY the retrieved
context provided below.

Grounding requirements:

1. Use only information explicitly supported by the retrieved context.
2. Do not use outside knowledge.
3. Do not invent facts, dates, statistics, policies, or requirements.
4. If the retrieved context does not provide enough information to answer
   the question, clearly state that the information is not available in
   the retrieved documents.
5. Keep the answer concise and directly relevant to the question.
6. Do not claim that information comes from a source unless that source
   appears in the retrieved context.

Retrieved context:

{context}

User question:

{question}
"""

        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        return response.output_text