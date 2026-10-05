from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:

    def __init__(self, embedding_model_name: str):

        self.embedding_model = EmbeddingModel(
            embedding_model_name
        )

        self.vector_store = VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ):

        query_embedding = self.embedding_model.encode_query(
            query
        )

        results = self.vector_store.search(
            query_embedding,
            top_k
        )

        return results