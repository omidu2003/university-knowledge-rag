from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, documents):
        return self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def encode_query(self, query: str):
        return self.model.encode(
            query,
            normalize_embeddings=True
        )