from fastembed import SparseTextEmbedding


SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"


class SparseEmbeddingService:
    def __init__(self) -> None:
        self.model = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME,
        )

    def embed_text(self, text: str) -> tuple[list[int], list[float]]:
        embedding = next(
            self.model.embed([text])
        )

        return (
            embedding.indices.tolist(),
            embedding.values.tolist(),
        )


sparse_embedding_service = SparseEmbeddingService()