from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    FieldCondition,
    Filter,
    MatchAny,
)

from backend.app.core.config import settings
from qdrant_client.models import PointIdsList

class VectorStore:
    def __init__(self) -> None:
        self.client = QdrantClient(
            url=settings.qdrant_url,
        )

        self.collection_name = settings.qdrant_collection

    def create_collection(self) -> None:
        collections = self.client.get_collections()

        existing_names = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=1024,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )

    def add_chunks(
        self,
        chunks: list[dict],
        dense_vectors: list[list[float]],
        sparse_vectors: list[tuple[list[int], list[float]]],
    ) -> None:
        points = []

        for chunk, dense_vector, sparse_vector in zip(
            chunks,
            dense_vectors,
            sparse_vectors,
            strict=True,
        ):
            sparse_indices, sparse_values = sparse_vector

            points.append(
                PointStruct(
                    id=chunk["point_id"],
                    vector={
                        "dense": dense_vector,
                        "sparse": SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    },
                    payload={
                        "document_id": chunk["document_id"],
                        "chunk_id": chunk["chunk_id"],
                        "filename": chunk["filename"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "page_number": chunk["page_number"],
                        "section": chunk["section"],
                    },
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        document_ids: list[str] | None = None,
    ) -> list:
        query_filter = None

        if document_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(
                            any=document_ids,
                        ),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        return results.points

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        limit: int = 5,
        document_ids: list[str] | None = None,
    ) -> list:
        query_filter = None

        if document_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(
                            any=document_ids,
                        ),
                    )
                ]
            )
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=20,
                    filter=query_filter,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse",
                    limit=20,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(
                fusion=Fusion.RRF,
            ),
            limit=limit,
            with_payload=True,
        )

        return results.points


    def delete_by_document_id(self, document_id: str) -> None:
        points = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match={"value": document_id},
                    )
                ]
            ),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )[0]

        point_ids = [point.id for point in points]

        if not point_ids:
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=point_ids,
            ),
        )


vector_store = VectorStore()