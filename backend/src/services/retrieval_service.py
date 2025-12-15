"""
Service for retrieving relevant content from Qdrant vector database.
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import os
from ..models.embedding import BookContentChunk
import math


class RetrievalService:
    """
    Service class for handling retrieval from Qdrant vector database.
    """
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise ValueError("QDRANT_URL environment variable is required")

        # Initialize Qdrant client
        if qdrant_api_key:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            # For local development
            self.client = QdrantClient(host="localhost", port=6333)

        self.collection_name = "book_embeddings"
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Ensure the book embeddings collection exists in Qdrant.
        """
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
        except:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
                # Adjust size based on the embedding model used (Cohere multilingual-v3.0 uses 1024 dimensions)
            )

    def store_embeddings(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Store chunks with their embeddings in Qdrant.
        Implements idempotent ingestion to prevent duplication based on checksum.
        Each chunk should have 'chunk' (BookContentChunk object) and 'embedding' (vector) properties.
        """
        points = []
        for i, item in enumerate(chunks):
            chunk = item['chunk']
            embedding = item['embedding']

            # Check if a chunk with the same checksum already exists to ensure idempotent ingestion
            existing_points = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="checksum",
                            match=models.MatchValue(value=chunk.checksum)
                        )
                    ]
                ),
                limit=1
            )

            # Only add the chunk if it doesn't already exist (idempotent behavior)
            if not existing_points[0]:  # If no existing point with this checksum
                # Create a Qdrant point with UUID based on checksum for consistency
                import uuid
                # Use the checksum to create a consistent UUID
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.checksum))

                point = models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "source_url": chunk.metadata.get("source_url", ""),
                        "chapter": chunk.metadata.get("chapter", ""),
                        "section": chunk.metadata.get("section", ""),
                        "position": chunk.metadata.get("position", 0),
                        "checksum": chunk.checksum
                    }
                )
                points.append(point)

        # Upload points to Qdrant if there are any new ones
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        return True

    def search(self, query_embedding: List[float], top_k: int = 5, selected_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks based on the query embedding.
        If selected_text is provided, focus the search on that context.
        """
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )

            results = []
            for hit in search_result:
                results.append({
                    "chunk_id": str(hit.id),
                    "text": hit.payload.get("text", ""),
                    "source_url": hit.payload.get("source_url", ""),
                    "chapter": hit.payload.get("chapter", ""),
                    "section": hit.payload.get("section", ""),
                    "relevance_score": hit.score
                })

            return results
        except Exception as e:
            # Log the error
            print(f"Qdrant search error: {str(e)}")
            # Return empty results gracefully instead of failing completely
            return []

    def search_by_text(self, query_text: str, embedding_service, top_k: int = 5, selected_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks by first embedding the query text.
        If selected_text is provided, perform focused retrieval on that specific content.
        """
        try:
            if selected_text:
                # For selected text queries, we focus on the provided text context
                # Generate embedding for the selected text to find semantically similar content
                selected_text_embedding = embedding_service.generate_embeddings([selected_text])[0]

                # Search in Qdrant for content most similar to the selected text
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=selected_text_embedding,
                    limit=20,  # Get more results to have a broader context of the selected text area
                )

                # Filter and rank results to ensure they're highly relevant to the selected text
                # First, collect all content that's related to the selected text
                selected_text_related_results = []
                for hit in search_result:
                    result_data = {
                        "chunk_id": str(hit.id),
                        "text": hit.payload.get("text", ""),
                        "source_url": hit.payload.get("source_url", ""),
                        "chapter": hit.payload.get("chapter", ""),
                        "section": hit.payload.get("section", ""),
                        "relevance_score": hit.score
                    }
                    # Add results that are part of the selected text's context
                    selected_text_related_results.append(result_data)

                # Now, for the actual query, we want to find the most relevant information
                # within the context of the selected text area
                query_embedding = embedding_service.generate_embeddings([query_text])[0]

                # Perform a focused search but only among the content related to selected text
                # We'll use the top results from the selected text context and rerank based on query relevance
                if selected_text_related_results:
                    # Rerank the selected text related results based on query relevance
                    reranked_results = []
                    for result in selected_text_related_results:
                        # Create embedding for the text to compare with query
                        text_embedding = embedding_service.generate_embeddings([result["text"]])[0]
                        # Calculate similarity between query and text using cosine similarity
                        similarity = self._cosine_similarity(query_embedding, text_embedding)
                        result["query_relevance_score"] = similarity
                        reranked_results.append(result)

                    # Sort by query relevance score
                    reranked_results.sort(key=lambda x: x["query_relevance_score"], reverse=True)

                    # Return top_k results that are most relevant to the query within the selected text context
                    return reranked_results[:top_k]
                else:
                    # If no related content found, return empty results to ensure grounding in selected text
                    return []
            else:
                # Standard search behavior
                query_embedding = embedding_service.generate_embeddings([query_text])[0]
                return self.search(query_embedding, top_k)
        except Exception as e:
            # Log the error
            print(f"Qdrant search_by_text error: {str(e)}")
            # Return empty results gracefully instead of failing completely
            return []

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

        # Calculate magnitudes
        magnitude_a = math.sqrt(sum(a * a for a in vec_a))
        magnitude_b = math.sqrt(sum(b * b for b in vec_b))

        # Calculate cosine similarity
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)