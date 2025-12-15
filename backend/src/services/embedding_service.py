"""
Service for generating embeddings using Cohere API.
"""
import cohere
from typing import List, Dict, Any
import os
from ..models.chunk import Chunk


class EmbeddingService:
    """
    Service class for handling embedding generation with Cohere.
    """
    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")
        self.client = cohere.Client(api_key)
        # Using the Cohere multilingual model as decided in research
        self.model = "embed-multilingual-v3.0"

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        """
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document"  # Optimize for search use case
        )
        return response.embeddings

    def embed_chunk(self, chunk: Chunk) -> List[float]:
        """
        Generate embedding for a single chunk.
        """
        embeddings = self.generate_embeddings([chunk.text])
        return embeddings[0] if embeddings else []

    def embed_chunks(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of chunks.
        Returns a list of dictionaries with chunk and embedding data.
        """
        if not chunks:
            return []

        # Extract texts for embedding
        texts = [chunk.text for chunk in chunks]
        embeddings = self.generate_embeddings(texts)

        # Pair each chunk with its embedding
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                'chunk': chunk,
                'embedding': embeddings[i] if i < len(embeddings) else []
            })

        return result