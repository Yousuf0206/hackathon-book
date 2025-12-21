"""
Embed API route for content ingestion.
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
import time

from ...services.content_service import ContentService
from ...services.embedding_service import EmbeddingService
from ...services.retrieval_service import RetrievalService


router = APIRouter()


class EmbedChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]


class EmbedRequest(BaseModel):
    chunks: List[EmbedChunk]


class EmbedResponse(BaseModel):
    processed_chunks: int
    failed_chunks: int
    processing_time_ms: float


@router.post("/embed", response_model=EmbedResponse)
async def embed_content(request: EmbedRequest) -> EmbedResponse:
    """
    Generate embeddings for book content and store in vector database.
    Implements idempotent ingestion to prevent duplication.
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.chunks:
            raise HTTPException(status_code=400, detail="No chunks provided for embedding")

        if len(request.chunks) > 100:  # Prevent massive ingestion requests
            raise HTTPException(status_code=400, detail="Too many chunks provided (max 100)")

        # Initialize services
        content_service = ContentService()
        embedding_service = EmbeddingService()
        retrieval_service = RetrievalService()

        processed_count = 0
        failed_count = 0

        # Process each chunk
        chunk_data = []
        for i, chunk_data_item in enumerate(request.chunks):
            try:
                # Validate chunk data
                if not chunk_data_item.text or not chunk_data_item.text.strip():
                    failed_count += 1
                    print(f"Skipping empty chunk at index {i}")
                    continue

                # Create a Chunk object from the request data
                from ...models.chunk import Chunk
                chunk = Chunk(
                    text=chunk_data_item.text,
                    source_url=chunk_data_item.metadata.get("source_url", ""),
                    chapter=chunk_data_item.metadata.get("chapter", ""),
                    position=chunk_data_item.metadata.get("position", 0),
                    section=chunk_data_item.metadata.get("section")
                )

                # Generate embedding for the chunk
                embedding = embedding_service.embed_chunk(chunk)

                chunk_data.append({
                    'chunk': chunk,
                    'embedding': embedding
                })
                processed_count += 1
            except Exception as e:
                failed_count += 1
                # Log the error but continue processing other chunks
                print(f"Failed to process chunk at index {i}: {str(e)}")

        # Store embeddings in Qdrant with idempotent behavior
        if chunk_data:
            try:
                retrieval_service.store_embeddings(chunk_data)
            except Exception as e:
                print(f"Failed to store embeddings in Qdrant: {str(e)}")
                # Don't fail the entire request if storage fails, just report it
                pass

        processing_time_ms = (time.time() - start_time) * 1000

        return EmbedResponse(
            processed_chunks=processed_count,
            failed_chunks=failed_count,
            processing_time_ms=processing_time_ms
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the error for debugging
        import traceback
        print(f"Unexpected error in embed_content: {str(e)}")
        print(traceback.format_exc())

        # Return user-friendly error message
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request. Please try again later.")