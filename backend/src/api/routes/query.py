"""
Query API route for retrieving relevant book content chunks.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os

from ...services.embedding_service import EmbeddingService
from ...services.retrieval_service import RetrievalService


router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return")
    selected_text: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    retrieval_time_ms: float


@router.post("/query", response_model=QueryResponse)
async def query_chunks(request: QueryRequest) -> QueryResponse:
    """
    Retrieve relevant book content chunks based on user query.
    Includes performance monitoring for response times.
    """
    import time
    start_time = time.time()

    try:
        # Validate user query
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > 1000:
            raise HTTPException(status_code=400, detail="Query must be less than 1000 characters")

        if request.selected_text and len(request.selected_text) > 5000:
            raise HTTPException(status_code=400, detail="Selected text must be less than 5000 characters")

        # Initialize services
        embedding_service = EmbeddingService()
        retrieval_service = RetrievalService()

        # Generate embedding for the query
        query_embedding = embedding_service.generate_embeddings([request.query])[0]

        # Search for relevant chunks
        results = retrieval_service.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            selected_text=request.selected_text
        )

        retrieval_time_ms = (time.time() - start_time) * 1000

        # Log performance metrics if retrieval time exceeds threshold (SC-008: <300ms Qdrant round-trip)
        if retrieval_time_ms > 300:
            print(f"WARNING: Retrieval time exceeded 300ms: {retrieval_time_ms:.2f}ms")

        return QueryResponse(
            query=request.query,
            results=results,
            retrieval_time_ms=retrieval_time_ms
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))