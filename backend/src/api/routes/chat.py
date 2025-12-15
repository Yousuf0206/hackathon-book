"""
Chat API route for question answering with source citations.
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
import time

from ...services.chat_service import ChatService


router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None
    session_id: Optional[str] = None


class Citation(BaseModel):
    chunk_id: str
    source_url: str
    chapter: str
    section: Optional[str] = None
    relevance_score: float
    text_snippet: str


class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]
    session_id: str
    response_time_ms: float


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Process user queries and return AI-generated responses with source citations.
    Includes performance monitoring for response times.
    """
    start_time = time.time()

    try:
        # Validate user query
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > 1000:
            raise HTTPException(status_code=400, detail="Query must be less than 1000 characters")

        if request.selected_text and len(request.selected_text) > 5000:
            raise HTTPException(status_code=400, detail="Selected text must be less than 5000 characters")

        # Initialize chat service
        chat_service = ChatService()

        # Process the query
        result = chat_service.process_query(
            query=request.query,
            selected_text=request.selected_text,
            session_id=request.session_id
        )

        response_time_ms = (time.time() - start_time) * 1000

        # Log performance metrics if response time exceeds threshold (SC-002: <800ms)
        if response_time_ms > 800:
            print(f"WARNING: Response time exceeded 800ms: {response_time_ms:.2f}ms")

        return ChatResponse(
            response=result["response"],
            citations=result["citations"],
            session_id=result["session_id"],
            response_time_ms=response_time_ms
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))