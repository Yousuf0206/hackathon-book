"""
Model for representing book content chunks with embeddings.
Based on the BookContentChunk entity from the data model.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4


class BookContentChunk(BaseModel):
    """
    Represents a chunk of book content with its embedding and metadata.
    """
    id: str
    text: str
    embedding: Optional[list] = None  # Vector embedding representation
    metadata: Dict[str, Any]  # Contains source_url, chapter, section, position, created_at
    checksum: str  # For idempotent ingestion

    class Config:
        # Allow extra fields for flexibility with metadata
        extra = "allow"