"""
Utilities for text chunking based on the requirements.
Implements chunking strategies to split book content appropriately.
"""
from pydantic import BaseModel
from typing import List, Optional
import re
from datetime import datetime


class Chunk(BaseModel):
    """
    Represents a text chunk with its properties.
    """
    text: str
    source_url: str
    chapter: str
    section: Optional[str] = None
    position: int
    checksum: str

    def __init__(self, text: str, source_url: str, chapter: str, position: int, **kwargs):
        import hashlib
        super().__init__(
            text=text,
            source_url=source_url,
            chapter=chapter,
            position=position,
            checksum=hashlib.md5((text + source_url + str(position)).encode()).hexdigest(),
            **kwargs
        )


class UserQuery(BaseModel):
    """
    Represents a user query with optional selected text context.
    """
    id: str = ""
    query_text: str
    selected_text: Optional[str] = None
    created_at: datetime = datetime.now()
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def __init__(self, query_text: str, **kwargs):
        import hashlib
        # Generate ID based on query text and timestamp if not provided
        if not kwargs.get('id'):
            id_source = query_text + str(datetime.now().timestamp())
            kwargs['id'] = hashlib.md5(id_source.encode()).hexdigest()[:16]

        super().__init__(query_text=query_text, **kwargs)

    def is_selected_text_query(self) -> bool:
        """
        Check if this is a selected text query.
        """
        return self.selected_text is not None and len(self.selected_text.strip()) > 0


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[Chunk]:
    """
    Split text into chunks of approximately max_tokens with overlap.
    This is a basic implementation - in a real system, you might want to use
    more sophisticated text splitting based on semantic boundaries.
    """
    # Basic implementation: split by sentences while respecting token limits
    # For simplicity, we'll use character count as a proxy for token count
    # In a real implementation, you would use a proper tokenizer

    # Split text into sentences
    sentences = re.split(r'[.!?]+\s+', text)

    chunks = []
    current_chunk = ""
    current_position = 0

    for sentence in sentences:
        # Check if adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) > max_tokens and current_chunk:
            # Add the current chunk to the list
            chunk = Chunk(
                text=current_chunk.strip(),
                source_url="",
                chapter="",
                position=current_position
            )
            chunks.append(chunk)
            current_position += 1

            # Start a new chunk with some overlap if possible
            if overlap > 0:
                # Find the last 'overlap' characters from the current chunk
                # and start the next chunk with the current sentence
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence

    # Add the last chunk if it has content
    if current_chunk.strip():
        chunk = Chunk(
            text=current_chunk.strip(),
            source_url="",
            chapter="",
            position=current_position
        )
        chunks.append(chunk)

    return chunks