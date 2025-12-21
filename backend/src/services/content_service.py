"""
Service for processing book content, including scraping, chunking, and ingestion.
"""
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import os
from ..models.chunk import Chunk, chunk_text
from ..models.embedding import BookContentChunk
from urllib.parse import urljoin, urlparse


class ContentService:
    """
    Service class for handling book content processing.
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def scrape_book_content(self, url: str) -> str:
        """
        Scrape text content from a book URL.
        This is a basic implementation - in a real system you might need to handle
        authentication, different content types, etc.
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            return text
        except Exception as e:
            raise Exception(f"Failed to scrape content from {url}: {str(e)}")

    def normalize_content(self, text: str, max_tokens: int = 500) -> List[Chunk]:
        """
        Normalize book content into appropriate text chunks (200-800 tokens).
        Using the chunking strategy decided in research (300-500 token chunks).
        """
        # Basic implementation - in a real system, you'd use proper tokenization
        # For now, using character count as a proxy
        chunks = chunk_text(text, max_tokens=max_tokens)

        # Set appropriate source information (would come from the actual source)
        for i, chunk in enumerate(chunks):
            if not chunk.source_url:
                chunk.source_url = "unknown"
            if not chunk.chapter:
                chunk.chapter = f"Chapter_{i//10 + 1}"  # Basic chapter assignment

        return chunks

    def process_book_content(self, url: str) -> List[Chunk]:
        """
        Complete pipeline: scrape, normalize, and chunk book content.
        """
        # Scrape content from the URL
        raw_content = self.scrape_book_content(url)

        # Normalize and chunk the content
        chunks = self.normalize_content(raw_content)

        return chunks

    def create_book_content_chunks(self, chunks: List[Chunk]) -> List[BookContentChunk]:
        """
        Convert processed chunks into BookContentChunk models with embeddings.
        """
        book_chunks = []
        for chunk in chunks:
            book_chunk = BookContentChunk(
                id=chunk.id if hasattr(chunk, 'id') else str(hash(chunk.checksum)),
                text=chunk.text,
                metadata={
                    "source_url": chunk.source_url,
                    "chapter": chunk.chapter,
                    "section": chunk.section,
                    "position": chunk.position,
                    "created_at": "2025-12-10T00:00:00Z"  # In real implementation, use actual timestamp
                },
                checksum=chunk.checksum
            )
            book_chunks.append(book_chunk)

        return book_chunks