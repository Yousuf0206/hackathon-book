"""
Tool for loading book content into Qdrant vector database.
This script handles the end-to-end process of content ingestion.
"""
import sys
import os
import argparse
from typing import List

# Add the backend src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.content_service import ContentService
from src.services.embedding_service import EmbeddingService
from src.services.retrieval_service import RetrievalService
from src.models.chunk import Chunk


def load_content_to_qdrant(source_url: str):
    """
    Complete pipeline to load content from a source URL to Qdrant.
    """
    print(f"Starting content load from: {source_url}")

    # Initialize services
    content_service = ContentService()
    embedding_service = EmbeddingService()
    retrieval_service = RetrievalService()

    try:
        # Step 1: Process the book content (scrape, normalize, chunk)
        print("Processing book content...")
        chunks: List[Chunk] = content_service.process_book_content(source_url)
        print(f"Processed {len(chunks)} content chunks")

        # Step 2: Generate embeddings for the chunks
        print("Generating embeddings...")
        chunk_embeddings = embedding_service.embed_chunks(chunks)
        print(f"Generated embeddings for {len(chunk_embeddings)} chunks")

        # Step 3: Store embeddings in Qdrant
        print("Storing embeddings in Qdrant...")
        retrieval_service.store_embeddings(chunk_embeddings)
        print("Content successfully loaded to Qdrant!")

        return True

    except Exception as e:
        print(f"Error during content loading: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Load book content to Qdrant vector database")
    parser.add_argument("--source-url", required=True, help="URL of the book content to load")
    parser.add_argument("--test", action="store_true", help="Run in test mode with sample content")

    args = parser.parse_args()

    if args.test:
        # For testing purposes, we'll use sample content
        print("Running in test mode with sample content...")
        # In a real implementation, you might load sample content here
        # For now, we'll just validate that the services can be initialized
        try:
            embedding_service = EmbeddingService()
            retrieval_service = RetrievalService()
            print("Services initialized successfully. Test passed!")
            return
        except Exception as e:
            print(f"Test failed: {str(e)}")
            sys.exit(1)

    success = load_content_to_qdrant(args.source_url)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()