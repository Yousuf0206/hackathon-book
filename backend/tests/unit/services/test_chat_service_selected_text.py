"""
Unit tests for selected-text question answering functionality in chat service using OpenRouter.
"""
import pytest
from unittest.mock import Mock, patch
from backend.src.services.chat_service import ChatService


class TestChatServiceSelectedText:
    """Test cases for selected-text functionality in ChatService."""

    @patch('backend.src.services.chat_service.openai.OpenAI')
    @patch('backend.src.services.chat_service.EmbeddingService')
    @patch('backend.src.services.chat_service.RetrievalService')
    def test_process_query_with_selected_text(self, mock_retrieval_service, mock_embedding_service, mock_openai_class):
        """Test processing a query with selected text context using OpenRouter."""
        # Setup
        chat_service = ChatService()

        # Mock the OpenAI client and its chat.completions.create method
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'This is a test response based on selected text.'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Mock embedding service
        mock_embedding_service.return_value.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        # Mock retrieval service
        mock_retrieval_service.return_value.search_by_text.return_value = [
            {
                "chunk_id": "selected_text",
                "text": "This is the selected text that the user highlighted.",
                "source_url": "test_url",
                "chapter": "Test Chapter",
                "section": "Test Section",
                "relevance_score": 1.0,
                "query_relevance_score": 0.9
            }
        ]

        # Execute
        result = chat_service.process_query(
            query="What does this selected text mean?",
            selected_text="This is the selected text that the user highlighted."
        )

        # Assert
        assert "response" in result
        assert "citations" in result
        assert "session_id" in result
        assert len(result["citations"]) > 0
        # Verify that the retrieval service was called with the selected text
        mock_retrieval_service.return_value.search_by_text.assert_called_once()

    @patch('backend.src.services.chat_service.openai.OpenAI')
    @patch('backend.src.services.chat_service.EmbeddingService')
    @patch('backend.src.services.chat_service.RetrievalService')
    def test_process_query_without_selected_text(self, mock_retrieval_service, mock_embedding_service, mock_openai_class):
        """Test processing a query without selected text (normal mode) using OpenRouter."""
        # Setup
        chat_service = ChatService()

        # Mock the OpenAI client and its chat.completions.create method
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'This is a test response based on general book content.'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Mock embedding service
        mock_embedding_service.return_value.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        # Mock retrieval service
        mock_retrieval_service.return_value.search.return_value = [
            {
                "chunk_id": "chunk_123",
                "text": "This is relevant book content.",
                "source_url": "test_url",
                "chapter": "Test Chapter",
                "section": "Test Section",
                "relevance_score": 0.8
            }
        ]

        # Execute
        result = chat_service.process_query(
            query="What is this book about?",
            selected_text=None
        )

        # Assert
        assert "response" in result
        assert "citations" in result
        assert "session_id" in result
        # Verify that the normal search method was called (not the selected text method)
        mock_retrieval_service.return_value.search.assert_called_once()

    def test_get_context_for_selected_text(self):
        """Test the _get_context_for_selected_text method."""
        # This test would require more complex mocking of internal methods
        # For now, we'll just verify the method exists and signature
        chat_service = ChatService()
        # The method should exist
        assert hasattr(chat_service, '_get_context_for_selected_text')

    def test_validate_selected_text_response_with_insufficient_info(self):
        """Test validation when response indicates insufficient information."""
        chat_service = ChatService()

        # Test response that indicates insufficient info
        response = "The provided context doesn't contain enough information to answer this question."
        context_chunks = [
            {
                "chunk_id": "selected_text",
                "text": "Sample selected text content.",
                "source_url": "",
                "chapter": "Selected Text",
                "section": "",
                "relevance_score": 1.0
            }
        ]

        result = chat_service._validate_selected_text_response(response, context_chunks)

        # Should return the response as-is since it indicates insufficient info
        assert result == response

    def test_validate_selected_text_response_with_content(self):
        """Test validation adds disclaimer for responses with content."""
        chat_service = ChatService()

        # Test response with actual content
        response = "Based on the selected text, the answer is X."
        context_chunks = [
            {
                "chunk_id": "selected_text",
                "text": "Sample selected text content.",
                "source_url": "test_url",
                "chapter": "Chapter 1",
                "section": "Section A",
                "relevance_score": 1.0
            }
        ]

        result = chat_service._validate_selected_text_response(response, context_chunks)

        # Should add a disclaimer note
        assert "*Note: This response is based solely on the provided text selection*" in result
        assert "(Chapter: Chapter 1)" in result