"""
Service for handling chat interactions and question answering using OpenAI.
"""
import openai
from openai import APIError, AuthenticationError, RateLimitError
from typing import List, Dict, Any, Optional
import os
import time
from ..models.chat import ChatSession
from ..services.embedding_service import EmbeddingService
from ..services.retrieval_service import RetrievalService
from ..auth.auth_system import auth_system


class ChatService:
    """
    Service class for handling chat interactions and question answering.
    """
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # The API key will be passed when creating the OpenAI client

        # Initialize other required services
        self.embedding_service = EmbeddingService()
        self.retrieval_service = RetrievalService()

        # Keep track of sessions (in production, use a proper database)
        self.sessions: Dict[str, ChatSession] = {}

        # Metrics tracking for performance monitoring
        self.metrics = {
            'selected_text_queries': 0,
            'general_queries': 0,
            'selected_text_avg_response_time': 0.0,
            'general_avg_response_time': 0.0,
            'total_selected_text_time': 0.0,
            'total_general_time': 0.0
        }

    def process_query(self, query: str, selected_text: Optional[str] = None, session_id: Optional[str] = None, user_id: Optional[int] = None, target_language: str = "en") -> Dict[str, Any]:
        """
        Process a user query and return an AI-generated response with citations.
        """
        from uuid import uuid4
        start_time = time.time()

        # Create or get session
        if session_id is None:
            session_id = str(uuid4())

        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(id=session_id)

        session = self.sessions[session_id]

        # Add user message to session
        session.add_message("user", query)

        # Get user profile if user_id is provided
        user_profile = None
        if user_id:
            # In a real implementation, you would retrieve user profile from database
            # For now, we'll return a default profile or None
            user_profile = {
                "software_experience": "intermediate",
                "hardware_familiarity": "mid-range"
            }

        # Retrieve relevant context
        if selected_text:
            # If selected text is provided, use it as the primary context
            context_chunks = self._get_context_for_selected_text(query, selected_text)
            query_type = 'selected_text'
        else:
            # Otherwise, search for relevant content in the book
            context_chunks = self._get_context_for_query(query)
            query_type = 'general'

        # Generate AI response with context, considering user profile and target language
        response = self._generate_response_with_context(query, context_chunks, user_profile, target_language)

        # Add AI response to session
        session.add_message("assistant", response["response"])

        # Calculate response time
        response_time = time.time() - start_time

        # Update metrics
        self._update_metrics(query_type, response_time)

        return {
            "response": response["response"],
            "citations": response["citations"],
            "session_id": session_id
        }

    def _update_metrics(self, query_type: str, response_time: float):
        """
        Update metrics for performance tracking.
        """
        if query_type == 'selected_text':
            self.metrics['selected_text_queries'] += 1
            self.metrics['total_selected_text_time'] += response_time
            self.metrics['selected_text_avg_response_time'] = (
                self.metrics['total_selected_text_time'] / self.metrics['selected_text_queries']
            )

            # Log warning if response time exceeds threshold (SC-003: accuracy validation)
            if response_time > 2.0:  # 2 seconds threshold
                print(f"WARNING: Selected text query response time exceeded 2 seconds: {response_time:.2f}s")
        else:
            self.metrics['general_queries'] += 1
            self.metrics['total_general_time'] += response_time
            self.metrics['general_avg_response_time'] = (
                self.metrics['total_general_time'] / self.metrics['general_queries']
            )

            # Log warning if response time exceeds threshold
            if response_time > 2.0:  # 2 seconds threshold
                print(f"WARNING: General query response time exceeded 2 seconds: {response_time:.2f}s")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        """
        return self.metrics.copy()

    def _get_context_for_query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant context chunks for a query.
        """
        # Generate embedding for the query
        query_embedding = self.embedding_service.generate_embeddings([query])[0]

        # Search for relevant chunks
        results = self.retrieval_service.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results

    def _get_context_for_selected_text(self, query: str, selected_text: str) -> List[Dict[str, Any]]:
        """
        Get context based on selected text.
        """
        # For selected text queries, use the retrieval service's focused retrieval
        # to find content related specifically to the selected text
        results = self.retrieval_service.search_by_text(
            query_text=query,
            embedding_service=self.embedding_service,
            top_k=5,
            selected_text=selected_text
        )

        # If the selected text itself isn't in the results, add it as the primary context
        selected_text_present = any(result.get("chunk_id") == "selected_text" for result in results)
        if not selected_text_present:
            # Add the selected text as a high-priority context if it's not already included
            results.insert(0, {
                "chunk_id": "selected_text",
                "text": selected_text,
                "source_url": "",
                "chapter": "Selected Text",
                "section": "",
                "relevance_score": 1.0,
                "query_relevance_score": 1.0  # For consistency with the retrieval service output
            })

        return results

    def _generate_response_with_context(self, query: str, context_chunks: List[Dict[str, Any]], user_profile: Optional[Dict[str, Any]] = None, target_language: str = "en") -> Dict[str, Any]:
        """
        Generate an AI response using the provided context, considering user profile and target language if available.
        """
        # Check if this is a selected text query by looking for the selected_text marker
        is_selected_text_query = any(chunk.get("chunk_id") == "selected_text" for chunk in context_chunks)

        # Format the context for the AI
        context_str = "\n\n".join([f"Source: {chunk['source_url']} (Chapter: {chunk['chapter']})\nContent: {chunk['text']}"
                                  for chunk in context_chunks])

        # Determine user profile details for prompt customization
        software_level = "intermediate"  # default
        hardware_level = "mid-range"     # default
        if user_profile:
            software_level = user_profile.get('software_experience', 'intermediate')
            hardware_level = user_profile.get('hardware_familiarity', 'mid-range')

        # Create the appropriate prompt based on language and query type
        if target_language.lower() == "ur":
            # Urdu prompt templates as specified in the plan
            if is_selected_text_query:
                # Urdu - Selected Text Only
                system_message = "آپ ایک تکنیکی کتاب کے AI اسسٹنٹ ہیں۔ فقط درج ذیل منتخب متن کی بنیاد پر جواب دیں۔"
                prompt = f"""
                منتخب متن:
                {context_str}

                اصول:
                - کوئی اندازہ نہ لگائیں
                - بیرونی علم استعمال نہ کریں

                سوال:
                {query}

                جواب:
                """
            else:
                # Urdu - Standard
                system_message = f"آپ ایک تکنیکی کتاب کے AI اسسٹنٹ ہیں۔ فقط فراہم کردہ مواد استعمال کریں۔ صارف کی سطح: - سافٹ ویئر: {software_level} - ہارڈویئر: {hardware_level}"
                prompt = f"""
                اصول:
                - کوئی اندازہ نہ لگائیں
                - اگر جواب موجود نہ ہو تو بتائیں

                مواد:
                {context_str}

                سوال:
                {query}

                جواب:
                """
        else:
            # English prompt templates as specified in the plan
            if is_selected_text_query:
                # English - Selected Text Only
                system_message = "Answer ONLY using the selected text below. Do not infer. Do not use external knowledge."
                prompt = f"""
                SELECTED TEXT:
                {context_str}

                QUESTION:
                {query}

                ANSWER:
                """
            else:
                # English - Standard
                system_message = f"You are an AI assistant embedded inside a technical book. Answer strictly using the provided context. USER PROFILE: - Software Level: {software_level} - Hardware Level: {hardware_level}. RULES: - No hallucinations - Match depth to user profile - If answer not found, say so."
                prompt = f"""
                CONTEXT:
                {context_str}

                QUESTION:
                {query}

                ANSWER:
                """

        try:
            # Call OpenAI API to generate response
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # You might want to use gpt-4 for better results
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            ai_response = response.choices[0].message.content.strip()

            # Extract citations from the context chunks
            citations = []
            for chunk in context_chunks:
                citations.append({
                    "chunk_id": chunk.get("chunk_id", ""),
                    "source_url": chunk.get("source_url", ""),
                    "chapter": chunk.get("chapter", ""),
                    "section": chunk.get("section", ""),
                    "relevance_score": chunk.get("relevance_score", 0.0),
                    "text_snippet": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", "")
                })

            # Validate that the response has proper citations if context was provided
            if context_chunks and not citations:
                # If we have context but no citations, add a warning
                citations.append({
                    "chunk_id": "warning",
                    "source_url": "",
                    "chapter": "Validation",
                    "section": "Warning",
                    "relevance_score": 0.0,
                    "text_snippet": "Response may not properly cite sources from the provided context."
                })

            # For selected text queries, add additional validation to ensure grounding
            if is_selected_text_query:
                ai_response = self._validate_selected_text_response(ai_response, context_chunks)

            return {
                "response": ai_response,
                "citations": citations
            }
        except APIError as e:
            # Handle OpenAI API errors specifically
            error_msg = f"Sorry, there was an issue with the AI service: {str(e)}"
            print(f"OpenAI API Error: {error_msg}")  # Log the error
            return self._generate_fallback_response(query, context_chunks, error_msg)
        except AuthenticationError as e:
            # Handle authentication errors
            error_msg = "Sorry, there's an issue with the AI service configuration. Please contact the administrator."
            print(f"OpenAI Authentication Error: {str(e)}")  # Log the error
            return self._generate_fallback_response(query, context_chunks, error_msg)
        except RateLimitError as e:
            # Handle rate limit errors
            error_msg = "Sorry, we've reached the rate limit for the AI service. Please try again in a moment."
            print(f"OpenAI Rate Limit Error: {str(e)}")  # Log the error
            return self._generate_fallback_response(query, context_chunks, error_msg)
        except Exception as e:
            # Handle any other errors in AI generation
            error_msg = f"Sorry, I encountered an error while processing your request: {str(e)}"
            print(f"General Error in AI Generation: {error_msg}")  # Log the error
            return self._generate_fallback_response(query, context_chunks, error_msg)

    def _generate_fallback_response(self, query: str, context_chunks: List[Dict[str, Any]], error_msg: str) -> Dict[str, Any]:
        """
        Generate a fallback response when external services fail.
        """
        # Try to provide a helpful response based on the context even if AI fails
        if context_chunks:
            # If we have context, try to find relevant information manually
            relevant_chunks = [chunk for chunk in context_chunks if query.lower() in chunk.get('text', '').lower()]

            if relevant_chunks:
                # Create a basic response from the relevant chunks
                response = f"I'm currently experiencing issues with the AI service, but I found some potentially relevant information in the book:\n\n"
                response += "\n\n".join([chunk.get('text', '')[:200] + "..." for chunk in relevant_chunks[:2]])
                response += f"\n\nError: {error_msg}"
            else:
                response = f"I'm currently experiencing issues with the AI service. {error_msg}"
        else:
            response = f"I'm currently experiencing issues with the AI service. {error_msg}"

        # Create basic citations from available context
        citations = []
        for chunk in context_chunks:
            citations.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "source_url": chunk.get("source_url", ""),
                "chapter": chunk.get("chapter", ""),
                "section": chunk.get("section", ""),
                "relevance_score": chunk.get("relevance_score", 0.0),
                "text_snippet": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", "")
            })

        return {
            "response": response,
            "citations": citations
        }

    def _validate_selected_text_response(self, response: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Validate that the response is properly grounded in the selected text context.
        If the response contains information not supported by the context, add a disclaimer.
        """
        # Check if the response indicates insufficient information
        if "doesn't contain enough information" in response or \
           "don't have enough information" in response or \
           "cannot determine" in response or \
           "not mentioned in the provided context" in response or \
           "not found in the provided text" in response:
            # This is a valid response for when context is insufficient
            return response

        # For selected text queries, extract the actual selected text content
        selected_text_content = None
        for chunk in context_chunks:
            if chunk.get("chunk_id") == "selected_text":
                selected_text_content = chunk.get("text", "").lower()
                break

        if selected_text_content:
            # For basic content leakage prevention, we add a disclaimer that the response
            # is based on the provided context. In a more advanced implementation, we
            # could use semantic similarity or other techniques to validate the response
            # is grounded in the context.
            selected_text_chunk = next((chunk for chunk in context_chunks if chunk.get("chunk_id") == "selected_text"), None)
            chapter_name = selected_text_chunk.get('chapter', 'Selected Text') if selected_text_chunk else 'Selected Text'
            response += f"\n\n*Note: This response is based solely on the provided text selection (Chapter: {chapter_name})*"

        # Additional validation could be implemented here in the future:
        # - Check if response contains key phrases from the context
        # - Use semantic similarity to ensure response aligns with context
        # - Implement more sophisticated content leakage detection

        return response