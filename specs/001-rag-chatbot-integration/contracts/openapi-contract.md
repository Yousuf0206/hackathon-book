# OpenAPI Contract: RAG Chatbot API

## /chat (POST)

**Purpose**: Process user queries and return AI-generated responses with source citations

### Request

```json
{
  "query": "What is the main concept discussed in chapter 3?",
  "selected_text": "Optional text selected by user for context-specific queries",
  "session_id": "Optional session identifier for conversation context"
}
```

### Response (Success 200)

```json
{
  "response": "The main concept discussed in chapter 3 is...",
  "citations": [
    {
      "chunk_id": "unique-identifier",
      "source_url": "https://book-url.com/chapter3",
      "chapter": "Chapter 3",
      "section": "3.2 Key Concepts",
      "relevance_score": 0.85,
      "text_snippet": "Original text that supports the response..."
    }
  ],
  "session_id": "session-identifier",
  "response_time_ms": 450
}
```

## /query (POST)

**Purpose**: Retrieve relevant book content chunks based on user query

### Request

```json
{
  "query": "Explain the concept of RAG systems",
  "top_k": 5,
  "selected_text": "Optional context for focused retrieval"
}
```

### Response (Success 200)

```json
{
  "query": "Explain the concept of RAG systems",
  "results": [
    {
      "chunk_id": "unique-identifier",
      "text": "Retrieved text content...",
      "source_url": "https://book-url.com/chapter2",
      "chapter": "Chapter 2",
      "section": "2.1 RAG Fundamentals",
      "relevance_score": 0.92
    }
  ],
  "retrieval_time_ms": 120
}
```

## /embed (POST)

**Purpose**: Generate embeddings for book content and store in vector database

### Request

```json
{
  "chunks": [
    {
      "text": "Text content to embed",
      "metadata": {
        "source_url": "https://book-url.com/chapter1",
        "chapter": "Chapter 1",
        "section": "1.1 Introduction",
        "position": 1
      }
    }
  ]
}
```

### Response (Success 200)

```json
{
  "processed_chunks": 15,
  "failed_chunks": 0,
  "processing_time_ms": 2500
}
```

## /health (GET)

**Purpose**: Check the health status of the API and its dependencies

### Response (Success 200)

```json
{
  "status": "healthy",
  "timestamp": "2025-12-10T10:00:00Z",
  "dependencies": {
    "qdrant": "connected",
    "cohere": "available",
    "openai": "available"
  }
}
```

## Error Responses

All endpoints follow this error response format (4xx/5xx):

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Optional technical details"
  }
}
```

### Common Error Codes

- `INVALID_INPUT`: Request data doesn't match schema
- `RATE_LIMIT_EXCEEDED`: User has exceeded rate limits
- `EXTERNAL_SERVICE_UNAVAILABLE`: Dependency (Qdrant, Cohere, OpenAI) unavailable
- `NO_RELEVANT_CONTENT`: No suitable content found for the query
- `INTERNAL_ERROR`: Unexpected server error