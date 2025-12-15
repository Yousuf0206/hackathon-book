# Data Model: RAG Chatbot Integration

## Entities

### BookContentChunk
- **id**: string (UUID) - Primary identifier for the chunk
- **text**: string - The actual text content of the chunk
- **embedding**: float[] - Vector embedding representation of the text
- **metadata**: object
  - **source_url**: string - URL where this content originated
  - **chapter**: string - Chapter title/identifier
  - **section**: string - Section within the chapter
  - **position**: integer - Position order within the document
  - **created_at**: datetime - When the chunk was created
- **checksum**: string - For idempotent ingestion (prevent duplicates)

### UserQuery
- **id**: string (UUID) - Primary identifier
- **query_text**: string - The user's original question
- **selected_text**: string (optional) - Text selected by user for context-specific queries
- **created_at**: datetime - When the query was made
- **user_id**: string (optional) - Identifier for the user (for rate limiting)
- **session_id**: string (optional) - For conversation context

### AIResponse
- **id**: string (UUID) - Primary identifier
- **query_id**: string - Reference to the original query
- **response_text**: string - The AI-generated response
- **source_citations**: array of objects
  - **chunk_id**: string - Reference to the BookContentChunk
  - **relevance_score**: float - How relevant this chunk was to the response
  - **text_snippet**: string - Snippet of the original text
  - **source_url**: string - URL of the original content
  - **chapter**: string - Chapter where the content appears
- **created_at**: datetime - When the response was generated
- **response_time_ms**: integer - How long the response took to generate

### ChatSession
- **id**: string (UUID) - Primary identifier
- **created_at**: datetime - When the session started
- **updated_at**: datetime - When the session was last updated
- **user_id**: string (optional) - User identifier for the session
- **messages**: array of objects
  - **id**: string (UUID) - Message identifier
  - **role**: enum (user, assistant) - Who sent the message
  - **content**: string - The message content
  - **timestamp**: datetime - When the message was sent

## Validation Rules

### BookContentChunk
- text must be 200-800 tokens
- embedding must be the correct dimension for the chosen model
- source_url must be a valid URL
- checksum must be unique to prevent duplication

### UserQuery
- query_text must not be empty
- query_text must be less than 1000 characters
- selected_text, if provided, must be less than 5000 characters

### AIResponse
- response_text must not be empty
- source_citations must contain at least one citation when response is generated from book content
- relevance_score must be between 0 and 1

### ChatSession
- Cannot have more than 50 messages without being archived
- Sessions older than 24 hours may be cleaned up

## State Transitions

### ChatSession
- **Active**: New session created, ready to receive messages
- **Inactive**: No activity for more than 1 hour
- **Archived**: Session completed and data preserved for analytics