---
id: 1
title: Create spec for RAG Chatbot Integration
stage: spec
date: 2025-12-10
surface: agent
model: claude-haiku-4-5-20251001
feature: rag-chatbot-integration
branch: 001-rag-chatbot-integration
user: user
command: /sp.specify
labels: [spec, rag, chatbot, ai]
links:
  spec: specs/001-rag-chatbot-integration/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/001-rag-chatbot-integration/spec.md
 - specs/001-rag-chatbot-integration/checklists/requirements.md
tests:
 - specs/001-rag-chatbot-integration/checklists/requirements.md
---

## Prompt

— RAG Chatbot Integration for AI/Spec-Driven Book

## Project Context
This project is Phase 2 of the Spec-Driven Book Creation initiative.
Phase 1 (Book writing + Docusaurus + GitHub Pages) is complete.
Now we are adding a fully functional **RAG Chatbot** integrated directly into the published book.

The chatbot must:
- Answer questions based on the entire book
- Answer questions based on **user-selected text only**
- Use OpenAI Agents SDK + FastAPI
- Use Neon Serverless Postgres + Qdrant Cloud
- Use Cohere embeddings for vectorization

---

# ✅ Spec 1 — Website Deployment URL → Embedding Generation → Qdrant Storage

### Objective
Extract the full book content from the deployed website URL, generate embeddings using **Cohere**, and store them in **Qdrant Cloud**.

### Requirements
- Crawl or fetch static content from the GitHub Pages Docusaurus website.
- Normalize content into clean text blocks (≈ 200–800 tokens per chunk).
- Generate embeddings using:
  - Model: `Cohere embed-multilingual-v3.0` or higher.
- Store vectors in Qdrant Cloud with metadata:
  - `chunk_text`
  - `source_url`
  - `chapter`
  - `position`
- Ensure idempotent ingestion (re-running should not duplicate vectors).

### Success Criteria
- 100% of book content stored in Qdrant.
- Vector count matches expected chunk count.
- No content duplication or empty chunks.
- Retrieval test returns relevant chunks.

---

# ✅ Spec 2 — Pipeline Validation & Quality Testing

### Objective
Verify that ingestion, embedding, and retrieval pipeline works correctly.

### Required Tests
- **Chunk quality test**: ensure chunk boundaries preserve meaning.
- **Embedding density test**: verify embedding vectors have expected shape.
- **Retrieval sanity test**:
  - Query: "What is the book about?"
  - Expect: High-similarity chunks from introduction.
- **Selected-text test**:
  - Pass user-selected text directly → retrieve related chunks → ensure relevance score exceeds threshold.
- **Latency test**:
  - Goal: Retrieval under 300ms Qdrant round-trip.

### Success Criteria
- All test cases pass with expected scores.
- Retrieval relevance > 0.75 similarity baseline.
- Chunking adjustments applied if failures occur.

---

# ✅ Spec 3 — Build OpenAI Agent + FastAPI Backend with Retrieval

### Objective
Create a retrieval-augmented agent using OpenAI Agents SDK and expose it via FastAPI.

### System Responsibilities
- Accept user queries
- Retrieve relevant vectors from Qdrant
- Construct augmented context:
  `user query + top-k chunks`
- Pass context to OpenAI Agent
- Return structured JSON response to frontend

### API Endpoints
- `POST /ask`: general question answering
- `POST /ask/selected`: question answering on selected text only
- `GET /health`: service heartbeat
- `POST /admin/reload`: reload vector index

### Internal Components
- Retrieval module
- Context builder
- Prompt templates
- Safety guards (max tokens, content filters)
- Async streaming output support

### Success Criteria
- Agent answers book questions consistently and accurately.
- Selected-text mode isolates itself to given input snippet.
- API response times < 800ms under normal load.

---

# ✅ Spec 4 — Frontend ↔ Backend Integration (Local + Production)

### Objective
Integrate FastAPI backend with the existing Docusaurus book frontend.

### Integration Tasks
- Establish local dev connection:
  - Docusaurus → FastAPI (localhost)
- Build production proxy (Cloudflare / GitHub Pages compatible)
- Create UI components:
  - Floating chatbot widget
  - Sidebar "Ask the Book" panel
  - Highlight-and-ask popover
- Implement interactive UX:
  - Text selection → send snippet to backend
  - Display streaming responses
  - Show source citations returned by agent

### Technical Expectations
- Use Fetch or Axios
- Handle CORS for GitHub Pages → API domain
- Include graceful error fallback:
  - Offline mode
  - No results found
  - Backend timeout

### Success Criteria
- Chatbot loads immediately on page visit.
- Selected-text questions work flawlessly.
- Respons

## Response snapshot

# Feature Specification: RAG Chatbot Integration for AI/Spec-Driven Book

**Feature Branch**: `001-rag-chatbot-integration`
**Created**: 2025-12-10
**Status**: Draft
**Input**: User description: "RAG Chatbot Integration for AI/Spec-Driven Book"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Book Question Answering (Priority: P1)

A book reader wants to ask questions about the book content and receive accurate, contextually relevant answers based on the entire book. The user types a question in the chat interface and receives a response with citations to the relevant book sections.

**Why this priority**: This is the core functionality that provides the primary value of the RAG system - enabling users to understand and interact with the book content through natural language queries.

**Independent Test**: User can ask a question about the book content and receive an accurate response with source citations within 2 seconds.

**Acceptance Scenarios**:

1. **Given** user is viewing the book, **When** user types a question about book content and submits it, **Then** the system returns an accurate answer with source citations from the book
2. **Given** user has a complex question spanning multiple book chapters, **When** user submits the question, **Then** the system returns a comprehensive answer synthesizing information from multiple sources

---

### User Story 2 - Selected Text Question Answering (Priority: P2)

A book reader wants to ask questions specifically about a selected portion of text in the book. The user highlights text in the book, asks a question about it, and receives an answer based only on that specific text.

**Why this priority**: This provides an advanced interaction model that allows users to deeply analyze specific sections of the book content.

**Independent Test**: User can select text, ask a question about it, and receive an answer based only on the selected text within 2 seconds.

**Acceptance Scenarios**:

1. **Given** user has selected text in the book, **When** user asks a question about the selected text, **Then** the system returns an answer based only on the selected text with relevant citations
2. **Given** user has selected text with multiple concepts, **When** user asks a specific question, **Then** the system provides a focused answer based on the selected context

---

### User Story 3 - Interactive Chat Interface (Priority: P3)

A book reader wants to have a conversational experience with the book content through a user-friendly chat interface that displays answers with source citations and allows follow-up questions.

**Why this priority**: This provides the essential UI/UX layer that makes the RAG functionality accessible and useful for readers.

**Independent Test**: User can interact with the chat interface, see streaming responses, and view source citations for the answers.

**Acceptance Scenarios**:

1. **Given** user has opened the chat interface, **When** user submits a question, **Then** the system displays a streaming response with source citations
2. **Given** user has received an answer, **When** user asks a follow-up question, **Then** the system maintains context and provides a relevant response

---

### Edge Cases

- What happens when the book content is not available or the vector database is unreachable?
- How does the system handle ambiguous or off-topic questions?
- How does the system respond when no relevant content is found for a user's question?
- What happens when the AI service is temporarily unavailable?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow users to ask questions about the book content and receive accurate answers
- **FR-002**: System MUST support both general book questions and questions about user-selected text only
- **FR-003**: System MUST retrieve relevant book content using vector search against stored embeddings
- **FR-004**: System MUST generate embeddings from book content using Cohere embedding models
- **FR-005**: System MUST store book content embeddings in a vector database (Qdrant Cloud)
- **FR-006**: System MUST provide source citations for all answers returned to users
- **FR-007**: System MUST handle text selection and context-specific queries
- **FR-008**: System MUST provide a web-based chat interface integrated with the book frontend
- **FR-009**: System MUST support streaming responses for better user experience
- **FR-010**: System MUST implement proper error handling for unavailable services
- **FR-011**: System MUST be idempotent in content ingestion to prevent duplication
- **FR-012**: System MUST normalize book content into appropriate text chunks (200-800 tokens)
- **FR-013**: System MUST support retrieval latency under 300ms for Qdrant round-trip
- **FR-014**: System MUST ensure retrieval relevance score exceeds 0.75 similarity baseline
- **FR-015**: System MUST provide API endpoints for general and selected-text question answering

### Key Entities

- **Book Content Chunk**: A segment of book text with associated metadata (source URL, chapter, position) used for retrieval
- **Vector Embedding**: Numerical representation of text content used for semantic similarity search
- **User Query**: A question or request from the user that requires information from the book
- **AI Response**: The generated answer to a user query with relevant citations to book content
- **Selected Text Context**: A specific portion of book content that the user has highlighted for focused questioning

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of book content is successfully stored in the vector database with proper metadata
- **SC-002**: User queries return relevant answers with source citations within 800ms under normal load
- **SC-003**: Text selection queries return answers based only on selected content with 95% accuracy
- **SC-004**: System achieves >75% relevance score on retrieved content compared to human-identified relevant sections
- **SC-005**: Users can successfully ask questions and receive answers in 95% of attempts (availability)
- **SC-006**: Book content is chunked with semantic boundaries preserved (quality of chunking)
- **SC-007**: The chat interface loads and is responsive on all major browsers and devices
- **SC-008**: Retrieval round-trip time to Qdrant remains under 300ms
- **SC-009**: The chatbot is available and responsive immediately on page visit

## Outcome

- ✅ Impact: Created comprehensive specification for RAG chatbot integration feature
- 🧪 Tests: Created requirements checklist to validate spec quality
- 📁 Files: Created spec.md and requirements checklist in feature directory
- 🔁 Next prompts: Ready for planning phase with `/sp.plan`
- 🧠 Reflection: Specification captures all key requirements from user description while maintaining focus on user value

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS - All validation criteria met
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Proceed to planning phase